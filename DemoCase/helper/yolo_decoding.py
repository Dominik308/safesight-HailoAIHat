import numpy as np
import cv2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def smart_sigmoid(x):
    """
    Apply sigmoid only if data appears to be logits (not already probabilities).
    Heuristic: if values are outside [0, 1], they are definitely logits.
    """
    if x.size == 0: return x
    if np.min(x) < 0 or np.max(x) > 1.0:
        return 1 / (1 + np.exp(-x))
    return x

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def decode_yolov5_segmentation(outputs, input_shape, conf_threshold=0.25, iou_threshold=0.45):
    """
    Decode raw YOLOv5 segmentation outputs
    outputs: list of tensors from Hailo
    input_shape: (H, W) of the model input (e.g. 640, 640)
    """
    # Identify the tensors
    proto_tensor = None
    
    # Standard YOLOv5n-seg anchors
    anchors = {
        8:  np.array([[10,13], [16,30], [33,23]]),   # P3
        16: np.array([[30,61], [62,45], [59,119]]),  # P4
        32: np.array([[116,90], [156,198], [373,326]]) # P5
    }
    
    # Find prototype mask
    for out in outputs:
        # Handle batch dimension
        if len(out.shape) == 4 and out.shape[0] == 1:
            out = out[0]
            
        if len(out.shape) == 3:
            if out.shape[0] == 32: # CHW
                proto_tensor = np.transpose(out, (1, 2, 0))
            elif out.shape[2] == 32: # HWC
                proto_tensor = out
    
    if proto_tensor is None:
        return None, None
        
    # Find detection heads
    # Expecting shapes like (20,20,351), (40,40,351), (80,80,351)
    # 351 = 3 * (5 + 80 + 32)
    heads = []
    for out in outputs:
        # Handle batch dimension
        if len(out.shape) == 4 and out.shape[0] == 1:
            out = out[0]
            
        if len(out.shape) == 3 and out.shape[2] == 351:
            stride = input_shape[0] // out.shape[0]
            heads.append((out, stride))
            
    if not heads:
        return None, None
        
    all_dets = []
    
    for output, stride in heads:
        grid_h, grid_w, _ = output.shape
        anchor_grid = anchors.get(stride)
        if anchor_grid is None: continue
        
        # Reshape to (grid_h, grid_w, 3, 117)
        # 117 = 4(box) + 1(conf) + 80(cls) + 32(mask)
        output = output.reshape(grid_h, grid_w, 3, 117)
        
        # Extract features
        # xy: sigmoid * 2 - 0.5 + grid
        # wh: (sigmoid * 2) ** 2 * anchor
        # conf: sigmoid
        # cls: sigmoid
        # mask: raw
        
        # Create grid
        yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
        grid = np.stack((xv, yv), axis=2).reshape(grid_h, grid_w, 1, 2)
        
        # Decode xy
        xy = (sigmoid(output[..., 0:2]) * 2 - 0.5 + grid) * stride
        
        # Decode wh
        wh = (sigmoid(output[..., 2:4]) * 2) ** 2 * anchor_grid.reshape(1, 1, 3, 2)
        
        # Decode conf
        conf = sigmoid(output[..., 4:5])
        
        # Decode class
        cls_probs = sigmoid(output[..., 5:85])
        
        # Mask coeffs
        mask_coeffs = output[..., 85:]
        
        # Flatten
        xy = xy.reshape(-1, 2)
        wh = wh.reshape(-1, 2)
        conf = conf.reshape(-1)
        cls_probs = cls_probs.reshape(-1, 80)
        mask_coeffs = mask_coeffs.reshape(-1, 32)
        
        # Filter by confidence
        # Objectness * Max Class Prob
        cls_max = np.max(cls_probs, axis=1)
        cls_idx = np.argmax(cls_probs, axis=1)
        scores = conf * cls_max
        
        mask = scores > conf_threshold
        if not np.any(mask):
            continue
            
        # Gather detections
        # [x, y, w, h, score, class_id, mask_coeffs...]
        xy_valid = xy[mask]
        wh_valid = wh[mask]
        scores_valid = scores[mask]
        cls_valid = cls_idx[mask]
        masks_valid = mask_coeffs[mask]
        
        # Convert xywh to xyxy
        x1 = xy_valid[:, 0] - wh_valid[:, 0] / 2
        y1 = xy_valid[:, 1] - wh_valid[:, 1] / 2
        x2 = xy_valid[:, 0] + wh_valid[:, 0] / 2
        y2 = xy_valid[:, 1] + wh_valid[:, 1] / 2
        
        dets = np.column_stack([x1, y1, x2, y2, scores_valid, cls_valid])
        dets = np.hstack([dets, masks_valid])
        
        all_dets.append(dets)
        
    if not all_dets:
        return proto_tensor, np.zeros((0, 38))
        
    all_dets = np.vstack(all_dets)
    
    # NMS
    keep = nms(all_dets[:, :4], all_dets[:, 4], iou_threshold)
    final_dets = all_dets[keep]
    
    return proto_tensor, final_dets

def nms(boxes, scores, iou_threshold):
    # Basic NMS implementation
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

def decode_yolov8_pose(outputs, input_shape, conf_threshold=0.25, iou_threshold=0.45):
    """
    Decode raw YOLOv8 Pose outputs (DFL + Keypoints)
    outputs: list of tensors from Hailo
    input_shape: (H, W) of the model input (e.g. 640, 640)
    """
    # Group tensors by scale
    # We expect 3 scales: 80x80 (P3), 40x40 (P4), 20x20 (P5)
    # Each scale has 3 tensors: Box (64), Cls (1), Kpts (51)
    
    heads = {} # stride -> {'box': tensor, 'cls': tensor, 'kpts': tensor}
    
    for out in outputs:
        # Handle batch dimension
        if len(out.shape) == 4 and out.shape[0] == 1:
            out = out[0]
            
        h, w, c = out.shape
        stride = input_shape[0] // h
        
        if stride not in heads:
            heads[stride] = {}
            
        if c == 64:
            heads[stride]['box'] = out
        elif c == 1:
            heads[stride]['cls'] = out
        elif c == 51:
            heads[stride]['kpts'] = out
            
    # Check if we have complete heads
    valid_strides = []
    for stride, components in heads.items():
        if 'box' in components and 'cls' in components and 'kpts' in components:
            valid_strides.append(stride)
            
    if not valid_strides:
        return None
        
    all_dets = []
    reg_max = 16
    
    # DFL integration range
    dfl_conv = np.arange(reg_max).reshape(1, reg_max)
    
    for stride in valid_strides:
        box_tensor = heads[stride]['box'] # (h, w, 64)
        cls_tensor = heads[stride]['cls'] # (h, w, 1)
        kpts_tensor = heads[stride]['kpts'] # (h, w, 51)
        
        h, w, _ = box_tensor.shape
        
        # Create grid
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid = np.stack((xv, yv), axis=2) # (h, w, 2)
        
        # Decode Class
        # Use smart_sigmoid to handle both logits and probabilities
        cls_scores = smart_sigmoid(cls_tensor) # (h, w, 1)
        
        # Filter by confidence early to save computation
        mask = cls_scores[..., 0] > conf_threshold
        if not np.any(mask):
            continue
            
        # Extract valid items
        grid_valid = grid[mask]
        box_valid = box_tensor[mask] # (N, 64)
        cls_valid = cls_scores[mask] # (N, 1)
        kpts_valid = kpts_tensor[mask] # (N, 51)
        
        # Decode Box (DFL)
        # Reshape to (N, 4, 16)
        box_valid = box_valid.reshape(-1, 4, reg_max)
        # Softmax
        box_valid = softmax(box_valid, axis=2)
        # Dot product with range
        dist = np.dot(box_valid, dfl_conv.T).reshape(-1, 4) # (N, 4)
        
        # dist is (l, t, r, b) relative to anchor center
        # x1 = grid_x - l
        # y1 = grid_y - t
        # x2 = grid_x + r
        # y2 = grid_y + b
        
        lt = dist[:, :2]
        rb = dist[:, 2:]
        
        x1y1 = (grid_valid - lt) * stride
        x2y2 = (grid_valid + rb) * stride
        
        boxes = np.hstack([x1y1, x2y2]) # (N, 4) xyxy
        
        # Decode Keypoints
        # Format: (x, y, vis) * 17
        kpts_valid = kpts_valid.reshape(-1, 17, 3)
        
        # Use raw values for x, y (unbounded regression) matching Ultralytics YOLOv8 Pose
        # Formula: (x * 2 + grid) * stride
        # We do NOT apply sigmoid to x/y as they need to be unbounded for limbs
        
        kpt_x = (kpts_valid[..., 0] * 2 + grid_valid[:, 0:1]) * stride
        kpt_y = (kpts_valid[..., 1] * 2 + grid_valid[:, 1:2]) * stride
        kpt_vis = sigmoid(kpts_valid[..., 2])
        
        kpts_decoded = np.stack([kpt_x, kpt_y, kpt_vis], axis=2).reshape(-1, 51)
        
        # Combine: [x1, y1, x2, y2, score, class_id(0), kpts(51)]
        # Total 4 + 1 + 1 + 51 = 57
        
        batch_dets = np.hstack([
            boxes, 
            cls_valid, 
            np.zeros_like(cls_valid), # Class 0
            kpts_decoded
        ])
        
        all_dets.append(batch_dets)
        
    if not all_dets:
        return np.zeros((0, 57))
        
    all_dets = np.vstack(all_dets)
    
    # NMS
    keep = nms(all_dets[:, :4], all_dets[:, 4], iou_threshold)
    final_dets = all_dets[keep]
    
    return final_dets
