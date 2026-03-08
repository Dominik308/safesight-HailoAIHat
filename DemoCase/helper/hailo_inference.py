"""
Hailo inference wrapper for .hef models
Provides interface compatible with YOLO results
"""
import numpy as np
import cv2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, 
                            InferVStreams, ConfigureParams, FormatType,
                            InputVStreamParams, OutputVStreamParams)


class HailoInference:
    """Wrapper for Hailo inference with .hef models"""
    
    def __init__(self, hef_path, batch_size=1, rotation=180):
        """
        Initialize Hailo inference
        
        Args:
            hef_path: Path to .hef model file
            batch_size: Batch size for inference (default: 1)
            rotation: Rotation angle in degrees (0, 90, 180, 270) to apply to bounding boxes
        """
        self.hef_path = hef_path
        self.batch_size = batch_size
        self.rotation = rotation  # Camera rotation angle
        self.target = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        
        # Model metadata
        self.input_shape = None
        self.output_shapes = []
        self.class_names = {}  # Will be populated based on model type
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Hailo device and load model"""
        # Load HEF
        self.hef = HEF(self.hef_path)
        
        # Get default device (Hailo-8)
        self.target = VDevice()
        
        # Configure network group
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        # Get input/output stream parameters
        input_vstreams_info = self.hef.get_input_vstream_infos()
        output_vstreams_info = self.hef.get_output_vstream_infos()
        
        # Check what format the model expects
        print(f"Input vstream info:")
        for info in input_vstreams_info:
            print(f"  Name: {info.name}, Shape: {info.shape}, Format: {info.format}")
        
        print(f"Output vstream info:")
        for info in output_vstreams_info:
            print(f"  Name: {info.name}, Shape: {info.shape}, Format: {info.format}")
        
        # Try with quantized format first (common for Hailo models)
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, quantized=True, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        
        # Get input/output info
        self.input_vstream_info = input_vstreams_info[0]
        self.output_vstream_info = output_vstreams_info
        
        # Get input shape (height, width, channels)
        self.input_shape = self.input_vstream_info.shape
        
        # Get output shapes
        for output_info in self.output_vstream_info:
            self.output_shapes.append(output_info.shape)
        
        print(f"Hailo model loaded: {self.hef_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output shapes: {self.output_shapes}")
        
        # Initialize class names (COCO dataset default)
        self._init_class_names()
    
    def _init_class_names(self):
        """Initialize class names for COCO dataset"""
        # Standard COCO class names
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
    
    def preprocess(self, frame):
        """
        Preprocess frame for Hailo inference
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Preprocessed frame ready for inference, and padding info
        """
        h, w, c = self.input_shape
        orig_h, orig_w = frame.shape[:2]
        
        # Calculate scaling to fit inside model input while maintaining aspect ratio
        scale = min(w / orig_w, h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize maintaining aspect ratio
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create padded image (letterbox)
        padded = np.ones((h, w, c), dtype=np.uint8) * 114  # Gray padding
        
        # Calculate padding offsets (center the image)
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension [batch, height, width, channels]
        batched = np.expand_dims(rgb, axis=0)
        
        # Store preprocessing info for postprocessing
        self.preprocess_info = {
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'orig_w': orig_w,
            'orig_h': orig_h
        }
        
        return batched
    
    def transform_point(self, x, y):
        """Transform normalized model coordinates to original frame pixel coordinates"""
        # x, y are normalized [0, 1] in model space
        
        model_h, model_w = self.input_shape[:2]
        scale = self.preprocess_info['scale']
        pad_x = self.preprocess_info['pad_x']
        pad_y = self.preprocess_info['pad_y']
        
        # Convert to model pixels
        px = x * model_w
        py = y * model_h
        
        # Remove padding
        px = px - pad_x
        py = py - pad_y
        
        # Scale back
        px = px / scale
        py = py / scale
        
        return int(px), int(py)

    def process_mask(self, mask):
        """Process segmentation mask to match original frame"""
        # mask is (mask_h, mask_w) - might be smaller than input_shape
        mask_h, mask_w = mask.shape[:2]
        input_h, input_w = self.input_shape[:2]
        
        scale = self.preprocess_info['scale']
        pad_x = self.preprocess_info['pad_x']
        pad_y = self.preprocess_info['pad_y']
        orig_w = self.preprocess_info['orig_w']
        orig_h = self.preprocess_info['orig_h']
        
        # Calculate scaling factor between mask and input
        # (Segmentation output might be smaller, e.g. 160x160 vs 640x640)
        ratio_x = mask_w / input_w
        ratio_y = mask_h / input_h
        
        # Scale padding and dimensions to mask space
        mask_pad_x = int(pad_x * ratio_x)
        mask_pad_y = int(pad_y * ratio_y)
        
        # Calculate valid region in mask (excluding padding)
        # The image was resized to (new_w, new_h) and placed at (pad_x, pad_y)
        # We need to calculate new_w/new_h in mask space
        mask_new_w = int(orig_w * scale * ratio_x)
        mask_new_h = int(orig_h * scale * ratio_y)
        
        # Crop the valid region from the mask
        y1 = max(0, mask_pad_y)
        y2 = min(mask_h, mask_pad_y + mask_new_h)
        x1 = max(0, mask_pad_x)
        x2 = min(mask_w, mask_pad_x + mask_new_w)
        
        if y2 <= y1 or x2 <= x1:
            return np.zeros((orig_h, orig_w), dtype=mask.dtype)
            
        cropped = mask[y1:y2, x1:x2]
        
        # Resize to original size
        resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return resized

    def postprocess(self, outputs, original_shape, conf_threshold=0.25):
        """
        Postprocess Hailo outputs to YOLO-like format
        
        Args:
            outputs: Raw outputs from Hailo inference
            original_shape: Original frame shape (h, w)
            conf_threshold: Confidence threshold for filtering detections
        
        Returns:
            HailoResult object compatible with YOLO results
        """
        boxes = []
        scores = []
        class_ids = []
        
        orig_h, orig_w = original_shape[:2]
        model_h, model_w = self.input_shape[:2]
        
        # Get preprocessing info (scale and padding)
        scale = self.preprocess_info['scale']
        pad_x = self.preprocess_info['pad_x']
        pad_y = self.preprocess_info['pad_y']
        
        # Handle different output formats
        if len(outputs) == 0:
            return HailoResult(boxes, scores, class_ids, self.class_names)
        
        # Try to find detection output
        detections = None
        for i, output in enumerate(outputs):
            # Convert list to numpy array if needed
            if isinstance(output, list):
                if len(output) > 0:
                    output = np.array(output[0]) if isinstance(output[0], (list, np.ndarray)) else np.array(output)
                else:
                    continue
            
            # Handle NMS postprocessed output: (num_classes, num_detections, 5) or (num_classes, 5, num_detections)
            # Format: [x, y, w, h, confidence] for each detection per class
            if len(output.shape) == 3:
                # Check which dimension is 5 (the features dimension)
                if output.shape[2] == 5:
                    # Format: (num_classes, num_detections, 5)
                    num_classes, num_detections, features = output.shape
                elif output.shape[1] == 5:
                    # Format: (num_classes, 5, num_detections) - need to transpose
                    output = np.transpose(output, (0, 2, 1))  # -> (num_classes, num_detections, 5)
                    num_classes, num_detections, features = output.shape
                else:
                    # Not an NMS format we recognize, continue to next check
                    if output.shape[0] == 1:
                        output = output[0]
                        # Check for YOLO output (box + conf + classes or box + conf + kpts)
                        # Pose: 4+1+51 = 56
                        # Detect: 4+1+80 = 85
                        if len(output.shape) == 2 and output.shape[1] >= 6:
                            detections = output
                            break
                    continue
                
                # Now process the NMS output
                all_detections = []
                total_raw_detections = 0
                for class_id in range(num_classes):
                    class_dets = output[class_id]  # (num_detections, 5)
                    # Filter out empty detections (where all values are 0)
                    valid_mask = np.any(class_dets != 0, axis=1)
                    valid_dets = class_dets[valid_mask]
                    
                    if len(valid_dets) > 0:
                        total_raw_detections += len(valid_dets)
                        # Add class_id column
                        class_ids_col = np.full((len(valid_dets), 1), class_id)
                        dets_with_class = np.hstack([valid_dets[:, :4], valid_dets[:, 4:5], class_ids_col])
                        all_detections.append(dets_with_class)
                
                if all_detections:
                    detections = np.vstack(all_detections)
                else:
                    return HailoResult(boxes, scores, class_ids, self.class_names)
                break
            
            # Common YOLO output formats:
            # - [1, num_boxes, 85] or [1, 85, num_boxes] (YOLO v5/v8)
            # - [1, num_boxes, 4+1+80] 
            # - Multiple outputs for different detection layers
            
            if len(output.shape) == 3:
                # Remove batch dimension if present
                if output.shape[0] == 1:
                    output = output[0]
                    
            if len(output.shape) == 2:
                # Check for transposed output [features, num_boxes]
                # Heuristic: num_boxes (anchors) is usually much larger than features
                if output.shape[0] < output.shape[1] and output.shape[0] in [56, 85, 117]:
                     output = output.T
                
                # output is [num_boxes, features]
                # Accept anything with enough features for a box (4) + conf (1) + class (1)
                if output.shape[1] >= 6:  
                    detections = output
                    break
        
        if detections is None or len(detections) == 0:
            return HailoResult(boxes, scores, class_ids, self.class_names)
        
        # Parse detections
        num_detections = 0
        for detection in detections:
            # Check if this is NMS format (6 values: coordinates + conf + class_id)
            if len(detection) == 6:
                # Hailo NMS is typically [y_min, x_min, y_max, x_max, conf, class_id]
                # Coordinates are normalized [0, 1]
                ymin, xmin, ymax, xmax, confidence, class_id = detection
                
                # Convert to pixel coordinates in MODEL space (640x640)
                x1 = xmin * model_w
                y1 = ymin * model_h
                x2 = xmax * model_w
                y2 = ymax * model_h
                
                # Remove padding offset (coordinates are in padded image space)
                x1 = x1 - pad_x
                y1 = y1 - pad_y
                x2 = x2 - pad_x
                y2 = y2 - pad_y
                
                # Scale back to original image size
                x1 = x1 / scale
                y1 = y1 / scale
                x2 = x2 / scale
                y2 = y2 / scale
                
                # Apply rotation transformation if needed
                if self.rotation == 90:
                    # 90° clockwise: (x, y) -> (height - y, x)
                    x1_rot = orig_h - y2
                    y1_rot = x1
                    x2_rot = orig_h - y1
                    y2_rot = x2
                    x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
                elif self.rotation == 180:
                    # 180°: (x, y) -> (width - x, height - y)
                    x1_rot = orig_w - x2
                    y1_rot = orig_h - y2
                    x2_rot = orig_w - x1
                    y2_rot = orig_h - y1
                    x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
                elif self.rotation == 270:
                    # 270° clockwise (90° counter-clockwise): (x, y) -> (y, width - x)
                    x1_rot = y1
                    y1_rot = orig_w - x2
                    x2_rot = y2
                    y2_rot = orig_w - x1
                    x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
                
                # Clamp final coordinates to image bounds
                # Note: after rotation, dimensions might be swapped
                max_x = orig_h if self.rotation in [90, 270] else orig_w
                max_y = orig_w if self.rotation in [90, 270] else orig_h
                x1 = max(0, min(x1, max_x))
                y1 = max(0, min(y1, max_y))
                x2 = max(0, min(x2, max_x))
                y2 = max(0, min(y2, max_y))
                
            # Standard YOLO format (85+ values) or Pose (56 values)
            elif len(detection) >= 6:
                x_center, y_center, width, height = detection[:4]
                
                # Check if we have objectness + class probs (YOLOv5/v8 Detect)
                # or just conf + keypoints (YOLOv8 Pose)
                
                if len(detection) >= 85: # Detect
                    objectness = detection[4]
                    class_probs = detection[5:85]  # 80 classes for COCO
                    class_id = np.argmax(class_probs)
                    class_conf = class_probs[class_id]
                    confidence = objectness * class_conf
                elif len(detection) >= 56: # Pose (4 box + 1 conf + 51 kpts)
                    # YOLOv8 Pose output: [x, y, w, h, conf, kpt1_x, kpt1_y, kpt1_conf, ...]
                    confidence = detection[4]
                    class_id = 0 # Person
                else:
                    # Fallback
                    confidence = detection[4]
                    class_id = 0
                
                # Convert to corner coordinates in model space
                x1 = (x_center - width / 2) * model_w
                y1 = (y_center - height / 2) * model_h
                x2 = (x_center + width / 2) * model_w
                y2 = (y_center + height / 2) * model_h
                
                # Remove padding
                x1 = x1 - pad_x
                y1 = y1 - pad_y
                x2 = x2 - pad_x
                y2 = y2 - pad_y
                
                # Scale to original
                x1 = x1 / scale
                y1 = y1 / scale
                x2 = x2 / scale
                y2 = y2 / scale
            else:
                continue
            
            if confidence < conf_threshold:
                continue
            
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            class_ids.append(class_id)
            num_detections += 1
        
        return HailoResult(boxes, scores, class_ids, self.class_names)
    
    def __call__(self, frame, conf=0.25, verbose=False):
        """
        Run inference on frame
        
        Args:
            frame: Input frame (BGR format)
            conf: Confidence threshold
            verbose: Whether to print verbose output
        
        Returns:
            List containing HailoResult object
        """
        # Preprocess
        input_data = self.preprocess(frame)
        
        if verbose:
            print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")
            print(f"Input vstream name: {self.input_vstream_info.name}")
        
        # Run inference
        with InferVStreams(self.network_group, self.input_vstreams_params, 
                          self.output_vstreams_params) as infer_pipeline:
            
            # Prepare input dictionary - pass numpy array with batch dimension
            # The array should already have batch dimension from preprocess: [1, H, W, C]
            input_dict = {self.input_vstream_info.name: input_data}
            
            # Run inference
            with self.network_group.activate(self.network_group_params):
                outputs = infer_pipeline.infer(input_dict)
        
        # Store raw outputs for pose/segmentation visualization
        self._last_outputs = {}
        for info in self.output_vstream_info:
            if isinstance(outputs, dict):
                self._last_outputs[info.name] = outputs[info.name]
        
        # Convert outputs to list - handle both dict and list returns
        output_list = []
        for info in self.output_vstream_info:
            if isinstance(outputs, dict):
                raw_output = outputs[info.name]
                
                # Handle list of arrays (batched output)
                if isinstance(raw_output, list):
                    if len(raw_output) > 0:
                        # Get first batch element
                        output_array = raw_output[0]
                        
                        # Check if this is a list of arrays (NMS per-class format)
                        if isinstance(output_array, list):
                            # This is the NMS format: list of 80 class arrays
                            # Convert to proper numpy array structure
                            # Each element is (num_detections_for_class, 5)
                            try:
                                # Stack with padding to handle different sizes
                                max_dets = max(len(arr) if isinstance(arr, (list, np.ndarray)) else 0 for arr in output_array)
                                
                                if max_dets > 0:
                                    # Create output array (num_classes, max_detections, 5)
                                    stacked = np.zeros((len(output_array), max_dets, 5))
                                    for class_idx, class_dets in enumerate(output_array):
                                        if isinstance(class_dets, (list, np.ndarray)) and len(class_dets) > 0:
                                            class_dets_array = np.array(class_dets)
                                            if len(class_dets_array.shape) == 2 and class_dets_array.shape[1] == 5:
                                                stacked[class_idx, :len(class_dets_array)] = class_dets_array
                                    output_array = stacked
                                else:
                                    # No detections
                                    output_array = np.zeros((len(output_array), 0, 5))
                            except Exception as e:
                                print(f"Error converting NMS output: {e}")
                                output_array = np.zeros((80, 0, 5))
                        
                        output_list.append(output_array)
                    else:
                        output_list.append(np.array([]))
                # Handle numpy array directly
                elif hasattr(raw_output, 'shape'):
                    # Remove batch dimension if present
                    if len(raw_output.shape) > 0 and raw_output.shape[0] == 1:
                        output_list.append(raw_output[0])
                    else:
                        output_list.append(raw_output)
                else:
                    output_list.append(np.array([]))
            else:
                output_list.append(np.array([]))
        
        # Debug: print output shapes only if verbose
        if verbose:
            print(f"Number of outputs: {len(output_list)}")
            for i, out in enumerate(output_list):
                print(f"  Output {i} shape: {out.shape}, dtype: {out.dtype}")
                if out.size > 0:
                    print(f"  Output {i} min/max: {out.min():.4f}/{out.max():.4f}")
        
        # Postprocess
        result = self.postprocess(output_list, frame.shape, conf)
        
        return [result]  # Return as list to match YOLO API
    
    def release(self):
        """Explicitly release Hailo resources"""
        if self.target is not None:
            try:
                self.target.release()
            except Exception as e:
                print(f"Error releasing Hailo device: {e}")
            self.target = None

    def __del__(self):
        """Cleanup Hailo resources"""
        self.release()


class HailoResult:
    """
    Result object compatible with YOLO results
    Mimics ultralytics Results structure
    """
    
    def __init__(self, boxes, scores, class_ids, class_names):
        """
        Initialize result object
        
        Args:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            scores: List of confidence scores
            class_ids: List of class IDs
            class_names: Dictionary mapping class IDs to names
        """
        self.boxes = HailoBoxes(boxes, scores, class_ids)
        self.names = class_names
        self.keypoints = None  # Not supported yet
        self.masks = None  # Not supported yet
    
    def __len__(self):
        return len(self.boxes.xyxy)


class HailoBoxes:
    """
    Boxes object compatible with YOLO boxes
    """
    
    def __init__(self, boxes, scores, class_ids):
        """
        Initialize boxes object
        
        Args:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            scores: List of confidence scores
            class_ids: List of class IDs
        """
        self._boxes = np.array(boxes) if len(boxes) > 0 else np.zeros((0, 4))
        self._scores = np.array(scores) if len(scores) > 0 else np.zeros((0,))
        self._class_ids = np.array(class_ids) if len(class_ids) > 0 else np.zeros((0,))
    
    @property
    def xyxy(self):
        """Get boxes in xyxy format"""
        return MockTensor(self._boxes)
    
    @property
    def conf(self):
        """Get confidence scores"""
        return MockTensor(self._scores)
    
    @property
    def cls(self):
        """Get class IDs"""
        return MockTensor(self._class_ids)


class MockTensor:
    """Mock tensor to mimic PyTorch tensor interface"""
    
    def __init__(self, data):
        # Ensure data is always a numpy array
        if isinstance(data, list):
            self._data = np.array(data)
        elif isinstance(data, np.ndarray):
            self._data = data
        else:
            self._data = np.array(data)
    
    def cpu(self):
        """Return self to mimic PyTorch .cpu() behavior"""
        return self
    
    def numpy(self):
        """Return underlying numpy array"""
        return self._data
    
    def __len__(self):
        return len(self._data)
    
    @property
    def shape(self):
        """Return shape of underlying array"""
        return self._data.shape


# COCO Keypoint definitions
COCO_KEYPOINTS = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

# Skeleton connections for pose visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def draw_pose_keypoints(frame, outputs, model, conf_threshold=0.5, danger_zone=None):
    """
    Draw pose estimation keypoints and skeleton on frame
    
    Args:
        frame: Input frame (will be modified in-place)
        outputs: Dictionary of output tensors from Hailo model
        model: HailoInference instance for coordinate transformation
        conf_threshold: Minimum confidence for drawing keypoints
        danger_zone: DangerZone object for checking intersections
        
    Returns:
        bool: True if any person is in the danger zone
    """
    any_danger = False
    
    # Debug output shapes
    print("DEBUG: draw_pose_keypoints outputs:")
    for k, v in outputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, list):
            print(f"  {k}: list len {len(v)}")
            
    # Try to decode raw YOLOv8 Pose outputs
    try:
        from helper.yolo_decoding import decode_yolov8_pose
        # Convert dict values to list
        output_list = []
        for v in outputs.values():
            if isinstance(v, list) and len(v) > 0:
                output_list.append(np.array(v[0]) if isinstance(v[0], (list, np.ndarray)) else np.array(v))
            elif hasattr(v, 'shape'):
                # Check if flattened and try to reshape
                if len(v.shape) == 1:
                    # Try to reshape based on known channel counts (64, 1, 51)
                    size = v.size
                    for c in [64, 1, 51]:
                        # h*w*c = size => h*w = size/c => h = sqrt(size/c)
                        if size % c == 0:
                            hw = size // c
                            h = int(np.sqrt(hw))
                            if h * h == hw:
                                # Found a square match
                                v = v.reshape(h, h, c)
                                break
                output_list.append(v)
                
        dets_decoded = decode_yolov8_pose(output_list, model.input_shape[:2], conf_threshold)
        
        if dets_decoded is not None and len(dets_decoded) > 0:
            print(f"DEBUG: Decoded YOLOv8 Pose: {dets_decoded.shape}")
            # dets_decoded format: [x1, y1, x2, y2, score, class_id, kpts(51)]
            
            for det in dets_decoded:
                # Filter for person class (class_id 0)
                class_id = int(det[5])
                if class_id != 0:
                    continue

                # --- Draw Bounding Box ---
                # det[0:4] are [x1, y1, x2, y2] in model pixels
                x1_m, y1_m, x2_m, y2_m = det[0], det[1], det[2], det[3]
                
                # Get preprocessing info
                scale = model.preprocess_info['scale']
                pad_x = model.preprocess_info['pad_x']
                pad_y = model.preprocess_info['pad_y']
                
                # Transform to original image pixels
                ox1 = (x1_m - pad_x) / scale
                oy1 = (y1_m - pad_y) / scale
                ox2 = (x2_m - pad_x) / scale
                oy2 = (y2_m - pad_y) / scale
                
                # Clip to image bounds
                h_img, w_img = frame.shape[:2]
                ox1 = max(0, min(w_img, ox1))
                oy1 = max(0, min(h_img, oy1))
                ox2 = max(0, min(w_img, ox2))
                oy2 = max(0, min(h_img, oy2))
                
                # Keypoints start at index 6
                keypoints = det[6:]
                
                # Helper to process a single person's keypoints
                valid_pts = []
                kpts_list = []
                
                # keypoints should be (17, 3)
                keypoints = keypoints.reshape(-1, 3)
                    
                for i in range(len(keypoints)):
                    kp_x, kp_y, kp_conf = keypoints[i]
                    
                    # Visibility threshold
                    if kp_conf > 0.5: 
                        # Transform point
                        # Check if coordinates are normalized (<=1.0)
                        if kp_x <= 1.0 and kp_y <= 1.0:
                             # Assume normalized to model input dimensions
                             kp_x = kp_x * model.input_shape[1]
                             kp_y = kp_y * model.input_shape[0]
                        
                        # Transform to original image pixels
                        px = (kp_x - pad_x) / scale
                        py = (kp_y - pad_y) / scale
                        
                        px = int(px)
                        py = int(py)
                        
                        kpts_list.append((px, py))
                        valid_pts.append((px, py))
                    else:
                        kpts_list.append(None)
                
                # Adjust wrists to extend towards fingers (approximate hand position)
                # Left arm: 7 (Elbow) -> 9 (Wrist)
                if len(kpts_list) > 9 and kpts_list[7] is not None and kpts_list[9] is not None:
                    p7 = kpts_list[7]
                    p9 = kpts_list[9]
                    kpts_list[9] = (int(p9[0] + (p9[0]-p7[0]) * 0.3), int(p9[1] + (p9[1]-p7[1]) * 0.3))
                    
                # Right arm: 8 (Elbow) -> 10 (Wrist)
                if len(kpts_list) > 10 and kpts_list[8] is not None and kpts_list[10] is not None:
                    p8 = kpts_list[8]
                    p10 = kpts_list[10]
                    kpts_list[10] = (int(p10[0] + (p10[0]-p8[0]) * 0.3), int(p10[1] + (p10[1]-p8[1]) * 0.3))
                
                # Rebuild valid_pts
                valid_pts = [pt for pt in kpts_list if pt is not None]

                # Check danger zone
                in_danger = False
                if valid_pts and danger_zone and danger_zone.has_zone():
                    for pt in valid_pts:
                        if danger_zone.check_point_in_zone(pt[0], pt[1]):
                            in_danger = True
                            break
                    
                    if in_danger:
                        any_danger = True
                
                kp_color = (0, 0, 255) if in_danger else (0, 255, 0)
                line_color = (0, 0, 255) if in_danger else (255, 0, 0)
                
                # Draw Bounding Box
                cv2.rectangle(frame, (int(ox1), int(oy1)), (int(ox2), int(oy2)), kp_color, 2)

                # Draw Label
                label = f"Person {det[4]*100:.0f}%"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(ox1), int(oy1) - 20), (int(ox1) + text_w, int(oy1)), kp_color, -1)
                cv2.putText(frame, label, (int(ox1), int(oy1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw keypoints
                for pt in valid_pts:
                    cv2.circle(frame, pt, 4, kp_color, -1)
                
                # Draw skeleton
                for connection in SKELETON_CONNECTIONS:
                    if connection[0] < len(kpts_list) and connection[1] < len(kpts_list):
                        pt1 = kpts_list[connection[0]]
                        pt2 = kpts_list[connection[1]]
                        if pt1 and pt2:
                            cv2.line(frame, pt1, pt2, line_color, 2)
            
            return any_danger
            
    except Exception as e:
        print(f"DEBUG: Error decoding YOLOv8 Pose: {e}")
        import traceback
        traceback.print_exc()
    
    # Helper to process a single person's keypoints
    def process_keypoints(keypoints, box_conf=1.0):
        nonlocal any_danger
        valid_pts = []
        kpts_list = []
        
        # keypoints should be (17, 3) or (51,)
        if len(keypoints.shape) == 1:
            keypoints = keypoints.reshape(-1, 3)
            
        for i in range(len(keypoints)):
            kp_x, kp_y, kp_conf = keypoints[i]
            
            # Visibility threshold
            if kp_conf > 0.5: 
                # Transform point
                # Check if coordinates are normalized (<=1.0) or pixels (>1.0)
                # This is a heuristic to handle different output formats
                if kp_x <= 1.0 and kp_y <= 1.0:
                    px, py = model.transform_point(kp_x, kp_y)
                else:
                    # Already in pixels (model space), just remove padding and scale
                    scale = model.preprocess_info['scale']
                    pad_x = model.preprocess_info['pad_x']
                    pad_y = model.preprocess_info['pad_y']
                    
                    px = (kp_x - pad_x) / scale
                    py = (kp_y - pad_y) / scale
                    
                    px = int(px)
                    py = int(py)

                kpts_list.append((px, py))
                valid_pts.append((px, py))
            else:
                kpts_list.append(None)
        
        # Adjust wrists to extend towards fingers (approximate hand position)
        # Left arm: 7 (Elbow) -> 9 (Wrist)
        if len(kpts_list) > 9 and kpts_list[7] is not None and kpts_list[9] is not None:
            p7 = kpts_list[7]
            p9 = kpts_list[9]
            kpts_list[9] = (int(p9[0] + (p9[0]-p7[0]) * 0.3), int(p9[1] + (p9[1]-p7[1]) * 0.3))
            
        # Right arm: 8 (Elbow) -> 10 (Wrist)
        if len(kpts_list) > 10 and kpts_list[8] is not None and kpts_list[10] is not None:
            p8 = kpts_list[8]
            p10 = kpts_list[10]
            kpts_list[10] = (int(p10[0] + (p10[0]-p8[0]) * 0.3), int(p10[1] + (p10[1]-p8[1]) * 0.3))
        
        # Rebuild valid_pts
        valid_pts = [pt for pt in kpts_list if pt is not None]

        # Check danger zone
        in_danger = False
        if valid_pts and danger_zone and danger_zone.has_zone():
            for pt in valid_pts:
                if danger_zone.check_point_in_zone(pt[0], pt[1]):
                    in_danger = True
                    break
            
            if in_danger:
                any_danger = True
        
        kp_color = (0, 0, 255) if in_danger else (0, 255, 0)
        line_color = (0, 0, 255) if in_danger else (255, 0, 0)
        
        # Draw Label (using top-most point)
        if valid_pts:
            top_pt = min(valid_pts, key=lambda p: p[1])
            label = f"Person {box_conf*100:.0f}%"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(top_pt[0]), int(top_pt[1]) - 20), (int(top_pt[0]) + text_w, int(top_pt[1])), kp_color, -1)
            cv2.putText(frame, label, (int(top_pt[0]), int(top_pt[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw keypoints
        for pt in valid_pts:
            cv2.circle(frame, pt, 4, kp_color, -1)
        
        # Draw skeleton
        for connection in SKELETON_CONNECTIONS:
            if connection[0] < len(kpts_list) and connection[1] < len(kpts_list):
                pt1 = kpts_list[connection[0]]
                pt2 = kpts_list[connection[1]]
                if pt1 and pt2:
                    cv2.line(frame, pt1, pt2, line_color, 2)

    for output_name, output_data in outputs.items():
        if output_data is None:
            continue
            
        # Handle list output (NMS)
        if isinstance(output_data, list):
            if len(output_data) > 0:
                output_data = np.array(output_data[0]) if isinstance(output_data[0], (list, np.ndarray)) else np.array(output_data)
            else:
                continue
        
        # Remove batch dimension if present
        if len(output_data.shape) == 3 and output_data.shape[0] == 1:
            output_data = output_data[0]
            
        # Case 1: Raw YOLOv8 Pose output [56, N] or [N, 56]
        # 56 = 4 (box) + 1 (conf) + 51 (17*3 keypoints)
        if len(output_data.shape) == 2:
            # Transpose if needed to get [N, 56]
            # If shape is [56, N], transpose to [N, 56]
            if output_data.shape[0] == 56 and output_data.shape[1] > 56:
                output_data = output_data.T
            
            if output_data.shape[1] == 56:
                # Sanity check: If we have too many detections (e.g. > 1000), 
                # it's likely a raw feature map (e.g. 8400 anchors) that hasn't been NMS'd.
                # Treating these as valid detections causes "green lines everywhere".
                if output_data.shape[0] > 1000:
                    print(f"DEBUG: Ignoring likely raw output of shape {output_data.shape}")
                    continue

                # Filter by confidence
                mask = output_data[:, 4] > conf_threshold
                detections = output_data[mask]
                
                for det in detections:
                    # Keypoints start at index 5
                    keypoints = det[5:]
                    process_keypoints(keypoints, det[4])
                continue

        # Case 2: Separated outputs (e.g. from NMS)
        # We might have a tensor of shape (N, 17, 3)
        if len(output_data.shape) == 3 and output_data.shape[1] == 17 and output_data.shape[2] == 3:
             for person_kpts in output_data:
                 process_keypoints(person_kpts)
             continue
             
        # Case 3: Flat buffer that needs reshaping (common in some Hailo examples)
        # If we see a large 1D array, it might be flattened keypoints
        # DISABLED: This is too dangerous and causes "green lines everywhere" if it misinterprets a feature map
        # if len(output_data.shape) == 1 and output_data.size % 51 == 0:
        #      # Assume just keypoints
        #      num_people = output_data.size // 51
        #      # Sanity check: if num_people is huge, it's probably a feature map, not detections
        #      if num_people < 100:
        #          reshaped = output_data.reshape(num_people, 17, 3)
        #          for person_kpts in reshaped:
        #              process_keypoints(person_kpts)
                                
    return any_danger
def draw_segmentation_masks(frame, outputs, model, conf_threshold=0.5, danger_zone=None, class_names=None):
    """
    Draw segmentation masks on frame
    
    Args:
        frame: Input frame (will be modified in-place)
        outputs: Dictionary of output tensors from Hailo model
        model: HailoInference instance for coordinate transformation
        conf_threshold: Minimum confidence for drawing masks
        danger_zone: DangerZone object for checking intersections
        class_names: Dictionary of class names (optional)
        
    Returns:
        bool: True if any object is in the danger zone
    """
    # Colors for different instances
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (255, 165, 0), (0, 128, 128)
    ]
    
    any_danger = False
    
    # Debug output shapes
    print("DEBUG: draw_segmentation_masks outputs:")
    for k, v in outputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, list):
            print(f"  {k}: list len {len(v)}")
            
    # Try to decode raw YOLOv5 Segmentation outputs
    try:
        from helper.yolo_decoding import decode_yolov5_segmentation
        # Convert dict values to list, handling potential list-wrapping
        output_list = []
        for v in outputs.values():
            if isinstance(v, list) and len(v) > 0:
                output_list.append(np.array(v[0]) if isinstance(v[0], (list, np.ndarray)) else np.array(v))
            elif hasattr(v, 'shape'):
                output_list.append(v)
                
        proto_decoded, dets_decoded = decode_yolov5_segmentation(output_list, model.input_shape[:2], conf_threshold)
        
        if proto_decoded is not None and dets_decoded is not None:
            print(f"DEBUG: Decoded YOLOv5 Seg: Proto {proto_decoded.shape}, Dets {dets_decoded.shape}")
            proto_tensor = proto_decoded
            detection_tensor = dets_decoded
        else:
            proto_tensor = None
            detection_tensor = None
    except Exception as e:
        print(f"DEBUG: Error decoding YOLOv5 Seg: {e}")
        proto_tensor = None
        detection_tensor = None
    
    # If decoding failed, try to identify tensors from raw outputs (NMS case)
    if proto_tensor is None or detection_tensor is None:
        # First pass: identify tensors
        for output_name, output_data in outputs.items():
            if output_data is None:
                continue
                
            # Handle list output
            if isinstance(output_data, list):
                if len(output_data) > 0:
                    output_data = np.array(output_data[0]) if isinstance(output_data[0], (list, np.ndarray)) else np.array(output_data)
                else:
                    continue
            
            # Remove batch dimension
            if len(output_data.shape) == 4 and output_data.shape[0] == 1:
                output_data = output_data[0]
            elif len(output_data.shape) == 3 and output_data.shape[0] == 1:
                output_data = output_data[0]
                
            # Check for Prototype Tensor: [32, 160, 160] or [160, 160, 32]
            # It usually has 32 channels and spatial dims around 160
            if len(output_data.shape) == 3:
                shape = output_data.shape
                if shape[0] == 32: # CHW
                    proto_tensor = np.transpose(output_data, (1, 2, 0)) # -> HWC
                elif shape[2] == 32: # HWC
                    proto_tensor = output_data
                
            # Check for Detection Tensor with Mask Coefficients: [N, 117] or [117, N]
            # 117 = 85 (box+conf+cls) + 32 (mask coeffs)
            elif len(output_data.shape) == 2:
                if output_data.shape[1] >= 38: # At least box(4)+conf(1)+cls(1)+coeffs(32) = 38
                    detection_tensor = output_data
                elif output_data.shape[0] >= 38:
                    detection_tensor = output_data.T
            
    # If we found both, perform Instance Segmentation logic
    if proto_tensor is not None and detection_tensor is not None:
        proto_h, proto_w, proto_c = proto_tensor.shape
        
        # Process Detections
        for i, det in enumerate(detection_tensor):
            # Extract basic info
            # Format: [x, y, w, h, conf, class_id, ...32 mask coeffs...]
            
            if len(det) < 38: continue
            
            conf = det[4]
            class_id = int(det[5])
            
            # FILTER: Only Person class (0)
            if class_id != 0:
                continue
                
            if conf < conf_threshold:
                continue
                
            # Extract mask coefficients (last 32 values)
            mask_coeffs = det[-32:]
            
            # --- Coordinate Transformation ---
            # det[0:4] are [x1, y1, x2, y2] in model pixels (e.g. 640x640)
            x1_m, y1_m, x2_m, y2_m = det[0], det[1], det[2], det[3]
            
            # Get preprocessing info
            scale = model.preprocess_info['scale']
            pad_x = model.preprocess_info['pad_x']
            pad_y = model.preprocess_info['pad_y']
            
            # Transform to original image pixels
            # (x - pad) / scale
            ox1 = (x1_m - pad_x) / scale
            oy1 = (y1_m - pad_y) / scale
            ox2 = (x2_m - pad_x) / scale
            oy2 = (y2_m - pad_y) / scale
            
            # Clip to image bounds
            h_img, w_img = frame.shape[:2]
            ox1 = max(0, min(w_img, ox1))
            oy1 = max(0, min(h_img, oy1))
            ox2 = max(0, min(w_img, ox2))
            oy2 = max(0, min(h_img, oy2))
            
            # --- Mask Processing ---
            # Calculate mask in prototype space (160x160)
            # We need to map the model-space box to the prototype space
            
            # Scale factor from model input (640) to proto (160)
            scale_proto_w = proto_w / model.input_shape[1]
            scale_proto_h = proto_h / model.input_shape[0]
            
            # Box in proto space
            px1 = int(x1_m * scale_proto_w)
            py1 = int(y1_m * scale_proto_h)
            px2 = int(x2_m * scale_proto_w)
            py2 = int(y2_m * scale_proto_h)
            
            # Clamp to proto dimensions
            px1 = max(0, min(proto_w, px1))
            py1 = max(0, min(proto_h, py1))
            px2 = max(0, min(proto_w, px2))
            py2 = max(0, min(proto_h, py2))
            
            if px2 <= px1 or py2 <= py1:
                continue
                
            # CROP the prototype mask
            cropped_proto = proto_tensor[py1:py2, px1:px2, :] # (h_crop, w_crop, 32)
            
            # Matrix Multiplication
            mask_data = np.dot(cropped_proto, mask_coeffs)
            
            # Sigmoid Activation
            mask_data = 1 / (1 + np.exp(-mask_data))
            
            # Threshold
            binary_mask = (mask_data > 0.5).astype(np.uint8)
            
            if not np.any(binary_mask):
                continue
            
            # Place back into full proto mask
            full_mask = np.zeros((proto_h, proto_w), dtype=np.uint8)
            full_mask[py1:py2, px1:px2] = binary_mask
            
            # Resize to original image size using process_mask
            # process_mask handles the padding removal and scaling
            resized_mask = model.process_mask(full_mask)
            
            # --- Danger Zone Logic ---
            in_danger = False
            if danger_zone and danger_zone.has_zone():
                h_frame, w_frame = frame.shape[:2]
                zone_mask = danger_zone.create_mask(h_frame, w_frame)
                if np.logical_and(resized_mask > 0, zone_mask).any():
                    in_danger = True
                    any_danger = True
            
            # Apply color
            color = (0, 0, 255) if in_danger else colors[i % len(colors)]
            
            # Blend Mask
            overlay = frame.copy()
            overlay[resized_mask > 0] = color
            frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            
            # Draw Box
            cv2.rectangle(frame, (int(ox1), int(oy1)), (int(ox2), int(oy2)), color, 2)
            
            # Draw Label
            names = class_names if class_names else model.class_names
            class_name = names.get(class_id, f"Class {class_id}")
            label = f"{class_name} {conf*100:.0f}%"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(ox1), int(oy1) - 20), (int(ox1) + text_w, int(oy1)), color, -1)
            cv2.putText(frame, label, (int(ox1), int(oy1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw Contour
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (255, 255, 255), 1)
            
        return any_danger

    # Fallback to previous logic if not Instance Segmentation
    for output_name, output_data in outputs.items():
        # Skip if output is empty
        if output_data is None:
            continue
            
        # Handle list output (NMS or batched)
        if isinstance(output_data, list):
            if len(output_data) > 0:
                output_data = np.array(output_data[0]) if isinstance(output_data[0], (list, np.ndarray)) else np.array(output_data)
            else:
                continue
            
        # Check for semantic segmentation format [batch, classes, H, W]
        if len(output_data.shape) == 4:
            # Remove batch dimension
            if output_data.shape[0] == 1:
                output_data = output_data[0]
            
            # output_data is now [channels/instances, H, W]
            if output_data.shape[1] < 32 or output_data.shape[2] < 32:
                continue
                
            num_masks = output_data.shape[0]
            
            # Avoid processing prototype mask (32 channels) as semantic segmentation
            if num_masks == 32:
                continue
            
            # Create overlay
            overlay = frame.copy()
            
            for i in range(min(num_masks, len(colors))):
                # Filter for person class if possible (assuming class 0 is person)
                if i != 0: 
                    continue
                    
                mask = output_data[i]
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                if not np.any(binary_mask):
                    continue
                
                resized_mask = model.process_mask(binary_mask)
                
                in_danger = False
                if danger_zone and danger_zone.has_zone():
                    h_frame, w_frame = frame.shape[:2]
                    zone_mask = danger_zone.create_mask(h_frame, w_frame)
                    if np.logical_and(resized_mask > 0, zone_mask).any():
                        in_danger = True
                        any_danger = True
                
                color = (0, 0, 255) if in_danger else colors[i % len(colors)]
                overlay[resized_mask > 0] = color
            
            frame[:] = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            
    return any_danger