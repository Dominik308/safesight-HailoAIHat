"""
Model type detection and handling for different YOLO models
"""
import cv2
import numpy as np
from typing import Optional


CLASS_COLOR_PALETTE = [
    (56, 56, 255),    # Red
    (151, 157, 255),  # Salmon
    (31, 112, 255),   # Orange
    (29, 178, 255),   # Yellow
    (49, 210, 207),   # Light green
    (10, 249, 72),    # Green
    (23, 204, 146),   # Teal
    (134, 219, 61),   # Lime
    (52, 147, 26),    # Forest
    (0, 212, 187),    # Aqua
]


def _resolve_class_name(result, cls_id: int) -> str:
    if hasattr(result, "names"):
        names = result.names
        if isinstance(names, dict):
            return names.get(cls_id, f"Class {cls_id}")
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return names[cls_id]
    return f"Class {cls_id}"


def _get_class_color(cls_id: Optional[int]) -> tuple[int, int, int]:
    if cls_id is None:
        return (0, 255, 0)
    return CLASS_COLOR_PALETTE[int(cls_id) % len(CLASS_COLOR_PALETTE)]


def _get_contrasting_text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def _draw_label(frame: np.ndarray, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    if not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    padding = 6
    box_width = text_width + padding
    box_height = text_height + baseline + padding

    x1 = int(max(0, min(x1, frame.shape[1] - box_width)))

    if y1 - box_height >= 0:
        top = y1 - box_height
        bottom = y1
    else:
        top = min(frame.shape[0] - box_height, max(0, y1))
        bottom = top + box_height

    cv2.rectangle(frame, (x1, top), (x1 + box_width, bottom), color, -1)
    text_color = _get_contrasting_text_color(color)
    text_org = (x1 + padding // 2, bottom - baseline - 2)
    cv2.putText(frame, text, text_org, font, scale, text_color, thickness, cv2.LINE_AA)


def get_model_type(model_name: str) -> str:
    """Determine model type from model filename"""
    name_lower = model_name.lower()

    if "hand" in name_lower:
        return "hand"
    elif "pose" in name_lower:
        return "pose"
    elif "segment" in name_lower or "seg" in name_lower:
        return "segment"
    else:
        return "detect"


def draw_detections(frame, results, danger_zone, model_type):
    """Draw YOLO detections on frame with danger zone checking"""
    if not results or len(results) == 0:
        return False

    result = results[0]
    danger_detected = False

    # Strategy depends on model type to ensure correct danger logic
    
    if model_type == "segment":
        # Segmentation handles its own drawing and danger check (mask-based)
        # It also handles labels and boxes if available
        if _draw_segmentation(frame, result, danger_zone):
            danger_detected = True
            
    elif model_type == "detect":
        # Standard detection handles its own drawing and danger check (box-based)
        if _draw_detection_boxes(frame, result, danger_zone):
            danger_detected = True
            
    elif model_type == "pose":
        # Pose needs detection boxes for labels, but we disable danger check there
        # to avoid triggering on bounding box overlap
        _draw_detection_boxes(frame, result, None)
        
        # Then draw pose with actual danger check (keypoint-based)
        if _draw_pose(frame, result, danger_zone):
            danger_detected = True
            
    elif model_type == "hand":
        # Hand needs detection boxes for labels, but we disable danger check there
        _draw_detection_boxes(frame, result, None)
        
        # Then draw hand pose with actual danger check
        if _draw_hand_pose(frame, result, danger_zone):
            danger_detected = True
            
    return danger_detected


def _draw_detection_boxes(frame, result, danger_zone):
    """Draw bounding boxes for detection models"""
    if not hasattr(result, 'boxes') or result.boxes is None:
        return False

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else None
    classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else None
    
    any_danger = False

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        cls_id = int(classes[i]) if classes is not None and i < len(classes) else None
        conf = float(confs[i]) if confs is not None and i < len(confs) else None

        # Only draw person detections (class 0 in COCO)
        class_name = _resolve_class_name(result, cls_id)
        if class_name != "person": 
             continue

        # Check if in danger zone
        in_danger = False
        if danger_zone and danger_zone.has_zone():
            in_danger = danger_zone.check_box_in_zone(
                x1, y1, x2, y2)
            if in_danger:
                any_danger = True

        # Color based on danger zone
        color = (0, 0, 255) if in_danger else _get_class_color(cls_id)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        if conf is not None and cls_id is not None:
            label = f"{class_name}: {conf * 100:.1f}%"
            _draw_label(frame, label, x1, y1, color)
            
    return any_danger


def _draw_segmentation(frame, result, danger_zone):
    """Draw segmentation masks"""
    if not hasattr(result, 'masks') or result.masks is None:
        return False

    mask_data = result.masks.data.cpu().numpy() if getattr(result.masks, "data", None) is not None else None
    mask_polygons = getattr(result.masks, "xy", None)

    boxes = None
    classes = None
    confs = None
    if hasattr(result, 'boxes') and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
            classes = result.boxes.cls.cpu().numpy()
        if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
            confs = result.boxes.conf.cpu().numpy()

    frame_h, frame_w = frame.shape[:2]
    overlay = frame.astype(np.float32).copy()

    mask_fill_alpha = 0.6
    overlay_alpha = 0.6

    draw_queue = []
    contour_queue = []
    
    any_danger = False

    total_masks = 0
    if mask_polygons is not None:
        total_masks = len(mask_polygons)
    elif mask_data is not None:
        total_masks = len(mask_data)

    for i in range(total_masks):
        mask_bool = None

        if mask_polygons is not None and i < len(mask_polygons):
            polygon = mask_polygons[i]
            if polygon is None or len(polygon) == 0:
                continue
            mask_u8 = np.zeros((frame_h, frame_w), dtype=np.uint8)
            polygon_pts = np.asarray(polygon, dtype=np.int32)
            cv2.fillPoly(mask_u8, [polygon_pts], 1)
            mask_bool = mask_u8.astype(bool)
        elif mask_data is not None and i < len(mask_data):
            mask = mask_data[i]
            mask_resized = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized > 0.5

        if mask_bool is None:
            continue

        if not mask_bool.any():
            continue

        cls_id = int(classes[i]) if classes is not None and i < len(classes) else None
        conf = float(confs[i]) if confs is not None and i < len(confs) else None

        # Filter for person class
        if cls_id is not None:
            class_name = _resolve_class_name(result, cls_id)
            if class_name != "person":
                continue

        x1 = y1 = x2 = y2 = None

        # Check if mask is in danger zone
        in_danger = False
        if danger_zone and danger_zone.has_zone():
            # Create zone mask if not already created (optimization: create once per frame)
            # But here we do it per mask for simplicity or pass it in
            zone_mask = danger_zone.create_mask(frame_h, frame_w)
            
            # Check intersection
            if np.logical_and(mask_bool, zone_mask).any():
                in_danger = True
                any_danger = True

        # Color based on danger zone
        color = (0, 0, 255) if in_danger else _get_class_color(cls_id)

        color_arr = np.array(color, dtype=np.float32)
        overlay_region = overlay[mask_bool]
        overlay[mask_bool] = overlay_region * (1.0 - mask_fill_alpha) + color_arr * mask_fill_alpha

        # Store contours to render sharply after blending
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contour_queue.append((contours, color))

        if boxes is not None and i < len(boxes):
            if x1 is None or y1 is None or x2 is None or y2 is None:
                x1, y1, x2, y2 = map(int, boxes[i])

            label_text = None
            if cls_id is not None:
                class_name = _resolve_class_name(result, cls_id)
                label_text = f"{class_name}: {conf * 100:.1f}%" if conf is not None else class_name

            draw_queue.append((x1, y1, x2, y2, color, label_text))

    if not contour_queue and not draw_queue:
        return any_danger

    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(overlay_uint8, overlay_alpha, frame, 1.0 - overlay_alpha, 0)
    frame[:] = blended

    for contours, color in contour_queue:
        cv2.drawContours(frame, contours, -1, color, 2)

    # Draw bounding boxes if available
    for x1, y1, x2, y2, color, label_text in draw_queue:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label_text:
            _draw_label(frame, label_text, x1, y1, color)
            
    return any_danger


def _draw_pose(frame, result, danger_zone):
    """Draw pose keypoints and skeleton"""
    if not hasattr(result, 'keypoints') or result.keypoints is None:
        return False

    keypoints = result.keypoints.xy.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') else None
    
    any_danger = False

    # COCO pose skeleton connections
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]

    for person_idx, person_kpts in enumerate(keypoints):
        # Check if person is in danger zone (check keypoints)
        in_danger = False
        if danger_zone and danger_zone.has_zone():
            # Check if any valid keypoint is in the zone
            for kp in person_kpts:
                if kp[0] > 0 and kp[1] > 0:  # Valid keypoint
                    if danger_zone.check_point_in_zone(int(kp[0]), int(kp[1])):
                        in_danger = True
                        break
            
            if in_danger:
                any_danger = True

        # Color based on danger zone
        skeleton_color = (0, 0, 255) if in_danger else (0, 255, 0)
        keypoint_color = (255, 0, 0) if in_danger else (0, 255, 255)

        # Draw skeleton connections
        for start_idx, end_idx in skeleton:
            if start_idx < len(person_kpts) and end_idx < len(person_kpts):
                start_point = person_kpts[start_idx]
                end_point = person_kpts[end_idx]

                # Check if both keypoints are valid (confidence > 0)
                if start_point[0] > 0 and start_point[1] > 0 and \
                   end_point[0] > 0 and end_point[1] > 0:
                    cv2.line(frame,
                             (int(start_point[0]), int(start_point[1])),
                             (int(end_point[0]), int(end_point[1])),
                             skeleton_color, 2)

        # Draw keypoints
        for keypoint in person_kpts:
            if keypoint[0] > 0 and keypoint[1] > 0:
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])),
                           4, keypoint_color, -1)

        # Draw bounding box if available
        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[person_idx])
            cv2.rectangle(frame, (x1, y1), (x2, y2), skeleton_color, 2)
            
    return any_danger


def _draw_hand_pose(frame, result, danger_zone):
    """Draw hand pose keypoints (21 keypoints per hand)"""
    if not hasattr(result, 'keypoints') or result.keypoints is None:
        return False

    keypoints = result.keypoints.xy.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') else None
    
    any_danger = False

    # Hand skeleton connections (MediaPipe hand model)
    hand_connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    for hand_idx, hand_kpts in enumerate(keypoints):
        # Check if hand is in danger zone (check keypoints)
        in_danger = False
        if danger_zone and danger_zone.has_zone():
            # Check if any valid keypoint is in the zone
            for kp in hand_kpts:
                if kp[0] > 0 and kp[1] > 0:  # Valid keypoint
                    if danger_zone.check_point_in_zone(int(kp[0]), int(kp[1])):
                        in_danger = True
                        break
            
            if in_danger:
                any_danger = True

        # Color based on danger zone
        skeleton_color = (0, 0, 255) if in_danger else (0, 255, 0)
        keypoint_color = (255, 0, 0) if in_danger else (0, 255, 255)

        # Normalize hand keypoints if needed (handle reversed ordering)
        hand_kpts = _normalize_hand_keypoints(hand_kpts)

        # Draw hand skeleton
        for start_idx, end_idx in hand_connections:
            if start_idx < len(hand_kpts) and end_idx < len(hand_kpts):
                start_point = hand_kpts[start_idx]
                end_point = hand_kpts[end_idx]

                if start_point[0] > 0 and start_point[1] > 0 and \
                   end_point[0] > 0 and end_point[1] > 0:
                    cv2.line(frame,
                             (int(start_point[0]), int(start_point[1])),
                             (int(end_point[0]), int(end_point[1])),
                             skeleton_color, 2)

        # Draw keypoints
        for i, keypoint in enumerate(hand_kpts):
            if keypoint[0] > 0 and keypoint[1] > 0:
                # Wrist is larger
                radius = 6 if i == 0 else 3
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])),
                           radius, keypoint_color, -1)

        # Draw bounding box if available
        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[hand_idx])
            cv2.rectangle(frame, (x1, y1), (x2, y2), skeleton_color, 2)
            
    return any_danger


def _normalize_hand_keypoints(hand_kpts):
    """
    Normalize hand keypoints to handle reversed ordering.
    Some hand models export fingers in reverse order for left hands.
    """
    if len(hand_kpts) != 21:
        return hand_kpts

    # Remap for reversed ordering: [0,17-20,13-16,9-12,5-8,1-4]
    REMAP_REVERSED = [0, 17, 18, 19, 20, 13, 14, 15,
                      16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]

    # Check if ordering looks reversed (thumb on wrong side)
    thumb_x = np.mean([hand_kpts[i][0]
                      for i in range(1, 5) if hand_kpts[i][0] > 0])
    pinky_x = np.mean([hand_kpts[i][0]
                      for i in range(17, 21) if hand_kpts[i][0] > 0])

    # If thumb is to the right of pinky, likely reversed
    if thumb_x > pinky_x and not np.isnan(thumb_x) and not np.isnan(pinky_x):
        return np.array([hand_kpts[i] for i in REMAP_REVERSED])

    return hand_kpts
