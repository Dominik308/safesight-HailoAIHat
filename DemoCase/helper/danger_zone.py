"""
Danger zone management for SafeSight DemoCase
Handles rectangle and polygon danger zones
"""
import cv2
import numpy as np


class DangerZone:
    """Manages danger zone configuration and checking"""
    
    def __init__(self):
        self.zone_shape = 'rectangle'  # 'rectangle' or 'polygon'
        self.rectangle = None  # [x1, y1, x2, y2]
        self.polygon = None    # [[x, y], [x, y], ...]
        
    def set_rectangle(self, x1, y1, x2, y2):
        """Set rectangular danger zone"""
        self.zone_shape = 'rectangle'
        self.rectangle = [int(x1), int(y1), int(x2), int(y2)]
        self.polygon = None
        
    def set_polygon(self, points):
        """Set polygon danger zone"""
        self.zone_shape = 'polygon'
        self.polygon = [[int(p[0]), int(p[1])] for p in points]
        self.rectangle = None
        
    def clear(self):
        """Clear danger zone"""
        self.rectangle = None
        self.polygon = None
        
    def has_zone(self):
        """Check if a zone is defined"""
        return self.rectangle is not None or self.polygon is not None
        
    def draw_on_frame(self, frame):
        """Draw the danger zone on a frame"""
        if self.zone_shape == 'rectangle' and self.rectangle:
            x1, y1, x2, y2 = self.rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif self.zone_shape == 'polygon' and self.polygon and len(self.polygon) >= 2:
            pts = self.polygon + [self.polygon[0]]
            for i in range(len(pts) - 1):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 2)
                
    def check_point_in_zone(self, x, y):
        """Check if a point is inside the danger zone"""
        if self.zone_shape == 'rectangle' and self.rectangle:
            x1, y1, x2, y2 = self.rectangle
            return x1 <= x <= x2 and y1 <= y <= y2
        elif self.zone_shape == 'polygon' and self.polygon:
            return self._point_in_polygon(x, y, self.polygon)
        return False
        
    def check_box_in_zone(self, x1, y1, x2, y2, mode='any_overlap'):
        """Check if a bounding box is in the danger zone"""
        if self.zone_shape == 'rectangle' and self.rectangle:
            return self._check_box_rectangle(x1, y1, x2, y2, mode)
        elif self.zone_shape == 'polygon' and self.polygon:
            return self._check_box_polygon(x1, y1, x2, y2, mode)
        return False
        
    def _check_box_rectangle(self, x1, y1, x2, y2, mode):
        """Check bounding box against rectangular zone"""
        zx1, zy1, zx2, zy2 = self.rectangle
        
        if mode == 'center':
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return zx1 <= cx <= zx2 and zy1 <= cy <= zy2
        elif mode == 'any_overlap':
            return not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2)
        elif mode == 'complete_inside':
            return x1 >= zx1 and y1 >= zy1 and x2 <= zx2 and y2 <= zy2
        
        return False
        
    def _check_box_polygon(self, x1, y1, x2, y2, mode):
        """Check bounding box against polygon zone"""
        if mode == 'center':
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return self._point_in_polygon(cx, cy, self.polygon)
        elif mode == 'any_overlap':
            # Check corners
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            if any(self._point_in_polygon(px, py, self.polygon) for px, py in corners):
                return True
            # Check center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return self._point_in_polygon(cx, cy, self.polygon)
        
        return False
        
    @staticmethod
    def _point_in_polygon(x, y, polygon):
        """Ray casting algorithm to check if point is in polygon"""
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside
        
    def count_detections_in_zone(self, results):
        """Count how many YOLO detections are in the danger zone"""
        if not self.has_zone() or not results:
            return 0
            
        count = 0
        result = results[0] if isinstance(results, (list, tuple)) else results
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                if self.check_box_in_zone(x1, y1, x2, y2, mode='center'):
                    count += 1
                    
        return count

    def create_mask(self, height, width):
        """Create a binary mask of the danger zone"""
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.zone_shape == 'rectangle' and self.rectangle:
            x1, y1, x2, y2 = self.rectangle
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
        elif self.zone_shape == 'polygon' and self.polygon:
            pts = np.array(self.polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)
