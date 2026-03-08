"""
Canvas widget for drawing danger zones
Supports both rectangle and polygon drawing
"""
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QPolygon
import numpy as np


class ZoneCanvas(QLabel):
    """Canvas for drawing danger zones on an image"""

    zone_completed = pyqtSignal(object)  # Emits zone data when complete

    def __init__(self):
        super().__init__()
        # self.setMinimumSize(900, 750)  # Removed fixed minimum size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #555555;
                border-radius: 10px;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)

        # Drawing state
        self.zone_shape = 'rectangle'
        self.original_image = None
        self.base_pixmap = None
        
        # Coordinate mapping state
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Rectangle mode
        self.drawing = False
        self.start_point = None
        self.end_point = None

        # Polygon mode
        self.polygon_points = []
        self.polygon_closed = False
        self.mouse_pos = None

    def load_image(self, frame: np.ndarray):
        """Load an image frame to draw on"""
        if frame is None or frame.size == 0:
            return

        self.original_image = frame.copy()
        
        # Convert to QPixmap once
        rgb_frame = self.original_image[:, :, ::-1]  # BGR to RGB
        h, w = rgb_frame.shape[:2]
        rgb_frame = rgb_frame.astype(np.uint8, copy=False)
        bytes_per_line = 3 * w
        if not rgb_frame.flags['C_CONTIGUOUS']:
            rgb_frame = np.ascontiguousarray(rgb_frame)
        qt_image = QImage(
            rgb_frame.tobytes(),
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        self.base_pixmap = QPixmap.fromImage(qt_image)
        
        self.reset_drawing()
        self.update()

    def set_zone_shape(self, shape: str):
        """Set zone shape: 'rectangle' or 'polygon'"""
        self.zone_shape = shape
        self.reset_drawing()
        self.update()

    def reset_drawing(self):
        """Reset all drawing state"""
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.polygon_points = []
        self.polygon_closed = False
        self.mouse_pos = None

    def get_zone_data(self):
        """Get the drawn zone data in original image coordinates"""
        if self.original_image is None:
            return None

        if self.zone_shape == 'rectangle':
            if self.start_point and self.end_point:
                # Convert from widget coords to image coords
                x1, y1 = self._widget_to_image_coords(self.start_point)
                x2, y2 = self._widget_to_image_coords(self.end_point)
                return {
                    'shape': 'rectangle',
                    'coords': [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                }
        elif self.zone_shape == 'polygon':
            if self.polygon_closed and len(self.polygon_points) >= 3:
                # Convert all points to image coords
                image_points = [self._widget_to_image_coords(
                    p) for p in self.polygon_points]
                return {
                    'shape': 'polygon',
                    'coords': image_points
                }
        return None

    def _widget_to_image_coords(self, widget_point):
        """Convert widget coordinates to original image coordinates"""
        if not self.base_pixmap or not self.original_image.size:
            return widget_point

        # Use stored offsets and scales
        adjusted_x = widget_point[0] - self.offset_x
        adjusted_y = widget_point[1] - self.offset_y

        # Scale to original image size
        return (int(adjusted_x * self.scale_x), int(adjusted_y * self.scale_y))

    def resizeEvent(self, event):
        """Handle resize events to update display"""
        # Just trigger a repaint, no logic needed
        super().resizeEvent(event)

    def paintEvent(self, event):
        """Paint the image and drawing"""
        super().paintEvent(event)
        
        if self.base_pixmap is None:
            return

        painter = QPainter(self)
        
        # Calculate scaling and offsets
        rect = self.contentsRect()
        scaled_pixmap = self.base_pixmap.scaled(
            rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        pixmap_rect = scaled_pixmap.rect()
        self.offset_x = (rect.width() - pixmap_rect.width()) // 2
        self.offset_y = (rect.height() - pixmap_rect.height()) // 2
        
        # Update scale factors
        img_w = self.base_pixmap.width()
        img_h = self.base_pixmap.height()
        self.scale_x = img_w / pixmap_rect.width() if pixmap_rect.width() > 0 else 1.0
        self.scale_y = img_h / pixmap_rect.height() if pixmap_rect.height() > 0 else 1.0

        # Draw image
        painter.drawPixmap(self.offset_x, self.offset_y, scaled_pixmap)

        # Draw current zone
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)

        if self.zone_shape == 'rectangle':
            if self.start_point and self.end_point:
                painter.drawRect(
                    self.start_point[0], self.start_point[1],
                    self.end_point[0] - self.start_point[0],
                    self.end_point[1] - self.start_point[1]
                )
        elif self.zone_shape == 'polygon':
            if len(self.polygon_points) > 0:
                # Draw lines between points
                for i in range(len(self.polygon_points) - 1):
                    painter.drawLine(
                        self.polygon_points[i][0], self.polygon_points[i][1],
                        self.polygon_points[i +
                                            1][0], self.polygon_points[i + 1][1]
                    )

                # Draw line to mouse if not closed
                if not self.polygon_closed and self.mouse_pos:
                    painter.drawLine(
                        self.polygon_points[-1][0], self.polygon_points[-1][1],
                        self.mouse_pos[0], self.mouse_pos[1]
                    )

                # Close polygon if complete
                if self.polygon_closed:
                    painter.drawLine(
                        self.polygon_points[-1][0], self.polygon_points[-1][1],
                        self.polygon_points[0][0], self.polygon_points[0][1]
                    )
                    # Fill with transparent red
                    brush = QBrush(QColor(255, 0, 0, 50))
                    painter.setBrush(brush)
                    qpoints = [QPoint(p[0], p[1]) for p in self.polygon_points]
                    painter.drawPolygon(QPolygon(qpoints))

                # Draw points
                painter.setBrush(QBrush(QColor(255, 0, 0)))
                for point in self.polygon_points:
                    painter.drawEllipse(point[0] - 4, point[1] - 4, 8, 8)

    def _update_display(self):
        """Trigger update"""
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press for drawing"""
        if self.base_pixmap is None:
            return

        pos = (event.pos().x(), event.pos().y())

        if self.zone_shape == 'rectangle':
            if event.button() == Qt.MouseButton.LeftButton:
                self.drawing = True
                self.start_point = pos
                self.end_point = pos
                self.update()
        elif self.zone_shape == 'polygon':
            if event.button() == Qt.MouseButton.LeftButton:
                if not self.polygon_closed:
                    self.polygon_points.append(pos)
                    self.update()
            elif event.button() == Qt.MouseButton.RightButton:
                # Right click removes last point
                if self.polygon_points:
                    self.polygon_points.pop()
                    self.polygon_closed = False
                    self.update()

        # super().mousePressEvent(event) # Don't call super if we handle it?

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing"""
        if self.base_pixmap is None:
            return

        pos = (event.pos().x(), event.pos().y())

        if self.zone_shape == 'rectangle':
            if self.drawing:
                self.end_point = pos
                self.update()
        elif self.zone_shape == 'polygon':
            if not self.polygon_closed:
                self.mouse_pos = pos
                self.update()

        # super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if self.zone_shape == 'rectangle':
            if self.drawing:
                self.drawing = False
                self.end_point = (event.pos().x(), event.pos().y())
                self.update()

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double click closes polygon"""
        if self.zone_shape == 'polygon' and len(self.polygon_points) >= 3:
            self.polygon_closed = True
            self.mouse_pos = None
            self.update()

        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        """Disable context menu"""
        if self.zone_shape == 'polygon':
            event.accept()
