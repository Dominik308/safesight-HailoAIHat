import cv2
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QDialog, QFileDialog, QMessageBox, QStyle
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QIcon
import numpy as np
from datetime import datetime


class VideoDisplay(QLabel):
    """
    Custom video display widget with touch support
    """
    double_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        # self.setFixedSize(1280, 720)
        # self.setMinimumSize(900, 750) # Removed fixed minimum size
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(),
            self.sizePolicy().verticalPolicy()
        )
        self.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #555555;
                border-radius: 10px;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        
        self.current_pixmap = None

        # Touch support
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

    def set_frame(self, frame: np.ndarray):
        """Display a numpy array frame"""
        if frame is None or frame.size == 0:
            return

        # Convert BGR to RGB
        rgb_frame = frame if len(frame.shape) == 2 else frame[:, :, ::-1]
        h, w = rgb_frame.shape[:2]

        # Create QImage from contiguous uint8 buffer
        rgb_frame = rgb_frame.astype(np.uint8, copy=False)
        if len(rgb_frame.shape) == 3:
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
        else:
            bytes_per_line = w
            if not rgb_frame.flags['C_CONTIGUOUS']:
                rgb_frame = np.ascontiguousarray(rgb_frame)
            qt_image = QImage(
                rgb_frame.tobytes(),
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_Grayscale8
            )

        self.current_pixmap = QPixmap.fromImage(qt_image)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_pixmap:
            painter = QPainter(self)
            rect = self.contentsRect()
            scaled_pixmap = self.current_pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Center the pixmap
            x = (rect.width() - scaled_pixmap.width()) // 2
            y = (rect.height() - scaled_pixmap.height()) // 2
            
            painter.drawPixmap(x, y, scaled_pixmap)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to toggle fullscreen"""
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


class ControlSlider(QWidget):
    """
    Custom slider widget with label and value display
    """
    value_changed = pyqtSignal(int)

    def __init__(self, label: str, min_val: int, max_val: int, default_val: int, suffix: str = ""):
        super().__init__()
        self.suffix = suffix
        self.setup_ui(label, min_val, max_val, default_val)

    def setup_ui(self, label, min_val, max_val, default_val):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)

        # Header with label and value
        header_layout = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setStyleSheet(
            "color: #f8f9fa; font-size: 14px; font-weight: bold;")

        self.value_label = QLabel(f"{default_val}{self.suffix}")
        self.value_label.setStyleSheet("color: #adb5bd; font-size: 14px;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        header_layout.addWidget(self.label)
        header_layout.addWidget(self.value_label)
        layout.addLayout(header_layout)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default_val)
        self.slider.setMinimumHeight(40)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 12px;
                background: #495057;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #f8f9fa;
                border: 2px solid #6c757d;
                width: 28px;
                height: 28px;
                margin: -8px 0;
                border-radius: 14px;
            }
            QSlider::handle:horizontal:pressed {
                background: #dee2e6;
            }
        """)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.value_label.setText(f"{value}{self.suffix}")
        self.value_changed.emit(value)

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.setValue(value)


class StatusIndicator(QWidget):
    """
    Status indicator with colored dot and text
    """

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)

        # Status dot
        self.dot = QLabel("●")
        self.dot.setStyleSheet("color: #ffc107; font-size: 16px;")

        # Status text
        self.text = QLabel("Stopped")
        self.text.setStyleSheet("color: #f8f9fa; font-size: 12px;")

        # FPS label
        self.fps_label = QLabel("0 FPS")
        self.fps_label.setStyleSheet("color: #6c757d; font-size: 12px;")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.dot)
        layout.addWidget(self.text)
        layout.addStretch()
        layout.addWidget(self.fps_label)

    def set_status(self, status: str, color: str = "#ffc107"):
        """Set status text and color"""
        self.text.setText(status)
        self.dot.setStyleSheet(f"color: {color}; font-size: 16px;")

    def set_fps(self, fps: float):
        """Update FPS display"""
        self.fps_label.setText(f"{fps:.1f} FPS")


class TouchButton(QPushButton):
    """
    Touch-optimized button with larger hit area
    """

    def __init__(self, text: str, style: str = "primary", icon: QIcon = None):
        super().__init__(text)
        self.setMinimumHeight(50)
        
        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(24, 24))

        styles = {
            "primary": "#0d6efd",
            "success": "#28a745",
            "danger": "#dc3545",
            "secondary": "#6c757d",
            "dark": "#343a40"
        }

        bg_color = styles.get(style, "#495057")

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: #ffffff;
                border: 2px solid {bg_color};
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(bg_color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(bg_color, 0.3)};
            }}
            QPushButton:disabled {{
                background-color: #495057;
                border-color: #495057;
                color: #6c757d;
            }}
        """)

    @staticmethod
    def _darken_color(hex_color: str, factor: float = 0.15) -> str:
        """Darken a hex color by a factor"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, int(c * (1 - factor))) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"


class SnapshotModal(QDialog):
    """
    Modal dialog to display a snapshot of the current video frame
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Snapshot")
        self.setModal(True)
        self.setStyleSheet("background-color: #1e1e1e;")

        # Layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Current Frame Snapshot")
        title.setStyleSheet(
            "color: #ffffff; font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #555555;
                border-radius: 10px;
            }
        """)
        # self.image_label.setMinimumSize(800, 600) # Removed fixed minimum size
        self.image_label.mousePressEvent = self._on_image_click
        layout.addWidget(self.image_label)

        # Button layout
        button_layout = QHBoxLayout()

        # Mark danger button
        self.mark_btn = TouchButton("Mark Danger", "danger", self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
        self.mark_btn.setCheckable(True)
        self.mark_btn.clicked.connect(self.toggle_marking)
        button_layout.addWidget(self.mark_btn)

        # Clear markers button
        self.clear_btn = TouchButton("Clear Marks", "secondary", self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self.clear_btn.clicked.connect(self.clear_markers)
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        # Save/Close button layout
        save_close_layout = QHBoxLayout()

        # Save button
        self.save_btn = TouchButton("Save Image", "primary", self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_btn.clicked.connect(self.save_snapshot)
        save_close_layout.addWidget(self.save_btn)

        # Close button
        self.close_btn = TouchButton("Close", "secondary", self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton))
        self.close_btn.clicked.connect(self.close)
        save_close_layout.addWidget(self.close_btn)

        layout.addLayout(save_close_layout)

        # Store frame and markers
        self.current_frame = None
        self.original_frame = None
        self.danger_markers = []
        self.marking_enabled = False

    def set_frame(self, frame: np.ndarray):
        """Set the frame to display"""
        if frame is None or frame.size == 0:
            return

        self.original_frame = frame.copy()
        self.current_frame = frame.copy()
        self._update_display()

    def _update_display(self):
        """Update the display with markers"""
        if self.current_frame is None:
            return

        # Draw markers on frame
        frame_with_markers = self._draw_markers(self.current_frame)

        # Convert BGR to RGB
        rgb_frame = frame_with_markers if len(
            frame_with_markers.shape) == 2 else frame_with_markers[:, :, ::-1]
        h, w = rgb_frame.shape[:2]

        # Create QImage from contiguous uint8 buffer
        rgb_frame = rgb_frame.astype(np.uint8, copy=False)
        if len(rgb_frame.shape) == 3:
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
        else:
            bytes_per_line = w
            if not rgb_frame.flags['C_CONTIGUOUS']:
                rgb_frame = np.ascontiguousarray(rgb_frame)
            qt_image = QImage(
                rgb_frame.tobytes(),
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_Grayscale8
            )

        # Scale to fit label while keeping aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def _on_image_click(self, event):
        """Handle click on image to mark danger"""
        if not self.marking_enabled or event.button() != Qt.MouseButton.LeftButton:
            return

        # Get click position
        click_pos = event.pos()

        # Convert to frame coordinates
        if self.image_label.pixmap() and self.current_frame is not None:
            frame_pos = self._widget_to_frame_coords(click_pos)
            if frame_pos:
                self.danger_markers.append(frame_pos)
                self._update_display()

    def _widget_to_frame_coords(self, widget_pos):
        """Convert widget coordinates to frame coordinates"""
        if not self.image_label.pixmap() or self.current_frame is None:
            return None

        # Get pixmap rect (centered in label)
        pixmap_rect = self.image_label.pixmap().rect()
        label_rect = self.image_label.rect()

        # Calculate offset (pixmap is centered)
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2

        # Adjust click position
        adjusted_x = widget_pos.x() - x_offset
        adjusted_y = widget_pos.y() - y_offset

        # Check if click is within pixmap bounds
        if 0 <= adjusted_x < pixmap_rect.width() and 0 <= adjusted_y < pixmap_rect.height():
            # Scale to original frame size
            frame_h, frame_w = self.current_frame.shape[:2]
            scale_x = frame_w / pixmap_rect.width()
            scale_y = frame_h / pixmap_rect.height()

            frame_x = int(adjusted_x * scale_x)
            frame_y = int(adjusted_y * scale_y)

            return (frame_x, frame_y)

        return None

    def _draw_markers(self, frame):
        """Draw danger markers on frame"""
        import cv2
        frame_copy = frame.copy()

        for marker in self.danger_markers:
            x, y = marker
            # Draw red circle
            cv2.circle(frame_copy, (x, y), 20, (0, 0, 255), 3)
            # Draw crosshair
            cv2.line(frame_copy, (x-15, y), (x+15, y), (0, 0, 255), 2)
            cv2.line(frame_copy, (x, y-15), (x, y+15), (0, 0, 255), 2)
            # Draw "DANGER" text (using FONT_HERSHEY_SIMPLEX instead of BOLD)
            cv2.putText(frame_copy, "DANGER", (x-30, y-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame_copy

    def toggle_marking(self):
        """Toggle danger marking mode"""
        self.marking_enabled = self.mark_btn.isChecked()
        if self.marking_enabled:
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.mark_btn.setText("Marking...")
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.mark_btn.setText("Mark Danger")

    def clear_markers(self):
        """Clear all danger markers"""
        self.danger_markers.clear()
        self._update_display()

    def save_snapshot(self):
        """Save the snapshot to a file"""
        if self.current_frame is None:
            return

        # Use frame with markers for saving
        frame_to_save = self._draw_markers(self.current_frame)

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"snapshot_{timestamp}.jpg"

        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Snapshot",
            default_filename,
            "JPEG Image (*.jpg);;PNG Image (*.png);;All Files (*)"
        )

        if filename:
            # Save the frame with markers
            cv2.imwrite(filename, frame_to_save)
            QMessageBox.information(
                self, "Success", f"Snapshot saved to:\n{filename}")
