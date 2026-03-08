"""
SafeSight Demo Case - PyQt5 GUI Application
Professional YOLO-based safety monitoring system
"""
from helper.model_mode import ModelMode
from zone_modal import ZoneModal
from video_processor import VideoProcessor
from widgets import VideoDisplay, ControlSlider, StatusIndicator, TouchButton, SnapshotModal
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QGroupBox, QScrollArea, QScroller, QMessageBox, QStyle
)
import sys
import os
import atexit

# GPIO setup for RPi 5 with Geekworm extension + Hailo Hat - Use gpiod
import gpiod
from gpiod.line import Direction, Value

# GPIO chip (gpiochip0 for RPi 5 with extension header)
chip = gpiod.Chip('/dev/gpiochip0')

# Pin assignments (BCM numbering)
pin_in1 = 23

# Request lines as outputs (active high for relays)
line_config = {
    pin_in1: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)
}

request = chip.request_lines(consumer="safesight", config=line_config)

# Create relay control class
class Relay:
    def __init__(self, request, offset):
        self.request = request
        self.offset = offset
        self.is_active = False
    
    def on(self):
        self.request.set_value(self.offset, Value.ACTIVE)  # 1 = HIGH = ON
        self.is_active = True
    
    def off(self):
        self.request.set_value(self.offset, Value.INACTIVE)  # 0 = LOW = OFF
        self.is_active = False

# Map relays
relays = {
    'red': Relay(request, pin_in1)
}

def all_relays_off():
    """Turn all relays OFF on exit"""
    print("Exiting: Turning all relays OFF")
    for relay in relays.values():
        relay.off()
    chip.close()

atexit.register(all_relays_off)

# Remove OpenCV's Qt plugin path to avoid conflicts
import cv2
cv2_path = cv2.__file__
plugin_path = os.path.join(os.path.dirname(cv2_path), 'qt', 'plugins')
if 'QT_PLUGIN_PATH' in os.environ:
    paths = os.environ['QT_PLUGIN_PATH'].split(':')
    paths = [p for p in paths if plugin_path not in p]
    os.environ['QT_PLUGIN_PATH'] = ':'.join(paths)


# Import custom modules
ICON_PATH = os.path.join(os.path.dirname(
    __file__), "assets", "SafeSightIcon.png")


class SafeSightGUI(QMainWindow):
    """Main application window"""

    # Model configurations (will be resolved to absolute paths in __init__)
    MODELS = {
        "yolov11n": "Models/yolov11n.hef",
        "yolov11s": "Models/yolov11s.hef",
        "yolov11m": "Models/yolov11m.hef",
        "yolov8s_pose": "Models/yolov8s_pose.hef",
        "yolov8m_pose": "Models/yolov8m_pose.hef",
        "yolov5n_seg": "Models/yolov5n_seg.hef",
        "yolov5m_seg": "Models/yolov5m_seg.hef",
        "best-hand-training.pt": "Models/best-hand-training.pt",
        "css-best.pt": "Models/css-best.pt",
        "hand-detection.pt": "Models/hand-detection.pt",
        "helmet_detection.pt": "Models/helmet_detection.pt",
        "yolo11n.pt": "Models/yolo11n.pt",
        "yolo11n-seg.pt": "Models/yolo11n-seg.pt",
        "yolo11n-pose.pt": "Models/yolo11n-pose.pt",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SafeSight - AI Safety Monitoring")
        self.setWindowIcon(QIcon(ICON_PATH))
        self.setStyleSheet("background-color: #1e1e1e; color: #f8f9fa;")

        # Enable touch support (PyQt6 enum namespace changes)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        
        # Resolve model paths to absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for key in self.MODELS:
            self.MODELS[key] = os.path.join(base_dir, self.MODELS[key])

        # Video processor
        self.video_processor = VideoProcessor()
        self.video_processor.frame_ready.connect(self.update_frame)
        self.video_processor.fps_update.connect(self.update_fps)
        self.video_processor.error_occurred.connect(self.show_error)
        self.video_processor.danger_detected_signal.connect(self.on_danger_detected)

        # Setup UI
        self.setup_ui()

        # Load default model if it exists
        default_index = self.model_combo.currentIndex()
        model_names = list(self.MODELS.keys())
        if default_index < len(model_names):
            model_path = self.MODELS[model_names[default_index]]
            if os.path.exists(model_path):
                self.on_model_changed(default_index)
            else:
                self.status_indicator.set_status("No model loaded", "#ffc107")

        self.showMaximized()

    def setup_ui(self):
        """Initialize the user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left: Video display
        self.setup_video_section(main_layout)

        # Right: Control panel
        self.setup_control_panel(main_layout)

    def setup_video_section(self, parent_layout):
        """Create video display area"""
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        # Video display
        self.video_display = VideoDisplay()
        self.video_display.double_clicked.connect(self.toggle_fullscreen)
        video_layout.addWidget(self.video_display)

        # Load placeholder
        self.load_placeholder()

        parent_layout.addWidget(video_widget, stretch=3)

    def setup_control_panel(self, parent_layout):
        """Create control panel with all controls"""
        # Scroll area for controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMaximumWidth(360)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        # Enable kinetic scrolling
        QScroller.grabGesture(
            scroll_area.viewport(),
            QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        # Controls widget
        controls_widget = QWidget()
        controls_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(15)

        # Title
        title = QLabel("Bedienfeld")
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #ffffff; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(title)

        # Status indicator
        self.status_indicator = StatusIndicator()
        controls_layout.addWidget(self.status_indicator)

        # Stream controls
        self.setup_stream_controls(controls_layout)

        # Model selection
        self.setup_model_selection(controls_layout)

        # Video adjustments
        self.setup_video_adjustments(controls_layout)

        # Detection settings
        self.setup_detection_settings(controls_layout)

        # Spacer
        controls_layout.addStretch()

        # Footer
        self.setup_footer(controls_layout)

        scroll_area.setWidget(controls_widget)
        parent_layout.addWidget(scroll_area, stretch=1)

    def setup_stream_controls(self, parent_layout):
        """Setup start/stop/fullscreen buttons"""
        group = QGroupBox("Stream-Steuerung")
        group.setStyleSheet(
            "QGroupBox { color: #ffffff; font-size: 14px; font-weight: bold; }")
        layout = QVBoxLayout(group)

        # Start/Stop buttons
        button_layout = QHBoxLayout()

        self.start_btn = TouchButton("Start", "success", self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_stream)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = TouchButton("Stopp", "danger", self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_btn.clicked.connect(self.stop_stream)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Fullscreen button
        self.fullscreen_btn = TouchButton("Vollbild", "dark", self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton))
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        layout.addWidget(self.fullscreen_btn)

        # Snapshot button
        self.snapshot_btn = TouchButton("Schnappschuss", "primary", self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.snapshot_btn.clicked.connect(self.show_snapshot)
        self.snapshot_btn.setEnabled(False)
        layout.addWidget(self.snapshot_btn)

        # Danger zone button
        self.danger_zone_btn = TouchButton("Gefahrenbereich", "danger", self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
        self.danger_zone_btn.clicked.connect(self.show_zone_modal)
        self.danger_zone_btn.setEnabled(False)
        layout.addWidget(self.danger_zone_btn)

        # Clear zone button
        self.clear_zone_btn = TouchButton("Zone löschen", "secondary", self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self.clear_zone_btn.clicked.connect(self.clear_danger_zone)
        self.clear_zone_btn.setEnabled(False)
        layout.addWidget(self.clear_zone_btn)
        
        # Relay control buttons
        relay_layout = QHBoxLayout()
        
        self.red_light_btn = TouchButton("Rot", "danger", self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical))
        self.red_light_btn.clicked.connect(lambda: self.toggle_relay('red', self.red_light_btn))
        relay_layout.addWidget(self.red_light_btn)
        
        layout.addLayout(relay_layout)

        parent_layout.addWidget(group)

    def setup_model_selection(self, parent_layout):
        """Setup model selection dropdown"""
        group = QGroupBox("KI-Modell")
        group.setStyleSheet(
            "QGroupBox { color: #ffffff; font-size: 14px; font-weight: bold; }")
        layout = QVBoxLayout(group)

        self.model_combo = QComboBox()
        self.model_mode = QComboBox()

        self.model_combo.setMinimumHeight(45)
        self.model_mode.setMinimumHeight(45)

        self.model_combo.addItems([
            "yolov11n", 
            "yolov11s", 
            "yolov11m",
            "yolov8s_pose", 
            "yolov8m_pose",
            "yolov5n_seg", 
            "yolov5m_seg",
            "best-hand-training.pt", 
            "css-best.pt",
            "hand-detection.pt", 
            "helmet_detection.pt",
            "yolo11n.pt", 
            "yolo11n-seg.pt", 
            "yolo11n-pose.pt"
        ])
        self.model_combo.setCurrentIndex(0)  # Default to Hailo model
        self.model_combo.setStyleSheet(ModelMode.set_dropdown_style())
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.model_combo)

        self.model_mode.addItems(ModelMode.fillModelModeDropdown(
            ModelMode.determineModeFromModel(self.model_combo.currentText())))

        self.model_mode.setStyleSheet(ModelMode.set_dropdown_style())
        # self.model_mode.currentIndexChanged.connect(self.on_model_mode_changed())
        layout.addWidget(self.model_mode)

        parent_layout.addWidget(group)

    def setup_video_adjustments(self, parent_layout):
        """Setup brightness and contrast sliders"""
        group = QGroupBox("Image Adjustments")
        group.setStyleSheet(
            "QGroupBox { color: #ffffff; font-size: 14px; font-weight: bold; }")
        layout = QVBoxLayout(group)

        # Brightness
        self.brightness_slider = ControlSlider("Brightness", 0, 100, 50)
        self.brightness_slider.value_changed.connect(
            self.video_processor.set_brightness)
        layout.addWidget(self.brightness_slider)

        # Contrast
        self.contrast_slider = ControlSlider("Contrast", 0, 100, 50)
        self.contrast_slider.value_changed.connect(
            self.video_processor.set_contrast)
        layout.addWidget(self.contrast_slider)

        # Gaussian Blur
        self.gaussian_slider = ControlSlider("Gaussian Blur", 0, 20, 0)
        self.gaussian_slider.value_changed.connect(
            self.video_processor.set_gaussian_blur)
        layout.addWidget(self.gaussian_slider)

        self.noise_slider = ControlSlider("Image Noise", 0, 100, 0)
        self.noise_slider.value_changed.connect(
            self.video_processor.set_gaussian_noise)
        layout.addWidget(self.noise_slider)

        parent_layout.addWidget(group)

    def setup_detection_settings(self, parent_layout):
        """Setup detection confidence threshold"""
        group = QGroupBox("Detection Settings")
        group.setStyleSheet(
            "QGroupBox { color: #ffffff; font-size: 14px; font-weight: bold; }")
        layout = QVBoxLayout(group)

        # Confidence threshold
        self.confidence_slider = ControlSlider("Confidence", 0, 100, 25, "%")
        self.confidence_slider.value_changed.connect(
            self.video_processor.set_confidence)
        layout.addWidget(self.confidence_slider)

        parent_layout.addWidget(group)

    def setup_footer(self, parent_layout):
        """Setup footer with version and info"""
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setSpacing(5)

        # Config button
        config_btn = TouchButton("Settings", "secondary", self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        config_btn.clicked.connect(self.show_settings)
        footer_layout.addWidget(config_btn)

        # Version
        version_label = QLabel("SafeSight v1.0")
        version_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_layout.addWidget(version_label)

        # Credits
        credits_label = QLabel("Powered by YOLOv11/v12")
        credits_label.setStyleSheet("color: #495057; font-size: 10px;")
        credits_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_layout.addWidget(credits_label)

        parent_layout.addWidget(footer)

    # Event handlers
    def start_stream(self):
        """Start video streaming"""
        # Get video source (default to camera 0)
        source = 0  # TODO: Add source selection dialog

        self.video_processor.set_source(source)
        self.video_processor.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.snapshot_btn.setEnabled(True)
        self.danger_zone_btn.setEnabled(True)
        self.clear_zone_btn.setEnabled(True)
        self.status_indicator.set_status("Streaming", "#28a745")

    def stop_stream(self):
        """Stop video streaming"""
        self.video_processor.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(False)
        self.danger_zone_btn.setEnabled(False)
        self.clear_zone_btn.setEnabled(False)
        self.status_indicator.set_status("Stopped", "#ffc107")

        # Reload placeholder
        self.load_placeholder()

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def show_snapshot(self):
        """Show snapshot modal with current frame"""
        current_frame = self.video_processor.get_current_frame()

        if current_frame is None:
            QMessageBox.warning(
                self, "No Frame", "No frame available to snapshot.")
            return

        # Create and show modal
        modal = SnapshotModal(self)
        modal.set_frame(current_frame)
        modal.exec()

    def show_zone_modal(self):
        """Show danger zone drawing modal"""
        current_frame = self.video_processor.get_current_frame()

        if current_frame is None:
            QMessageBox.warning(self, "Kein Bild",
                                "Kein Bild zum Zeichnen verfügbar.")
            return

        # Create and show zone modal
        modal = ZoneModal(self)
        modal.load_snapshot(current_frame)
        modal.zone_saved.connect(self.on_zone_saved)
        modal.exec()

    def on_zone_saved(self, zone_data):
        """Handle saved danger zone"""
        if zone_data is None:
            return

        if zone_data['shape'] == 'rectangle':
            x1, y1, x2, y2 = zone_data['coords']
            self.video_processor.danger_zone.set_rectangle(x1, y1, x2, y2)
            self.status_indicator.set_status(
                f"Rechteck-Zone: ({x1},{y1})-({x2},{y2})", "#dc3545")
        elif zone_data['shape'] == 'polygon':
            points = zone_data['coords']
            self.video_processor.danger_zone.set_polygon(points)
            self.status_indicator.set_status(
                f"Polygon-Zone: {len(points)} Punkte", "#dc3545")

    def clear_danger_zone(self):
        """Clear the danger zone"""
        self.video_processor.danger_zone.clear()
        self.status_indicator.set_status("Zone gelöscht", "#28a745")
    
    def set_relay_state(self, relay_name, state):
        """Set relay state (True=ON, False=OFF) and update UI"""
        try:
            relay = relays.get(relay_name)
            if relay is None:
                return
            
            # Only act if state changes
            if relay.is_active != state:
                if state:
                    relay.on()
                else:
                    relay.off()
                
                # Update UI button if it exists
                if relay_name == 'red':
                    if hasattr(self, 'red_light_btn'):
                        btn = self.red_light_btn
                        if state:
                            btn.setText("Red ON")
                            # Only update status if we are in streaming mode
                            if self.video_processor.running:
                                self.status_indicator.set_status("DANGER DETECTED!", "#dc3545")
                        else:
                            btn.setText("Red OFF")
                            if self.video_processor.running:
                                self.status_indicator.set_status("Streaming", "#28a745")

        except Exception as e:
            print(f"Error setting relay {relay_name}: {e}")

    def on_danger_detected(self, detected):
        """Handle danger detection signal"""
        # Only control relay if danger zone is set
        if self.video_processor.danger_zone.has_zone():
            self.set_relay_state('red', detected)

    def toggle_relay(self, relay_name, button):
        """Toggle relay on/off"""
        try:
            relay = relays.get(relay_name)
            if relay is None:
                self.show_error(f"Relay '{relay_name}' not found")
                return
            
            # Toggle the relay
            if relay.is_active:
                relay.off()
                # Update button appearance for OFF state
                button.setText("Red OFF")
                self.status_indicator.set_status(f"{relay_name.capitalize()} light OFF", "#6c757d")
            else:
                relay.on()
                # Update button appearance for ON state
                button.setText("Red ON")
                self.status_indicator.set_status(f"{relay_name.capitalize()} light ON", "#ffc107")
        
        except Exception as e:
            self.show_error(f"Error controlling {relay_name} relay: {str(e)}")

    def on_model_changed(self, index):
        """Handle model selection change"""
        model_names = list(self.MODELS.keys())
        if index < len(model_names):
            model_key = model_names[index]
            model_path = self.MODELS[model_key]

            if os.path.exists(model_path):
                self.video_processor.load_model(model_path)
                self.status_indicator.set_status(
                    f"Model: {model_key}", "#0d6efd")
                self.model_mode.clear()
                self.model_mode.addItems(ModelMode.fillModelModeDropdown(
                    ModelMode.determineModeFromModel(model_key)))

            else:
                self.show_error(f"Model not found: {model_path}")

    def on_model_mode_changed(self, index):
        """Handle model danger zone mode change"""
        pass

    def update_frame(self, frame):
        """Update video display with new frame"""
        self.video_display.set_frame(frame)

    def update_fps(self, fps):
        """Update FPS display"""
        self.status_indicator.set_fps(fps)

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.status_indicator.set_status("Error", "#dc3545")

    def show_settings(self):
        """Show settings dialog"""
        QMessageBox.information(
            self,
            "Settings",
            "Settings dialog coming soon!\n\n"
            "Future features:\n"
            "- Video source selection\n"
            "- Recording options\n"
            "- Alert configurations\n"
            "- Network settings"
        )

    def load_placeholder(self):
        """Load placeholder image"""
        placeholder_path = ""
        if os.path.exists(placeholder_path):
            pixmap = QPixmap(placeholder_path)
            self.video_display.setPixmap(
                pixmap.scaled(
                    self.video_display.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

    def closeEvent(self, event):
        """Handle window close"""
        self.video_processor.stop()
        # Turn off all relays
        all_relays_off()
        event.accept()


def main():
    """Application entry point"""
    app = QApplication(sys.argv)

    # Set application-wide style
    app.setStyle("Fusion")

    window = SafeSightGUI()

    # Optional: Auto-start stream (uncomment and set source)
    # window.start_stream(0)  # Camera 0
    # window.start_stream("http://10.0.1.1:5000/video_feed")  # Network stream

    # Enable application-wide touch event synthesis
    app.setAttribute(
        Qt.ApplicationAttribute.AA_SynthesizeTouchForUnhandledMouseEvents, True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
