"""
Modal dialog for creating danger zones
"""
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QMessageBox, QStyle
from PyQt6.QtCore import Qt, pyqtSignal
from zone_canvas import ZoneCanvas
from widgets import TouchButton


class ZoneModal(QDialog):
    """Modal for drawing danger zones"""

    zone_saved = pyqtSignal(object)  # Emits zone data when saved

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gefahrenbereich festlegen")
        self.setModal(True)
        self.setStyleSheet("background-color: #1e1e1e;")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 900)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Gefahrenbereich festlegen")
        title.setStyleSheet(
            "color: #ffffff; font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Shape selector
        shape_layout = QHBoxLayout()
        shape_label = QLabel("Form:")
        shape_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        shape_layout.addWidget(shape_label)

        self.shape_selector = QComboBox()
        self.shape_selector.addItems(["Rechteck", "Polygon"])
        self.shape_selector.setStyleSheet("""
            QComboBox {
                background-color: #3a3a3a;
                color: #f8f9fa;
                border: 2px solid #495057;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
                min-height: 35px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                color: #f8f9fa;
                selection-background-color: #495057;
            }
        """)
        self.shape_selector.currentIndexChanged.connect(self.on_shape_changed)
        shape_layout.addWidget(self.shape_selector)
        shape_layout.addStretch()
        layout.addLayout(shape_layout)

        # Instructions
        self.instructions = QLabel(
            "Klicken und ziehen Sie, um ein Rechteck zu zeichnen")
        self.instructions.setStyleSheet(
            "color: #adb5bd; font-size: 12px; font-style: italic;")
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)

        # Canvas
        self.canvas = ZoneCanvas()
        layout.addWidget(self.canvas, 1) # Add stretch factor of 1

        # Buttons
        button_layout = QHBoxLayout()

        self.clear_btn = TouchButton("Zurücksetzen", "secondary", self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.clear_btn.clicked.connect(self.clear_zone)
        button_layout.addWidget(self.clear_btn)

        self.save_btn = TouchButton("Speichern", "success", self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_btn.clicked.connect(self.save_zone)
        button_layout.addWidget(self.save_btn)

        self.cancel_btn = TouchButton("Abbrechen", "danger", self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def load_snapshot(self, frame):
        """Load a frame to draw on"""
        self.canvas.load_image(frame)

    def on_shape_changed(self, index):
        """Handle shape selector change"""
        shape = 'rectangle' if index == 0 else 'polygon'
        self.canvas.set_zone_shape(shape)

        if shape == 'rectangle':
            self.instructions.setText(
                "Klicken und ziehen Sie, um ein Rechteck zu zeichnen")
        else:
            self.instructions.setText(
                "Klicken Sie, um Polygonpunkte zu setzen. "
                "Doppelklick zum Abschließen. Rechtsklick zum Entfernen des letzten Punktes."
            )

    def clear_zone(self):
        """Clear the current drawing"""
        self.canvas.reset_drawing()
        self.canvas._update_display()

    def save_zone(self):
        """Save the drawn zone"""
        zone_data = self.canvas.get_zone_data()
        if zone_data is None:
            if self.canvas.zone_shape == 'rectangle':
                QMessageBox.warning(self, "Keine Zone",
                                    "Bitte zeichnen Sie eine Zone!")
            else:
                QMessageBox.warning(
                    self, "Polygon nicht abgeschlossen",
                    "Bitte schließen Sie das Polygon mit einem Doppelklick ab!"
                )
            return

        self.zone_saved.emit(zone_data)
        self.accept()
