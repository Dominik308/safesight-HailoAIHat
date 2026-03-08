"""
Video processing module with YOLO integration
Handles video capture, YOLO inference, and frame processing
"""
from helper.danger_zone import DangerZone
from helper.model_handler import get_model_type, draw_detections
from helper.hailo_inference import HailoInference
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex
from ultralytics import YOLO
import sys
import os
import time
sys.path.append(os.path.dirname(__file__))


class VideoProcessor(QThread):
    """
    Background thread for video capture and YOLO processing
    Emits processed frames to the GUI
    """
    frame_ready = pyqtSignal(np.ndarray)
    fps_update = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    danger_detected_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.model = None
        self.model_type = "detect"  # Track current model type
        self.is_hailo_model = False  # Track if using Hailo
        self.camera_rotation = 0  # Camera rotation angle (0, 90, 180, 270)
        self.running = False
        self.mutex = QMutex()
        
        # Processing settings
        self.brightness = 0
        self.contrast = 1.0
        self.gaussian_blur = 0
        self.noise_level = 0
        self.conf_threshold = 0.25
        self.source = None

        # FPS calculation
        self.frame_count = 0
        self.fps = 0.0

        # Store current frame
        self.current_frame = None

        # Danger zone
        self.danger_zone = DangerZone()

    def load_model(self, model_path):
        """Load YOLO or Hailo model from path"""
        try:
            self.mutex.lock()
            
            # Unload existing model if present
            if self.model is not None:
                if hasattr(self.model, 'release'):
                    print("Releasing previous Hailo model...")
                    self.model.release()
                self.model = None
                # Force garbage collection to ensure resources are freed
                import gc
                gc.collect()
            
            # Check if this is a Hailo model
            if model_path.endswith('.hef') or model_path.endswith('.har'):
                print(f"Loading Hailo model: {model_path}")
                
                # Check if .har needs extraction/compilation
                if model_path.endswith('.har'):
                    self.mutex.unlock()
                    self.error_occurred.emit(
                        "Fehler: .har-Dateien sind Archive, keine kompilierten Modelle.\n\n"
                        "Sie benötigen eine .hef (Hailo Executable Format) Datei.\n\n"
                        "Bitte kompilieren Sie das Modell mit dem Hailo Dataflow Compiler\n"
                        "oder besorgen Sie sich ein vorkompiliertes .hef-Modell."
                    )
                    return
                
                # Determine model type from filename for .hef (detect/segment/pose)
                self.model_type = get_model_type(model_path)
                self.is_hailo_model = True
                
                # Use HailoInference for all .hef models
                self.model = HailoInference(model_path, rotation=self.camera_rotation)
                print(f"Hailo model loaded: {model_path} (type: {self.model_type})")
            else:
                # Standard YOLO model
                self.is_hailo_model = False
                
                # Determine model type from filename
                self.model_type = get_model_type(model_path)
                
                # Determine task for YOLO
                if self.model_type in ("pose", "hand"):
                    task = "pose"
                elif self.model_type == "segment":
                    task = "segment"
                else:
                    task = "detect"
                
                # Load model with explicit task
                self.model = YOLO(model_path, task=task)
                print(f"YOLO model loaded: {model_path} (type: {self.model_type}, task: {task})")
            
            self.mutex.unlock()
        except Exception as e:
            self.mutex.unlock()
            self.error_occurred.emit(f"Modell konnte nicht geladen werden: {str(e)}")

    def set_source(self, source):
        """Set video source (camera index or URL)"""
        self.source = source

    def set_brightness(self, value):
        """Set brightness adjustment (-50 to 50)"""
        self.brightness = value - 50

    def set_contrast(self, value):
        """Set contrast adjustment (0.0 to 2.0)"""
        self.contrast = value / 50.0

    def set_gaussian_blur(self, value):
        """Set Gaussian blur kernel size (0 to 20)"""
        self.gaussian_blur = value

    def set_gaussian_noise(self, value):
        """Set Gaussian noise level (0 to 100)"""
        self.noise_level = value

    def set_confidence(self, value):
        """Set YOLO confidence threshold (0.0 to 1.0)"""
        self.conf_threshold = value / 100.0

    def run(self):
        """Main processing loop"""
        if self.source is None:
            self.error_occurred.emit("Keine Videoquelle festgelegt")
            return

        # Open video capture
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.error_occurred.emit(
                f"Videoquelle konnte nicht geöffnet werden: {self.source}")
            return

        self.running = True
        fps_counter = cv2.getTickCount()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.error_occurred.emit("Frame konnte nicht gelesen werden")
                break

            # Apply adjustments
            frame = self.adjust_frame(frame)

            # Draw danger zone
            self.danger_zone.draw_on_frame(frame)

            # Run YOLO/Hailo inference if model is loaded
            self.mutex.lock()
            if self.model is not None:
                try:
                    # Run inference (works for all model types)
                    results = self.model(
                        frame, conf=self.conf_threshold, verbose=False)

                    danger_detected = False

                    # Draw detections based on model type
                    if results and len(results) > 0:
                        if self.is_hailo_model:
                            # For Hailo models, check if we need special visualization
                            if self.model_type == "pose":
                                # Draw pose keypoints directly from raw outputs
                                from helper.hailo_inference import draw_pose_keypoints
                                if hasattr(self.model, '_last_outputs') and self.model._last_outputs:
                                    danger_detected = draw_pose_keypoints(
                                        frame, 
                                        self.model._last_outputs, 
                                        model=self.model,
                                        conf_threshold=self.conf_threshold, 
                                        danger_zone=self.danger_zone
                                    )
                            elif self.model_type == "segment":
                                # Draw segmentation masks directly from raw outputs
                                from helper.hailo_inference import draw_segmentation_masks
                                if hasattr(self.model, '_last_outputs') and self.model._last_outputs:
                                    danger_detected = draw_segmentation_masks(
                                        frame, 
                                        self.model._last_outputs, 
                                        model=self.model,
                                        conf_threshold=self.conf_threshold, 
                                        danger_zone=self.danger_zone,
                                        class_names=self.model.class_names
                                    )
                            else:
                                # Standard detection drawing
                                danger_detected = draw_detections(frame, results, self.danger_zone, self.model_type)
                        else:
                            # Standard YOLO drawing
                            danger_detected = draw_detections(frame, results, self.danger_zone, self.model_type)
                    
                    self.danger_detected_signal.emit(danger_detected)

                except Exception as e:
                    import traceback
                    print(f"Inference error: {e}")
                    print(f"Full traceback:")
                    traceback.print_exc()
            self.mutex.unlock()

            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_tick = cv2.getTickCount()
                self.fps = 30 / ((current_tick - fps_counter) /
                                 cv2.getTickFrequency())
                self.fps_update.emit(self.fps)
                fps_counter = current_tick

            # Store current frame for snapshot
            self.current_frame = frame.copy()

            # Emit frame
            self.frame_ready.emit(frame)

        # Cleanup
        if self.cap:
            self.cap.release()

    def adjust_frame(self, frame):
        """Apply Gaussian blur/noise, brightness and contrast adjustments"""
        if self.gaussian_blur > 0:
            frame = cv2.GaussianBlur(
                frame, (self.gaussian_blur * 2 + 1, self.gaussian_blur * 2 + 1), 0)
        if self.noise_level > 0:
            noise = np.random.normal(
                0, self.noise_level/50, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)

        return cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)

    def get_current_frame(self):
        """Get the current frame for snapshot"""
        return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        """Stop processing"""
        self.running = False
        self.wait()
