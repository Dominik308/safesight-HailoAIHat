# SafeSight – AI Safety Monitoring with Hailo AI Hat

Real-time AI-powered safety monitoring system running on a Raspberry Pi 5 with the Hailo-8 AI accelerator. SafeSight performs live object detection, segmentation, and pose estimation to monitor configurable danger zones and trigger hardware relay alerts.

## Features

- **Real-time YOLO inference** – supports detection, segmentation, pose estimation, and hand detection models
- **Hailo-8 acceleration** – hardware-accelerated inference via `.hef` models for low-latency processing
- **PyTorch fallback** – also runs standard Ultralytics `.pt` models on CPU/GPU
- **Danger zone monitoring** – define rectangular or polygon zones on the video feed; triggers alerts when objects enter
- **Multiple detection modes** – center point, any overlap, complete inside, mask overlap, keypoint-based checks
- **GPIO relay control** – drives a physical relay (BCM pin 23) for external warning devices (lights, sirens)
- **Touch-friendly GUI** – PyQt6 interface with kinetic scrolling, fullscreen toggle, and on-screen controls
- **Adjustable image processing** – brightness, contrast, Gaussian blur, noise, and confidence threshold sliders
- **Snapshot export** – capture and save the current processed frame
- **Camera rotation** – 0°, 90°, 180°, 270° rotation support

## Hardware Requirements

| Component | Details |
|---|---|
| Raspberry Pi 5 | Main compute board |
| Hailo-8 AI Hat | PCIe AI accelerator for `.hef` model inference |
| Geekworm GPIO extension | GPIO breakout (gpiochip0) |
| Relay module | Connected to BCM pin 23 |
| USB/CSI camera | Video input source |

## Software Requirements

- Python 3.10+
- Raspberry Pi OS (64-bit)
- Hailo Runtime & `hailo_platform` Python package
- `gpiod` library for GPIO access on RPi 5

### Python Dependencies

```
PyQt6
opencv-python
numpy
ultralytics
hailo_platform
gpiod
```

## Project Structure

```
├── start_safesight.sh          # Launch script (activates venv & runs app)
├── DemoCase/
│   ├── app.py                  # Main application & GUI (SafeSightGUI)
│   ├── video_processor.py      # Video capture & YOLO inference thread
│   ├── widgets.py              # Custom PyQt6 widgets (VideoDisplay, sliders, buttons)
│   ├── zone_modal.py           # Dialog for creating danger zones
│   ├── zone_canvas.py          # Canvas widget for drawing rectangle/polygon zones
│   ├── helper/
│   │   ├── hailo_inference.py  # Hailo .hef model wrapper
│   │   ├── model_handler.py    # Model type detection & drawing utilities
│   │   ├── model_mode.py       # UI mode/style helpers per model type
│   │   ├── danger_zone.py      # Danger zone geometry & overlap checks
│   │   └── yolo_decoding.py    # Raw YOLOv5/v8/v11 output decoding for Hailo
│   └── Models/                 # Model files (.hef and .pt) – not tracked by git
```

## Supported Models

### Hailo (.hef) – hardware-accelerated

| Model | Type |
|---|---|
| yolov11n / yolov11s / yolov11m | Detection |
| yolov8s_pose / yolov8m_pose | Pose estimation |
| yolov5n_seg / yolov5m_seg | Segmentation |

### Ultralytics (.pt) – CPU/GPU

| Model | Type |
|---|---|
| yolo11n.pt | Detection |
| yolo11n-seg.pt | Segmentation |
| yolo11n-pose.pt | Pose estimation |
| best-hand-training.pt / hand-detection.pt | Hand detection |
| css-best.pt | Custom CSS detection |
| helmet_detection.pt | Helmet detection |

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd safesight-HailoAIHat
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install PyQt6 opencv-python numpy ultralytics gpiod
```

> **Note:** `hailo_platform` must be installed according to Hailo's documentation for your device.

### 4. Add model files

Place your `.hef` and `.pt` model files into `DemoCase/Models/`. These are not included in the repository due to their size.

### 5. Run the application

```bash
cd DemoCase
python3 app.py
```

Or use the provided launch script:

```bash
./start_safesight.sh
```

## Usage

1. Select a model from the dropdown in the control panel.
2. Choose a video source (camera index or file).
3. Optionally define a danger zone via the zone dialog (rectangle or polygon).
4. Adjust confidence threshold and image processing sliders as needed.
5. When an object enters the danger zone, the relay is activated and a visual alert is shown.

## License

This project is licensed under the [GNU AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html) to comply with the [Ultralytics YOLO license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

This project is intended for **educational and non-commercial use only**.
