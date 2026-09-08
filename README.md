# Borrame

Real-time computer vision system built with Python, OpenCV, and MediaPipe. The application generates an interactive invisibility effect by detecting when the user closes their fist, replacing their body region with a previously captured static background, while simultaneously providing hand skeletal tracking and spatial metrics.

---

## Key Features

- **Background Segmentation and Replacement**: Dynamic real-time replacement of the user's segmented area with minimal latency.
- **Hand Skeletal Tracking (MediaPipe Hand Landmarker)**:
  - Detection of 21 anatomical keypoints of the hand.
  - Continuous Euclidean distance calculation between fingertips and the wrist joint.
  - Gesture state classification (open hand vs. closed fist).
  - Temporal hysteresis filter (*debouncer*) to stabilize state transitions and mitigate false positives due to occlusion.
- **Human Presence Detection (Bounding Box)**: Continuous localization of the user within the frame using TensorFlow Lite-based models.
- **User Interface (Tkinter)**: Integrated dark-themed control panel, guided background capture wizard, and visual overlay selectors.
- **Live Information Display (HUD)**: On-screen video overlay featuring status indicators, frames per second (FPS), and quick controls.

---

## Repository Structure

```text
├── main.py                     # Graphical user interface and main loop
├── borrame.py                  # Processing module and invisibility logic
├── utils.py                    # Hand classifiers, detectors, and helper utilities
├── requirements.txt            # Project dependency specifications
├── efficientdet_lite0.tflite   # Object/person detection model
├── hand_landmarker.task        # MediaPipe hand landmarker model
├── selfie_segmenter.tflite     # Silhouette segmentation model
├── .gitignore                  # Git ignore patterns
└── README.md                   # Project technical documentation
```

---

## System Requirements

- **Python**: Version 3.9 or higher.
- **Webcam**: Recommended minimum resolution of 720p (1280x720).
- **Operating System**: Compatible with Windows, Linux, and macOS.

---

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alejandrocol-dev/BorradorPY.git
   cd BorradorPY
   ```

2. **Create and activate a virtual environment (recommended)**:
   - On Windows:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - On Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

```bash
python main.py
```

---

## Operational Workflow

1. **Step 1 - Background Capture**:
   - Click the `1. Capturar Fondo` button.
   - Clear the camera's field of view during the 5-second countdown.
2. **Step 2 - Start the System**:
   - Click the `2. Iniciar Borrame` button.
   - Position yourself in front of the camera within the frame.
3. **Gesture Control**:
   - **Open hand**: Normal display with skeletal overlay and measurement vectors.
   - **Closed fist**: Activates the invisibility effect over the user.

---

## Real-Time Controls (Camera Mode)

| Key | Function |
|:---:|---|
| `B` | Toggle bounding box visibility. |
| `H` | Toggle hand skeletal landmarks visibility. |
| `C` | Recapture background frame on the fly. |
| `Q` / `Esc` | Exit capture mode and return to the main interface. |

---

## License

This project is distributed under the terms of the MIT License. For more information, see the [LICENSE](LICENSE) file.
