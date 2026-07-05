# VisionX - Deep Learning Gaze Tracker

VisionX is an Edge AI gaze-tracking system that allows you to control a simulated dashboard interface using only your webcam. It uses [L2CS-Net](https://github.com/ahmed-nabil/L2CS-Net), a robust deep learning model for gaze estimation (ResNet50 based), to accurately estimate gaze direction even under varying head poses.

No cloud processing is required; everything runs locally on your machine.

## Features
- **Deep Learning Estimation:** Uses L2CS-Net for high-accuracy gaze pitch and yaw extraction.
- **Web UI Dashboard:** A sleek, non-scrolling, gaze-interactive web dashboard with magnetic snap effects and a modern glassmorphic aesthetic.
- **Auto-Calibration:** Calibrates the system using an intuitive GUI by tracking your gaze on on-screen targets.
- **Smart Filtering:** Utilizes Median and One-Euro filters to eliminate cursor jitter, distinguishing between steady gazes and quick glances.
- **Passive Recalibration:** Accuracy drifts up as you use it. Every confirmed action slightly nudges the calibration.
- **Quality Gating:** Only relies on frames with confident detections, pausing the cursor during blinks and lost tracking instead of wildly jumping.

## Prerequisites
- **Python 3.8 or higher**
- A webcam

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dibyanshusamal9/VisionX.git
   cd VisionX
   ```

2. **Set up a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This might take a while due to PyTorch and other deep learning dependencies.)*

## Running the Application

Execute the main script to start the gaze tracker:

```bash
python main.py
```

## Controls & Usage

### 1. Calibration Phase
When you first run the app, you will need to calibrate it.
- **Look at the yellow dot** on the screen and **hold your head and gaze steady**.
- Keep your natural driving/sitting posture.
- A progress bar will fill up; once it fills, the point is automatically captured.
- **Shortcuts during calibration:**
  - `C`: Capture point manually
  - `S`: Skip the current point
  - `Q`: Quit

### 2. Normal Operation Phase
Once calibrated, the Python server will track your gaze and transmit it via WebSockets.
- Open `web_ui/index.html` in your web browser to view the interactive dashboard.
- The web interface features magnetic buttons. Your gaze will act as the mouse cursor.
- **SPACEBAR (in Python window)**: Confirm selection (Acts like a steering wheel thumb-button / click).
- `R (in Python window)`: Recalibrate (if you feel accuracy drifting).
- `Q (in Python window)`: Quit the application.

## Troubleshooting
- **Face/eyes not detected:** Ensure you are in a well-lit room and facing the webcam.
- **Cursor moves while my eyes are still:** Your head might be shifting slightly. Try to press `R` to recalibrate and make sure to move your head naturally towards the dots during calibration.
- **Webcam not opening:** Check the `CAM_INDEX` variable in `main.py` and change it (from `0` to `1` or `2`) if you are using an external camera.

## License
MIT License
