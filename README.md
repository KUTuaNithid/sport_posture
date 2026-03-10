# Running Posture Correction AI

A computer vision application built with Python, Streamlit, and Ultralytics YOLO to analyze side-profile running videos and provide actionable biomechanical feedback.

## Features

- **Automated Pose Estimation**: Uses YOLO-Pose (v8, v11, etc.) to track body keypoints with high accuracy.
- **Biomechanical Metrics Tracking**:
  - **Torso Lean**: Calculates the forward or backward lean of the runner based on running direction.
  - **Knee Flexion**: Computes the external knee angle during the foot strike phase.
  - **Overstride Ratio**: Calculates overstriding as a percentage of the runner's physical leg length to normalize the measurement against camera distance.
- **Moving Average Smoothing**: Averages kinematic data across a 5-frame window on impact to prevent single-frame flickering.
- **Intel GPU Acceleration**: Built-in toggle to use OpenVINO for significantly faster inference on Intel CPUs and Integrated GPUs.
- **Rich Debug Visualization**: Outputs a fully annotated slow-motion gallery of the strike window to visually verify the AI's math, along with an annotated full-length MP4 video.

## Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:KUTuaNithid/sport_posture.git
   cd sport_posture
   ```

2. Create a virtual environment (Python 3.11 recommended):
   ```bash
   python -m venv venv311
   ```

3. Activate the environment:
   - **Windows (PowerShell)**: `.\venv311\Scripts\activate`
   - **Mac/Linux**: `source venv311/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the Streamlit web interface:

```bash
python -m streamlit run app.py
```

1. Open your browser to the given Local URL (usually `http://localhost:8502`).
2. Upload a clear, side-profile video of your run (\`.mp4\`, \`.mov\`, or \`.avi\`).
3. Select your YOLO model (e.g., `yolo26x-pose.pt` for maximum accuracy, or `yolov8n-pose.pt` for speed).
4. (Optional) Enable **Intel GPU Acceleration (OpenVINO)** if you have supported Intel hardware.
5. (Optional) Fine-tune the "Minimum Keypoint Confidence" slider to filter out false detections.
6. Click **Analyze Posture**!

## Project Structure

- `app.py`: The Streamlit graphical frontend.
- `analyzer.py`: The core video processing, YOLO inference, and vector mathematics backend.
- `requirements.txt`: Python package dependencies.

## Hardware Support

If OpenVINO acceleration is toggled on the UI, the app will automatically convert standard PyTorch `.pt` models to optimized OpenVINO formatting on the first run, routing the inference workload to your Intel CPU/iGPU for massively improved performance.
