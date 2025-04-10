✅ Project Setup Guide (from scratch)
🧱 Prerequisites

    Python version: Python 3.10+ (recommended)

    Git: Installed and configured

    GitHub repo: Make sure your repository is public or accessible via token if private.

    External dependencies (see requirements.txt)

📦 Step 1: Clone the repository

git clone https://github.com/STdmiV/Project_the_Pill.git
cd Project_the_Pill

🧰 Step 2: Set up a virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/macOS

📦 Step 3: Install all dependencies

pip install -r requirements.txt

🗃 Step 4: Required folders & files

Ensure that the following directories exist and contain valid data:

    app_data/:

        camera_calibration.npz — created after calibration (see below)

        detection_params.json — created automatically after first run or GUI tuning

        identification_params.json — optional, stores known object categories

        detected_objects.csv — auto-generated after each session

If missing, these files will be generated automatically during runtime.
🖥 How to Use the Application

Launch the GUI:

python src/main.py

📸 Step 1: Calibration (only once per setup)
🧭 Why it's needed:

To correct camera distortion using ChArUco markers.
🧪 How to calibrate:

    Open the Calibration tab.

    Select video or camera source:

        For video: load a .mp4 or .avi file with ChArUco board views.

        For live camera: use webcam index (usually 0).

    Click "Start Calibration".

    The system will automatically detect markers frame-by-frame.

    Once enough frames are collected (usually 10–30), the process stops and camera_calibration.npz is saved.

Tips:

    Use a printed ChArUco board (5×7 squares, marker size: 15.4mm, square size: 30.4mm).

    Move the board slowly to different angles.

    Calibration is needed only once per camera.

🎯 Step 2: Object Recognition (Working Mode)

    Go to "Working Mode" tab.

    Choose input source:

        "Working video" — select a prepared pill video.

        "Camera" — select a webcam index.

    Click Start.

📐 Working Area Selection

    The first few frames are used to detect a working area (e.g., A4 paper).

    You'll be prompted with "Yes/No" confirmation buttons for each candidate.

    Once you select "Yes", the working area is locked and used as a mask for detection.

🤖 Detection and Tracking

Once the working area is confirmed:

    Objects (e.g., pills) are detected and classified.

    ID, category, dimensions, and positions are shown in real time.

    Each detection is saved to app_data/detected_objects.csv.

📤 Modbus Integration (Optional)

    Go to "Modbus TCP" tab.

    Enter robot IP and port (default 192.168.0.10:502).

    Click Connect.

    Enable "Send data via Modbus" in the "Working Mode" tab.

    Detected object data will be sent to the robot automatically.

⚙️ Detection Parameters (Advanced Users)

To fine-tune edge detection:

    Use the right panel of the GUI to change:

        SCALE

        BLUR_KERNEL

        CANNY_LOW / HIGH

        MIN_AREA, MAX_AREA

    You can also toggle a debug window to visualize:

        Canny edge result

        Detected contours

Changes are live and saved automatically to detection_params.json.
💾 Saving Data

The following files are auto-generated:

    detected_objects.csv — per-session object log

    detection_params.json — saved tuning parameters

    camera_calibration.npz — saved calibration matrix
