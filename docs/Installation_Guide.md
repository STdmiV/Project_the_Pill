# Pill Project Pilot GUI – Getting Started Tutorial

This guide will walk you through setting up and running the Pill Project Pilot GUI from scratch using VSCode. It covers downloading the code, installing dependencies, launching the application, and understanding each GUI tab.

---

### Set Up Your Environment

1. **Download and Install Python**:
   - Download Python from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/).
   - During installation, ensure you check the box **"Add Python to PATH"**.
   - Verify the installation by running the following command in a terminal:
     ```bash
     python --version
     ```
     This should display the installed Python version.

2. **Download and Install VSCode**:
   - If you don’t already have VSCode installed, download it from [https://code.visualstudio.com/](https://code.visualstudio.com/) and follow the installation instructions for your operating system.

3. **Install Recommended VSCode Extensions**:
   - Open the Extensions view in VSCode (`Ctrl+Shift+X`) and search for the following extensions:
     - **Python (by Microsoft)**: Essential for Python development.
     - **Pylance (by Microsoft)**: Improves IntelliSense and type checking.
     - **GitLens**: Enhances Git capabilities within VSCode.
     - **Python Docstring Generator**: Helps create properly formatted docstrings.
     - *(Optional)* **Bracket Pair Colorizer**: Improves readability of nested code.
     - *(Optional)* **Prettier - Code Formatter**: Maintains consistent code style.

4. **Create a Folder for the Project**:
   - Open your file explorer and create a new folder where you want to store the project (e.g., `Pill_Project`).

5. **Clone the Repository**:
   - Open VSCode and launch the integrated terminal (`Ctrl + `` `) or use any terminal of your choice.
   - Navigate to the folder you just created:
     ```bash
     cd path/to/your/folder
     ```
   - Clone the repository into this folder:
     ```bash
     git clone https://github.com/STdmiV/Project_the_Pill
     ```
   - Navigate into the project directory:
     ```bash
     cd Project_the_Pill
     ```

6. **Open the Project in VSCode**:
   - In the terminal, run:
     ```bash
     code .
     ```
   - This opens the project folder in VSCode.

---

## 2. Install Dependencies

### Create and Activate a Virtual Environment (Recommended)
1. Create a virtual environment named `venv`:
    ```bash
    python -m venv venv
    ```
2. Activate the virtual environment:
    - **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    - **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```
    You should see `(venv)` prefixed in your terminal prompt.

### Install Required Packages
1. With the virtual environment activated, install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    This installs libraries like OpenCV, NumPy, PyQt5, pyModbusTCP, etc.

---

## 3. Launch the Application

### Run the Application
1. Ensure your virtual environment is active.
2. Run the main script:
    ```bash
    python main.py
    ```

### Exit the Application
- Close the main GUI window. The application will:
  - Stop running background threads.
  - Disconnect from the Modbus server (if connected).
  - Save detection parameters (`detection_params.json`).
  - Release resources cleanly.

---

## 4. GUI Overview: Tab Descriptions and Usage

### Working Mode Tab
- **Purpose**: Real-time video processing, object detection, and optional Modbus data transmission.
- **Key Features**:
  - **Source Selection**: Choose `Working video` (file) or `Camera` (index).
  - **Start/Stop Controls**: Start or stop the processing loop.
  - **Video Display**: Shows live feed with detected objects and annotations.
  - **Modbus Option**: Enable `Send data via Modbus` for PLC communication.
  - **Debug Views**: Enable `Show Debug Views Below` for intermediate processing steps.
- **How to Use**:
  1. Select the source and specify the video path or camera index.
  2. Click `Start` and confirm the detected working area.
  3. Adjust detection parameters live using the right panel.
  4. Click `Stop` to halt processing.

### Modbus TCP Tab
- **Purpose**: Configure and manage Modbus TCP communication.
- **Key Features**:
  - **Network Scan**: Discover devices on the local subnet.
  - **Connection Controls**: Connect or disconnect from the Modbus server.
  - **PLC Data Structure Settings**: Configure Modbus addresses and parameters.
  - **Save Settings**: Save configuration to `modbus_config.json`.
- **How to Use**:
  1. Enter the target IP and port or use the `Scan` button.
  2. Configure PLC settings and save them.
  3. Click `Connect` to establish communication.

### Calibration Tab
- **Purpose**: Calibrate the camera using a Charuco board pattern.
- **Key Features**:
  - **Source Selection**: Choose `Calibration video` or `Camera`.
  - **Live Preview**: Displays detected markers and captured frames.
  - **Calibration Controls**: Start or stop the calibration process.
- **How to Use**:
  1. Select the source and provide the video path or camera index.
  2. Click `Start Calibration` and ensure the board is detected reliably.
  3. Once finished, calibration data is saved to `camera_calibration.npz`.

---

## 5. Recommended VSCode Extensions
- **Python (by Microsoft)**: Essential for Python development.
- **Pylance (by Microsoft)**: Improves IntelliSense and type checking.
- **GitLens**: Enhances Git capabilities within VSCode.
- **Python Docstring Generator**: Helps create properly formatted docstrings.
- *(Optional)* **Bracket Pair Colorizer**: Improves readability of nested code.
- *(Optional)* **Prettier - Code Formatter**: Maintains consistent code style.

To install, open the Extensions view (`Ctrl+Shift+X`), search for the extension name, and click `Install`.

---
