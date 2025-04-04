#main.py


import sys
import numpy as np
import cv2
import os
import socket # For Modbus TCP connection
import threading # For threading
import time # For time management
import logging # For logging
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout, 
    QComboBox, QFileDialog, QMessageBox, QSizePolicy, QCheckBox, QSlider
)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from robot_comm import RobotComm  # Robot communication module
from calib import calibrate_camera # Camera calibration module
import variables  # Global variables and constants
from ultimate import WorkingArea, ObjectDetector

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pill Project Pilot GUI")
        self.resize(800, 600)
        
        self.detection_params = {
            # Default detection parameters
            # These parameters can be adjusted via the GUI
            # and are used for object detection and working area processing.
            "SCALE": 0.5,
            "BLUR_KERNEL": 5,
            "CANNY_LOW": 50,
            "CANNY_HIGH": 150,
            "MIN_AREA": 1000,
            "MAX_AREA": 20000
        }
        self.load_detection_params()
        
         # Initialize working_area_mask and conversion_factor for working area detection
        self.working_area_mask = None
        # Flag to toggle display between overlay_frame (search phase) and objects_overlayed (after detection)
        self.display_overlay_search = False  # Default: show final detection result
        # Initialize working area processor
        self.working_area_processor = WorkingArea(
            detection_params=self.detection_params,
            confirmation_callback=self.confirm_working_area,
            parent=self
        )
        # Central widget and main vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top horizontal layout for tabs and buttons
        top_layout = QHBoxLayout()

        # Tab widget with multiple functional views
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_working_mode_tab(), "Working Mode")
        self.tabs.addTab(QWidget(), "Data Collection")
        self.tabs.addTab(QWidget(), "Data Analysis")
        self.tabs.addTab(self.create_modbus_tab(), "Modbus TCP")  
        self.tabs.addTab(self.create_calibration_tab(), "Calibration") 

        
        top_layout.addWidget(self.tabs)

                # --- Parameter Controls Panel ---
        self.param_control_panel = self.create_param_controls()
        top_layout.addLayout(self.param_control_panel)
        # --- End Parameter Controls Panel ---



        # Add top layout to the main vertical layout
        main_layout.addLayout(top_layout)


    def create_modbus_tab(self):
        """
        Create the "Modbus TCP" configuration tab.
        Allows user to enter IP/Port and connect/disconnect from robot via Modbus TCP.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Input fields for IP and port
        form_layout = QFormLayout()
        self.ip_input = QLineEdit("192.168.0.10")
        self.port_input = QLineEdit("502")
        form_layout.addRow("IP Address:", self.ip_input)
        form_layout.addRow("Port:", self.port_input)
        
        self.device_selector = QComboBox() # Dropdown for IP selection
        self.refresh_button = QPushButton("Scan Network") # Scan for Modbus devices
        self.refresh_button.clicked.connect(self.scan_network_for_modbus) 

        layout.addWidget(self.device_selector)
        layout.addWidget(self.refresh_button)

        # When user selects an IP from the dropdown, fill it into the IP input
        self.device_selector.currentTextChanged.connect(lambda ip: self.ip_input.setText(ip))


        layout.addLayout(form_layout)

        # Buttons for connect and disconnect
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        self.status_label = QLabel("Disconnected")

        self.connect_button.clicked.connect(self.connect_modbus)
        self.disconnect_button.clicked.connect(self.disconnect_modbus)

        layout.addWidget(self.connect_button)
        layout.addWidget(self.disconnect_button)
        layout.addWidget(self.status_label)

        return tab

    def connect_modbus(self):
        """
        Attempt to connect to Modbus TCP server using provided IP and port.
        Update connection status accordingly.
        """
        ip = self.ip_input.text()
        port = int(self.port_input.text())

        self.robot = RobotComm(host=ip, port=port)
        self.robot.connect()
        if self.robot.connected:
            self.status_label.setText("✅ Connected")
        else:
            self.status_label.setText("❌ Connection failed")

    def disconnect_modbus(self):
        """
        Disconnect from Modbus TCP server and update connection status.
        """
        if hasattr(self, 'robot') and self.robot.connected:
            self.robot.disconnect()
            self.status_label.setText("🔌 Disconnected")


    def scan_network_for_modbus(self):
        """
        Scan local network (192.168.0.*) for devices responding on port 502 (Modbus TCP).
        Found IPs are added to the device_selector combo box.
        """
        from concurrent.futures import ThreadPoolExecutor
        subnet = "192.168.0."
        port = 502
        self.device_selector.clear()
        self.status_label.setText("Scanning...")

        def check_ip(ip):
            try:
                with socket.create_connection((ip, port), timeout=0.3):
                    return ip
            except:
                return None

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(check_ip, f"{subnet}{i}") for i in range(1, 255)]
            results = [f.result() for f in futures if f.result()]

        if results:
            self.device_selector.addItems(results)
            self.status_label.setText(f"Found {len(results)} device(s)")
        else:
            self.status_label.setText("No devices found")




            
    def create_calibration_tab(self):
        """
        Create calibration tab with compact layout:
        - Source selection (camera or video)
        - Dynamic controls for selected source
        - Calibration video preview
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Source selection
        self.source_selector = QComboBox()
        self.source_selector.addItems(["Calibration video", "Camera"])
        self.source_selector.currentIndexChanged.connect(self.update_calibration_source)

        # Horizontal layout for "Calibration Source" + QComboBox
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Calibration Source:"))
        source_row.addWidget(self.source_selector)
        layout.addLayout(source_row)


        # Horizontal layout for Calibration Video Path
        self.calib_video_path_input = QLineEdit(variables.CALIBRATION_VIDEO_PATH)
        self.calib_video_path_input.setReadOnly(True)
        self.calib_video_path_input.mousePressEvent = self.open_video_file_from_input

        self.calib_video_browse_button = QPushButton("...")
        self.calib_video_browse_button.setFixedWidth(30)
        self.calib_video_browse_button.clicked.connect(self.browse_calibration_video)

        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Calibration Video Path:"))
        video_row.addWidget(self.calib_video_path_input)
        video_row.addWidget(self.calib_video_browse_button)

        self.video_layout = QVBoxLayout()
        self.video_layout.addLayout(video_row)


        # Camera index input
        self.camera_index_input = QLineEdit(str(variables.CAMERA_INDEX))
        self.camera_index_input.editingFinished.connect(self.update_camera_index)

        self.camera_layout = QVBoxLayout()
        self.camera_layout.addWidget(QLabel("Camera Index:"))
        self.camera_layout.addWidget(self.camera_index_input)

        # Stack layouts (we toggle visibility later)
        layout.addLayout(self.video_layout)
        layout.addLayout(self.camera_layout)

        # Live preview of calibration
        self.calib_video_label = QLabel("Calibration video will appear here")
        
        self.calib_video_label.setMinimumSize(320, 240)
        self.calib_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.calib_video_label.setStyleSheet("background-color: black;")

        # Calibration control buttons (Start/Stop) in one line
        calib_button_row = QHBoxLayout()
        self.calib_start_button = QPushButton("Start Calibration")
        self.calib_stop_button = QPushButton("Stop Calibration")

        self.calib_start_button.clicked.connect(self.start_calibration)
        self.calib_stop_button.clicked.connect(self.stop_calibration)

        calib_button_row.addWidget(self.calib_start_button)
        calib_button_row.addWidget(self.calib_stop_button)
        layout.addLayout(calib_button_row)

        layout.addWidget(self.calib_video_label)

        # Set default visibility
        self.update_calibration_source()




        return tab

    def create_working_mode_tab(self):
        
        """
        Create the Working Mode tab with controls:
        - Source selection (video or camera)
        - Video path input or camera index
        - Video display label
        - Start/Stop buttons
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Source selection
        self.working_source_selector = QComboBox()
        self.working_source_selector.addItems(["Working video", "Camera"])
        self.working_source_selector.currentIndexChanged.connect(self.update_working_source)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Working Source:"))
        source_row.addWidget(self.working_source_selector)
        layout.addLayout(source_row)

        # Video path row
        self.working_video_path_input = QLineEdit(variables.WORKING_VIDEO_PATH)
        self.working_video_path_input.setReadOnly(True)
        self.working_video_path_input.mousePressEvent = self.open_working_video_file_from_input

        browse_button = QPushButton("...")
        browse_button.setFixedWidth(30)
        browse_button.clicked.connect(self.browse_working_video)

        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Working Video Path:"))
        video_row.addWidget(self.working_video_path_input)
        video_row.addWidget(browse_button)
        self.working_video_layout = QVBoxLayout()
        self.working_video_layout.addLayout(video_row)

        # Camera index input
        self.working_camera_index_input = QLineEdit(str(variables.CAMERA_INDEX))
        self.working_camera_layout = QVBoxLayout()
        self.working_camera_layout.addWidget(QLabel("Camera Index:"))
        self.working_camera_layout.addWidget(self.working_camera_index_input)

        layout.addLayout(self.working_video_layout)
        layout.addLayout(self.working_camera_layout)

        # Video display
        self.working_video_label = QLabel("Working mode video will appear here")
        self.working_video_label.setMinimumSize(320, 240)
        self.working_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.working_video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.working_video_label)

        # Start/Stop buttons with Modbus checkbox
        button_row = QHBoxLayout()

        self.working_start_button = QPushButton("Start")
        self.working_stop_button = QPushButton("Stop")
        self.modbus_checkbox = QCheckBox("Send data via Modbus")

        self.working_start_button.clicked.connect(self.start_working_mode)
        self.working_stop_button.clicked.connect(self.stop_working_mode)

        button_row.addWidget(self.working_start_button)
        button_row.addWidget(self.working_stop_button)
        button_row.addWidget(self.modbus_checkbox)
        layout.addLayout(button_row)


        # Set default visibility
        self.update_working_source()

        return tab

    def create_param_controls(self):
        from functools import partial

        panel_layout = QVBoxLayout()
        panel_layout.setAlignment(Qt.AlignTop)

        self.debug_checkbox = QCheckBox("Show Debug View")
        self.debug_checkbox.stateChanged.connect(self.toggle_debug_window)
        panel_layout.addWidget(self.debug_checkbox)

        param_specs = {
            "SCALE": (0.1, 1.5, 0.01),
            "BLUR_KERNEL": (1, 15, 2),  # must be odd
            "CANNY_LOW": (0, 255, 1),
            "CANNY_HIGH": (0, 255, 1),
            "MIN_AREA": (0, 5000, 50),
            "MAX_AREA": (1000, 100000, 500)
        }

        self.param_widgets = {}

        for name, (min_val, max_val, step) in param_specs.items():
            value = self.detection_params[name]
            container = QVBoxLayout()
            container.setSpacing(2)

            # 1. Label
            label = QLabel(f"{name}")
            label.setAlignment(Qt.AlignCenter)
            container.addWidget(label)

            # 2. Horizontal row: - value +
            hbox = QHBoxLayout()
            minus_btn = QPushButton("-")
            minus_btn.setFixedWidth(20)
            plus_btn = QPushButton("+")
            plus_btn.setFixedWidth(20)
            value_label = QLabel(str(value))
            value_label.setFixedWidth(50)
            value_label.setAlignment(Qt.AlignCenter)
            hbox.addWidget(minus_btn)
            hbox.addWidget(value_label)
            hbox.addWidget(plus_btn)
            container.addLayout(hbox)

            # 3. Slider (scaled by step size)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val / step))
            slider.setMaximum(int(max_val / step))
            slider.setValue(int(value / step))
            slider.setFixedWidth(120)
            container.addWidget(slider, alignment=Qt.AlignCenter)

            # 4. Update function
            def update_value(param_name, param_step, param_min, param_max, value_lbl, slider_obj, delta=0, direct_value=None):
                if direct_value is not None:
                    new_val = direct_value
                else:
                    cur_val = self.detection_params[param_name]
                    new_val = cur_val + delta * param_step

                # Special handling for BLUR_KERNEL
                if param_name == "BLUR_KERNEL":
                    new_val = int(new_val)
                    if new_val % 2 == 0:
                        new_val += 1 if delta >= 0 else -1

                # Clamp to min/max
                new_val = max(param_min, min(param_max, new_val))

                # Update global variable
                self.detection_params[param_name] = round(new_val, 2) if isinstance(param_step, float) else int(new_val)

                # Update label and slider
                value_lbl.setText(str(round(new_val, 2)))
                slider_obj.setValue(int(new_val / param_step))
                print(f"[PARAM] {param_name} set to {new_val}")

            # Bind buttons
            def make_step_callback(param, step_size, min_v, max_v, label_widget, slider_widget, delta):
                def callback():
                    update_value(param, step_size, min_v, max_v, label_widget, slider_widget, delta)
                return callback

            minus_btn.clicked.connect(make_step_callback(name, step, min_val, max_val, value_label, slider, -1))
            plus_btn.clicked.connect(make_step_callback(name, step, min_val, max_val, value_label, slider, 1))
            

            # Bind slider
            def slider_callback(val, n=name, s=step, mn=min_val, mx=max_val, lbl=value_label, sld=slider):
                actual_val = val * s
                update_value(n, s, mn, mx, lbl, sld, direct_value=actual_val)

            slider.valueChanged.connect(slider_callback)

            # Add to main panel
            panel_layout.addLayout(container)
            self.param_widgets[name] = (label, value_label, slider)

        return panel_layout


    def toggle_debug_window(self, state):
        if state == Qt.Checked:
            self.debug_window = DebugWindow()
            self.debug_window.show()
        else:
            if hasattr(self, 'debug_window'):
                self.debug_window.close()
                del self.debug_window
    
    
    def browse_video_file(self):
        """
        Open a file dialog to select a calibration video file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Calibration Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path_input.setText(file_path)
    def browse_calibration_video(self):
        """
        Let user select a video file and update variables.CALIBRATION_VIDEO_PATH accordingly.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Calibration Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.calib_video_path_input.setText(file_path)
            variables.CALIBRATION_VIDEO_PATH = file_path
            print(f"Updated CALIBRATION_VIDEO_PATH to: {file_path}")

    def update_calibration_source(self):
        """
        Show/hide relevant controls depending on selected calibration source.
        """
        use_video = self.source_selector.currentText() == "Calibration video"

        def toggle_layout_visibility(layout, visible):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                widget = item.widget()
                if widget:
                    widget.setVisible(visible)
                elif isinstance(item, QHBoxLayout) or isinstance(item, QVBoxLayout):
                    toggle_layout_visibility(item, visible)

        toggle_layout_visibility(self.video_layout, use_video)
        toggle_layout_visibility(self.camera_layout, not use_video)


    def update_camera_index(self):
        """
        Update variables.CAMERA_INDEX from the camera index input field.
        """
        try:
            index = int(self.camera_index_input.text())
            variables.CAMERA_INDEX = index
            print(f"Updated CAMERA_INDEX to: {index}")
        except ValueError:
            print("Invalid camera index entered.")

    def open_video_file_from_input(self, event):
        """
        Trigger file browser when clicking the input field.
        """
        self.browse_calibration_video()

    def start_calibration(self):
        """
        Launch threaded calibration process with GUI video streaming.
        """
        from calib import calibrate_camera_gui

        source = self.source_selector.currentText()
        if source == "Calibration video":
            source_type = "video"
            video_path = variables.CALIBRATION_VIDEO_PATH
            camera_index = 0
        else:
            source_type = "camera"
            video_path = None
            camera_index = variables.CAMERA_INDEX

        def frame_callback(frame):
            from PyQt5.QtGui import QImage, QPixmap
            import cv2
            # Resize frame with aspect ratio preserved to fit QLabel
            label_w = self.calib_video_label.width()
            label_h = self.calib_video_label.height()
            frame_h, frame_w = frame.shape[:2]
            scale = min(label_w / frame_w, label_h / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            # Create black canvas and paste resized frame centered
            canvas = np.zeros((label_h, label_w, 3), dtype=np.uint8)
            y_offset = (label_h - new_h) // 2
            x_offset = (label_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            # Convert to RGB
            if canvas.shape[2] == 3:
                frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            else:
                print("[ERROR] Unexpected number of channels in frame:", canvas.shape)
                return QPixmap()
            
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.calib_video_label.setPixmap(pixmap)

        def done_callback(camera_matrix, dist_coeffs):
            if camera_matrix is not None:
                print("Calibration successful.")
            else:
                print("Calibration cancelled or failed.")

        # Run calibration in a background thread
        import threading
        self._calib_stop_flag = False

        def thread_func():
            # Reset stop flag before calibration starts
            self._calib_stop_flag = False

            calibrate_camera_gui(
                source_type=source_type,
                camera_index=camera_index,
                video_path=video_path,
                max_captures=variables.MAX_CAPTURES,

                frame_callback=lambda f: None if self._calib_stop_flag else frame_callback(f),
                done_callback=done_callback,
                stop_flag_getter=lambda: self._calib_stop_flag  # <-- This line is new
            )

        self._calib_thread = threading.Thread(target=thread_func, daemon=True)
        self._calib_thread.start()


    def stop_calibration(self):
        """
        Signal calibration loop to stop (non-blocking).
        """
        self._calib_stop_flag = True
        print("Calibration stop requested.")

    def closeEvent(self, event):
        """
        Called when the window is closed (e.g. via 'X' button).
        Ensures proper shutdown of threads, devices, and saves state if needed.
        """
        print("Shutting down application...")

        # Stop calibration thread if running
        try:
            self._calib_stop_flag = True
        except AttributeError:
            pass

        # Disconnect from Modbus robot if connected
        if hasattr(self, 'robot') and self.robot.connected:
            self.robot.disconnect()
            print("Disconnected from Modbus robot.")

        # Additional cleanup (e.g. video capture, temp files)
        # TODO: stop any other active threads here
        self.save_detection_params()
        print("All processes terminated. Application will close now.")
        event.accept()

    def update_working_source(self):
        """
        Show/hide relevant inputs for working source (video or camera).
        """
        use_video = self.working_source_selector.currentText() == "Working video"

        def toggle_layout_visibility(layout, visible):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                widget = item.widget()
                if widget:
                    widget.setVisible(visible)
                elif isinstance(item, QHBoxLayout) or isinstance(item, QVBoxLayout):
                    toggle_layout_visibility(item, visible)

        toggle_layout_visibility(self.working_video_layout, use_video)
        toggle_layout_visibility(self.working_camera_layout, not use_video)

    def browse_working_video(self):
        """
        Open file dialog to choose a working video.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Working Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.working_video_path_input.setText(file_path)
            variables.WORKING_VIDEO_PATH = file_path
            print(f"Updated WORKING_VIDEO_PATH to: {file_path}")

    def open_working_video_file_from_input(self, event):
        """
        Trigger video file browse dialog on line edit click.
        """
        self.browse_working_video()

    def frame_to_qpixmap(self, frame, label_w, label_h):
        
        import numpy as np
        import cv2

        frame_h, frame_w = frame.shape[:2]
        scale = min(label_w / frame_w, label_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.zeros((label_h, label_w, 3), dtype=np.uint8)
        y_offset = (label_h - new_h) // 2
        x_offset = (label_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(image)

    def start_working_mode(self):
        """
        Start object detection and Modbus/console output based on selected source.
        """

        self._working_stop_flag = False
        self._working_frame_counter = 0
        self._working_records = []

        # Select video source
        source = self.working_source_selector.currentText()
        if source == "Working video":
            video_path = self.working_video_path_input.text()
            cap = cv2.VideoCapture(video_path)
        else:
            try:
                index = int(self.working_camera_index_input.text())
            except ValueError:
                print("Invalid camera index.")
                return
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            print("Failed to open video source.")
            return

        data = np.load(variables.CALIBRATION_FILE)
        cameraMatrix = data["cameraMatrix"]
        distCoeffs = data["distCoeffs"]

        
        # Check if identification config file exists
        if not os.path.exists(variables.IDENTIFICATION_CONFIG):
            print(f"Identification config {variables.IDENTIFICATION_CONFIG} not found. Continuing without recognition.")
            identification_config_path = None
        else:
            identification_config_path = variables.IDENTIFICATION_CONFIG
        
        
        if cameraMatrix is None or distCoeffs is None:
            print("Calibration data is missing.")
            return

       # Read first frame to detect working area (in main thread)
        # Loop until working area is confirmed
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame or end of video.")
                return

            confirmed = self.prepare_working_area(frame)
            if confirmed:
                break  # Working area confirmed, continue
            else:
                print("[DEBUG] Working area not confirmed; reading next frame...")


        # Now that working area is confirmed, initialize detector


        self._detector = ObjectDetector(
            identification_config_path,
            self.detection_params
        )


        
        def thread_func():
            while not self._working_stop_flag:
                ret, frame = cap.read()
                if not ret:
                    print("Video ended or read failed.")
                    break


                
                # --- Working Area Detection Step ---
                # If the working area mask is not defined, attempt to detect it
                
                    # detect_working_aif self.working_area_mask is None:rea should be imported from its module
                if self.working_area_mask is None:
                    result = self.working_area_processor.objectDetection(frame)
                    if result is None:
                        print("Working area not detected or not confirmed; retrying...")
                        continue
                    else:
                        overlay_frame, working_mask, conversion_factor = result
                        self.working_area_mask = working_mask
                        self.last_overlay_frame = overlay_frame  # Save the overlay frame for toggling display
                        print("Working area detected and confirmed. Conversion factor:", conversion_factor)


                # --- End Working Area Detection ---



                # Process the frame using the object detector; pass the working area mask
                _, corrected_frame, _, _, _ = self._detector.WorkingDetect(frame)
                result = self._detector.process_frame(corrected_frame, self._working_frame_counter, mask=self.working_area_mask, mode="test")
                if result is None:
                    print("No detections returned; skipping this frame.")
                    continue
                edges, objects_overlayed, detections = result

                if hasattr(self, 'debug_window'):
                    contours_img = np.zeros_like(edges)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contours_img, contours, -1, 255, 1)
                    self.debug_window.update_images(edges, contours_img)


                self._working_frame_counter += 1

                # Choose which image to display based on the toggle flag:
                # If the user wants to see the working area search result, display overlay_frame;
                # Otherwise, display the object detection result (objects_overlayed).
                if self.display_overlay_search:
                    display_image = self.last_overlay_frame  # overlay_frame was obtained during working area detection
                else:
                    display_image = objects_overlayed

                pixmap = self.frame_to_qpixmap(display_image, self.working_video_label.width(), self.working_video_label.height())
                self.working_video_label.setPixmap(pixmap)

                # Send data via Modbus if enabled, else print detection details to terminal
 
                import variables  # если не импортирован в этом месте
                if self.modbus_checkbox.isChecked() and hasattr(self, "robot") and self.robot.connected:
                    send_robot_data(detections, conversion_factor=variables.CONVERSION_FACTOR, robot_comm=self.robot)


                else:
                    for d in detections:
                        print(f"Detected ID {d.get('track_id')} | Cat: {d.get('predicted_category')} "
                            f"| Center: {d['center']} | Size: {d['width']:.1f}x{d['height']:.1f}")



    


            cap.release()
            print("Working mode finished.")


        self._working_thread = threading.Thread(target=thread_func, daemon=True)
        self._working_thread.start()

    def stop_working_mode(self):
        """
        Stop object detection and video processing.
        """
        self._working_stop_flag = True
        print("Working mode stop requested.")

    def confirm_working_area(self, overlay_frame):
        """
        Show the detected working area in the main QLabel with confirmation buttons.
        This runs synchronously in the main thread.
        """
        from PyQt5.QtCore import QEventLoop
        self._confirmation_result = None  # temporary storage

        # Display the overlay frame in the QLabel
        pixmap = self.frame_to_qpixmap(overlay_frame, self.working_video_label.width(), self.working_video_label.height())
        self.working_video_label.setPixmap(pixmap)

        # Create confirmation buttons dynamically
        self.confirm_button_yes = QPushButton("Yes")
        self.confirm_button_no = QPushButton("No")

        # Connect signals to result and stop loop
        loop = QEventLoop()

        def accept():
            self._confirmation_result = True
            loop.quit()

        def reject():
            self._confirmation_result = False
            loop.quit()

        self.confirm_button_yes.clicked.connect(accept)
        self.confirm_button_no.clicked.connect(reject)

        # Add to layout under the video
        # Remove previous confirmation layout if it exists
        if hasattr(self, "working_confirm_layout") and self.working_confirm_layout is not None:
            while self.working_confirm_layout.count():
                item = self.working_confirm_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        # Recreate layout
        self.working_confirm_layout = QHBoxLayout()
        self.working_confirm_layout.addWidget(self.confirm_button_yes)
        self.working_confirm_layout.addWidget(self.confirm_button_no)

        # Add to GUI under the video
        self.working_video_label.parent().layout().addLayout(self.working_confirm_layout)


        # Wait for user response
        loop.exec_()

        # Clean up: remove buttons
        self.confirm_button_yes.deleteLater()
        self.confirm_button_no.deleteLater()

        return self._confirmation_result

    def prepare_working_area(self, first_frame):
        """
        Detect working area in the first frame and confirm it from the GUI thread.
        """
        result = self.working_area_processor.objectDetection(first_frame)
        if result is None:
            print("Working area not detected or not confirmed.")
            return False
        else:
            overlay_frame, working_mask, conversion_factor = result
            self.working_area_mask = working_mask
            self.last_overlay_frame = overlay_frame
            print("Working area detected and confirmed. Conversion factor:", conversion_factor)
            return True
    
    def load_detection_params(self):
        """
        Load detection parameters from JSON file if it exists.
        """
        try:
            with open(variables.PARAMETERS_CONFIG, "r") as f:
                params = json.load(f)
                for key, val in params.items():
                    self.detection_params[key] = val

        except FileNotFoundError:
            print("[INFO] No saved parameter config found.")
        except Exception as e:
            print(f"[ERROR] Failed to load detection parameters: {e}")

    def save_detection_params(self):
        """
        Save current detection parameters to a JSON file.
        """
        keys = ["SCALE", "BLUR_KERNEL", "CANNY_LOW", "CANNY_HIGH", "MIN_AREA", "MAX_AREA"]
        params = {k: self.detection_params[k] for k in keys}

        try:
            with open(variables.PARAMETERS_CONFIG, "w") as f:
                json.dump(params, f, indent=2)
            print("[SAVE] Detection parameters saved.")
        except Exception as e:
            print(f"[ERROR] Failed to save detection parameters: {e}")

def send_robot_data(detections, conversion_factor, robot_comm, category_mapping=None):
    if category_mapping is None:
        category_mapping = {
            "circular_white": 1,
            "circular_pink": 2,
            "circular_black": 3,
            "rhombus_white": 4,
            "rhombus_pink": 5,
            "rhombus_black": 6,
        }
    for det in detections:
        x_mm = det['center'][0]
        y_mm = det['center'][1]
        width_mm = det['width']
        height_mm = det['height']
        angle = det['angle']
        category_code = category_mapping.get(det['predicted_category'], 0)
        robot_comm.send_data(
            obj_id=det.get('track_id', 0),
            x_mm=x_mm,
            y_mm=y_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            angle=angle,
            category_code=category_code
        )

import cv2
import numpy as np
import random
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

class DebugWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Debug View")
        self.resize(800, 600)
        self._init_ui()
        self.canny_image: np.ndarray | None = None
        self.contour_image: np.ndarray | None = None

    def _init_ui(self) -> None:
        """Initializes the UI with a vertical layout without external captions."""
        layout = QVBoxLayout(self)
        self.canny_label = QLabel()
        self.canny_label.setAlignment(Qt.AlignCenter)
        self.contour_label = QLabel()
        self.contour_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.canny_label)
        layout.addWidget(self.contour_label)

    def update_images(self, canny_img: np.ndarray, contour_img: np.ndarray) -> None:
        """
        Updates the displayed images.

        :param canny_img: The image after Canny edge detection.
        :param contour_img: The binary image containing contours.
        """
        self.canny_image = canny_img
        self.contour_image = contour_img
        self._update_display()

    def _update_display(self) -> None:
        """Updates the display by overlaying captions, scaling, and updating labels."""
        if self.canny_image is not None:
            # Overlay caption on canny image
            canny_with_caption = self._overlay_caption(self.canny_image, "Canny Image")
            self.canny_label.setPixmap(self._fit_pixmap(canny_with_caption, self.canny_label))
        if self.contour_image is not None:
            # Colorize contours, overlay caption
            colored_contours = self._colorize_contours(self.contour_image)
            contours_with_caption = self._overlay_caption(colored_contours, "Contours Image")
            self.contour_label.setPixmap(self._fit_pixmap(contours_with_caption, self.contour_label))

    def _fit_pixmap(self, image: np.ndarray, label: QLabel) -> QPixmap:
        """
        Scales the image to fit within the given QLabel while preserving aspect ratio.

        :param image: The source image as a numpy array.
        :param label: The QLabel to display the image.
        :return: A QPixmap of the scaled image.
        """
        h, w = image.shape[:2]
        label_w, label_h = label.width(), label.height()
        scale = min(label_w / w, label_h / h) if w and h else 1
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2 or resized.shape[2] == 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(resized.data, new_w, new_h, resized.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _overlay_caption(self, image: np.ndarray, caption: str) -> np.ndarray:
        """
        Overlays a caption text onto the image.

        :param image: Source image as a numpy array.
        :param caption: Text to overlay.
        :return: Image with the caption overlay.
        """
        img = image.copy()
        # Convert grayscale to BGR if necessary
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size, _ = cv2.getTextSize(caption, font, font_scale, thickness)
        text_width, text_height = text_size
        x, y = 10, text_height + 10
        # Draw a filled rectangle as background for the caption
        cv2.rectangle(img, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (0, 0, 0), cv2.FILLED)
        # Put the caption text over the rectangle
        cv2.putText(img, caption, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img

    def _colorize_contours(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Finds contours in a binary image and fills each with a random color.

        :param binary_image: A binary image (e.g., output from Canny or a drawn contour mask).
        :return: A color image with contours filled with random colors.
        """
        colored = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(colored, [cnt], -1, color, thickness=cv2.FILLED)
            cv2.drawContours(colored, [cnt], -1, (0, 0, 0), thickness=1)
        return colored

    def resizeEvent(self, event) -> None:
        """Handles window resize events to update the display."""
        self._update_display()
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())