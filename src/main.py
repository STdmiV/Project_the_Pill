# Okay, let's refactor the main.py file incorporating the new Modbus logic (PLC request flag + multi-object packet sending) and implementing thread-safe GUI updates using signals and slots.

# Key Changes:

# Signals/Slots: Added pyqtSignal and @pyqtSlot for safe GUI updates from the worker thread.

# Modbus Logic in thread_func: Replaced the old sending logic with the new one: check flag, build packet, send packet, reset flag.

# Robustness: Added more error handling (try...except), logging, and checks for prerequisites (calibration, video source).

# Cleanup: Improved thread stopping and resource release.

# Readability: Used variables as var alias, added logging.

# main.py

import sys
import numpy as np
import cv2
import os
import socket # For Modbus TCP connection
import threading # For threading
import time # For time management
import logging # For logging
import json
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout,
    QComboBox, QFileDialog, QMessageBox, QSizePolicy, QCheckBox, QSlider
)
# IMPORTANT: Import QtCore elements for signals/slots and event loop
from PyQt5.QtCore import Qt, QEventLoop, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from robot_comm import RobotComm  # Robot communication module
# Assuming calibrate_camera is not directly used, but calib.py has calibrate_camera_gui
from calib import calibrate_camera_gui
import variables as var # Use alias for clarity
from ultimate import WorkingArea, ObjectDetector, undistort_frame # Import undistort_frame if needed elsewhere

# --- Define category mapping here or ensure it's accessible ---
# Example: Using the one from variables.py
category_mapping = var.OBJECT_CATEGORIES

# Setup basic logging (configure once, preferably at the start)
# Moved configuration to the `if __name__ == '__main__':` block for clarity

class MainWindow(QMainWindow):
    # --- Define Signals for thread-safe GUI updates ---
    updateWorkingPixmapSignal = pyqtSignal(QPixmap)
    showMessageSignal = pyqtSignal(str, str)
    calibrationUpdateSignal = pyqtSignal(QPixmap)
    calibrationDoneSignal = pyqtSignal(bool)
    updateWorkingPixmapSignal = pyqtSignal(QPixmap)
    updateCannyPixmapSignal = pyqtSignal(QPixmap)
    updateContourPixmapSignal = pyqtSignal(QPixmap)   
    threadStoppedSignal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pill Project Pilot GUI")
        # Increased size slightly for better layout
        self.resize(1200, 800)

        # --- State Variables ---
        self.detection_params = {
            "SCALE": 0.75, # Initial reasonable default
            "BLUR_KERNEL": 5,
            "CANNY_LOW": 10,
            "CANNY_HIGH": 50,
            "MIN_AREA": 100,
            "MAX_AREA": 50000,
            "CONVERSION_FACTOR": 1.0 # Will be calculated
        }
        self.load_detection_params() # Load saved params over defaults

        self.working_area_mask = None
        self.last_overlay_frame = None # Frame shown during confirmation
        self.display_overlay_search = False # Controlled by user? Add checkbox if needed.

        self.robot: RobotComm | None = None # Type hinting
        self._detector: ObjectDetector | None = None

        # Threading control
        self._working_stop_flag = False
        self._working_thread: threading.Thread | None = None
        self._calib_stop_flag = False
        self._calib_thread: threading.Thread | None = None
        self.last_detections = [] # Store last detections for potential reuse

        # Initialize processors
        self.working_area_processor = WorkingArea(
            detection_params=self.detection_params, # Pass the dictionary
            calibration_file=var.CALIBRATION_FILE,
            confirmation_callback=self.confirm_working_area,
            parent=self # Pass self if WorkingArea needs to interact back
        )

        # --- Build UI ---
        self._init_ui()
        self.load_modbus_settings() # Load Modbus settings on startup
        
        # --- Connect Signals to Slots ---
        self.updateWorkingPixmapSignal.connect(self.setWorkingPixmap) # Update working video label
        self.showMessageSignal.connect(self.showMessageBox) # Show message box
        self.calibrationUpdateSignal.connect(self.setCalibrationPixmap) # Update calibration preview
        self.calibrationDoneSignal.connect(self.handleCalibrationResult) # Handle calibration result
        self.updateCannyPixmapSignal.connect(self.setCannyPixmap) # Update Canny debug label
        self.updateContourPixmapSignal.connect(self.setContourPixmap) # Update Contour debug label

    def _init_ui(self):
        """Helper method to initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top horizontal layout for tabs and controls
        top_layout = QHBoxLayout()

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_working_mode_tab(), "Working Mode")
        self.tabs.addTab(QWidget(), "Data Collection") # Placeholder
        self.tabs.addTab(QWidget(), "Data Analysis") # Placeholder
        self.tabs.addTab(self.create_modbus_tab(), "Modbus TCP")
        self.tabs.addTab(self.create_calibration_tab(), "Calibration")
        top_layout.addWidget(self.tabs, stretch=3) # Give more space to tabs

        # --- Parameter Controls Panel ---
        self.param_control_panel = self.create_param_controls()
        # Wrap controls in a QWidget for better layout management
        param_widget = QWidget()
        param_widget.setLayout(self.param_control_panel)
        param_widget.setFixedWidth(200) # Give parameters a fixed width
        top_layout.addWidget(param_widget, stretch=1) # Less space for controls

        main_layout.addLayout(top_layout)

    # =========================================================================
    # GUI Creation Methods (create_modbus_tab, create_calibration_tab, etc.)
    # =========================================================================

    def create_modbus_tab(self):
        """Creates the Modbus TCP configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignTop)

        # Network Scan section
        scan_layout = QHBoxLayout()
        scan_layout.addWidget(QLabel("Scan Network:"))
        self.device_selector = QComboBox() # Dropdown for IP selection
        scan_layout.addWidget(self.device_selector)
        self.refresh_button = QPushButton("Scan") # Scan for Modbus devices
        self.refresh_button.setFixedWidth(60)
        self.refresh_button.clicked.connect(self.scan_network_for_modbus)
        scan_layout.addWidget(self.refresh_button)
        layout.addLayout(scan_layout)

        # Input fields for IP and port
        form_layout = QFormLayout()
        self.ip_input = QLineEdit(var.MODBUS_TCP_HOST) # Use default from variables
        self.port_input = QLineEdit(str(var.MODBUS_TCP_PORT)) # Use default from variables
        form_layout.addRow("IP Address:", self.ip_input)
        form_layout.addRow("Port:", self.port_input)
        layout.addLayout(form_layout)
        
        layout.addWidget(QLabel("--- PLC Data Structure Settings ---")) # Separator
        plc_form_layout = QFormLayout()

        # Fields for the new parameters
        self.req_flag_addr_input = QLineEdit() # Initial value set during load
        self.req_flag_is_coil_check = QCheckBox()
        self.max_objects_input = QLineEdit()
        self.num_objects_addr_input = QLineEdit()
        self.obj_data_start_addr_input = QLineEdit()
        # Optional: self.regs_per_obj_input = QLineEdit(str(var.MODBUS_REGISTERS_PER_OBJECT))

        plc_form_layout.addRow("Request Flag Addr:", self.req_flag_addr_input)
        plc_form_layout.addRow("Flag is Coil:", self.req_flag_is_coil_check)
        plc_form_layout.addRow("Max Objects/Packet:", self.max_objects_input)
        plc_form_layout.addRow("Num Objects Addr:", self.num_objects_addr_input)
        plc_form_layout.addRow("Object Data Start Addr:", self.obj_data_start_addr_input)
        # Optional: plc_form_layout.addRow("Registers/Object:", self.regs_per_obj_input)

        layout.addLayout(plc_form_layout)

        # Add a save button specifically for Modbus settings
        self.save_modbus_button = QPushButton("Save Modbus Settings")
        self.save_modbus_button.clicked.connect(self.save_modbus_settings)
        layout.addWidget(self.save_modbus_button, alignment=Qt.AlignCenter) # Center button maybe

        # Connection Buttons and Status
        button_layout = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        button_layout.addWidget(self.connect_button)
        button_layout.addWidget(self.disconnect_button)
        layout.addLayout(button_layout)

        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Connections
        self.device_selector.currentTextChanged.connect(lambda ip: self.ip_input.setText(ip) if ip else None)
        self.connect_button.clicked.connect(self.connect_modbus)
        self.disconnect_button.clicked.connect(self.disconnect_modbus)

        layout.addStretch() # Push elements to the top
        return tab

    def create_calibration_tab(self):
        """Creates the Calibration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignTop)

        # --- Source Selection ---
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Calibration Source:"))
        self.source_selector = QComboBox()
        self.source_selector.addItems(["Calibration video", "Camera"])
        self.source_selector.currentIndexChanged.connect(self.update_calibration_source)
        source_row.addWidget(self.source_selector)
        layout.addLayout(source_row)

        # --- Video Source Controls ---
        self.video_layout_widget = QWidget() # Use a widget to easily hide/show
        video_layout = QVBoxLayout(self.video_layout_widget)
        video_layout.setContentsMargins(0,0,0,0)
        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Video Path:"))
        self.calib_video_path_input = QLineEdit(var.CALIBRATION_VIDEO_PATH)
        self.calib_video_path_input.setReadOnly(True)
        # Make read-only field clickable
        self.calib_video_path_input.mousePressEvent = lambda event: self.browse_calibration_video()
        video_row.addWidget(self.calib_video_path_input)
        self.calib_video_browse_button = QPushButton("...")
        self.calib_video_browse_button.setFixedWidth(30)
        self.calib_video_browse_button.clicked.connect(self.browse_calibration_video)
        video_row.addWidget(self.calib_video_browse_button)
        video_layout.addLayout(video_row)
        layout.addWidget(self.video_layout_widget)

        # --- Camera Source Controls ---
        self.camera_layout_widget = QWidget() # Use a widget to easily hide/show
        camera_layout = QHBoxLayout(self.camera_layout_widget)
        camera_layout.setContentsMargins(0,0,0,0)
        camera_layout.addWidget(QLabel("Camera Index:"))
        self.camera_index_input = QLineEdit(str(var.CAMERA_INDEX))
        self.camera_index_input.setFixedWidth(50)
        # Connect signal when editing is finished
        self.camera_index_input.editingFinished.connect(self.update_camera_index)
        camera_layout.addWidget(self.camera_index_input)
        camera_layout.addStretch()
        layout.addWidget(self.camera_layout_widget)

        # --- Calibration Controls ---
        calib_button_row = QHBoxLayout()
        self.calib_start_button = QPushButton("Start Calibration")
        self.calib_stop_button = QPushButton("Stop Calibration")
        self.calib_start_button.clicked.connect(self.start_calibration)
        self.calib_stop_button.clicked.connect(self.stop_calibration)
        self.calib_stop_button.setEnabled(False) # Disable stop initially
        calib_button_row.addWidget(self.calib_start_button)
        calib_button_row.addWidget(self.calib_stop_button)
        layout.addLayout(calib_button_row)

        # --- Calibration Preview ---
        self.calib_video_label = QLabel("Calibration preview will appear here")
        self.calib_video_label.setMinimumSize(640, 360) # Reasonable minimum size
        self.calib_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.calib_video_label.setAlignment(Qt.AlignCenter)
        self.calib_video_label.setStyleSheet("background-color: #333; color: white;")
        layout.addWidget(self.calib_video_label)

        self.update_calibration_source() # Set initial visibility
        return tab

    def create_working_mode_tab(self):
        """Creates the Working Mode tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        #layout.setAlignment(Qt.AlignTop) # Let video expand

        # --- Source Selection ---
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Working Source:"))
        self.working_source_selector = QComboBox()
        self.working_source_selector.addItems(["Working video", "Camera"])
        self.working_source_selector.currentIndexChanged.connect(self.update_working_source)
        source_row.addWidget(self.working_source_selector)
        layout.addLayout(source_row)

        # --- Video Source Controls ---
        self.working_video_layout_widget = QWidget()
        working_video_layout = QVBoxLayout(self.working_video_layout_widget)
        working_video_layout.setContentsMargins(0,0,0,0)
        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Video Path:"))
        self.working_video_path_input = QLineEdit(var.WORKING_VIDEO_PATH)
        self.working_video_path_input.setReadOnly(True)
        self.working_video_path_input.mousePressEvent = lambda event: self.browse_working_video()
        video_row.addWidget(self.working_video_path_input)
        browse_button = QPushButton("...")
        browse_button.setFixedWidth(30)
        browse_button.clicked.connect(self.browse_working_video)
        video_row.addWidget(browse_button)
        working_video_layout.addLayout(video_row)
        layout.addWidget(self.working_video_layout_widget)

        # --- Camera Source Controls ---
        self.working_camera_layout_widget = QWidget()
        working_camera_layout = QHBoxLayout(self.working_camera_layout_widget)
        working_camera_layout.setContentsMargins(0,0,0,0)
        working_camera_layout.addWidget(QLabel("Camera Index:"))
        self.working_camera_index_input = QLineEdit(str(var.CAMERA_INDEX))
        self.working_camera_index_input.setFixedWidth(50)
        # Add connection if needed: self.working_camera_index_input.editingFinished.connect(...)
        working_camera_layout.addWidget(self.working_camera_index_input)
        working_camera_layout.addStretch()
        layout.addWidget(self.working_camera_layout_widget)

        # --- Video Display ---
        self.working_video_label = QLabel("Working mode video will appear here")
        self.working_video_label.setMinimumSize(640, 480) # Larger minimum
        self.working_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.working_video_label.setAlignment(Qt.AlignCenter)
        self.working_video_label.setStyleSheet("background-color: #222; color: white;")
        layout.addWidget(self.working_video_label) # Let it expand

        self.debug_views_widget = QWidget() # Widget to hold debug labels
        debug_layout = QHBoxLayout(self.debug_views_widget) # Arrange side-by-side
        debug_layout.setContentsMargins(0, 5, 0, 5) # Add vertical margin

        self.canny_debug_label = QLabel("Canny")
        self.canny_debug_label.setAlignment(Qt.AlignCenter)
        self.canny_debug_label.setStyleSheet("background-color: #555; color: white;")
        self.canny_debug_label.setMinimumSize(320, 180) # Smaller minimum size
        self.canny_debug_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.contour_debug_label = QLabel("Contours")
        self.contour_debug_label.setAlignment(Qt.AlignCenter)
        self.contour_debug_label.setStyleSheet("background-color: #555; color: white;")
        self.contour_debug_label.setMinimumSize(320, 180)
        self.contour_debug_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        debug_layout.addWidget(self.canny_debug_label)
        debug_layout.addWidget(self.contour_debug_label)

        # Initially hide the debug views
        self.debug_views_widget.setVisible(False)
        layout.addWidget(self.debug_views_widget)

        # --- Control Buttons ---
        button_row = QHBoxLayout()
        self.working_start_button = QPushButton("Start")
        self.working_stop_button = QPushButton("Stop")
        self.modbus_checkbox = QCheckBox("Send data via Modbus")
        self.working_start_button.clicked.connect(self.start_working_mode)
        self.working_stop_button.clicked.connect(self.stop_working_mode)
        self.working_stop_button.setEnabled(False) # Disable stop initially
        button_row.addWidget(self.working_start_button)
        button_row.addWidget(self.working_stop_button)
        button_row.addStretch()
        button_row.addWidget(self.modbus_checkbox)
        # Checkbox to control debug view visibility
        self.show_debug_views_checkbox = QCheckBox("Show Debug Views Below")
        self.show_debug_views_checkbox.stateChanged.connect(
            lambda state: self.debug_views_widget.setVisible(state == Qt.Checked)
        )
        button_row.insertWidget(2, self.show_debug_views_checkbox) # Insert before stretch
        layout.addLayout(button_row)

        self.update_working_source() # Set initial visibility
        return tab

    def create_param_controls(self):
        """Creates the parameter control panel layout."""
        from functools import partial # Keep import local if only used here

        panel_layout = QVBoxLayout()
        panel_layout.setAlignment(Qt.AlignTop)
        panel_layout.setSpacing(10) # Add spacing between controls


        # Parameter Specifications
        param_specs = {
            # Name: (Min, Max, Step, Decimals)
            "BLUR_KERNEL": (1, 15, 2, 0),  # Step 2 for odd numbers
            "CANNY_LOW": (0, 255, 1, 0),
            "CANNY_HIGH": (0, 255, 1, 0),
            "MIN_AREA": (0, 10000, 50, 0), # Adjusted range
            "MAX_AREA": (1000, 100000, 500, 0),
            "CONVERSION_FACTOR": (0.01, 10.0, 0.01, 4) # Allow manual adjustment? Risky. Read-only?
            # "SCALE" parameter seems less relevant for processing now, consider removing?
        }

        self.param_widgets = {} # Store widgets for potential future access

        for name, (min_val, max_val, step, decimals) in param_specs.items():
            # Ensure parameter exists in self.detection_params, if not add default
            if name not in self.detection_params:
                 # If CONVERSION_FACTOR was missing, default to 1.0
                 self.detection_params[name] = min_val if name != "CONVERSION_FACTOR" else 1.0
            value = self.detection_params[name]

            container_widget = QWidget() # Group elements for better styling/margins
            container_layout = QVBoxLayout(container_widget)
            container_layout.setContentsMargins(0,0,0,0)
            container_layout.setSpacing(2)

            # 1. Label
            label = QLabel(f"{name}")
            label.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(label)

            # 2. Horizontal row: [-] [value] [+]
            hbox = QHBoxLayout()
            minus_btn = QPushButton("-")
            minus_btn.setFixedWidth(25)
            plus_btn = QPushButton("+")
            plus_btn.setFixedWidth(25)

            # Use QLineEdit for value display/editing if needed, otherwise QLabel
            value_label = QLabel(f"{value:.{decimals}f}")
            value_label.setFixedWidth(60) # Wider for decimals
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet("background-color: white; border: 1px solid grey;")

            hbox.addWidget(minus_btn)
            hbox.addWidget(value_label)
            hbox.addWidget(plus_btn)
            container_layout.addLayout(hbox)

            # 3. Slider
            # Slider works with integers, so scale values
            slider_min = int(min_val / step)
            slider_max = int(max_val / step)
            slider_val = int(value / step)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(slider_min)
            slider.setMaximum(slider_max)
            slider.setValue(slider_val)
            # slider.setFixedWidth(120) # Let it expand slightly
            slider.setTickPosition(QSlider.TicksBelow) # Add ticks
            slider.setTickInterval(max(1, (slider_max - slider_min) // 10)) # Sensible tick interval

            container_layout.addWidget(slider) # No alignment needed if expanding

            # Store widgets associated with the parameter name
            self.param_widgets[name] = {
                'label': value_label,
                'slider': slider,
                'min': min_val,
                'max': max_val,
                'step': step,
                'decimals': decimals
            }

            # --- Connect Signals ---
            # Use partial to pass parameters to the update function
            minus_btn.clicked.connect(partial(self.update_param_value, name, -1))
            plus_btn.clicked.connect(partial(self.update_param_value, name, 1))
            slider.valueChanged.connect(partial(self.update_param_from_slider, name))

            panel_layout.addWidget(container_widget)

        panel_layout.addStretch() # Push controls to the top
        return panel_layout

    # =========================================================================
    # Parameter Update Logic
    # =========================================================================

    def update_param_value(self, param_name, direction):
        """Updates parameter value based on +/- button clicks."""
        if param_name not in self.param_widgets: return

        widgets = self.param_widgets[param_name]
        step = widgets['step']
        min_val = widgets['min']
        max_val = widgets['max']
        decimals = widgets['decimals']
        current_value = self.detection_params.get(param_name, min_val)

        new_value = current_value + direction * step

        # Special handling for BLUR_KERNEL (must be odd)
        if param_name == "BLUR_KERNEL":
            new_value = int(new_value)
            if new_value < 1: new_value = 1 # Min blur kernel is 1
            if new_value % 2 == 0:
                # If moving up (+1), go to next odd. If moving down (-1), go to previous odd.
                new_value += 1 if direction > 0 else -1
                # Ensure it doesn't go below 1 after adjustment
                new_value = max(1, new_value)


        # Clamp to min/max
        new_value = max(min_val, min(max_val, new_value))

        # Update the dictionary and UI
        self.detection_params[param_name] = new_value
        widgets['label'].setText(f"{new_value:.{decimals}f}")
        widgets['slider'].setValue(int(new_value / step)) # Update slider position
        # logging.debug(f"[PARAM] {param_name} set to {new_value} by button")

    def update_param_from_slider(self, param_name, slider_value):
        """Updates parameter value based on slider movement."""
        if param_name not in self.param_widgets: return

        widgets = self.param_widgets[param_name]
        step = widgets['step']
        min_val = widgets['min']
        max_val = widgets['max']
        decimals = widgets['decimals']

        new_value = slider_value * step

        # Special handling for BLUR_KERNEL (must be odd)
        # Adjust value *after* calculating from slider if needed
        if param_name == "BLUR_KERNEL":
             new_value = int(new_value)
             if new_value < 1: new_value = 1
             if new_value % 2 == 0:
                 # If slider moves, bias towards the next higher odd number? Or lower?
                 # Let's bias higher for simplicity. But this might feel jumpy.
                 # A better approach might be to disallow even values in the slider setup.
                 # For now, just ensure it's at least 1. We rely on button logic for strict odd steps.
                 new_value = max(1, new_value) # Ensure minimum 1
                 # Let's actually force it odd if possible within range
                 if new_value + 1 <= max_val:
                      new_value += 1
                 elif new_value -1 >= min_val:
                      new_value -=1


        # Clamp again just in case slider calculation is slightly off
        new_value = max(min_val, min(max_val, new_value))

        # Update the dictionary and UI
        self.detection_params[param_name] = new_value
        widgets['label'].setText(f"{new_value:.{decimals}f}")
        # logging.debug(f"[PARAM] {param_name} set to {new_value} by slider")

    # =========================================================================
    # GUI Slots for Thread Signals
    # =========================================================================

    @pyqtSlot(QPixmap)
    def setWorkingPixmap(self, pixmap):
        """Thread-safe slot to update the working video label."""
        if not pixmap.isNull():
            self.working_video_label.setPixmap(pixmap)
        else:
            logging.warning("Received null QPixmap for working video label.")

    @pyqtSlot(QPixmap)
    def setCannyPixmap(self, pixmap):
        """Thread-safe slot to update the Canny debug label."""
        if not pixmap.isNull() and self.show_debug_views_checkbox.isChecked():
            self.canny_debug_label.setPixmap(pixmap)
        # else: # Optionally clear if checkbox is off or pixmap is null
        #     self.canny_debug_label.setText("Canny")

    @pyqtSlot(QPixmap)
    def setContourPixmap(self, pixmap):
        """Thread-safe slot to update the Contour debug label."""
        if not pixmap.isNull() and self.show_debug_views_checkbox.isChecked():
            self.contour_debug_label.setPixmap(pixmap)
        # else:
        #     self.contour_debug_label.setText("Contours")

    @pyqtSlot(str, str)
    def showMessageBox(self, title, message):
        """Thread-safe slot to show a message box."""
        # Use appropriate icon based on title?
        if "error" in title.lower():
            QMessageBox.critical(self, title, message)
        elif "warning" in title.lower():
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    @pyqtSlot(QPixmap)
    def setCalibrationPixmap(self, pixmap):
        """Thread-safe slot to update the calibration preview label."""
        if not pixmap.isNull():
            self.calib_video_label.setPixmap(pixmap)
        else:
            logging.warning("Received null QPixmap for calibration label.")

    @pyqtSlot(bool)
    def handleCalibrationResult(self, success):
        """Handles the result of the calibration process."""
        self.calib_start_button.setEnabled(True)
        self.calib_stop_button.setEnabled(False)
        if success:
            logging.info("Calibration process finished successfully.")
            self.showMessageSignal.emit("Calibration", "Calibration completed successfully and data saved.")
        else:
            logging.warning("Calibration process failed or was cancelled.")
            self.showMessageSignal.emit("Calibration", "Calibration failed or was cancelled.")
        # Clean up thread reference
        self._calib_thread = None


    # =========================================================================
    # Action Handlers (Connect/Disconnect Modbus, Start/Stop Modes)
    # =========================================================================

    def connect_modbus(self):
        """Attempts to connect to the Modbus TCP server."""
        if self.robot is not None and self.robot.connected:
            logging.warning("Already connected to Modbus.")
            self.showMessageSignal.emit("Modbus TCP", "Already connected.")
            return

        ip = self.ip_input.text()
        port_str = self.port_input.text()
        try:
            port = int(port_str)
            if not (0 < port < 65536):
                raise ValueError("Port number out of range")
        except ValueError:
            logging.error(f"Invalid Modbus port number: {port_str}")
            self.showMessageSignal.emit("Error", f"Invalid port number: {port_str}")
            return

        logging.info(f"Attempting to connect to Modbus server at {ip}:{port}...")
        self.status_label.setText("Status: Connecting...")
        QApplication.processEvents() # Update UI immediately

        # Perform connection in a separate thread to avoid freezing GUI? Not strictly necessary for quick connect.
        # For now, keep it synchronous. Add thread if connects take too long.
        try:
            # Use configured timeout
            self.robot = RobotComm(host=ip, port=port, timeout=var.MODBUS_TIMEOUT)
            self.robot.connect() # This method handles its own logging/printing

            if self.robot.connected:
                self.status_label.setText("Status: ✅ Connected")
                logging.info("Modbus connection successful.")
            else:
                self.status_label.setText("Status: ❌ Connection failed")
                logging.error("Modbus connection failed.")
                self.showMessageSignal.emit("Modbus Error", f"Failed to connect to {ip}:{port}.")
                self.robot = None # Clear instance if connection failed
        except Exception as e:
            self.status_label.setText("Status: ❌ Error")
            logging.error(f"Exception during Modbus connection: {e}", exc_info=True)
            self.showMessageSignal.emit("Modbus Error", f"An error occurred during connection:\n{e}")
            self.robot = None

    def disconnect_modbus(self):
        """Disconnects from the Modbus TCP server."""
        if self.robot is None or not self.robot.connected:
            logging.warning("Not connected to Modbus.")
            # self.showMessageSignal.emit("Modbus TCP", "Not currently connected.")
            self.status_label.setText("Status: Disconnected") # Ensure status is correct
            return

        logging.info("Disconnecting from Modbus server...")
        try:
            self.robot.disconnect() # Handles its own logging
            self.status_label.setText("Status: 🔌 Disconnected")
            logging.info("Modbus disconnected successfully.")
        except Exception as e:
            self.status_label.setText("Status: ⚠️ Error during disconnect")
            logging.error(f"Exception during Modbus disconnection: {e}", exc_info=True)
            self.showMessageSignal.emit("Modbus Error", f"An error occurred during disconnection:\n{e}")
        finally:
             # Clear the instance even if disconnect raised an error
             self.robot = None


    def scan_network_for_modbus(self):
        """Scans the local network for Modbus TCP devices (port 502)."""
        from concurrent.futures import ThreadPoolExecutor # Keep import local

        # Determine subnet based on current IP? Or assume common ones?
        # For simplicity, assume 192.168.0.x or 192.168.1.x
        # A more robust way would involve getting local IP and subnet mask
        base_ip = self.ip_input.text()
        parts = base_ip.split('.')
        if len(parts) == 4:
            subnet_base = ".".join(parts[:3]) + "."
        else:
             subnet_base = "192.168.0." # Default fallback
             logging.warning("Could not determine subnet from IP input, using default.")

        port = 502 # Standard Modbus TCP port
        self.device_selector.clear()
        self.device_selector.addItem("Scanning...")
        self.status_label.setText("Status: Scanning network...")
        self.refresh_button.setEnabled(False) # Disable scan button during scan
        QApplication.processEvents()

        def check_ip(ip_to_check):
            try:
                # Short timeout for scanning
                with socket.create_connection((ip_to_check, port), timeout=0.2):
                    logging.debug(f"Modbus device found at {ip_to_check}")
                    return ip_to_check # Return IP if connection successful
            except socket.timeout:
                logging.debug(f"Timeout connecting to {ip_to_check}:{port}")
                return None
            except OSError as e: # Handle cases like "Host unreachable"
                 logging.debug(f"OS Error connecting to {ip_to_check}:{port} - {e}")
                 return None
            except Exception as e:
                logging.debug(f"Unexpected error connecting to {ip_to_check}:{port} - {e}")
                return None

        # Run scan in background thread to keep GUI responsive
        def scan_thread_func():
            found_ips = []
            # Scan range 1-254
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(check_ip, f"{subnet_base}{i}") for i in range(1, 255)]
                for future in futures:
                     result = future.result()
                     if result:
                         found_ips.append(result)

            # Update GUI from main thread after scan finishes
            # Use QTimer.singleShot or signals if this becomes complex
            self.device_selector.clear() # Clear "Scanning..."
            if found_ips:
                self.device_selector.addItems(sorted(found_ips)) # Sort IPs
                self.status_label.setText(f"Status: Found {len(found_ips)} device(s)")
                logging.info(f"Network scan found {len(found_ips)} potential Modbus devices.")
                # Optionally set current IP input to the first found device
                # self.ip_input.setText(found_ips[0])
            else:
                self.status_label.setText("Status: No devices found")
                logging.info("Network scan completed. No devices found responding on port 502.")
            self.refresh_button.setEnabled(True) # Re-enable scan button

        # Start the scan thread
        scan_thread = threading.Thread(target=scan_thread_func, daemon=True)
        scan_thread.start()


    def start_calibration(self):
        """Starts the camera calibration process in a background thread."""
        if self._calib_thread is not None and self._calib_thread.is_alive():
            logging.warning("Calibration is already running.")
            self.showMessageSignal.emit("Calibration", "Calibration process is already running.")
            return

        source = self.source_selector.currentText()
        video_path = None
        camera_index = 0
        source_type = "camera" # Default to camera

        if source == "Calibration video":
            source_type = "video"
            video_path = self.calib_video_path_input.text()
            if not os.path.exists(video_path):
                logging.error(f"Calibration video file not found: {video_path}")
                self.showMessageSignal.emit("Error", f"Calibration video file not found:\n{video_path}")
                return
        else: # Camera source
            try:
                camera_index = int(self.camera_index_input.text())
            except ValueError:
                logging.error(f"Invalid camera index: {self.camera_index_input.text()}")
                self.showMessageSignal.emit("Error", "Invalid camera index for calibration.")
                return

        logging.info(f"Starting calibration with source type: {source_type}")
        self.calib_start_button.setEnabled(False)
        self.calib_stop_button.setEnabled(True)
        self._calib_stop_flag = False

        # Define callbacks for the calibration function
        def frame_callback_gui(frame):
            """Processes frame and emits signal to update GUI."""
            if frame is None: return
            try:
                 # Use the same conversion function, adapt size to calib label
                 pixmap = self.frame_to_qpixmap(frame, self.calib_video_label.width(), self.calib_video_label.height())
                 if not pixmap.isNull():
                      self.calibrationUpdateSignal.emit(pixmap) # Emit signal
            except Exception as e:
                 logging.error(f"Error processing calibration frame for GUI: {e}")

        def done_callback_gui(camera_matrix, dist_coeffs):
            """Handles calibration completion and emits signal."""
            success = camera_matrix is not None and dist_coeffs is not None
            self.calibrationDoneSignal.emit(success) # Emit signal with result

        # --- Calibration Thread ---
        def thread_func_calib():
            logging.info("Calibration thread started.")
            try:
                 # Call the GUI version of calibration function
                 calibrate_camera_gui(
                     source_type=source_type,
                     camera_index=camera_index,
                     video_path=video_path,
                     max_captures=var.MAX_CAPTURES,
                     frame_callback=lambda f: None if self._calib_stop_flag else frame_callback_gui(f),
                     done_callback=done_callback_gui,
                     stop_flag_getter=lambda: self._calib_stop_flag
                 )
            except Exception as e:
                 logging.error(f"Exception in calibration thread: {e}", exc_info=True)
                 # Signal failure if thread crashes
                 self.calibrationDoneSignal.emit(False)
            logging.info("Calibration thread finished.")

        # Start the thread
        self._calib_thread = threading.Thread(target=thread_func_calib, daemon=True)
        self._calib_thread.start()


    def stop_calibration(self):
        """Signals the calibration thread to stop."""
        if self._calib_thread is None or not self._calib_thread.is_alive():
             logging.warning("Calibration is not running, cannot stop.")
             return
        logging.info("Calibration stop requested.")
        self._calib_stop_flag = True
        self.calib_stop_button.setEnabled(False) # Disable stop button after clicking
        # Don't disable start button here, wait for done_callback


    def start_working_mode(self):
        """
        Starts the working mode: initiates video capture, confirms working area,
        and launches the main processing thread.
        Refactored for Signals/Slots and new Modbus logic.
        """
        if self._working_thread is not None and self._working_thread.is_alive():
            logging.warning("Working mode thread is already running.")
            self.showMessageSignal.emit("Warning", "Working mode is already running.")
            return

        # --- Reset State ---
        self._working_stop_flag = False
        self._detector = None
        self.working_area_mask = None
        self.last_detections = []
        self.working_start_button.setEnabled(False) # Disable start button
        self.working_stop_button.setEnabled(True)  # Enable stop button

        # --- 1. Select Video Source ---
        source = self.working_source_selector.currentText()
        cap = None
        source_name = ""
        try:
            if source == "Working video":
                video_path = self.working_video_path_input.text()
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Working video file not found: {video_path}")
                cap = cv2.VideoCapture(video_path)
                source_name = video_path
            else: # Camera source
                index = int(self.working_camera_index_input.text())
                cap = cv2.VideoCapture(index)
                source_name = f"Camera Index {index}"

            if not cap or not cap.isOpened():
                raise IOError(f"Failed to open video source: {source_name}")
            logging.info(f"Using working source: {source_name}")

        except (FileNotFoundError, ValueError, IOError, Exception) as e:
            logging.error(f"Error initializing video source: {e}")
            self.showMessageSignal.emit("Error", f"Error initializing video source:\n{e}")
            if cap: cap.release()
            self.working_start_button.setEnabled(True) # Re-enable start button
            self.working_stop_button.setEnabled(False)
            return

        # --- 2. Load Calibration Data ---
        try:
            if not os.path.exists(var.CALIBRATION_FILE):
                 raise FileNotFoundError(f"Calibration file not found: {var.CALIBRATION_FILE}")
            # No need to load matrix/coeffs here, undistort_frame does it. Just check existence.
            logging.info(f"Found calibration data file: {var.CALIBRATION_FILE}")
        except Exception as e:
            logging.error(f"Calibration file check failed: {e}")
            self.showMessageSignal.emit("Error", f"Calibration file error:\n{e}\nPlease run calibration.")
            cap.release()
            self.working_start_button.setEnabled(True)
            self.working_stop_button.setEnabled(False)
            return

        # --- 3. Confirm Working Area (runs in main thread) ---
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            logging.error("Failed to read the first frame from the video source.")
            self.showMessageSignal.emit("Error", "Failed to read the first frame.")
            cap.release()
            self.working_start_button.setEnabled(True)
            self.working_stop_button.setEnabled(False)
            return

        # Undistort the frame for working area detection
        undistorted_first_frame = undistort_frame(first_frame, var.CALIBRATION_FILE)
        if undistorted_first_frame is first_frame: # Check if undistortion worked
             logging.warning("Undistortion failed for first frame, using original for working area detection.")

        confirmed = self.prepare_working_area(undistorted_first_frame) # Pass undistorted frame
        if not confirmed:
            logging.warning("Working area confirmation failed or was cancelled.")
            cap.release()
            self.working_start_button.setEnabled(True)
            self.working_stop_button.setEnabled(False)
            return
        # If confirmed, self.working_area_mask and self.detection_params["CONVERSION_FACTOR"] are set

        # --- 4. Initialize Object Detector ---
        identification_config_path = None
        if not os.path.exists(var.IDENTIFICATION_CONFIG):
            logging.warning(f"Identification config {var.IDENTIFICATION_CONFIG} not found.")
            # self.showMessageSignal.emit("Warning", f"Identification config not found:\n{var.IDENTIFICATION_CONFIG}\nObject recognition may be limited.")
        else:
            identification_config_path = var.IDENTIFICATION_CONFIG
            logging.info(f"Using identification config: {identification_config_path}")

        try:
            # Pass the updated detection_params (includes CONVERSION_FACTOR)
            self._detector = ObjectDetector(
                identification_config_path=identification_config_path,
                detection_params=self.detection_params,
            )
            logging.info("Object detector initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize ObjectDetector: {e}", exc_info=True)
            self.showMessageSignal.emit("Error", f"Failed to initialize ObjectDetector:\n{e}")
            cap.release()
            self.working_start_button.setEnabled(True)
            self.working_stop_button.setEnabled(False)
            return


        # --- 5. Define and Start the Worker Thread ---
        # Accept mb_params dictionary as the second argument

        # Define Modbus parameters
        modbus_params = {
            'req_addr': int(self.req_flag_addr_input.text()),
            'is_coil': self.req_flag_is_coil_check.isChecked(),
            'max_objs': int(self.max_objects_input.text()),
            'num_obj_addr': int(self.num_objects_addr_input.text()),
            'data_start_addr': int(self.obj_data_start_addr_input.text())
        }

        # --- 5. Define and Start the Worker Thread ---
        # Accept mb_params dictionary as the second argument
        def thread_func(video_capture, mb_params):
            """Worker function to process video frames, handle Modbus, and update GUI via signals."""
            logging.info("Working mode thread started.")
            frame_counter = 0
            # Use local variable for detections within the loop iteration
            current_detections = []

            # Helper function to generate pixmap for debug views (avoids code repetition)
            def _prepare_debug_pixmap(image_data, caption, label_widget):
                if image_data is not None and image_data.size > 0:
                    try:
                        # Ensure it's 8-bit if needed (e.g., Canny edges)
                        if image_data.dtype != np.uint8:
                            image_data = image_data.astype(np.uint8)

                        # Add caption overlay
                        img_with_caption = self._overlay_caption(image_data, caption) # Assumes _overlay_caption exists in MainWindow

                        # Get label dimensions (use defaults if not available yet)
                        lbl_w = label_widget.width() if label_widget.width() > 0 else 320
                        lbl_h = label_widget.height() if label_widget.height() > 0 else 180

                        # Convert to QPixmap
                        pixmap = self.frame_to_qpixmap(img_with_caption, lbl_w, lbl_h) # Assumes frame_to_qpixmap exists in MainWindow

                        if not pixmap.isNull():
                            return pixmap
                        else:
                            logging.warning(f"Failed to create pixmap for {caption}")
                            return None
                    except Exception as e_prep:
                        logging.error(f"Error preparing pixmap for {caption}: {e_prep}")
                        return None
                else:
                    # logging.debug(f"No valid image data for {caption}") # Optional debug
                    return None


            while not self._working_stop_flag: # Use the instance flag
                ret, frame = video_capture.read()
                if not ret or frame is None:
                    logging.info("Video source ended or read failed.")
                    break # Exit loop

                try:
                    # --- Frame Processing ---
                    if self._detector is None:
                        logging.error("Object detector is not initialized. Stopping thread.")
                        break

                    result = self._detector.process_frame(
                        frame, frame_counter, working_area_mask=self.working_area_mask, mode="process"
                    )

                    display_image = None
                    edges = None
                    current_detections = [] # Reset detections for this frame

                    if result is None or len(result) != 3:
                        logging.warning(f"Frame {frame_counter}: process_frame returned invalid result.")
                        display_image = undistort_frame(frame, var.CALIBRATION_FILE) # Show undistorted original
                    else:
                        edges, objects_overlayed, current_detections = result

                        # --- Select Image for Main Display ---
                        if self.display_overlay_search and self.last_overlay_frame is not None:
                            display_image = self.last_overlay_frame
                        else:
                            display_image = objects_overlayed

                    # --- Update Main Video Display (via Signal) ---
                    if display_image is not None:
                        label_w = self.working_video_label.width()
                        label_h = self.working_video_label.height()
                        if label_w > 0 and label_h > 0:
                            main_pixmap = self.frame_to_qpixmap(display_image, label_w, label_h)
                            if not main_pixmap.isNull():
                                self.updateWorkingPixmapSignal.emit(main_pixmap)
                        # else: logging.debug("Main label size zero") # Reduce log noise

                    # --- Update Debug Views Below (via Signals) ---
                    # Only process if the checkbox is checked (reduces overhead)
                    if self.show_debug_views_checkbox.isChecked():
                        # --- Prepare Canny Pixmap ---
                        canny_pixmap = _prepare_debug_pixmap(edges, "Canny", self.canny_debug_label)
                        if canny_pixmap:
                            self.updateCannyPixmapSignal.emit(canny_pixmap)

                        # --- Prepare Contours Pixmap ---
                        if edges is not None and edges.ndim == 2 and edges.size > 0:
                            try:
                                # Create colored contour image (needs grayscale edges)
                                contours_img_color = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
                                contours_found, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in contours_found:
                                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                                    cv2.drawContours(contours_img_color, [cnt], -1, color, thickness=cv2.FILLED)
                                    cv2.drawContours(contours_img_color, [cnt], -1, (0, 0, 0), thickness=1)

                                contour_pixmap = _prepare_debug_pixmap(contours_img_color, "Contours", self.contour_debug_label)
                                if contour_pixmap:
                                    self.updateContourPixmapSignal.emit(contour_pixmap)
                            except Exception as e_dbg_contour:
                                logging.error(f"Error preparing contour pixmap: {e_dbg_contour}")


                    # --- MODBUS COMMUNICATION LOGIC ---
                    if self.modbus_checkbox.isChecked() and self.robot is not None and self.robot.connected:
                        try:
                            # 1. Read Request Flag
                            request_flag_value = self.robot.read_request_flag(
                                mb_params['req_addr'], mb_params['is_coil']
                            )

                            # 2. Check if Flag is Set
                            if request_flag_value is not None and request_flag_value != 0:
                                logging.info(f"Frame {frame_counter}: PLC requested data (Flag={request_flag_value}). Preparing packet...")

                                # 3. Prepare Packet
                                packet_data = []
                                send_success = False

                                if current_detections:
                                    num_to_send = min(len(current_detections), mb_params['max_objs'])
                                    packet_data = [num_to_send]

                                    for i in range(num_to_send):
                                        det = current_detections[i]
                                        category_code = category_mapping.get(det.get('predicted_category', 'unknown'), 0)
                                        center_x, center_y = det.get('center', [0, 0])
                                        width, height = det.get('width', 0), det.get('height', 0)
                                        angle = det.get('angle', 0)
                                        track_id = det.get('track_id', 0)

                                        object_data = [
                                            int(track_id), int(center_x * 100), int(center_y * 100),
                                            int(width * 100), int(height * 100), int(angle * 100),
                                            int(category_code)
                                        ]
                                        if len(object_data) != var.MODBUS_REGISTERS_PER_OBJECT:
                                            logging.error(f"Data mismatch obj {i}! Exp {var.MODBUS_REGISTERS_PER_OBJECT}, got {len(object_data)}. Aborting.")
                                            packet_data = None; break
                                        packet_data.extend(object_data)
                                else:
                                    packet_data = [0] # Send count 0 if no detections
                                    logging.info(f"Frame {frame_counter}: No objects detected, preparing packet with count 0.")

                                # 4. Send Packet
                                if packet_data is not None:
                                    logging.info(f"Frame {frame_counter}: Sending packet for {packet_data[0]} objects.")
                                    send_success = self.robot.client.write_multiple_registers(
                                        mb_params['num_obj_addr'], packet_data
                                    )
                                    if not send_success: logging.error(f"Frame {frame_counter}: Modbus write failed.")
                                    else: logging.info(f"Frame {frame_counter}: Packet sent successfully.")

                                # 5. Reset PLC Request Flag
                                if send_success or (packet_data is not None and packet_data[0] == 0) :
                                    logging.info(f"Frame {frame_counter}: Resetting PLC request flag.")
                                    reset_success = self.robot.reset_request_flag(
                                        mb_params['req_addr'], mb_params['is_coil']
                                    )
                                    if not reset_success: logging.warning(f"Frame {frame_counter}: Failed to reset PLC flag.")

                            # else: # Optional log if flag isn't set
                            #     logging.debug(f"Frame {frame_counter}: PLC flag not set or Modbus off.")

                        except Exception as e_modbus:
                            logging.error(f"Frame {frame_counter}: Error during Modbus comm: {e_modbus}", exc_info=True)

                    # --- Non-Modbus Output ---
                    elif not self.modbus_checkbox.isChecked() and current_detections:
                        print(f"--- Frame {frame_counter} Detections ---")
                        for d in current_detections:
                            center = d.get('center', [0,0])
                            print(f"  ID {d.get('track_id','N/A')} | Cat: {d.get('predicted_category','unk')} "
                                f"| C: ({center[0]:.1f},{center[1]:.1f})mm "
                                f"| Sz: {d.get('width',0):.1f}x{d.get('height',0):.1f}mm")

                    frame_counter += 1
                    time.sleep(0.005) # Yield CPU

                except Exception as e_loop:
                    logging.error(f"Error in working thread loop (Frame {frame_counter}): {e_loop}", exc_info=True)
                    # self.showMessageSignal.emit("Thread Error", f"Critical error in processing loop:\n{e_loop}")
                    break # Stop processing on major error

            # --- Loop Finished ---
            if video_capture: video_capture.release()
            logging.info("Working mode thread finished. Video capture released.")
            # Maybe signal main thread to update button states if stop wasn't called explicitly?
            # self.threadStoppedSignal.emit() # Requires defining this signal
        # --- Start the Thread ---
        # Ensure modbus_params dictionary is created BEFORE this line in start_working_mode
        self._working_thread = threading.Thread(target=thread_func, args=(cap, modbus_params), name="WorkingModeThread", daemon=True)
        self._working_thread.start()
        logging.info("Working mode thread initiated.")


    def stop_working_mode(self):
        """Signals the working mode thread to stop and updates UI."""
        if self._working_thread is None or not self._working_thread.is_alive():
             logging.info("Working mode is not running.")
             # Ensure UI state is correct if stop is called mistakenly
             self.working_start_button.setEnabled(True)
             self.working_stop_button.setEnabled(False)
             return

        logging.info("Working mode stop requested.")
        self._working_stop_flag = True
        # Wait briefly for the thread to finish
        self._working_thread.join(timeout=1.0) # Wait up to 1 second

        if self._working_thread.is_alive():
             logging.warning("Working thread did not stop within the timeout.")
             # Force stop? Not directly possible in Python threads.
        else:
             logging.info("Working thread stopped.")

        # Clean up and update UI regardless of join timeout
        self._working_thread = None
        self.working_start_button.setEnabled(True)
        self.working_stop_button.setEnabled(False)
        # Optional: Clear video label?
        # self.working_video_label.clear()
        # self.working_video_label.setText("Working mode stopped.")

    def confirm_working_area(self, overlay_frame):
        """
        Shows the detected working area overlay and waits for user confirmation.
        Attempts to guarantee immediate visual/structural removal of previous widgets.
        """
        # --- Force Removal of Previous Confirmation Widget ---
        if hasattr(self, 'working_confirm_layout_widget') and self.working_confirm_layout_widget is not None:
            logging.debug("Attempting to forcefully remove previous confirmation widget.")
            try:
                old_widget = self.working_confirm_layout_widget
                if old_widget: # Check if the instance is valid

                    # 1. Hide it immediately
                    old_widget.hide()
                    logging.debug(f"Hid old widget: {old_widget}")

                    # 2. Remove it from its parent layout
                    parent = old_widget.parentWidget()
                    if parent and parent.layout():
                        logging.debug(f"Attempting to remove old widget from layout: {parent.layout()}")
                        # removeWidget returns the item, doesn't delete the widget itself
                        parent.layout().removeWidget(old_widget)
                        logging.debug(f"Removed old widget from layout.")
                    else:
                        logging.warning("Could not find parent layout for old confirmation widget during removal.")

                    # 3. Schedule for actual deletion
                    old_widget.deleteLater()
                    logging.debug(f"Scheduled old widget for deletion.")

                # 4. Clear the reference in MainWindow IMMEDIATELY
                self.working_confirm_layout_widget = None
                logging.debug("Set self.working_confirm_layout_widget to None after forced removal.")

            except RuntimeError as e_runtime:
                 # Catch potential "wrapped C/C++ object of type ... has been deleted"
                 logging.error(f"RuntimeError removing old widget (might already be deleted): {e_runtime}")
                 self.working_confirm_layout_widget = None # Ensure reference is cleared
            except Exception as e_force_remove:
                logging.error(f"Error during forceful removal of previous confirmation widget: {e_force_remove}")
                # Attempt to clear reference anyway
                self.working_confirm_layout_widget = None
        # --- End of Force Removal ---


        self._confirmation_result = None

        # --- Display Overlay ---
        try:
             pixmap = self.frame_to_qpixmap(overlay_frame, self.working_video_label.width(), self.working_video_label.height())
             if not pixmap.isNull():
                  self.working_video_label.setPixmap(pixmap)
             else:
                  logging.error("Failed to create pixmap for working area confirmation.")
                  self.working_video_label.setText("Error displaying confirmation frame.")
                  return False
        except Exception as e:
             logging.error(f"Error displaying working area confirmation frame: {e}")
             self.working_video_label.setText("Error displaying confirmation frame.")
             return False

        # --- Create NEW Confirmation Controls ---
        confirm_widget = QWidget()
        # Assign the NEW widget instance to the attribute
        self.working_confirm_layout_widget = confirm_widget
        logging.debug(f"Created new confirmation widget: {self.working_confirm_layout_widget}")

        confirm_layout = QHBoxLayout(self.working_confirm_layout_widget)
        confirm_layout.setContentsMargins(10, 5, 10, 5)

        confirm_label = QLabel("<b>Confirm Working Area?</b>")
        confirm_button_yes = QPushButton("✅ Yes")
        confirm_button_no = QPushButton("❌ No")
        confirm_button_yes.setStyleSheet("background-color: lightgreen;")
        confirm_button_no.setStyleSheet("background-color: lightcoral;")

        confirm_layout.addWidget(confirm_label)
        confirm_layout.addStretch()
        confirm_layout.addWidget(confirm_button_yes)
        confirm_layout.addWidget(confirm_button_no)

        # --- Event Loop for Waiting ---
        loop = QEventLoop()

        def accept_action():
            self._confirmation_result = True
            logging.info("User accepted the working area.")
            loop.quit()

        def reject_action():
            self._confirmation_result = False
            logging.info("User rejected the working area.")
            loop.quit()

        confirm_button_yes.clicked.connect(accept_action)
        confirm_button_no.clicked.connect(reject_action)

        # --- Add NEW Controls to GUI ---
        # Determine parent layout dynamically EACH time
        parent_layout = self.working_video_label.parentWidget().layout()
        widget_added = False
        if isinstance(parent_layout, QVBoxLayout):
             index = -1
             for i in range(parent_layout.count()):
                 item = parent_layout.itemAt(i)
                 if item and item.widget() == self.working_video_label:
                      index = i
                      break
             if index != -1:
                 # Add the NEW widget
                 parent_layout.insertWidget(index + 1, self.working_confirm_layout_widget)
                 widget_added = True
                 logging.debug(f"Inserted new widget at index {index+1} in {parent_layout}")
             else:
                 logging.warning("Could not find working_video_label index, adding confirmation widget at the end.")
                 parent_layout.addWidget(self.working_confirm_layout_widget)
                 widget_added = True
                 logging.debug(f"Added new widget at the end of {parent_layout}")

             # Make the new widget visible (it might be hidden by default sometimes)
             self.working_confirm_layout_widget.show()

        else:
             logging.error("Cannot add confirmation buttons: Parent layout not a QVBoxLayout or not found.")


        # --- Wait for User Response ---
        if widget_added:
            logging.info("Waiting for user confirmation of working area...")
            loop.exec_()
        else:
            logging.error("Confirmation widget could not be added to layout. Aborting confirmation.")
            self._confirmation_result = False
            # Clean up the widget we just created but couldn't add
            if self.working_confirm_layout_widget:
                self.working_confirm_layout_widget.deleteLater()
                self.working_confirm_layout_widget = None


        # --- Clean Up CURRENT Widget (after user response) ---
        # We repeat the forceful removal logic for the widget we just handled
        if hasattr(self, 'working_confirm_layout_widget') and self.working_confirm_layout_widget is not None:
            logging.debug("Cleaning up CURRENT confirmation widget after user response.")
            try:
                current_widget = self.working_confirm_layout_widget
                if current_widget:
                    current_widget.hide()
                    parent = current_widget.parentWidget()
                    if parent and parent.layout():
                        parent.layout().removeWidget(current_widget)
                    current_widget.deleteLater()
                self.working_confirm_layout_widget = None
                logging.debug("Cleared reference to current widget after cleanup.")
            except RuntimeError as e_runtime_end:
                 logging.error(f"RuntimeError cleaning up current widget: {e_runtime_end}")
                 self.working_confirm_layout_widget = None
            except Exception as e_cleanup_end:
                 logging.error(f"Error cleaning up current confirmation widget: {e_cleanup_end}")
                 self.working_confirm_layout_widget = None
        # --- End of Cleanup ---

        return self._confirmation_result

    def prepare_working_area(self, frame_to_process):
        """
        Detects working area in the given frame and calls user confirmation.
        Args:
            frame_to_process: The (preferably undistorted) frame to analyze.
        Returns:
            True if area detected and confirmed, False otherwise.
        """
        if self.working_area_processor is None:
            logging.error("Working area processor not initialized.")
            return False

        try:
            result = self.working_area_processor.objectDetection(frame_to_process)
        except Exception as e:
             logging.error(f"Exception during working area detection: {e}", exc_info=True)
             self.showMessageSignal.emit("Error", f"Error during working area detection:\n{e}")
             return False

        if result is None:
            logging.warning("Working area detection failed or was rejected by user.")
            # Confirmation callback handles user rejection message/logging
            return False
        else:
            overlay_frame, working_mask, conversion_factor = result
            # Store results in MainWindow state
            self.working_area_mask = working_mask
            self.last_overlay_frame = overlay_frame # Used for display toggle maybe
            # Update detection_params directly (WorkingArea already does this, but be explicit)
            self.detection_params["CONVERSION_FACTOR"] = conversion_factor
            # Update the conversion factor slider/label in the GUI
            if "CONVERSION_FACTOR" in self.param_widgets:
                widgets = self.param_widgets["CONVERSION_FACTOR"]
                widgets['label'].setText(f"{conversion_factor:.{widgets['decimals']}f}")
                # Avoid division by zero if step is zero
                step = widgets['step']
                if step != 0:
                     widgets['slider'].setValue(int(conversion_factor / step))
            logging.info("Working area detected and confirmed.")
            return True


    def load_detection_params(self):
        """Loads detection parameters from JSON file."""
        try:
            if os.path.exists(var.PARAMETERS_CONFIG):
                with open(var.PARAMETERS_CONFIG, "r") as f:
                    params = json.load(f)
                    # Update only existing keys, be careful not to overwrite calculated ones like CONVERSION_FACTOR?
                    # Or just load all saved ones. Let's load all defined ones.
                    for key, val in params.items():
                        if key in self.detection_params: # Only load known params
                            # Apply type constraints if necessary (e.g., BLUR_KERNEL must be int)
                            if key == "BLUR_KERNEL":
                                val = int(val)
                                if val % 2 == 0: val = max(1, val -1) # Ensure odd
                            self.detection_params[key] = val
                    logging.info(f"Loaded detection parameters from {var.PARAMETERS_CONFIG}")
            else:
                 logging.info("No saved parameter config found, using defaults.")

        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to load detection parameters from {var.PARAMETERS_CONFIG}: {e}")
            # Don't show message box here, just log error

    def save_detection_params(self):
        """Saves current detection parameters to JSON file."""
        # Define which parameters to save (exclude temporary/calculated ones if needed)
        keys_to_save = ["BLUR_KERNEL", "CANNY_LOW", "CANNY_HIGH", "MIN_AREA", "MAX_AREA"]
        # Optionally save SCALE if it's actually used and adjustable
        # Optionally save CONVERSION_FACTOR if manual adjustment is desired, but usually calculated
        params_to_save = {k: self.detection_params[k] for k in keys_to_save if k in self.detection_params}

        try:
            with open(var.PARAMETERS_CONFIG, "w") as f:
                json.dump(params_to_save, f, indent=4) # Use indent for readability
            logging.info(f"Detection parameters saved to {var.PARAMETERS_CONFIG}")
        except Exception as e:
            logging.error(f"Failed to save detection parameters to {var.PARAMETERS_CONFIG}: {e}")
            self.showMessageSignal.emit("Error", f"Failed to save parameters:\n{e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def frame_to_qpixmap(self, frame: np.ndarray, target_w: int, target_h: int) -> QPixmap:
        """
        Converts an OpenCV frame (NumPy array) to a QPixmap, scaled to fit
        the target dimensions while maintaining aspect ratio. Handles potential errors.
        Args:
            frame: Input OpenCV frame (BGR).
            target_w: Target width of the display label.
            target_h: Target height of the display label.
        Returns:
            QPixmap object, or an empty QPixmap on error.
        """
        if frame is None or frame.size == 0 or target_w <= 0 or target_h <= 0:
            logging.warning("Invalid frame or target dimensions for QPixmap conversion.")
            return QPixmap() # Return empty pixmap

        try:
            frame_h, frame_w = frame.shape[:2]
            if frame_w <= 0 or frame_h <= 0: return QPixmap()

            # Calculate scaling factor
            scale = min(target_w / frame_w, target_h / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)

            # Resize the frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create a black canvas and center the resized frame
            # Ensure canvas has 3 channels (BGR)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            # Convert canvas BGR to RGB for QImage
            frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w

            # Create QImage and QPixmap
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            return pixmap

        except Exception as e:
            logging.error(f"Error in frame_to_qpixmap: {e}", exc_info=True)
            return QPixmap() # Return empty pixmap on error


    def update_calibration_source(self):
        """Shows/hides controls based on selected calibration source."""
        use_video = self.source_selector.currentText() == "Calibration video"
        self.video_layout_widget.setVisible(use_video)
        self.camera_layout_widget.setVisible(not use_video)

    def update_working_source(self):
        """Shows/hides controls based on selected working source."""
        use_video = self.working_source_selector.currentText() == "Working video"
        self.working_video_layout_widget.setVisible(use_video)
        self.working_camera_layout_widget.setVisible(not use_video)

    def browse_calibration_video(self):
        """Opens dialog to select calibration video."""
        # Use parent directory of current path as starting point?
        start_dir = os.path.dirname(var.CALIBRATION_VIDEO_PATH) if os.path.exists(var.CALIBRATION_VIDEO_PATH) else ""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Calibration Video", start_dir, "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.calib_video_path_input.setText(file_path)
            # Optionally update var.CALIBRATION_VIDEO_PATH immediately, but maybe wait until start?
            # var.CALIBRATION_VIDEO_PATH = file_path
            logging.info(f"Selected calibration video path: {file_path}")


    def browse_working_video(self):
        """Opens dialog to select working video."""
        start_dir = os.path.dirname(var.WORKING_VIDEO_PATH) if os.path.exists(var.WORKING_VIDEO_PATH) else ""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Working Video", start_dir, "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.working_video_path_input.setText(file_path)
            # Optionally update var.WORKING_VIDEO_PATH immediately
            # var.WORKING_VIDEO_PATH = file_path
            logging.info(f"Selected working video path: {file_path}")

    def update_camera_index(self):
        """Validates and updates the camera index from input."""
        try:
            index = int(self.camera_index_input.text())
            if index < 0: raise ValueError("Index cannot be negative")
            # Optionally update var.CAMERA_INDEX immediately
            # var.CAMERA_INDEX = index
            logging.info(f"Camera index updated to: {index}")
        except ValueError:
            logging.error(f"Invalid camera index entered: {self.camera_index_input.text()}")
            self.showMessageSignal.emit("Error", "Invalid camera index. Please enter a non-negative integer.")
            # Reset to previous valid value?
            # self.camera_index_input.setText(str(var.CAMERA_INDEX))

    def closeEvent(self, event):
        """Handles the window close event for proper shutdown."""
        logging.info("Shutdown requested. Cleaning up...")

        # Stop working thread if running
        self.stop_working_mode() # Ensures flag is set and waits briefly

        # Stop calibration thread if running
        self.stop_calibration() # Sets flag
        if self._calib_thread is not None and self._calib_thread.is_alive():
             self._calib_thread.join(timeout=0.5) # Wait briefly
             if self._calib_thread.is_alive():
                 logging.warning("Calibration thread did not stop quickly.")
        self._calib_thread = None

        # Disconnect from Modbus server
        self.disconnect_modbus() # Ensures robot instance is cleaned up

        # Save detection parameters
        self.save_detection_params()

        logging.info("Cleanup finished. Application will close now.")
        event.accept() # Allow window to close

    def save_modbus_settings(self):
        """Saves Modbus GUI settings to a JSON file."""
        settings = {
            "host": self.ip_input.text(),
            "port": self.port_input.text(),
            "req_addr": self.req_flag_addr_input.text(),
            "is_coil": self.req_flag_is_coil_check.isChecked(),
            "max_objs": self.max_objects_input.text(),
            "num_obj_addr": self.num_objects_addr_input.text(),
            "data_start_addr": self.obj_data_start_addr_input.text()
            # "regs_per_obj": self.regs_per_obj_input.text() # If added
        }
        # Use a dedicated config file for Modbus settings
        config_path = os.path.join(var.APP_DIR, "modbus_config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(settings, f, indent=4)
            logging.info(f"Modbus settings saved to {config_path}")
        except Exception as e:
            logging.error(f"Failed to save Modbus settings: {e}")
            # Optionally inform user via message signal if save fails critically?
            # self.showMessageSignal.emit("Error", f"Could not save Modbus settings:\n{e}")




    def load_modbus_settings(self):
        """Loads Modbus GUI settings from JSON file or uses defaults."""
        config_path = os.path.join(var.APP_DIR, "modbus_config.json")
        # Define defaults using values from variables.py initially
        defaults = {
            "host": var.MODBUS_TCP_HOST,
            "port": str(var.MODBUS_TCP_PORT),
            "req_addr": str(var.MODBUS_DATA_REQUEST_ADDR),
            "is_coil": var.MODBUS_IS_REQUEST_FLAG_COIL,
            "max_objs": str(var.MODBUS_MAX_OBJECTS),
            "num_obj_addr": str(var.MODBUS_NUM_OBJECTS_ADDR),
            "data_start_addr": str(var.MODBUS_OBJECT_DATA_START_ADDR)
            # "regs_per_obj": str(var.MODBUS_REGISTERS_PER_OBJECT) # If added
        }
        settings = defaults.copy() # Start with defaults

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    loaded_settings = json.load(f)
                    # Update settings with loaded values, keeping defaults for missing keys
                    for key in defaults.keys():
                        if key in loaded_settings:
                            # Basic type validation might be needed here if file gets corrupted
                            settings[key] = loaded_settings[key]
                logging.info(f"Loaded Modbus settings from {config_path}")
            else:
                logging.info("Modbus config file not found, using defaults.")

        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            logging.error(f"Failed to load Modbus settings: {e}. Using defaults.")
            settings = defaults.copy() # Ensure defaults are used on error

        # --- Apply settings to the GUI widgets ---
        try:
            self.ip_input.setText(settings["host"])
            self.port_input.setText(str(settings["port"])) # Ensure string
            self.req_flag_addr_input.setText(str(settings["req_addr"]))
            # Ensure 'is_coil' is treated as boolean
            is_coil_val = settings["is_coil"]
            self.req_flag_is_coil_check.setChecked(bool(is_coil_val))
            self.max_objects_input.setText(str(settings["max_objs"]))
            self.num_objects_addr_input.setText(str(settings["num_obj_addr"]))
            self.obj_data_start_addr_input.setText(str(settings["data_start_addr"]))
            # if hasattr(self, 'regs_per_obj_input'): # Check if widget exists
            #     self.regs_per_obj_input.setText(str(settings.get("regs_per_obj", var.MODBUS_REGISTERS_PER_OBJECT)))
        except KeyError as e:
            logging.error(f"Missing key in loaded/default Modbus settings: {e}")
        except Exception as e_apply:
            logging.error(f"Error applying loaded Modbus settings to GUI: {e_apply}")    

    def _overlay_caption(self, image: np.ndarray, caption: str) -> np.ndarray:
        """ Overlays a caption text onto the image. """
        img_copy = image.copy()
        # Ensure it's BGR for drawing color text/rect
        if img_copy.ndim == 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        elif img_copy.shape[2] == 1: # Handle single channel grayscale that thinks it's color
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)


        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        # Use white text with black background for contrast
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)

        # Get text size to position background
        (text_width, text_height), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
        text_height += baseline # Add baseline for better positioning

        # Position at top-left
        x, y = 5, 5 # Padding from border
        # Draw background rectangle
        cv2.rectangle(img_copy, (x, y), (x + text_width + 4, y + text_height + 4), bg_color, cv2.FILLED)
        # Put text on top
        cv2.putText(img_copy, caption, (x + 2, y + text_height + 2 - baseline // 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return img_copy
    
 
# =========================================================================
# Main Execution Block
# =========================================================================
if __name__ == '__main__':
    # Configure logging (File and Console)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(threadName)-10s] %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set root level (e.g., INFO or DEBUG)

    # Clear existing handlers (if re-running in interactive environment)
    #for handler in root_logger.handlers[:]:
    #    root_logger.removeHandler(handler)

    # Create directory if it doesn't exist
    if not os.path.exists(var.APP_DIR):
        try:
            os.makedirs(var.APP_DIR)
        except OSError as e:
             print(f"Error creating app data directory {var.APP_DIR}: {e}")
             sys.exit(1) # Exit if cannot create log dir

    # File handler (Append mode)
    try:
        file_handler = logging.FileHandler(var.LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
         print(f"Error setting up file logger to {var.LOG_FILE}: {e}")


    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    logging.info("================ Application Starting ================")

    # --- Run Application ---
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    exit_code = app.exec_()
    logging.info(f"================ Application Exited (Code: {exit_code}) ================")
    sys.exit(exit_code)
