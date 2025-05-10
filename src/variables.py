# variables.py


import os
import cv2

# Conversion factor: 1 pixel = 1 mm (adjust as needed)
CONVERSION_FACTOR = 1.0

# Base directory for data storage
APP_DIR = os.path.join(os.getcwd(), "app_data")
if not os.path.exists(APP_DIR):
    os.makedirs(APP_DIR)

# Paths for generated files
CALIBRATION_FILE = os.path.join(APP_DIR,"camera_calibration.npz")
LOG_FILE = os.path.join(APP_DIR, "error_log.txt")
DATA_COLLECTION_CSV = os.path.join(APP_DIR, "collected_data.csv")
IDENTIFICATION_CONFIG = os.path.join(APP_DIR, "identification_params.json")
PARAMETERS_CONFIG = os.path.join(APP_DIR, "detection_params.json")

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
MAX_DISTANCE_MM = 10.0
MAX_FRAMES_TO_FIND_WORKING_AREA = 200 # Number of frames to try for WA detection


# Default Video Paths
CALIBRATION_VIDEO_PATH = r'C:\Users\Stas\Videos\Pillproject\cali.mp4'

#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\cylinder_black.mp4"
#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\cylinder_pink.mp4"
#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\cylinder_white.mp4"
#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\rhombus_black.mp4"
#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\rhombus_pink.mp4"
#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\rhombus_white.mp4"
#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\circular_black.mp4"
#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\circular_pink.mp4"
WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\white_circle.mp4"
#WORKING_VIDEO_PATH = r"C:\Users\Stas\Videos\Pillproject\All.mp4"

# Camera configuration
source_type = "video"  # Set to "video" to use video file
#source_type = "camera" # Set to "camera" to use webcam

CAMERA_INDEX = 0  # Default webcam index, 0 - builtin, 1 - external

CAMERA_WIDTH = 1920  # Camera frame width
CAMERA_HEIGHT = 1080  # Camera frame height
CAMERA_FPS = 30  # Camera frame rate
CAMERA_ROTATION = 0  # Camera rotation (0, 90, 180, 270)
CAMERA_FLIP = False  # Flip camera image horizontally
CAMERA_MIRROR = False  # Mirror camera image vertically

TARGET_WIDTH = 1280  # Target frame width for processing
TARGET_HEIGHT_DISPLAY = 720  # Target frame height for processing
TARGET_ROW_HEIGHT = 30  # Target row height for display
BOTTOM_HEIGHT = 100  # Bottom row height for display

# Camera Calibration Settings
MAX_CAPTURES = 100  # Maximum frames for calibration
FRAME_INTERVAL = 60  # Interval between frames when capturing from video
DICT_TYPE = cv2.aruco.DICT_5X5_1000  # ArUco dictionary type
SQUARE_LENGTH = 0.0304  # mm (length of chessboard squares)
MARKER_LENGTH = 0.0154  # mm (length of ArUco markers within squares)
SQUARES_X = 5  # Number of squares horizontally on ChArUco board
SQUARES_Y = 7  # Number of squares vertically on ChArUco board
MIN_CHARUCO_CORNERS = 12  # Minimum corners per frame to accept

# Object Detection Parameters (initial defaults, adjustable via GUI)
SCALE = 0.75  # Image scaling factor
BLUR_KERNEL = 3  # Gaussian blur kernel size (must be odd)
CANNY_LOW = 10  # Lower threshold for Canny edge detection
CANNY_HIGH = 50  # Higher threshold for Canny edge detection
MIN_AREA = 100  # Min contour area (pixels)
MAX_AREA = 50000  # Max contour area (pixels)

# Working area side limits as percentage of frame size
WORKING_AREA_MIN_SIZE_RATIO = 0.01  # 20% of frame width/height
WORKING_AREA_MAX_SIZE_RATIO = 1  # 90% of frame width/height


# Object Tracking Parameters
MAX_LOST_FRAMES = 30  # Frames before a track is considered lost
MAX_DISTANCE = 50  # Maximum pixel distance between frames to track an object
DATA_GATHERING_REQUEUE_DELAY = 2.0 # Seconds to wait before asking again about a skipped/accepted object ID

# Reference object (for scale calibration, e.g., an A4 paper)
REFERENCE_OBJECT_WIDTH_MM = 210.0  # Width of A4 in mm
REFERENCE_OBJECT_HEIGHT_MM = 297.0  # Height of A4 in mm
REFERENCE_ASPECT_RATIO = REFERENCE_OBJECT_WIDTH_MM / REFERENCE_OBJECT_HEIGHT_MM
ASPECT_RATIO_TOLERANCE = 0.01  # Allowed deviation in aspect ratio detection

      
# Modbus TCP configuration (PLC/Robot Communication)
MODBUS_TCP_HOST = "192.168.0.10" # Default PLC/Robot IP
MODBUS_TCP_PORT = 502            # Default Modbus TCP Port
MODBUS_TIMEOUT = 5               # Connection timeout in seconds

# --- Additions for PLC Data Request ---
MODBUS_DATA_REQUEST_ADDR = 100   # <== CHANGE THIS: Address of the PLC register/coil for requesting data
MODBUS_IS_REQUEST_FLAG_COIL = False # <== CHANGE THIS: True if the request flag is a Coil, False if a Holding Register

# --- Additions for Multi-Object Packets ---
MODBUS_MAX_OBJECTS = 5              # <== CHANGE THIS: Max objects per packet PLC can handle
MODBUS_NUM_OBJECTS_ADDR = 200       # <== CHANGE THIS: Address for register holding the number of objects
MODBUS_OBJECT_DATA_START_ADDR = 201 # <== CHANGE THIS: Start address for the data of the first object
MODBUS_REGISTERS_PER_OBJECT = 7     # Number of registers used per object (id, x, y, w, h, angle, category) - Usually stays 7 based on your old send_data


# Object categories (example category codes for robot communication)
OBJECT_CATEGORIES = {
    "unknown": 0,
    "Circle_White": 1,
    "circle_red": 2,
    "rectangle_white": 3,
    "rectangle_red": 4,
    "oval_white": 5,
    "oval_red": 6,
    # Extend as needed...
}
