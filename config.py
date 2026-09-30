"""Central configuration for the Niryo conveyor sorting system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROBOT_IP = "192.168.10.10"       # TODO: replace with Ned2 IP
CAMERA_ID = 0                    # external USB camera index
CONVEYOR_SERIAL_PORT = "COM3"    # TODO: replace
CONVEYOR_BAUDRATE = 115200

# Fixed physical station: calibrate these on the real setup.
PICK_POSE = (0.20, 0.00, 0.12, 0.0, 1.57, 0.0)
BIN_POSES = {
    "RED":    (0.20, 0.25, 0.12, 0.0, 1.57, 0.0),
    "GREEN":  (0.20, 0.35, 0.12, 0.0, 1.57, 0.0),
    "BLUE":   (0.20, 0.45, 0.12, 0.0, 1.57, 0.0),
    "YELLOW": (0.20, 0.55, 0.12, 0.0, 1.57, 0.0),
}

# Camera ROI around the fixed pick station: x1, y1, x2, y2.
ROI = (100, 80, 540, 400)
MIN_CONTOUR_AREA = 500

COLOR_CLASSES = ["RED", "GREEN", "BLUE", "YELLOW"]
N_COLOR_CLUSTERS = 4
MODEL_DIR = ROOT / "models" / "color"
SCALER_PATH = MODEL_DIR / "color_scaler.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_color.pkl"
MAPPING_PATH = MODEL_DIR / "color_mapping.json"
