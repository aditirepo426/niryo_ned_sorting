import time
import csv
from pathlib import Path

from config import (
    ROBOT_IP, CAMERA_ID, CONVEYOR_SERIAL_PORT, CONVEYOR_BAUDRATE,
    ROI, MIN_CONTOUR_AREA, PICK_POSE, BIN_POSES, MODEL_DIR
)
from src.conveyor.conveyor_controller import ConveyorController
from src.vision.camera import USBCamera
from src.vision.object_detector import detect_object
from src.classification.classify_object import ColorClassifier
from src.robot.robot_controller import RobotController

SETTLE_TIME = 0.35
LOG_PATH = Path("logs/sorting_log.csv")

def log_result(row):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def run():
    conveyor = ConveyorController(CONVEYOR_SERIAL_PORT, CONVEYOR_BAUDRATE)
    camera = USBCamera(CAMERA_ID)
    robot = RobotController(ROBOT_IP, PICK_POSE, BIN_POSES)
    classifier = ColorClassifier(MODEL_DIR)
    try:
        robot.connect()
        conveyor.start()
        print("System ready. Waiting for objects...")
        while True:
            if not conveyor.wait_for_object(timeout=60):
                print("No object detected in the last 60 seconds.")
                continue
            time.sleep(SETTLE_TIME)
            frame = camera.read()
            detection = detect_object(frame, ROI, MIN_CONTOUR_AREA)
            if detection is None:
                print("Object not detected in camera ROI; restarting conveyor.")
                conveyor.start()
                continue
            color, cluster = classifier.predict(detection["roi_frame"], detection["mask"])
            shape = detection["shape"]
            print(f"Detected: shape={shape}, color={color}, cluster={cluster}")
            if color == "UNKNOWN" or shape == "UNKNOWN":
                log_result({"shape":shape, "color":color, "cluster":cluster, "status":"REJECT_UNKNOWN"})
                conveyor.start()
                continue
            robot.pick()
            robot.place(color)
            log_result({"shape":shape, "color":color, "cluster":cluster, "status":"SORTED"})
            conveyor.start()
    except KeyboardInterrupt:
        print("Stopping system.")
    finally:
        try:
            conveyor.stop()
        except Exception:
            pass
        camera.release()
        conveyor.close()
        robot.shutdown()

if __name__ == "__main__":
    run()
