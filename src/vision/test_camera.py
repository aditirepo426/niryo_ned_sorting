from config import CAMERA_ID
from src.vision.camera import USBCamera
import cv2

cam = USBCamera(CAMERA_ID)
try:
    while True:
        frame = cam.read()
        cv2.imshow("USB Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cam.release()
    cv2.destroyAllWindows()
