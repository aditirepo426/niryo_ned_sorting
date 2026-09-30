import cv2

class USBCamera:
    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open USB camera {camera_id}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("USB camera frame capture failed")
        return frame

    def release(self):
        self.cap.release()
