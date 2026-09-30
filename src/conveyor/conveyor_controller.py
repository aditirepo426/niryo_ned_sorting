import time
import serial

class ConveyorController:
    def __init__(self, port, baudrate=115200, timeout=0.2):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)

    def start(self):
        self.ser.write(b"START\n")

    def stop(self):
        self.ser.write(b"STOP\n")

    def wait_for_object(self, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line == "OBJECT_DETECTED":
                return True
        return False

    def close(self):
        self.ser.close()
