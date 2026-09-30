"""PyNiryo wrapper. Pose values are placeholders until the physical station is calibrated."""
from pyniryo import NiryoRobot

class RobotController:
    def __init__(self, ip, pick_pose, bin_poses):
        self.robot = NiryoRobot(ip)
        self.pick_pose = pick_pose
        self.bin_poses = bin_poses

    def connect(self):
        self.robot.calibrate_auto()
        self.robot.update_tool()

    def move_pose(self, pose):
        x,y,z,roll,pitch,yaw = pose
        self.robot.move_pose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)

    def pick(self):
        self.move_pose(self.pick_pose)
        self.robot.close_gripper(speed=500, hold_torque=100)

    def place(self, color):
        if color not in self.bin_poses:
            raise ValueError(f"No bin pose configured for {color}")
        self.move_pose(self.bin_poses[color])
        self.robot.open_gripper(speed=500)

    def shutdown(self):
        try:
            self.robot.close_connection()
        except Exception:
            pass
