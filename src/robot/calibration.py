"""Manual calibration helper for the physical Ned2 station."""
def print_pose_template(name, pose):
    print(f"{name} = ({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}, {pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f})")

if __name__ == "__main__":
    print("Calibrate physically, then copy measured pose values into config.py.")
