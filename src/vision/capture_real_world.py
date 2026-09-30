"""Capture real camera images for physical-world testing.

Run this with one known object colour at a time and type the label when prompted.
Images are deliberately kept separate from the synthetic Kaggle dataset.
"""
import argparse
import csv
from pathlib import Path
import cv2

def main(camera_id, output_dir):
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.csv"
    new_file = not labels_path.exists()
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}")
    with labels_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["image","label"])
        index = len(list(image_dir.glob("*.jpg")))
        print("Press C to capture, Q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("Real-world dataset capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                label = input("Label [RED/GREEN/BLUE/YELLOW]: ").strip().upper()
                if label not in {"RED","GREEN","BLUE","YELLOW"}:
                    print("Invalid label.")
                    continue
                name = f"{index:06d}.jpg"
                cv2.imwrite(str(image_dir / name), frame)
                writer.writerow([name, label])
                f.flush()
                index += 1
                print("Saved", name)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--output-dir", default="dataset/real_world")
    a = p.parse_args()
    main(a.camera_id, a.output_dir)
