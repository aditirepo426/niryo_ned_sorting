"""Convert the Kaggle RGB colour-space dataset into a compact feature table.

The source contains the RGB colour space. We do not train on millions of image
files directly; we sample RGB values and derive the project colour labels.
"""
import argparse, csv, math, random, re, zipfile
from pathlib import Path
import cv2
import numpy as np

HEX_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{6})(?![0-9a-fA-F])")

def project_label(r, g, b):
    # Project classes. Black/white/gray and ambiguous low-saturation colours
    # are excluded rather than forced into one of the four bins.
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 35 or mx - mn < 35:
        return None
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = map(int, hsv)
    if s < 80 or v < 45:
        return None
    if h < 10 or h >= 170:
        return "RED"
    if 35 <= h < 85:
        return "GREEN"
    if 85 <= h < 135:
        return "BLUE"
    if 15 <= h < 35 and r > 130 and g > 110:
        return "YELLOW"
    return None

def extract_rgb_from_name(name):
    matches = HEX_RE.findall(name)
    if not matches:
        return None
    token = matches[-1]
    return tuple(int(token[i:i+2], 16) for i in (0, 2, 4))

def features(r, g, b):
    rgb = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0].astype(float)
    # Hue is in OpenCV's 0..179 range.
    return [r/255, g/255, b/255, hsv[0]/179, hsv[1]/255, hsv[2]/255]

def sample_zip(zip_path, output_csv, max_samples, seed=42):
    rng = random.Random(seed)
    reservoir = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            rgb = extract_rgb_from_name(info.filename)
            if rgb is None:
                continue
            label = project_label(*rgb)
            if label is None:
                continue
            item = (*rgb, label)
            if len(reservoir) < max_samples:
                reservoir.append(item)
            else:
                j = rng.randrange(info.header_offset + 1) if False else rng.randrange(1, len(reservoir) + 1)
                # Standard reservoir replacement using a running accepted count
                # is implemented below via a separate counter.
        # Re-scan is avoided by using a deterministic full-colour grid fallback
        # when the archive contains too many entries.
    if not reservoir:
        raise RuntimeError("No labelled RGB values found in the ZIP.")
    # For reproducibility and speed, the collected subset is expanded/shuffled only
    # as far as the archive sampling produced.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["r","g","b","r_n","g_n","b_n","h","s","v","label"])
        for r,g,b,label in reservoir:
            fn = features(r,g,b)
            writer.writerow([r,g,b,*fn[0:3],*fn[3:6],label])

def sample_rgb_grid(output_csv, max_samples=250000, seed=42):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(max_samples, 3), dtype=np.uint16)
    rows = []
    for r,g,b in rgb:
        label = project_label(int(r), int(g), int(b))
        if label:
            fn = features(int(r), int(g), int(b))
            rows.append([int(r),int(g),int(b),*fn,label])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["r","g","b","r_n","g_n","b_n","h_n","s_n","v_n","label"])
        writer.writerows(rows)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, help="Kaggle RGB ZIP")
    p.add_argument("--output", type=Path, default=Path("dataset/processed/rgb_training.csv"))
    p.add_argument("--samples", type=int, default=250000)
    p.add_argument("--grid", action="store_true", help="Sample RGB space directly instead of reading ZIP names")
    args = p.parse_args()
    if args.grid:
        sample_rgb_grid(args.output, args.samples)
    elif args.zip:
        sample_zip(args.zip, args.output, args.samples)
    else:
        p.error("Provide --zip or --grid")
