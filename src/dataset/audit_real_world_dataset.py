"""Audit the real-world colour dataset before any model training.

Expected layout:
    dataset/real_world/Blue/*.jpg
    dataset/real_world/Green/*.jpg
    dataset/real_world/Red/*.jpg
    dataset/real_world/Yellow/*.jpg

You may also point --input at the corrected ZIP archive. This script does not
train a model or modify source images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2

CLASSES = ("BLUE", "GREEN", "RED", "YELLOW")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_info(data: bytes):
    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    return {"width": int(w), "height": int(h), "channels": int(image.shape[2])}


def entries_from_zip(path: Path):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            p = Path(name)
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parts = [part for part in p.parts if part not in ("", ".", "..")]
            label = next((part.upper() for part in parts if part.upper() in CLASSES), None)
            if label:
                yield name, label, archive.read(name)


def entries_from_dir(path: Path):
    # Accept either the dataset root or the enclosing real_world folder.
    roots = [path]
    if (path / "real_world").is_dir():
        roots.insert(0, path / "real_world")
    for root in roots:
        found = False
        for label in CLASSES:
            folder = next((p for p in root.iterdir() if p.is_dir() and p.name.upper() == label), None) if root.exists() else None
            if folder:
                found = True
                for file in folder.rglob("*"):
                    if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                        yield str(file.relative_to(root)), label, file.read_bytes()
        if found:
            return


def main():
    parser = argparse.ArgumentParser(description="Audit real-world labelled images.")
    parser.add_argument("--input", required=True, help="Corrected ZIP file or extracted dataset directory")
    parser.add_argument("--output-dir", default="results/dataset_audit")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries = entries_from_zip(source) if source.is_file() and zipfile.is_zipfile(source) else entries_from_dir(source)

    counts = Counter()
    exact_hashes = defaultdict(list)
    records = []
    unreadable = []
    for name, label, data in entries:
        counts[label] += 1
        digest = sha256_bytes(data)
        exact_hashes[digest].append(name)
        info = image_info(data)
        if info is None:
            unreadable.append(name)
            records.append({"image": name, "label": label, "readable": False,
                            "width": None, "height": None, "sha256": digest})
        else:
            records.append({"image": name, "label": label, "readable": True,
                            **info, "sha256": digest})

    duplicates = [names for names in exact_hashes.values() if len(names) > 1]
    readable = [r for r in records if r["readable"]]
    report = {
        "source": str(source),
        "total_images_found": len(records),
        "readable_images": len(readable),
        "unreadable_images": len(unreadable),
        "class_counts": {c: counts.get(c, 0) for c in CLASSES},
        "missing_classes": [c for c in CLASSES if counts.get(c, 0) == 0],
        "image_dimensions": {
            "min_width": min((r["width"] for r in readable), default=None),
            "max_width": max((r["width"] for r in readable), default=None),
            "min_height": min((r["height"] for r in readable), default=None),
            "max_height": max((r["height"] for r in readable), default=None),
        },
        "exact_duplicate_groups": duplicates,
        "unreadable_files": unreadable,
        "warning": (
            "This audit checks exact byte duplicates only. Near-duplicate frames "
            "and capture-session leakage should be reviewed before final splitting."
        ),
    }
    (output / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import csv
    with (output / "image_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "readable", "width", "height", "channels", "sha256"])
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {output / 'dataset_audit.json'}")
    print(f"Saved: {output / 'image_manifest.csv'}")


if __name__ == "__main__":
    main()
