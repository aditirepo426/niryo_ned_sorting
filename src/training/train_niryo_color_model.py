"""Train/evaluate the task-specific colour model from real-camera images.

Expected folder structure:
dataset/real_world/
  BLUE/*.jpg
  GREEN/*.jpg
  RED/*.jpg
  YELLOW/*.jpg

Use independent capture sessions for the final test set. The default split is
image-level and is for development only; it must not be reported as a
session-independent benchmark when images are correlated.
"""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.classification.niryo_color_model import extract_features, FEATURE_VERSION


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset(root):
    X, y, paths = [], [], []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = class_dir.name.upper()
        if label not in {"BLUE", "GREEN", "RED", "YELLOW"}:
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in EXTS:
                continue
            image = cv2.imread(str(path))
            if image is None:
                print(f"WARNING: unreadable image skipped: {path}")
                continue
            try:
                X.append(extract_features(image))
                y.append(label)
                paths.append(str(path))
            except ValueError as exc:
                print(f"WARNING: {path}: {exc}")
    if not X:
        raise SystemExit(f"No readable labelled images found under {root}")
    return np.vstack(X), np.asarray(y), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/real_world")
    ap.add_argument("--model-out", default="models/color/niryo_color_svm.joblib")
    ap.add_argument("--report-out", default="models/color/niryo_color_metrics.json")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, paths = load_dataset(Path(args.data))
    counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    try:
        X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
            X, y, paths, test_size=args.test_size, random_state=args.seed,
            stratify=y
        )
    except ValueError as exc:
        raise SystemExit(f"Cannot create stratified split: {exc}")

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", SVC(C=3.0, kernel="rbf", gamma="scale",
                           class_weight="balanced", probability=True,
                           random_state=args.seed)),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    report = classification_report(y_test, pred, labels=["BLUE", "GREEN", "RED", "YELLOW"],
                                   output_dict=True, zero_division=0)
    metrics = {
        "model": "Niryo circular-HSV/RGB statistical features + RBF-SVM",
        "feature_version": FEATURE_VERSION,
        "samples_total": int(len(y)),
        "class_counts": counts,
        "split": {"method": "stratified image-level random split; not session-independent",
                  "seed": args.seed, "train": int(len(y_train)), "test": int(len(y_test))},
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix_labels": ["BLUE", "GREEN", "RED", "YELLOW"],
        "confusion_matrix": confusion_matrix(y_test, pred,
            labels=["BLUE", "GREEN", "RED", "YELLOW"]).tolist(),
        "classification_report": report,
        "test_paths": p_test,
        "warning": "Small or correlated image-level test sets do not establish field generalization."
    }
    model_path, report_path = Path(args.model_out), Path(args.report_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"test_paths", "classification_report"}}, indent=2))
    print("Classification report:")
    print(classification_report(y_test, pred, labels=["BLUE", "GREEN", "RED", "YELLOW"], zero_division=0))
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {report_path}")


if __name__ == "__main__":
    main()
