# Real-world ML rebuild: Phase 1

This phase audits the labelled webcam images before training. It does not use the Kaggle RGB dataset and does not change the source images.

## 1. Dataset contract

Four initial colour classes:

- `Blue/`
- `Green/`
- `Red/`
- `Yellow/`

The current archive has 159 images (39 Blue, 39 Green, 41 Red, 40 Yellow). The current physical objects are cuboids; multi-shape recognition is a later phase after new shapes are introduced and labelled.

## 2. Run the audit

Copy `real_world_yellow_corrected.zip` into the repository workspace, then run:

```bash
python -m pip install opencv-python numpy
python -m src.dataset.audit_real_world_dataset --input real_world_yellow_corrected.zip
```

Or extract it into `dataset/real_world/` and run:

```bash
python -m src.dataset.audit_real_world_dataset --input dataset/real_world
```

The script writes:

- `results/dataset_audit/dataset_audit.json`: class counts, readability, image-size range and duplicate checks.
- `results/dataset_audit/image_manifest.csv`: one row per image with label, dimensions and SHA-256.

## 3. Do not train before reviewing the audit

The image names suggest that some images may have been captured within the same short burst. An image-level random split could put nearly identical frames in training and test sets, inflating test scores. The audit detects byte-identical files; it cannot by itself detect near-duplicates or capture-session leakage. Review the contact sheet and group images from the same capture burst/session before final splitting.

## 4. Training protocol to follow

1. Freeze the labels and inspect mislabeled/blurred images.
2. Make a group-aware train/validation/test split (target approximately 70/15/15, adjusted if the number of independent capture groups is small).
3. Establish a simple OpenCV/HSV baseline.
4. Train a small PyTorch CNN from random initial weights using only the training partition. Keep augmentation limited to plausible camera changes.
5. Tune settings using validation only; evaluate the test set once after the model and threshold choices are frozen.
6. Report accuracy, macro precision/recall/F1, per-class metrics, confusion matrix, inference latency, and the number of independent capture groups.
7. Collect a new, independent test set under the actual conveyor station lighting before making claims about robot sorting reliability.

## 5. Important scope distinction

A folder-classification CNN predicts the colour of a cropped object. It is not a YOLO detector and does not locate an object in an arbitrary full camera frame. The current project uses an IR sensor to stop the object at a fixed station, so a fixed ROI plus contour segmentation may be a simpler first deployment. If localisation varies or multiple objects can appear, annotate bounding boxes and evaluate a detector separately.

No model accuracy is claimed by this phase; it only establishes dataset quality and a reproducible starting point.
