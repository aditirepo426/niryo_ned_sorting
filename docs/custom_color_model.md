# Niryo task-specific colour model (development branch)

This branch adds a compact supervised colour model designed for the fixed-camera, small-block conveyor setup. It does not use K-Means or a neural network. It combines robust RGB/HSV summary statistics, saturation-weighted hue histograms, and circular hue encoding, followed by a standardized RBF-SVM classifier.

## Train

Arrange labelled real-camera images into class folders:

```
dataset/real_world/BLUE/
dataset/real_world/GREEN/
dataset/real_world/RED/
dataset/real_world/YELLOW/
```

Then run:

```bash
python -m src.training.train_niryo_color_model --data dataset/real_world
```

The script saves `models/color/niryo_color_svm.joblib` and `models/color/niryo_color_metrics.json`.

## Inference

```python
from src.classification.niryo_color_model import NiryoColorModel
model = NiryoColorModel("models/color/niryo_color_svm.joblib", reject_below=0.60)
result = model.predict(frame_bgr, object_mask)  # mask must match frame dimensions
print(result["label"], result["confidence"])
```

The confidence threshold is a configurable abstention rule, not a calibrated probability guarantee. Tune it on a validation set and report both accepted-sample accuracy and rejection rate.

## Important evaluation note

The trainer's built-in split is stratified by image, not by capture session. It is suitable for a first development check only. For a credible paper result, collect multiple sessions containing every colour, keep a complete session untouched for final testing, and ensure near-duplicate frames remain in the same partition. Report accuracy, macro-F1, per-class metrics, confusion matrix, rejection rate, and inference latency. Do not report metrics until the code is run on the final dataset.
