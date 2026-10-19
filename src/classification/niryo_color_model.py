"""Compact, task-specific colour classifier for the Niryo conveyor.

The model is trained from labelled real-camera images; no pretrained weights,
K-Means, or neural network are used. Features combine circular hue encoding
with robust RGB/HSV statistics and a saturation-weighted hue histogram.
"""
from pathlib import Path
import joblib
import cv2
import numpy as np

FEATURE_VERSION = "niryo_hsv_rgb_circular_v1"
CLASS_NAMES = ("BLUE", "GREEN", "RED", "YELLOW")


def _select_pixels(frame_bgr, mask=None):
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame")
    h, w = frame_bgr.shape[:2]
    if mask is not None:
        if mask.shape[:2] != (h, w):
            raise ValueError("Mask dimensions must match frame")
        pixels = frame_bgr[mask > 0]
    else:
        # For offline folder training: central crop limits background influence.
        y0, y1 = int(0.20 * h), max(int(0.80 * h), int(0.20 * h) + 1)
        x0, x1 = int(0.20 * w), max(int(0.80 * w), int(0.20 * w) + 1)
        crop = frame_bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        candidate = (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 20) & (hsv[:, :, 2] <= 250)
        pixels = crop[candidate]
        if len(pixels) < max(20, int(crop.shape[0] * crop.shape[1] * 0.05)):
            pixels = crop.reshape(-1, 3)
    if len(pixels) < 10:
        raise ValueError("Too few foreground pixels; check object mask/ROI")
    return pixels.reshape(-1, 1, 3).astype(np.uint8)


def extract_features(frame_bgr, mask=None):
    """Return a fixed-length feature vector; input frame is OpenCV BGR."""
    px_bgr = _select_pixels(frame_bgr, mask)
    px_rgb = px_bgr[:, :, ::-1]
    px_hsv = cv2.cvtColor(px_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    rgb = px_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    hue = px_hsv[:, 0] * (2.0 * np.pi / 180.0)  # OpenCV hue is [0,179]
    sat = px_hsv[:, 1] / 255.0
    val = px_hsv[:, 2] / 255.0

    # Circular hue statistics prevent the red hue wrap-around discontinuity.
    weights = sat + 1e-4
    hue_sin = float(np.average(np.sin(hue), weights=weights))
    hue_cos = float(np.average(np.cos(hue), weights=weights))
    hue_concentration = float(np.hypot(hue_sin, hue_cos))

    def stats(a):
        return np.concatenate([
            np.mean(a, axis=0),
            np.median(a, axis=0),
            np.std(a, axis=0),
            np.percentile(a, 10, axis=0),
            np.percentile(a, 90, axis=0),
        ])

    # 12-bin hue histogram weighted by saturation suppresses grey/background pixels.
    hist, _ = np.histogram(px_hsv[:, 0], bins=12, range=(0, 180), weights=weights)
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return np.concatenate([
        stats(rgb), stats(np.column_stack([sat, val])),
        np.array([hue_sin, hue_cos, hue_concentration], dtype=np.float32),
        hist,
    ]).astype(np.float32)


class NiryoColorModel:
    """Inference wrapper for a fitted sklearn Pipeline saved by the trainer."""
    def __init__(self, model_path="models/color/niryo_color_svm.joblib",
                 reject_below=0.60):
        self.model_path = Path(model_path)
        self.reject_below = float(reject_below)
        self.pipeline = joblib.load(self.model_path)
        self.classes_ = tuple(self.pipeline.named_steps["classifier"].classes_)

    def predict(self, frame_bgr, mask=None):
        features = extract_features(frame_bgr, mask).reshape(1, -1)
        probabilities = self.pipeline.predict_proba(features)[0]
        idx = int(np.argmax(probabilities))
        confidence = float(probabilities[idx])
        label = str(self.classes_[idx]) if confidence >= self.reject_below else "UNKNOWN"
        return {"label": label, "raw_label": str(self.classes_[idx]),
                "confidence": confidence, "feature_version": FEATURE_VERSION}
