import json
from pathlib import Path
import joblib
import numpy as np
from src.vision.color_features import extract_color_features

class ColorClassifier:
    def __init__(self, model_dir="models/color"):
        d = Path(model_dir)
        self.scaler = joblib.load(d / "color_scaler.pkl")
        self.model = joblib.load(d / "kmeans_color.pkl")
        self.mapping = json.loads((d / "color_mapping.json").read_text())

    def predict(self, frame_bgr, mask):
        x = extract_color_features(frame_bgr, mask).reshape(1, -1)
        cluster = int(self.model.predict(self.scaler.transform(x))[0])
        return self.mapping.get(str(cluster), "UNKNOWN"), cluster
