"""Evaluate the saved K-Means colour model on a labelled CSV."""
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = ["r_n","g_n","b_n","h_n","s_n","v_n"]

def main(csv_path, model_dir):
    df = pd.read_csv(csv_path).dropna(subset=FEATURES + ["label"])
    scaler = joblib.load(model_dir / "color_scaler.pkl")
    km = joblib.load(model_dir / "kmeans_color.pkl")
    mapping = json.loads((model_dir / "color_mapping.json").read_text())
    X = scaler.transform(df[FEATURES].to_numpy(float))
    clusters = km.predict(X)
    pred = [mapping.get(str(c), "UNKNOWN") for c in clusters]
    print(classification_report(df["label"], pred, labels=["RED","GREEN","BLUE","YELLOW"], zero_division=0))
    print("Accuracy:", accuracy_score(df["label"], pred))
    print("Confusion matrix:")
    print(confusion_matrix(df["label"], pred, labels=["RED","GREEN","BLUE","YELLOW"]))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model-dir", default="models/color")
    a = p.parse_args()
    main(Path(a.csv), Path(a.model_dir))
