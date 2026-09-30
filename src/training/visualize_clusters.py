import argparse
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

FEATURES = ["r_n","g_n","b_n","h_n","s_n","v_n"]

def main(csv_path, model_dir, output):
    df = pd.read_csv(csv_path).dropna(subset=FEATURES)
    scaler = joblib.load(Path(model_dir) / "color_scaler.pkl")
    km = joblib.load(Path(model_dir) / "kmeans_color.pkl")
    X = scaler.transform(df[FEATURES])
    xy = PCA(n_components=2, random_state=42).fit_transform(X)
    clusters = km.predict(X)
    plt.figure(figsize=(8,6))
    plt.scatter(xy[:,0], xy[:,1], c=clusters, s=5, alpha=0.4)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("K-Means colour clusters")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model-dir", default="models/color")
    p.add_argument("--output", default="results/color/clusters.png")
    a = p.parse_args()
    main(a.csv, a.model_dir, a.output)
