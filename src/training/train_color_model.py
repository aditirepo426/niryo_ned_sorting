"""Train K-Means colour model and save scaler/model/cluster mapping."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["r_n","g_n","b_n","h_n","s_n","v_n"]

def main(csv_path, model_dir, clusters=4):
    df = pd.read_csv(csv_path).dropna(subset=FEATURES + ["label"])
    X = df[FEATURES].to_numpy(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=clusters, random_state=42, n_init=20)
    ids = km.fit_predict(Xs)

    mapping = {}
    for cid in range(clusters):
        labels = df.loc[ids == cid, "label"]
        if labels.empty:
            mapping[str(cid)] = "UNKNOWN"
        else:
            mapping[str(cid)] = labels.value_counts().idxmax()

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "color_scaler.pkl")
    joblib.dump(km, model_dir / "kmeans_color.pkl")
    (model_dir / "color_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    df["cluster"] = ids
    print("Samples:", len(df))
    print("Cluster mapping:", mapping)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="dataset/processed/rgb_training.csv")
    p.add_argument("--model-dir", default="models/color")
    p.add_argument("--clusters", type=int, default=4)
    a = p.parse_args()
    main(Path(a.csv), Path(a.model_dir), a.clusters)
