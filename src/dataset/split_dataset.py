import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["label"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "rgb_training.csv", index=False)
    val.to_csv(output_dir / "rgb_validation.csv", index=False)
    test.to_csv(output_dir / "rgb_test.csv", index=False)
    print(f"train={len(train)}, validation={len(val)}, test={len(test)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="dataset/processed")
    a = p.parse_args()
    main(Path(a.input), Path(a.output_dir))
