"""Predict compressive strength for new mixes.

Usage:  python src/predict.py --model linear_fe --input examples/mixes.csv [--output predictions.csv]
The input CSV needs the columns: cement, slag, fly_ash, water, superplasticizer, coarse_agg,
fine_agg (all kg/m^3) and age_days.
"""
import argparse

import joblib
import pandas as pd

from common import MODEL_NAMES, RAW_FEATURES, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="linear_fe", choices=MODEL_NAMES)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    a = ap.parse_args()

    X = pd.read_csv(a.input)
    missing = [c for c in RAW_FEATURES if c not in X.columns]
    if missing:
        raise SystemExit(f"input is missing columns: {missing}")
    model = joblib.load(ROOT / "models" / f"{a.model}.joblib")
    X["predicted_strength_mpa"] = model.predict(X[RAW_FEATURES]).round(1)
    print(X.to_string(index=False))
    if a.output:
        X.to_csv(a.output, index=False)


if __name__ == "__main__":
    main()
