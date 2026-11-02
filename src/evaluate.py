"""Score a trained model on the held-out test split (run once, for the final model choice).

Usage:  python src/evaluate.py --model hgb
        python src/evaluate.py --model linear_fe --on train   # pipeline sanity check, test stays unseen
Output: results/<split>_<model>.json
"""
import argparse
import json

import joblib
import pandas as pd

from common import MODEL_NAMES, RAW_FEATURES, ROOT, TARGET, get_split, load_data
from train import scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--on", default="test", choices=["test", "train"])
    a = ap.parse_args()

    df = load_data()
    tr, te = get_split(df)
    d = df.iloc[te if a.on == "test" else tr]
    model = joblib.load(ROOT / "models" / f"{a.model}.joblib")
    y, p = d[TARGET].to_numpy(), model.predict(d[RAW_FEATURES])
    s = scores(y, p)
    pd.DataFrame({"mix_id": d["mix_id"], "age_days": d["age_days"], "measured": y, "predicted": p}).to_csv(
        ROOT / "results" / f"{a.on}_predictions_{a.model}.csv", index=False)
    s = {k: float(v) for k, v in s.items()}
    (ROOT / "results" / f"{a.on}_{a.model}.json").write_text(json.dumps({"model": a.model, "split": a.on,
                                                                           "n": len(d), **s}, indent=2))
    print(f"{a.model} on {a.on} (n={len(d)}): MAE {s['MAE']:.2f}  RMSE {s['RMSE']:.2f}  R2 {s['R2']:.3f}")


if __name__ == "__main__":
    main()
