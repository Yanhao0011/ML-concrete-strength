"""Cross-validate a model on the training split, then fit it on the whole training split.

Usage:  python src/train.py --model linear_fe
Output: models/<model>.joblib, results/cv_<model>.json, results/cv_<model>.png
The test split is NOT touched here (see src/evaluate.py).
"""
import argparse
import json

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import MODEL_NAMES, RAW_FEATURES, ROOT, SEED, TARGET, get_split, load_data, make_model


def scores(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p)}


def per_fold(y, p, folds):
    s = [scores(y[f], p[f]) for f in folds]
    return {k: {"mean": float(np.mean([d[k] for d in s])), "std": float(np.std([d[k] for d in s]))} for k in s[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="linear_fe", choices=MODEL_NAMES)
    a = ap.parse_args()

    df = load_data()
    tr, _ = get_split(df)
    d = df.iloc[tr].reset_index(drop=True)
    X, y, g = d[RAW_FEATURES], d[TARGET].to_numpy(), d["mix_id"].to_numpy()

    # main CV: grouped by mix (honest); random KFold only to show the leakage effect
    gkf = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, g))
    rkf = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    p_g = cross_val_predict(make_model(a.model), X, y, cv=gkf)
    p_r = cross_val_predict(make_model(a.model), X, y, cv=rkf)
    out = {"model": a.model, "n_train": len(d), "n_train_mixes": int(d["mix_id"].nunique()),
           "cv_grouped_by_mix": per_fold(y, p_g, [f[1] for f in gkf]),
           "cv_random_kfold": per_fold(y, p_r, [f[1] for f in rkf])}

    model = make_model(a.model).fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / f"{a.model}.joblib")
    (ROOT / "results" / f"cv_{a.model}.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y, p_g, s=8, alpha=0.6, color="#2a78d6")
    lim = [0, max(y.max(), p_g.max()) * 1.05]
    ax.plot(lim, lim, color="0.3", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Measured strength (MPa)", ylabel="Predicted strength (MPa)",
           title=f"{a.model}: 5-fold CV grouped by mix")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"cv_{a.model}.png", dpi=200)

    m = out["cv_grouped_by_mix"]
    print(f"{a.model}: grouped CV  MAE {m['MAE']['mean']:.2f}+/-{m['MAE']['std']:.2f}  "
          f"RMSE {m['RMSE']['mean']:.2f}+/-{m['RMSE']['std']:.2f}  R2 {m['R2']['mean']:.3f}+/-{m['R2']['std']:.3f}")
    r = out["cv_random_kfold"]
    print(f"{' ' * len(a.model)}  random CV   MAE {r['MAE']['mean']:.2f}  RMSE {r['RMSE']['mean']:.2f}  "
          f"R2 {r['R2']['mean']:.3f}  (leaky reference)")


if __name__ == "__main__":
    main()
