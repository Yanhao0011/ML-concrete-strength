"""Permutation importance of the 8 raw inputs, measured on the held-out folds of the grouped CV
(training split only, so the test set stays unseen).

Usage:  python src/importance.py --model hgb_tuned
Output: results/importance_<model>.json, results/importance_<model>.png
"""
import argparse
import json

import matplotlib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import MODEL_NAMES, RAW_FEATURES, ROOT, SEED, TARGET, get_split, load_data, make_model

LABELS = {"cement": "Cement", "slag": "Slag", "fly_ash": "Fly ash", "water": "Water",
          "superplasticizer": "Superplasticizer", "coarse_agg": "Coarse aggregate",
          "fine_agg": "Fine aggregate", "age_days": "Age"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb_tuned", choices=MODEL_NAMES)
    a = ap.parse_args()

    df = load_data()
    tr, _ = get_split(df)
    d = df.iloc[tr].reset_index(drop=True)
    X, y = d[RAW_FEATURES], d[TARGET]
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, d["mix_id"])
    imp = []
    for fit_idx, val_idx in folds:
        m = make_model(a.model).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        r = permutation_importance(m, X.iloc[val_idx], y.iloc[val_idx], n_repeats=20, random_state=SEED,
                                   scoring="neg_mean_absolute_error")
        imp.append(r.importances_mean)          # increase in MAE (MPa) when the input is shuffled
    imp = np.array(imp)
    res = {f: {"mean": float(imp[:, i].mean()), "std_over_folds": float(imp[:, i].std())}
           for i, f in enumerate(RAW_FEATURES)}
    (ROOT / "results" / f"importance_{a.model}.json").write_text(json.dumps(res, indent=2))

    order = np.argsort(imp.mean(0))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.barh([LABELS[RAW_FEATURES[i]] for i in order], imp.mean(0)[order], xerr=imp.std(0)[order],
            color="#2a78d6", ecolor="0.4", capsize=2)
    ax.set_xlabel("Increase in MAE when shuffled (MPa)")
    ax.set_title("Permutation importance (grouped CV)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"importance_{a.model}.png", dpi=200)
    for i in order[::-1]:
        print(f"{RAW_FEATURES[i]:>18}: {imp[:, i].mean():6.2f} +/- {imp[:, i].std():.2f} MPa")


if __name__ == "__main__":
    main()
