"""Two-panel figure for the report: (a) test predictions of the baseline and the selected model,
(b) permutation importance. Also prints the test error by strength range and by age.

Usage:  python src/make_figures.py   (after evaluate.py and importance.py)
Output: results/fig_report.png
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ROOT
from importance import LABELS

FINAL, BASE = "hgb_tuned", "linear_fe"
R = ROOT / "results"


def main():
    f = pd.read_csv(R / f"test_predictions_{FINAL}.csv")
    b = pd.read_csv(R / f"test_predictions_{BASE}.csv")
    mf, mb = json.loads((R / f"test_{FINAL}.json").read_text()), json.loads((R / f"test_{BASE}.json").read_text())
    imp = json.loads((R / f"importance_{FINAL}.json").read_text())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
    lim = [0, 85]
    ax.plot(lim, lim, color="0.3", lw=1)
    ax.scatter(b.measured, b.predicted, s=10, color="#eb6834", alpha=0.6,
               label=f"Linear + features (MAE {mb['MAE']:.2f})")
    ax.scatter(f.measured, f.predicted, s=10, color="#2a78d6", alpha=0.7,
               label=f"Gradient boosting (MAE {mf['MAE']:.2f})")
    ax.set(xlim=lim, ylim=lim, xlabel="Measured strength (MPa)", ylabel="Predicted strength (MPa)")
    ax.set_title("(a) Held-out test mixes (n = 195)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", frameon=False)

    names = sorted(imp, key=lambda k: imp[k]["mean"])
    bx.barh([LABELS[k] for k in names], [imp[k]["mean"] for k in names],
            xerr=[imp[k]["std_over_folds"] for k in names], color="#2a78d6", ecolor="0.4", capsize=2)
    bx.set_xlabel("Increase in MAE when shuffled (MPa)")
    bx.set_title("(b) Permutation importance", fontsize=10)
    for a_ in (ax, bx):
        a_.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(R / "fig_report.png", dpi=300)

    f["abs_err"] = (f.predicted - f.measured).abs()
    print("test MAE of", FINAL, "by measured strength:")
    print(f.groupby(pd.cut(f.measured, [0, 20, 40, 60, 90]), observed=True).abs_err.agg(["count", "mean"]).round(2))
    print("by age:")
    print(f.groupby(pd.cut(f.age_days, [0, 7, 28, 365]), observed=True).abs_err.agg(["count", "mean"]).round(2))


if __name__ == "__main__":
    main()
