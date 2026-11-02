"""Grid search with 5-fold CV grouped by mix, on the training split only.

Usage:  python src/tune.py --model hgb      (or rf)
Output: results/best_params_<model>.json  -> then train with: python src/train.py --model <model>_tuned
"""
import argparse
import json

from sklearn.model_selection import GridSearchCV, GroupKFold

from common import RAW_FEATURES, ROOT, SEED, TARGET, get_split, load_data, make_model

GRIDS = {
    "hgb": {"histgradientboostingregressor__learning_rate": [0.03, 0.1],
            "histgradientboostingregressor__max_iter": [300, 1000],
            "histgradientboostingregressor__max_leaf_nodes": [15, 31],
            "histgradientboostingregressor__min_samples_leaf": [5, 20],
            "histgradientboostingregressor__l2_regularization": [0.0, 1.0]},
    "rf": {"randomforestregressor__max_features": [0.33, 0.5, 1.0],
           "randomforestregressor__min_samples_leaf": [1, 2, 4]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(GRIDS))
    a = ap.parse_args()

    df = load_data()
    tr, _ = get_split(df)
    d = df.iloc[tr].reset_index(drop=True)
    cv = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(d, groups=d["mix_id"]))
    gs = GridSearchCV(make_model(a.model), GRIDS[a.model], cv=cv, n_jobs=-1,
                      scoring="neg_mean_absolute_error").fit(d[RAW_FEATURES], d[TARGET])
    out = {"model": a.model, "cv": "GroupKFold(5) by mix, training split", "n_candidates": len(gs.cv_results_["params"]),
           "best_cv_MAE": -gs.best_score_, "best_params": gs.best_params_}
    (ROOT / "results" / f"best_params_{a.model}.json").write_text(json.dumps(out, indent=2))
    print(f"{a.model}: best grouped-CV MAE {-gs.best_score_:.2f} from {out['n_candidates']} candidates")
    for k, v in gs.best_params_.items():
        print(f"   {k.split('__')[1]} = {v}")


if __name__ == "__main__":
    main()
