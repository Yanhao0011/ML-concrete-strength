"""Shared pieces: data loading, features, the fixed train/test split, and the model registry."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "concrete.csv"
SPLIT_JSON = ROOT / "data" / "split.json"
SEED = 42

MIX_COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"]
RAW_FEATURES = MIX_COLS + ["age_days"]
TARGET = "strength_mpa"


def load_data():
    """Deduplicated data with a mix id (same 7 ingredient amounts = same mix, any age)."""
    if not DATA_CSV.exists():
        raise SystemExit("data/concrete.csv not found - run: python data/download.py")
    df = pd.read_csv(DATA_CSV).drop_duplicates().reset_index(drop=True)
    df["mix_id"] = df.groupby(MIX_COLS, sort=True).ngroup()
    return df


def engineer(X):
    """Raw inputs + water/binder ratio + log(age): the two relations engineers use for strength."""
    X = pd.DataFrame(X, columns=RAW_FEATURES).copy()
    X["w_b_ratio"] = X["water"] / (X["cement"] + X["slag"] + X["fly_ash"])
    X["log_age"] = np.log(X["age_days"])
    return X


def get_split(df):
    """Fixed 80/20 split grouped by mix, so no mix appears in both train and test.
    Saved to data/split.json on first use; later runs reuse it."""
    if SPLIT_JSON.exists():
        s = json.loads(SPLIT_JSON.read_text())
        if s["n_rows"] == len(df):
            return np.array(s["train"]), np.array(s["test"])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["mix_id"]))
    SPLIT_JSON.write_text(json.dumps({"seed": SEED, "grouped_by": "mix_id", "n_rows": len(df),
                                      "train": tr.tolist(), "test": te.tolist()}))
    return tr, te


def tuned_params(base):
    p = ROOT / "results" / f"best_params_{base}.json"
    if not p.exists():
        raise SystemExit(f"{p.name} not found - run: python src/tune.py --model {base}")
    return json.loads(p.read_text())["best_params"]


def make_model(name):
    if name.endswith("_tuned"):
        return make_model(name[:-6]).set_params(**tuned_params(name[:-6]))
    fe = FunctionTransformer(engineer)
    models = {
        # baselines
        "linear_raw": make_pipeline(StandardScaler(), LinearRegression()),
        "linear_fe": make_pipeline(fe, StandardScaler(), LinearRegression()),
        # stronger models (next step of the plan)
        "rf": make_pipeline(fe, RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)),
        "hgb": make_pipeline(fe, HistGradientBoostingRegressor(random_state=SEED)),
    }
    if name not in models:
        raise SystemExit(f"unknown model '{name}', choose from {list(models)}")
    return models[name]


MODEL_NAMES = ["linear_raw", "linear_fe", "rf", "hgb", "rf_tuned", "hgb_tuned"]
