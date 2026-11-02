# Predicting concrete compressive strength from the mix design

CV7001 Machine Learning Mini-Project — Yanhao Ning (NTU).

## Problem

Concrete strength is normally checked by casting and crushing test cubes after 7, 28 or more days.
A model that predicts strength from the mix proportions and age can screen candidate mix designs
before any specimen is cast.

- **Inputs (8):** cement, blast-furnace slag, fly ash, water, superplasticizer, coarse aggregate and
  fine aggregate (all kg/m³ of concrete), and age (days).
- **Output:** compressive strength (MPa). This is a regression task.

## Data

UCI Machine Learning Repository, *Concrete Compressive Strength* (Yeh, 1998), licence CC BY 4.0.
It has 1030 laboratory tests; 25 rows are exact duplicates and are removed, leaving 1005 tests of
428 distinct mixes (the same mix is often tested at several ages).

```bash
python data/download.py
```

The script downloads the official archive, checks the SHA-256 of `Concrete_Data.xls`, and writes
`data/concrete.csv` with readable column names.

## Evaluation design

- **Test split:** 20% of the *mixes* (195 tests), chosen once with seed 42 and stored in
  `data/split.json`. Splitting by mix means that no mix appears in both training and test sets, so
  the score measures performance on genuinely new mixes. The test split is used only once, for the
  final model.
- **Model selection:** 5-fold cross-validation on the training split, also grouped by mix
  (`GroupKFold`, seed 42). For reference, `train.py` also reports ordinary random K-fold.
- **Metrics:** MAE, RMSE (MPa) and R².

## Models

| Name | Description |
|---|---|
| `linear_raw` | Linear regression on the 8 standardised inputs (simplest baseline) |
| `linear_fe` | Linear regression plus two engineering features: water/binder ratio and log(age) |
| `rf` | Random forest, 500 trees, same features as `linear_fe` |
| `hgb` | Histogram gradient boosting, same features as `linear_fe` |

All models are scikit-learn pipelines defined in `src/common.py`.

## How to run

```bash
pip install -r requirements.txt
python data/download.py

# baselines and default tree models: 5-fold CV grouped by mix on the training split
python src/train.py --model linear_raw
python src/train.py --model linear_fe
python src/train.py --model rf
python src/train.py --model hgb

# grid search (grouped CV, training split only), then retrain with the best settings
python src/tune.py --model hgb
python src/tune.py --model rf
python src/train.py --model hgb_tuned
python src/train.py --model rf_tuned

# permutation importance on the CV validation folds (test set still unseen)
python src/importance.py --model hgb_tuned

# final, one-off scoring on the held-out test mixes, then the report figure
python src/evaluate.py --model hgb_tuned
python src/evaluate.py --model linear_fe    # baseline, for the comparison in the figure
python src/make_figures.py

# use the selected model on new mixes
python src/predict.py --model hgb_tuned --input examples/mixes.csv
```

`python src/evaluate.py --model <name> --on train` scores the training split instead; use it to
check the pipeline without touching the test set.

## Results

**Model selection rule (fixed before the test set was opened):** lowest mean MAE in the grouped
5-fold CV. This selected `hgb_tuned`. All models were then scored once on the test split for reference.

| Model | Grouped CV MAE (MPa) | Grouped CV R² | Random CV R² (leaky) | Test MAE (MPa) | Test RMSE (MPa) | Test R² |
|---|---|---|---|---|---|---|
| `linear_raw` | 8.39 ± 0.72 | 0.574 | 0.569 | 7.52 | 9.58 | 0.648 |
| `linear_fe` | 5.62 ± 0.40 | 0.803 | 0.812 | 5.32 | 6.57 | 0.835 |
| `rf` | 4.15 ± 0.30 | 0.865 | 0.898 | 3.55 | 5.01 | 0.904 |
| `hgb` | 3.99 ± 0.40 | 0.873 | 0.916 | 3.55 | 4.94 | 0.907 |
| `rf_tuned` | 4.06 ± 0.38 | 0.874 | 0.904 | 3.44 | 4.81 | 0.911 |
| **`hgb_tuned` (selected)** | **3.87 ± 0.47** | **0.875** | 0.923 | **3.46** | **5.02** | **0.904** |

Main observations:
- The two engineering features (water/binder ratio, log age) cut the linear-model error by a third.
- Tree models reduce the test MAE further to about 3.5 MPa. The four tree variants differ by less
  than the fold-to-fold scatter, so tuning gave only a small gain.
- For tree models, ordinary random K-fold overstates R² (0.90–0.92 vs 0.87 grouped), because tests of
  the same mix at other ages leak into training. Grouping by mix avoids this.
- Permutation importance ranks cement and age highest, then slag, water and fly ash; the aggregates
  matter little, which agrees with concrete technology.
- Test error grows with strength: MAE 2.6–2.9 MPa below 40 MPa, 4.3 MPa at 40–60 MPa and 6.5 MPa above
  60 MPa (only 13 test samples there).

Outputs: `results/cv_*.json`, `results/test_*.json`, `results/best_params_*.json`,
`results/importance_hgb_tuned.json`, `results/fig_report.png`.

## Reproducibility

- Python 3.13.15, Windows 11, CPU only; package versions are pinned in `requirements.txt`.
- Seed 42 for the split, the CV folds, the tree models and the permutations; the split itself is saved in `data/split.json`.
- The trained model files (`models/*.joblib`) are not committed. `train.py` recreates them in seconds.

## References

- Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural
  networks. *Cement and Concrete Research*, 28(12), 1797–1808.
- Yeh, I.-C. (2007). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5PK67
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
