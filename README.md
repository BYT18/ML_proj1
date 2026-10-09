# ML Project 1 — Predicting coronary heart disease (MICHD) from BRFSS data

Binary classification of MICHD from 321 survey features (328,135 training
respondents, 8.8% positive). Only Python's standard library and NumPy are used.

**Final model:** one-hot / quantile-binned features + L2-regularized logistic
regression (λ = 1e-4, Newton's method) + F1-optimal decision threshold.
5-fold CV: **F1 = 0.440 ± 0.008**, accuracy 0.880, AUC 0.865.

## Reproducing the submission

```bash
mkdir data   # put x_train.csv, y_train.csv, x_test.csv here (names unchanged)
python run.py   # ~30 s after the first load; writes submission.csv
```

The first run parses the CSVs (~40 s) and caches them in `data/cache.npz`.
The pipeline is deterministic, so `run.py` reproduces the submitted file exactly.

## Files

| File | Content |
|---|---|
| `implementations.py` | The six required methods (GD/SGD/least squares/ridge/logistic/regularized logistic). |
| `helpers.py` | Course-provided CSV loading and submission writing. |
| `preprocessing.py` | Data loading/caching and the `FeatureBuilder` feature pipeline. |
| `model.py` | Newton solver for regularized logistic regression, metrics (F1, accuracy, AUC), F1 threshold search, k-fold cross-validation. |
| `experiments.py` | Baselines, ablation study and λ sweep (`python experiments.py ablation` / `lambda`); results in `results/`. |
| `run.py` | Trains the final model on all training data and writes `submission.csv`. |

## Pipeline

1. **Column filtering.** 18 survey bookkeeping columns (interview date,
   sequence numbers, sampling weights, phone screening) are dropped. Constant
   and fully-missing columns are dropped.
2. **Categorical columns** (≤ 30 distinct values): one-hot encoding of every
   category seen in ≥ 0.1% of training rows, plus a "missing" indicator. BRFSS
   skip patterns make missingness informative, so it is kept as a category.
3. **Numeric columns** (> 30 distinct values):
   - BRFSS special codes in raw questions are cleaned: "none" (88/888) → 0,
     "don't know"/"refused" (7/9/77/99/777/999/7777/9999) → missing, applied
     only when the code lies above every regular answer of that question;
   - missing values → training median, plus a missing indicator;
   - clipping to the [0.5, 99.5] percentiles, then standardization;
   - 10 quantile-bin indicators so the linear model can fit non-linear effects.
   This gives 1,733 features + bias.
4. **Model.** Logistic regression with penalty λ‖w‖² (bias unpenalized),
   minimized with Newton's method and a backtracking line search (converges in
   ~8 iterations, ~10 s on the full data).
5. **Decision threshold.** The leaderboard metric is F1 and the classes are
   imbalanced, so we predict MICHD when the predicted probability exceeds the
   threshold that maximizes F1 on the training data (p ≈ 0.21) instead of 0.5.

All statistics (categories, medians, bins, means, thresholds) are fitted on
training data only. In cross-validation they are re-fitted inside each fold.

## Cross-validation results (5 folds, same folds for every row)

| Configuration | F1 | Accuracy | AUC |
|---|---|---|---|
| Ridge regression, raw features | 0.429 | 0.875 | 0.858 |
| Reg. logistic regression, GD (300 it., γ=0.5), raw | 0.416 | 0.864 | 0.850 |
| Reg. logistic regression, Newton, raw | 0.436 | 0.877 | 0.864 |
| + drop admin columns | 0.436 | 0.880 | 0.864 |
| + one-hot categoricals | 0.439 | 0.876 | 0.866 |
| + quantile bins (**final**) | **0.440** | 0.880 | 0.865 |
| Final features, ridge regression | 0.432 | 0.876 | 0.858 |
| Final model with threshold 0.5 instead of tuned | 0.267 | – | – |

"Raw" means median-imputed, special-code-cleaned, standardized columns. The
λ sweep (1e-6 … 1e-2) is in `results/lambda.csv`: F1 is flat for
λ ∈ [1e-5, 1e-4] and drops at λ = 1e-2 (0.422). With λ = 0 the Hessian is
singular because the one-hot blocks are collinear.
