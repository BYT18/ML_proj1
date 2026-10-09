"""Cross-validation experiments: baselines, ablation study and lambda sweep.

Usage:
    python experiments.py ablation   # baselines + one change at a time
    python experiments.py lambda     # regularization sweep for the final model

Results are printed and appended to results/<experiment>.csv.
"""

import os
import sys
import time

import numpy as np

from implementations import ridge_regression, reg_logistic_regression
from model import best_threshold, cross_validate, f1_score, logistic_newton
from preprocessing import FeatureBuilder, add_bias, load_data

K_FOLDS = 5
LAMBDA = 1e-4


def features(**opts):
    """Return a make_features callable for cross_validate with given options."""

    def make(x_tr, x_va, names):
        fb = FeatureBuilder(**opts).fit(x_tr, names)
        return add_bias(fb.transform(x_tr)), add_bias(fb.transform(x_va))

    return make


def ridge(lambda_):
    """Ridge regression on labels in {-1, 1}; the score is the linear output."""

    def fit_predict(y, tx_tr, tx_va):
        tx64 = tx_tr.astype(np.float64)
        w, _ = ridge_regression(2 * y - 1, tx64, lambda_)
        return tx64 @ w, tx_va.astype(np.float64) @ w

    return fit_predict


def logistic_gd(lambda_, max_iters=300, gamma=0.5):
    """Regularized logistic regression with plain gradient descent (Step 2 code)."""

    def fit_predict(y, tx_tr, tx_va):
        tx64 = tx_tr.astype(np.float64)
        w0 = np.zeros(tx64.shape[1])
        w, _ = reg_logistic_regression(y, tx64, lambda_, w0, max_iters, gamma)
        return tx64 @ w, tx_va.astype(np.float64) @ w

    return fit_predict


def logistic(lambda_):
    """Regularized logistic regression solved by Newton's method."""

    def fit_predict(y, tx_tr, tx_va):
        w = logistic_newton(y, tx_tr, lambda_)
        return tx_tr @ w.astype(tx_tr.dtype), tx_va @ w.astype(tx_va.dtype)

    return fit_predict


def report(name, res, path, t):
    """Print one result line and append it to a CSV file."""
    line = (
        f"{name},{res['f1']:.4f},{res['f1_std']:.4f},{res['acc']:.4f},{res['auc']:.4f}"
    )
    print(f"{line}  ({t:.0f}s)", flush=True)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("config,f1,f1_std,accuracy,auc\n")
        f.write(line + "\n")


def run(configs, path):
    """Cross-validate each (name, make_features, fit_predict) configuration."""
    x, _, y, _, _, names = load_data()
    y01 = (y == 1).astype(np.float64)
    out = {}
    for name, make, fit in configs:
        t = time.time()
        res = cross_validate(x, y01, names, make, fit, k=K_FOLDS)
        report(name, res, path, time.time() - t)
        out[name] = res
    return out, y01


def ablation():
    """Baselines on raw features, then add one modification at a time."""
    raw = dict(one_hot=False, bins=0, drop_admin=False)
    configs = [
        ("ridge | raw features", features(**raw), ridge(1e-4)),
        ("logistic GD | raw features", features(**raw), logistic_gd(1e-4)),
        ("logistic Newton | raw features", features(**raw), logistic(LAMBDA)),
        (
            "+ drop admin columns",
            features(one_hot=False, bins=0, drop_admin=True),
            logistic(LAMBDA),
        ),
        (
            "+ one-hot categoricals",
            features(one_hot=True, bins=0, drop_admin=True),
            logistic(LAMBDA),
        ),
        (
            "+ quantile bins (full model)",
            features(one_hot=True, bins=10, drop_admin=True),
            logistic(LAMBDA),
        ),
        (
            "full model, keep admin columns",
            features(one_hot=True, bins=10, drop_admin=False),
            logistic(LAMBDA),
        ),
        (
            "ridge | full features",
            features(one_hot=True, bins=10, drop_admin=True),
            ridge(1e-4),
        ),
    ]
    out, y01 = run(configs, "results/ablation.csv")

    # Effect of tuning the decision threshold vs the default 0.5 cut-off,
    # evaluated on the out-of-fold probabilities of the full model.
    oof = out["+ quantile bins (full model)"]["oof"]
    f1_default = f1_score(y01, (oof > 0).astype(int))  # score 0 <=> p = 0.5
    thr, f1_tuned = best_threshold(y01, oof)
    p_thr = 1 / (1 + np.exp(-thr))
    print(
        f"threshold p=0.5: F1={f1_default:.4f} | tuned p={p_thr:.3f}: F1={f1_tuned:.4f}"
    )
    with open("results/threshold.csv", "w") as f:
        f.write("threshold,f1\n")
        f.write(f"0.5,{f1_default:.4f}\n{p_thr:.4f},{f1_tuned:.4f}\n")


def lambda_sweep():
    """Regularization strength of the final model."""
    make = features(one_hot=True, bins=10, drop_admin=True)
    configs = [
        (f"lambda={lam:g}", make, logistic(lam))
        for lam in [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]
    ]
    run(configs, "results/lambda.csv")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    {"ablation": ablation, "lambda": lambda_sweep}[sys.argv[1]]()
