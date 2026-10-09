"""Train the final model on the full training set and write submission.csv.

Pipeline (see README.md and preprocessing.py / model.py for details):
    1. drop survey bookkeeping columns, clean BRFSS special answer codes;
    2. one-hot encode categorical columns (incl. a "missing" indicator);
       standardize numeric columns and add 10 quantile-bin indicators;
    3. L2-regularized logistic regression (lambda = 1e-4) solved by Newton;
    4. predict MICHD when the score exceeds the F1-optimal threshold found on
       the training set.

The whole procedure is deterministic, so re-running it reproduces the
submitted predictions exactly.
"""

import numpy as np

from helpers import create_csv_submission
from model import best_threshold, logistic_newton
from preprocessing import FeatureBuilder, add_bias, load_data

LAMBDA = 1e-4


def main():
    x_train, x_test, y_train, _, test_ids, names = load_data("data")
    y01 = (y_train == 1).astype(np.float64)

    fb = FeatureBuilder(one_hot=True, bins=10, drop_admin=True).fit(x_train, names)
    tx_train = add_bias(fb.transform(x_train))
    tx_test = add_bias(fb.transform(x_test))
    print(f"design matrix: {tx_train.shape}")

    w = logistic_newton(y01, tx_train, LAMBDA)
    with np.errstate(all="ignore"):
        s_train = tx_train @ w.astype(tx_train.dtype)
        s_test = tx_test @ w.astype(tx_test.dtype)

    thr, f1_train = best_threshold(y01, s_train)
    print(f"threshold p={1 / (1 + np.exp(-thr)):.3f}, training F1={f1_train:.4f}")

    y_pred = np.where(s_test > thr, 1, -1)
    print(f"predicted positive rate on test: {np.mean(y_pred == 1):.3f}")
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print("wrote submission.csv")


if __name__ == "__main__":
    main()
