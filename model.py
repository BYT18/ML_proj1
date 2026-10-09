"""Training, evaluation and cross-validation utilities.

The main model is L2-regularized logistic regression (same objective as
``reg_logistic_regression`` in implementations.py) solved with Newton's method,
which converges in a handful of iterations on ~2000 one-hot features where
plain gradient descent would need thousands of steps.
"""

import numpy as np

from implementations import sigmoid


def logistic_newton(y, tx, lambda_, max_iters=30, tol=1e-8, sample_weight=None):
    """L2-regularized logistic regression solved by Newton's method.

    Minimizes  1/N sum_i s_i [log(1 + exp(x_i^T w)) - y_i x_i^T w] + lambda_ ||w||^2
    where s_i are optional per-sample weights. The bias (column 0) is not
    penalized. A backtracking line search guarantees monotone decrease.

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), feature matrix whose first column is the bias.
        lambda_: L2 regularization strength.
        max_iters: maximum number of Newton steps.
        tol: stop when the Newton decrement falls below this value.
        sample_weight: optional shape (N,) weights.

    Returns:
        w: shape (D,), fitted weights.
    """
    n, d = tx.shape
    s = np.ones(n) if sample_weight is None else sample_weight
    pen = np.full(d, 2 * lambda_)
    pen[0] = 0.0

    def objective(w):
        z = matvec(tx, w)
        return np.sum(s * (np.logaddexp(0, z) - y * z)) / n + 0.5 * np.sum(pen * w**2)

    w = np.zeros(d)
    f = objective(w)
    for _ in range(max_iters):
        p = sigmoid(matvec(tx, w))
        grad = matvec(tx.T, s * (p - y)) / n + pen * w
        hess = hessian(tx, s * p * (1 - p) / n) + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        decrement = np.sum(grad * step)
        if decrement / 2 < tol:
            break
        t = 1.0
        while True:  # backtracking line search
            w_new = w - t * step
            f_new = objective(w_new)
            if f_new <= f - 0.25 * t * decrement or t < 1e-6:
                break
            t *= 0.5
        w, f = w_new, f_new
    return w


def matvec(a, v):
    """a @ v computed in a's dtype (avoids upcasting a large float32 matrix).

    numpy 1.26 with Apple Accelerate emits spurious floating-point warnings on
    float32 matmuls, so they are silenced here.
    """
    with np.errstate(all="ignore"):
        return (a @ v.astype(a.dtype)).astype(np.float64)


def hessian(tx, d, chunk=40000):
    """Compute tx^T diag(d) tx in row chunks (float32 matmul, float64 sum)."""
    out = np.zeros((tx.shape[1], tx.shape[1]))
    with np.errstate(all="ignore"):
        for i in range(0, tx.shape[0], chunk):
            xc = tx[i : i + chunk]
            out += (xc * d[i : i + chunk, None].astype(xc.dtype)).T @ xc
    return out


def f1_score(y_true, y_pred):
    """F1 score for binary labels in {0, 1}."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return np.mean(y_true == y_pred)


def best_threshold(y_true, scores):
    """Threshold on the scores that maximizes F1 (exact search over sorted scores).

    Returns:
        (threshold, best F1)
    """
    order = np.argsort(-scores)
    s, yt = scores[order], y_true[order]
    tp = np.cumsum(yt)
    k = np.arange(1, len(yt) + 1)  # number predicted positive
    f1 = 2 * tp / (k + yt.sum())
    # Only cut between distinct score values.
    valid = np.r_[s[1:] != s[:-1], True]
    f1 = np.where(valid, f1, -1)
    i = np.argmax(f1)
    thr = (s[i] + s[i + 1]) / 2 if i + 1 < len(s) else s[i] - 1e-12
    return thr, f1[i]


def auc(y_true, scores):
    """Area under the ROC curve via the rank-sum (Mann-Whitney) statistic."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def kfold_indices(n, k, seed=1):
    """Random partition of range(n) into k folds."""
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k)


def cross_validate(x_raw, y01, names, make_features, fit_predict, k=5, seed=1):
    """K-fold CV returning out-of-fold scores and per-fold metrics.

    The feature pipeline is re-fitted on each training split so that no
    statistic of the validation fold leaks into training. The F1 threshold is
    chosen on the training split's own scores and applied to the held-out fold.

    Args:
        x_raw: raw feature matrix.
        y01: labels in {0, 1}.
        names: raw column names.
        make_features: callable(x_tr, x_va, names) -> (tx_tr, tx_va).
        fit_predict: callable(y_tr, tx_tr, tx_va) -> (scores_tr, scores_va).

    Returns:
        dict with mean/std of F1, accuracy and AUC, and the out-of-fold scores.
    """
    folds = kfold_indices(len(y01), k, seed)
    oof = np.zeros(len(y01))
    f1s, accs, aucs = [], [], []
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        tx_tr, tx_va = make_features(x_raw[tr], x_raw[va], names)
        s_tr, s_va = fit_predict(y01[tr], tx_tr, tx_va)
        thr, _ = best_threshold(y01[tr], s_tr)
        pred = (s_va > thr).astype(int)
        f1s.append(f1_score(y01[va], pred))
        accs.append(accuracy(y01[va], pred))
        aucs.append(auc(y01[va], s_va))
        oof[va] = s_va
    return {
        "f1": np.mean(f1s),
        "f1_std": np.std(f1s),
        "acc": np.mean(accs),
        "auc": np.mean(aucs),
        "oof": oof,
    }
