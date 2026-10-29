"""Extensions of the final model that did NOT improve F1 (report, "Extensions").

Usage:
    python extensions.py cv      # 5-fold CV, paired per fold with the final model:
                                 # OOF threshold, max_cat=70, bin averaging, bagging
    python extensions.py split   # held-out 20% split: interaction features,
                                 # one-hidden-layer neural network, ensembles

The 'cv' experiment takes ~12 minutes, 'split' ~2 minutes.
"""

import sys
import time

import numpy as np

from model import auc, best_threshold, f1_score, kfold_indices, logistic_newton
from preprocessing import FeatureBuilder, add_bias, load_data

LAMBDA = 1e-4


# ---------------------------------------------------------------------------
# 5-fold CV experiments
# ---------------------------------------------------------------------------


def fit_scores(x_tr, y_tr, x_va, names, **opts):
    """Fit features + logistic regression on train; return (train, val) scores."""
    fb = FeatureBuilder(**opts).fit(x_tr, names)
    a, b = add_bias(fb.transform(x_tr)), add_bias(fb.transform(x_va))
    w = logistic_newton(y_tr, a, LAMBDA).astype(np.float32)
    return (a @ w).astype(float), (b @ w).astype(float)


def inner_oof(x_tr, y_tr, names, k=4):
    """Out-of-fold scores on the training split, used to choose the threshold."""
    oof = np.zeros(len(y_tr))
    inner = kfold_indices(len(y_tr), k, seed=2)
    for i in range(k):
        va = inner[i]
        tr = np.concatenate([inner[j] for j in range(k) if j != i])
        _, oof[va] = fit_scores(x_tr[tr], y_tr[tr], x_tr[va], names)
    return oof


def cv_experiments():
    """Compare cheap extensions with the final model on the same 5 folds."""
    x, _, yraw, _, _, names = load_data()
    y = (yraw == 1).astype(float)
    folds = kfold_indices(len(y), 5, seed=1)
    f1s, aucs = {}, {}

    def record(name, y_va, s_va, thr):
        f1s.setdefault(name, []).append(f1_score(y_va, (s_va > thr).astype(int)))
        aucs.setdefault(name, []).append(auc(y_va, s_va))

    for f in range(5):
        t0 = time.time()
        va = folds[f]
        tr = np.concatenate([folds[j] for j in range(5) if j != f])
        x_tr, y_tr, x_va, y_va = x[tr], y[tr], x[va], y[va]

        # Final model: threshold chosen on its own training scores.
        s_tr, s_va = fit_scores(x_tr, y_tr, x_va, names)
        record("current", y_va, s_va, best_threshold(y_tr, s_tr)[0])

        # Threshold chosen on inner out-of-fold scores instead.
        oof = inner_oof(x_tr, y_tr, names)
        record("OOF threshold", y_va, s_va, best_threshold(y_tr, oof)[0])

        # More columns treated as categorical.
        a_tr, a_va = fit_scores(x_tr, y_tr, x_va, names, max_cat=70)
        record("max_cat=70", y_va, a_va, best_threshold(y_tr, a_tr)[0])

        # Average of models with 5, 10 and 20 quantile bins.
        views_tr, views_va = [s_tr], [s_va]
        for bins in [5, 20]:
            b_tr, b_va = fit_scores(x_tr, y_tr, x_va, names, bins=bins)
            views_tr.append(b_tr)
            views_va.append(b_va)
        m_tr, m_va = np.mean(views_tr, 0), np.mean(views_va, 0)
        record("avg bins 5/10/20", y_va, m_va, best_threshold(y_tr, m_tr)[0])

        # Bagging: 5 models on bootstrap resamples, scores averaged.
        rng = np.random.default_rng(f)
        boot_tr, boot_va = [], []
        for _ in range(5):
            bi = rng.integers(0, len(y_tr), len(y_tr))
            fb = FeatureBuilder().fit(x_tr[bi], names)
            a = add_bias(fb.transform(x_tr[bi]))
            w = logistic_newton(y_tr[bi], a, LAMBDA).astype(np.float32)
            boot_tr.append((add_bias(fb.transform(x_tr)) @ w).astype(float))
            boot_va.append((add_bias(fb.transform(x_va)) @ w).astype(float))
        bag_tr, bag_va = np.mean(boot_tr, 0), np.mean(boot_va, 0)
        record("bootstrap bag x5", y_va, bag_va, best_threshold(y_tr, bag_tr)[0])
        print(f"fold {f} done ({time.time() - t0:.0f}s)", flush=True)

    base = np.array(f1s["current"])
    print(f"\n{'variant':20s} {'F1':>7s} {'AUC':>7s} {'dF1':>8s}  dF1 per fold")
    for name, v in f1s.items():
        d = np.array(v) - base
        per_fold = " ".join(f"{z:+.4f}" for z in d)
        print(
            f"{name:20s} {np.mean(v):7.4f} {np.mean(aucs[name]):7.4f} "
            f"{d.mean():+8.4f}  {per_fold}"
        )


# ---------------------------------------------------------------------------
# Held-out split experiments: interactions and a neural network
# ---------------------------------------------------------------------------


def mlp(x, x_va, y, h=64, epochs=6, lr=1e-3, wd=1e-5, bs=512, seed=0, drop=0.0):
    """One-hidden-layer ReLU network trained with Adam on the logistic loss.

    Returns the logits for x and x_va.
    """
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    w1 = (rng.standard_normal((d, h)) * np.sqrt(2 / d)).astype(np.float32)
    b1 = np.zeros(h, np.float32)
    w2 = (rng.standard_normal(h) * np.sqrt(1 / h)).astype(np.float32)
    b2 = np.array([np.log(y.mean() / (1 - y.mean()))], np.float32)
    params = [w1, b1, w2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    y32 = y.astype(np.float32)
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        for i in range(0, len(y), bs):
            bi = perm[i : i + bs]
            xb, yb = x[bi], y32[bi]
            hidden = xb @ params[0] + params[1]
            mask = hidden > 0
            hr = hidden * mask
            if drop:
                dm = (rng.random(hr.shape) > drop).astype(np.float32) / (1 - drop)
                hr = hr * dm
            z = hr @ params[2] + params[3][0]
            p = 1 / (1 + np.exp(-z))
            g = (p - yb) / len(bi)  # dLoss/dz
            g_w2 = hr.T @ g
            g_b2 = np.array([g.sum()], np.float32)
            g_h = np.outer(g, params[2])
            if drop:
                g_h = g_h * dm
            g_h = g_h * mask
            grads = [
                xb.T @ g_h + wd * params[0],
                g_h.sum(0),
                g_w2 + wd * params[2],
                g_b2,
            ]
            t += 1
            for k in range(4):  # Adam update
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                m_hat = m[k] / (1 - 0.9**t)
                v_hat = v[k] / (1 - 0.999**t)
                params[k] = params[k] - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def logits(z):
        return np.maximum(z @ params[0] + params[1], 0) @ params[2] + params[3][0]

    return logits(x), logits(x_va)


def split_experiments():
    """Interactions, neural networks and ensembles on one 80/20 split."""
    xtr, _, yraw, _, _, names = load_data()
    y = (yraw == 1).astype(float)
    idx = np.random.default_rng(0).permutation(len(y))
    tr, va = idx[:262000], idx[262000:]
    fb = FeatureBuilder().fit(xtr[tr], names)
    a = add_bias(fb.transform(xtr[tr]))
    b = add_bias(fb.transform(xtr[va]))

    def evaluate(name, s_tr, s_va):
        thr, _ = best_threshold(y[tr], s_tr)
        f1 = f1_score(y[va], (s_va > thr).astype(int))
        print(f"{name:42s} F1 {f1:.4f}  AUC {auc(y[va], s_va):.4f}", flush=True)

    w = logistic_newton(y[tr], a, LAMBDA)
    s_lr_tr, s_lr_va = a @ w.astype(np.float32), b @ w.astype(np.float32)
    evaluate("logistic (final model)", s_lr_tr, s_lr_va)

    # Interactions: (standardized age group, male indicator) x the 60
    # features with the largest absolute logistic weight.
    age = xtr[:, names.index("_AGEG5YR")]
    age = np.where(age == 14, np.nan, age)  # 14 = don't know / refused
    age = np.where(np.isnan(age), np.nanmedian(age[tr]), age)
    age_z = ((age - age[tr].mean()) / age[tr].std()).astype(np.float32)
    male = (xtr[:, names.index("SEX")] == 1).astype(np.float32)
    top = np.argsort(-np.abs(w[1:]))[:60] + 1
    base = add_bias(fb.transform(xtr))[:, top]
    inter = np.hstack([base * age_z[:, None], base * male[:, None]])
    a2, b2 = np.hstack([a, inter[tr]]), np.hstack([b, inter[va]])
    for lam in [1e-4, 3e-4]:
        w2 = logistic_newton(y[tr], a2, lam).astype(np.float32)
        evaluate(f"logistic + 120 interactions, lambda={lam:g}", a2 @ w2, b2 @ w2)

    # Neural networks, alone and averaged with logistic regression.
    for h, epochs, drop, lr in [
        (64, 6, 0.0, 1e-3),
        (64, 15, 0.0, 5e-4),
        (128, 15, 0.3, 5e-4),
        (256, 20, 0.5, 3e-4),
    ]:
        s_tr, s_va = mlp(a, b, y[tr], h=h, epochs=epochs, drop=drop, lr=lr)
        evaluate(f"MLP h={h} epochs={epochs} dropout={drop}", s_tr, s_va)
        evaluate(
            "  0.5 * logistic + 0.5 * MLP",
            0.5 * s_lr_tr + 0.5 * s_tr,
            0.5 * s_lr_va + 0.5 * s_va,
        )


if __name__ == "__main__":
    np.seterr(all="ignore")  # spurious matmul warnings with Apple Accelerate
    {"cv": cv_experiments, "split": split_experiments}[sys.argv[1]]()
