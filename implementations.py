"""Implementations of the six basic ML methods required for Project 1.

All functions return a tuple ``(w, loss)`` where ``w`` is the final weight
vector (1D array) and ``loss`` is the corresponding (unregularized) loss.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Losses and gradients
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Mean squared error with the 1/2 factor: L(w) = 1/(2N) * ||y - tx w||^2.

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), feature matrix.
        w: shape (D,), weights.

    Returns:
        scalar loss.
    """
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Gradient of the MSE loss: -1/N * tx^T (y - tx w).

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), feature matrix.
        w: shape (D,), weights.

    Returns:
        shape (D,), gradient.
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Numerically stable logistic function 1 / (1 + exp(-t))."""
    out = np.empty_like(t, dtype=float)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def compute_logistic_loss(y, tx, w):
    """Average negative log-likelihood for logistic regression with y in {0, 1}.

    Uses log(1 + exp(z)) - y z, computed stably with np.logaddexp.

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), feature matrix.
        w: shape (D,), weights.

    Returns:
        scalar loss.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the average logistic loss: 1/N * tx^T (sigmoid(tx w) - y).

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), feature matrix.
        w: shape (D,), weights.

    Returns:
        shape (D,), gradient.
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ---------------------------------------------------------------------------
# Required methods
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent on the MSE loss.

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), feature matrix.
        initial_w: shape (D,), initial weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), final weights.
        loss: scalar, MSE loss of the final weights.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (mini-batch size 1).

    At each step one sample is drawn uniformly at random and the weights are
    updated with the gradient of the loss on that sample only.

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), feature matrix.
        initial_w: shape (D,), initial weights.
        max_iters: number of stochastic steps.
        gamma: step size.

    Returns:
        w: shape (D,), final weights.
        loss: scalar, MSE loss of the final weights on the full dataset.
    """
    w = np.array(initial_w, dtype=float)
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)
        w = w - gamma * compute_mse_gradient(y[i : i + 1], tx[i : i + 1], w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression by solving the normal equations tx^T tx w = tx^T y.

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), feature matrix.

    Returns:
        w: shape (D,), optimal weights.
        loss: scalar, MSE loss.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression via normal equations.

    Minimizes 1/(2N) ||y - tx w||^2 + lambda_ ||w||^2, whose solution is
    (tx^T tx + 2 N lambda_ I) w = tx^T y.

    Args:
        y: shape (N,), targets.
        tx: shape (N, D), feature matrix.
        lambda_: regularization strength.

    Returns:
        w: shape (D,), optimal weights.
        loss: scalar, MSE loss (without the penalty term).
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using full-batch gradient descent (y in {0, 1}).

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), feature matrix.
        initial_w: shape (D,), initial weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), final weights.
        loss: scalar, negative log-likelihood of the final weights.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression using gradient descent (y in {0, 1}).

    Minimizes the average logistic loss + lambda_ ||w||^2.

    Args:
        y: shape (N,), labels in {0, 1}.
        tx: shape (N, D), feature matrix.
        lambda_: regularization strength.
        initial_w: shape (D,), initial weights.
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: shape (D,), final weights.
        loss: scalar, negative log-likelihood (without the penalty term).
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
