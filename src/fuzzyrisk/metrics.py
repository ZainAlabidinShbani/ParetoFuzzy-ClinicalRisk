"""Discrimination, calibration and Pareto-front metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

EPS = 1e-12


def auroc(y, p) -> float:
    """Mann-Whitney AUROC with mid-ranks for ties (fast, no sklearn overhead)."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def brier(y, p) -> float:
    y = np.asarray(y, float)
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def log_loss(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error (L1, sample-weighted) for the positive class.

    strategy='uniform'  : equal-width bins on [0, 1]  (Naeini et al. 2015; Guo et al. 2017)
    strategy='quantile' : equal-mass bins (adaptive ECE; Nixon et al. 2019)
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        raise ValueError(strategy)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / y.size)


def fit_logistic_l2(A, y, lam: float = 1.0, penalize_intercept: bool = False,
                    max_iter: int = 50, tol: float = 1e-8, w0=None):
    """Newton-Raphson for L2-penalised logistic regression.

    Minimises  sum_i logloss(y_i, sigmoid(b + A_i w)) + lam/2 ||w||^2.
    Returns (b, w). Deterministic; used both for TSK consequent estimation and
    for logistic recalibration (calibration slope/intercept).
    """
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    n, k = A.shape
    Z = np.hstack([np.ones((n, 1)), A])
    pen = np.full(k + 1, lam, float)
    if not penalize_intercept:
        pen[0] = 0.0
    theta = np.zeros(k + 1)
    if w0 is not None:
        theta[:] = w0
    else:
        pbar = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        theta[0] = np.log(pbar / (1 - pbar))
    for _ in range(max_iter):
        z = np.clip(Z @ theta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-z))
        g = Z.T @ (p - y) + pen * theta
        W = p * (1 - p)
        H = (Z * W[:, None]).T @ Z + np.diag(pen) + 1e-9 * np.eye(k + 1)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        theta -= step
        if np.max(np.abs(step)) < tol:
            break
    return float(theta[0]), theta[1:].copy()


def calibration_slope_intercept(y, p):
    """Logistic recalibration: logit P(y=1) = a + b * logit(p).

    Returns (intercept a, slope b) of the unpenalised recalibration model
    (Van Calster et al. 2016; Steyerberg et al. 2010). Ideal: a=0, b=1.
    Note: the slope/intercept pair is reported jointly (``calibration-in-the-large``
    is reported separately as mean(p) - mean(y)).
    """
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))[:, None]
    a, b = fit_logistic_l2(lp, y, lam=0.0)
    return a, float(b[0])


def all_metrics(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, float)
    a, b = calibration_slope_intercept(y, p)
    return {
        "auroc": auroc(y, p),
        "pr_auc": pr_auc(y, p),
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "ece": ece(y, p, 10, "uniform"),
        "ece_adaptive": ece(y, p, 10, "quantile"),
        "cal_intercept": a,
        "cal_slope": b,
        "cal_in_the_large": float(p.mean() - y.mean()),
    }


# --------------------------------------------------------------------------- #
# Pareto utilities (minimisation)
# --------------------------------------------------------------------------- #
def dominates(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_mask(F) -> np.ndarray:
    """Boolean mask of the non-dominated rows of F (all objectives minimised)."""
    F = np.asarray(F, float)
    n = F.shape[0]
    mask = np.ones(n, bool)
    for i in range(n):
        if not mask[i]:
            continue
        le = np.all(F <= F[i], axis=1)
        lt = np.any(F < F[i], axis=1)
        if np.any(le & lt):
            mask[i] = False
    return mask


def hypervolume(F, ref) -> float:
    """Exact hypervolume of the points in F w.r.t. reference point ``ref`` (minimisation).

    Uses pymoo's implementation; points not strictly better than ``ref`` in
    every objective contribute nothing.
    """
    from pymoo.indicators.hv import HV

    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    F = F[np.all(F < ref, axis=1)]
    if F.size == 0:
        return 0.0
    return float(HV(ref_point=ref)(F))
