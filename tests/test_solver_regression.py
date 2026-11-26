"""Regression test for Deviation D3: the penalised-logistic Newton solver must not diverge.

Plain Newton diverged (intercept ~ -1e12) on Thyroid rule bases: a rule fired fully on a small,
high-risk subgroup in imbalanced data, the first step overshot, and later steps on saturated logits
exploded. The synthetic design below reproduces that pattern; the damped solver must match sklearn.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from fuzzyrisk.metrics import calibration_slope_intercept, fit_logistic_l2


def thyroid_like(seed=1, n=3000):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    x1 = np.where(u < 0.07, 1.0, np.where(u < 0.09, rng.random(n), 0.0))
    x2 = np.clip(1 - np.abs(rng.normal(0, 1, n)), 0, 1)
    x3 = np.where(rng.random(n) < 0.04, rng.random(n), 0.0)
    x4 = (rng.random(n) < 0.58) * 1.0
    y = (rng.random(n) < np.where(x1 > 0.5, 0.66, 0.006)).astype(float)
    return np.column_stack([x1, x2, x3, x4]), y


def test_damped_newton_matches_sklearn_on_divergent_case():
    A, y = thyroid_like()
    for lam in (0.01, 0.1, 1.0, 10.0):
        b, w = fit_logistic_l2(A, y, lam=lam)
        sk = LogisticRegression(C=1 / lam, tol=1e-12, max_iter=10000).fit(A, y)
        assert np.isfinite(b) and abs(b) < 20
        assert abs(b - sk.intercept_[0]) < 1e-4
        assert np.max(np.abs(w - sk.coef_[0])) < 1e-4


def test_solver_objective_not_worse_than_start():
    A, y = thyroid_like(seed=3)
    b, w = fit_logistic_l2(A, y, lam=0.01)
    z = b + A @ w
    J = np.sum(np.logaddexp(0, z) - y * z) + 0.005 * np.sum(w * w)
    pb = y.mean()
    z0 = np.full_like(y, np.log(pb / (1 - pb)))
    J0 = np.sum(np.logaddexp(0, z0) - y * z0)
    assert J <= J0


def test_calibration_slope_identity_on_calibrated_probs():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.random(p.size) < p).astype(float)
    a, s = calibration_slope_intercept(y, p)
    assert abs(s - 1) < 0.05 and abs(a) < 0.05
