"""Pareto non-dominance validity and rule-base interpretability extraction."""
import numpy as np
import pytest

from fuzzyrisk.encoding import GenomeSpec
from fuzzyrisk.fis import MIN_GAP, repair_centres, triangular_partition
from fuzzyrisk.metrics import dominates, hypervolume, non_dominated_mask
from fuzzyrisk.models import FuzzyRiskClassifier
from fuzzyrisk.optim import FuzzyCrossover, FuzzyMutation, FuzzySampling, select_eps_compact


# ----------------------------- Pareto ------------------------------------- #
def test_non_dominated_mask_handcrafted():
    F = np.array([[1, 5], [2, 3], [3, 3], [4, 1], [2, 4], [1, 5], [5, 5]], float)
    m = non_dominated_mask(F)
    # [3,3] dominated by [2,3]; [2,4] by [2,3]; [5,5] by all; duplicates of [1,5] both kept
    assert m.tolist() == [True, True, False, True, False, True, False]


def test_non_dominated_mask_matches_bruteforce():
    rng = np.random.default_rng(0)
    F = rng.random((80, 3))
    m = non_dominated_mask(F)
    for i in range(len(F)):
        dominated = any(dominates(F[j], F[i]) for j in range(len(F)) if j != i)
        assert m[i] == (not dominated)


def test_hypervolume_known_values():
    assert hypervolume([[0.5, 0.5]], [1, 1]) == pytest.approx(0.25)
    assert hypervolume([[0.25, 0.75], [0.75, 0.25]], [1, 1]) == pytest.approx(0.3125)
    assert hypervolume([[0.5, 0.5, 0.5]], [1, 1, 1]) == pytest.approx(0.125)
    assert hypervolume([[1.5, 0.1]], [1, 1]) == 0.0  # outside reference box


def test_nsga2_front_is_mutually_nondominated_and_consistent(synth, cols):
    m = FuzzyRiskClassifier("nsga2", pop_size=20, n_gen=5, R_max=6, random_state=1, **cols)
    m.fit(synth.X.iloc[:180], synth.y[:180])
    F = m.front_F_
    assert non_dominated_mask(F).all()
    # stored objective values must equal a fresh recomputation from the genome
    for x, f, met in zip(m.front_X_, F, m.front_metrics_):
        np.testing.assert_allclose(f, [1 - met["auroc"], met["brier"], met["complexity"]], atol=1e-12)
    # the deployed model's complexity equals the selected front member's
    assert m.complexity_ == pytest.approx(m.front_metrics_[m.selected_index_]["complexity"])
    assert 0.0 <= m.inner_hv_ <= 1.0


def test_select_eps_compact():
    fm = [{"auroc": 0.85, "complexity": 9.0, "brier": 0.15},
          {"auroc": 0.845, "complexity": 3.0, "brier": 0.16},
          {"auroc": 0.845, "complexity": 3.0, "brier": 0.155},
          {"auroc": 0.80, "complexity": 1.0, "brier": 0.19}]
    assert select_eps_compact(fm, eps=0.01) == 2
    assert select_eps_compact(fm, eps=0.0) == 0


# -------------------------- interpretability ------------------------------ #
def _spec():
    # features: 0,1 continuous; 2,3 binary
    return GenomeSpec(np.array(["continuous", "continuous", "binary", "binary"]), R_max=6, L_max=3,
                      feature_names=["age", "bmi", "sex=male", "smoker=yes"])


def test_decode_counts_distinct_active_rules():
    s = _spec()
    x = np.zeros(s.n_var)
    x[: s.n_mf] = [0.1, 0.5, 0.9, 0.2, 0.4, 0.8]
    A = np.zeros((s.R_max, s.d), int)
    act = np.zeros(s.R_max, int)
    A[0] = [3, 0, 0, 0]; act[0] = 1          # age High
    A[1] = [3, 0, 0, 0]; act[1] = 1          # duplicate -> removed
    A[2] = [0, 0, 0, 0]; act[2] = 1          # empty -> inactive
    A[3] = [1, 2, 2, 0]; act[3] = 0          # inactive bit
    A[4] = [1, 0, 0, 2]; act[4] = 1          # age Low AND smoker Yes
    A[5] = [2, 1, 2, 2]; act[5] = 1          # 4 conditions > L_max=3 -> truncated to 3
    s.write_rules(x, A, act)
    rb, slots = s.decode(x)
    assert rb.n_rules == 3
    assert slots.tolist() == [0, 4, 5]
    assert [len(r) for r in rb.rules] == [1, 2, 3]
    assert rb.mean_antecedents == pytest.approx(2.0)
    assert rb.total_conditions == 6
    assert rb.complexity() == pytest.approx(3 + 2.0)
    rb.weights = np.array([1.0, -0.5, 0.25]); rb.bias = -1.0
    text = "\n".join(rb.describe())
    assert "IF age is High THEN +1.000" in text
    assert "IF age is Low AND smoker=yes is Yes THEN -0.500" in text


def test_strong_partition_and_centre_repair():
    rng = np.random.default_rng(0)
    xs = np.linspace(-0.2, 1.2, 501)
    for _ in range(200):
        c = repair_centres(rng.random(3))
        assert np.all(np.diff(c) >= MIN_GAP - 1e-12) and c[0] >= 0 and c[2] <= 1
        M = triangular_partition(xs, c)
        np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-12)
        assert M.min() >= 0 and M.max() <= 1
        # each set reaches full membership at its centre
        for k in range(3):
            assert triangular_partition(np.array([c[k]]), c)[0, k] == pytest.approx(1.0)


def test_operators_respect_bounds_and_rule_length():
    s = _spec()
    rng = np.random.default_rng(0)

    class P:  # minimal stand-in for a pymoo problem
        n_var = s.n_var

    X = FuzzySampling(s, np.tile([0.2, 0.5, 0.8], (2, 1)))._do(P, 30, random_state=rng)
    xl, xu = s.bounds()
    for _ in range(20):
        pairs = rng.integers(0, len(X), size=(2, 15))
        Y = FuzzyCrossover(s)._do(P, X[pairs], random_state=rng).reshape(-1, s.n_var)
        X = FuzzyMutation(s)._do(P, Y, random_state=rng)
        assert np.all(X >= xl - 1e-12) and np.all(X <= xu + 1e-12)
        for x in X:
            assert (s.ante_matrix(x) > 0).sum(axis=1).max() <= s.L_max


def test_empty_rule_base_is_intercept_only():
    """Regression: an empty rule base must return one probability per row."""
    s = _spec()
    x = np.zeros(s.n_var)
    x[: s.n_mf] = [0.1, 0.5, 0.9, 0.2, 0.4, 0.8]
    rb, _ = s.decode(x)
    assert rb.n_rules == 0 and rb.complexity() == 0.0
    rb.bias = 0.3
    P = rb.predict_proba(np.random.default_rng(0).random((7, 4)))
    assert P.shape == (7, 2)
    np.testing.assert_allclose(P[:, 1], 1 / (1 + np.exp(-0.3)))
