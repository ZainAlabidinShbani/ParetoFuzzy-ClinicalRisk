"""Valid probability outputs and deterministic seeding."""
import numpy as np
import pytest

from conftest import FAST_CFG
from fuzzyrisk.evaluation import model_factory
from fuzzyrisk.models import FuzzyRiskClassifier
from test_leakage import ALL_MODELS

TR, TE = np.arange(0, 180), np.arange(180, 240)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_predict_proba_valid(synth, name):
    m = model_factory(name, synth, 0, FAST_CFG).fit(synth.X.iloc[TR], synth.y[TR])
    P = m.predict_proba(synth.X.iloc[TE])
    assert P.shape == (len(TE), 2)
    assert np.all(np.isfinite(P))
    assert P.min() >= 0.0 and P.max() <= 1.0
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
    assert set(np.unique(m.predict(synth.X.iloc[TE]))) <= {0, 1}


def test_fis_probabilities_valid_on_extreme_inputs(synth, cols):
    m = FuzzyRiskClassifier("nsga2", pop_size=10, n_gen=2, R_max=5, random_state=0, **cols)
    m.fit(synth.X.iloc[TR], synth.y[TR])
    X = synth.X.iloc[TE].copy()
    X["age"] = np.where(np.arange(len(X)) % 2, 1e9, -1e9)
    X["glu"] = np.nan
    P = m.predict_proba(X)
    assert np.all(np.isfinite(P)) and P.min() >= 0 and P.max() <= 1


@pytest.mark.parametrize("alg,obj,cons", [
    ("nsga2", ("auroc", "brier", "complexity"), "hybrid"),
    ("ga", ("auroc",), "hybrid"),
    ("ga", ("accuracy",), "evolved"),
])
def test_fuzzy_deterministic_seeding(synth, cols, alg, obj, cons):
    kw = dict(pop_size=14, n_gen=4, R_max=6, **cols)
    a = FuzzyRiskClassifier(alg, obj, cons, random_state=3, **kw).fit(synth.X.iloc[TR], synth.y[TR])
    b = FuzzyRiskClassifier(alg, obj, cons, random_state=3, **kw).fit(synth.X.iloc[TR], synth.y[TR])
    np.testing.assert_array_equal(a.front_X_, b.front_X_)
    np.testing.assert_array_equal(a.front_F_, b.front_F_)
    np.testing.assert_array_equal(a.predict_proba(synth.X.iloc[TE]), b.predict_proba(synth.X.iloc[TE]))
    c = FuzzyRiskClassifier(alg, obj, cons, random_state=4, **kw).fit(synth.X.iloc[TR], synth.y[TR])
    assert not np.array_equal(a.front_X_, c.front_X_) or a.front_X_.shape != c.front_X_.shape


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM", "CatBoost", "MLP", "DecisionTree"])
def test_baseline_deterministic_seeding(synth, name):
    p = [model_factory(name, synth, 7, FAST_CFG).fit(synth.X.iloc[TR], synth.y[TR])
         .predict_proba(synth.X.iloc[TE]) for _ in range(2)]
    np.testing.assert_array_equal(p[0], p[1])
