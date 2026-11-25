"""Zero train/test leakage."""
import numpy as np
import pytest

from conftest import FAST_CFG
from fuzzyrisk.baselines import TunedModel
from fuzzyrisk.evaluation import model_factory, run_cv
from fuzzyrisk.models import FuzzyRiskClassifier
from fuzzyrisk.preprocessing import TabularPreprocessor

ALL_MODELS = ["LogReg", "DecisionTree", "XGBoost", "LightGBM", "CatBoost", "MLP", "XGBoost+Platt",
              "XGBoost+Isotonic", "SO-GA-Fuzzy(classic)", "SO-GA-Fuzzy(hybrid)", "MO-Fuzzy(NSGA-II)"]


@pytest.mark.parametrize("mode", ["linear", "tree", "fuzzy"])
def test_preprocessor_fits_on_train_only(synth, cols, mode):
    X = synth.X
    tr, te = np.arange(0, 180), np.arange(180, 240)
    pp = TabularPreprocessor(mode, **cols).fit(X.iloc[tr])
    # statistics equal those computed from the training rows alone
    for c in synth.numeric_cols:
        v = X[c].iloc[tr].to_numpy(float)
        assert pp.medians_[c] == pytest.approx(np.nanmedian(v))
        vi = np.where(np.isnan(v), np.nanmedian(v), v)
        assert pp.lo_[c] == pytest.approx(np.percentile(vi, 1))
        assert pp.hi_[c] == pytest.approx(np.percentile(vi, 99))
        assert pp.means_[c] == pytest.approx(vi.mean())
    assert pp.n_fit_rows_ == len(tr)
    state = {k: dict(v) for k, v in vars(pp).items() if k.endswith("_") and isinstance(v, dict)}
    Z_tr = pp.transform(X.iloc[tr])
    # transforming (wildly perturbed) test rows must not change any fitted state
    Xte = X.iloc[te].copy()
    for c in synth.numeric_cols:
        Xte[c] = Xte[c] * 1000 + 1e6
    Xte["cp"] = "unseen_level"
    pp.transform(Xte)
    after = {k: dict(v) for k, v in vars(pp).items() if k.endswith("_") and isinstance(v, dict)}
    assert after == state
    np.testing.assert_array_equal(pp.transform(X.iloc[tr]), Z_tr)


def test_fuzzy_mode_universe_is_unit_interval(synth, cols):
    pp = TabularPreprocessor("fuzzy", **cols).fit(synth.X.iloc[:150])
    Xte = synth.X.iloc[150:].copy()
    Xte["age"] = Xte["age"] * 50  # far outside the training range
    Z = pp.transform(Xte)
    assert Z.min() >= 0.0 and Z.max() <= 1.0


def test_outer_splits_disjoint_and_exhaustive(synth):
    seen_test, fit_calls = [], []

    def spy(name, fold, X_tr, X_te, model):
        assert set(X_tr.index).isdisjoint(set(X_te.index))
        orig = model.fit

        def recording_fit(X, y):
            fit_calls.append((name, fold, set(X.index), set(X_te.index)))
            return orig(X, y)

        model.fit = recording_fit
        if name == "LogReg":
            seen_test.append(np.asarray(X_te.index))

    recs, preds, _, _ = run_cv(synth, ["LogReg", "MO-Fuzzy(NSGA-II)"], seed=0, n_folds=4,
                               cfg=FAST_CFG, log=lambda *a: None, spy=spy)
    allte = np.concatenate(seen_test)
    assert sorted(allte.tolist()) == list(range(len(synth.y)))  # each row tested exactly once
    for name, fold, fit_rows, test_rows in fit_calls:
        assert fit_rows.isdisjoint(test_rows), f"{name} fold {fold} saw test rows in fit"
    # predictions file references each row once per model
    for m, g in preds.groupby("model"):
        assert g["row"].is_unique and len(g) == len(synth.y)


def test_fitted_preprocessors_saw_only_training_rows(synth, cols):
    tr = np.arange(0, 160)
    fz = FuzzyRiskClassifier("nsga2", pop_size=10, n_gen=2, R_max=5, random_state=0, **cols)
    fz.fit(synth.X.iloc[tr], synth.y[tr])
    assert fz.prep_.n_fit_rows_ == len(tr)
    tm = TunedModel("logreg", random_state=0, **cols).fit(synth.X.iloc[tr], synth.y[tr])
    assert tm.model_.named_steps["prep"].n_fit_rows_ == len(tr)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_predictions_are_row_independent(synth, name):
    """No transductive leakage: a test row's probability must not depend on the
    other rows in the prediction batch."""
    tr, te = np.arange(0, 180), np.arange(180, 240)
    m = model_factory(name, synth, 0, FAST_CFG).fit(synth.X.iloc[tr], synth.y[tr])
    p_all = m.predict_proba(synth.X.iloc[te])[:, 1]
    p_sub = m.predict_proba(synth.X.iloc[te[:7]])[:, 1]
    np.testing.assert_allclose(p_sub, p_all[:7], rtol=0, atol=1e-12)
