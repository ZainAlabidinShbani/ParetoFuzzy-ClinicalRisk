"""Metric correctness, consequent solver, dataset registry and held-out guard."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from fuzzyrisk import data as D
from fuzzyrisk.metrics import auroc, brier, calibration_slope_intercept, ece, fit_logistic_l2


def test_auroc_and_brier_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500), 1)  # many ties
    assert auroc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert brier(y, p) == pytest.approx(brier_score_loss(y, p))


def test_ece_values():
    y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75])
    assert ece(y, p) == pytest.approx(0.0)
    p2 = np.full(10, 0.5)  # one bin, frequency 0.7 -> |0.5 - 0.7| = 0.2
    assert ece(y, p2) == pytest.approx(0.2)
    assert ece(y, p2, strategy="quantile") == pytest.approx(0.2)


def test_calibration_slope_intercept_on_calibrated_data():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.random(20000) < p).astype(int)
    a, b = calibration_slope_intercept(y, p)
    assert abs(a) < 0.05 and abs(b - 1) < 0.05
    # overconfident predictions -> slope < 1
    lp = np.log(p / (1 - p))
    a2, b2 = calibration_slope_intercept(y, 1 / (1 + np.exp(-2 * lp)))
    assert b2 == pytest.approx(0.5, abs=0.05)


def test_newton_logistic_matches_sklearn():
    rng = np.random.default_rng(2)
    A = rng.random((300, 5))
    y = (rng.random(300) < 1 / (1 + np.exp(-(A @ [2, -1, 0.5, 0, 1] - 1)))).astype(int)
    b0, w = fit_logistic_l2(A, y, lam=1.0)
    sk = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(A, y)
    np.testing.assert_allclose(w, sk.coef_[0], atol=1e-5)
    assert b0 == pytest.approx(sk.intercept_[0], abs=1e-5)


def test_registry_dev_heldout_separation():
    assert len(D.dev_datasets()) == 2
    assert len(D.heldout_datasets()) == 4
    assert set(D.dev_datasets()).isdisjoint(D.heldout_datasets())


@pytest.mark.parametrize("name", D.heldout_datasets())
def test_heldout_locked(name, monkeypatch, tmp_path):
    # guard fires before any network access
    monkeypatch.setattr(D, "_fetch_raw", lambda *a, **k: pytest.fail("held-out data was fetched"))
    with pytest.raises(D.HeldOutAccessError):
        D.load_dataset(name)
    monkeypatch.setattr(D, "PREREG_FILE", tmp_path / "PREREGISTRATION.md")
    with pytest.raises(D.HeldOutAccessError):
        D.load_dataset(name, allow_heldout=True)  # no preregistration file
    (tmp_path / "PREREGISTRATION.md").write_text("status: draft\n", encoding="utf-8")
    with pytest.raises(D.HeldOutAccessError):
        D.load_dataset(name, allow_heldout=True)  # not locked


def test_heldout_not_present_on_disk():
    # Deviation D2 (PROGRESS.md): pre-unlock invariant; after the locked preregistration
    # the held-out files are fetched by design.
    if D.preregistration_locked():
        pytest.skip("held-out datasets unlocked by the locked preregistration")
    for name in D.heldout_datasets():
        assert not (D.DATA_RAW / f"{name}.csv").exists()


@pytest.mark.network
@pytest.mark.parametrize("name,n,n_pos", [("pima_diabetes", 768, 268), ("heart_cleveland", 303, 138)])
def test_dev_datasets_checksums(name, n, n_pos):
    ds = D.load_dataset(name)  # raises ChecksumError on any md5 / sha256 mismatch
    assert len(ds.y) == n and int(ds.y.sum()) == n_pos
    assert ds.sha256 == D._load_pins()[name]


def _fake_ckd(n=400, n_pos=250, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    sch = D.REGISTRY["kidney_ckd"]["schema"]
    df = {}
    for c in sch["numeric"]:
        v = rng.normal(50, 10, n).round(1).astype(object)
        v[rng.choice(n, 5, replace=False)] = "?"           # missing token
        v[rng.choice(n, 2, replace=False)] = "\t43"        # stray tab (UCI artefact)
        df[{"wbcc": "wc", "rbcc": "rc"}.get(c, c)] = v     # exercise aliases
    for c, lv in sch["binary"].items():
        v = rng.choice(lv, n).astype(object)
        v[0] = "\t" + lv[1]
        v[1] = "?"
        df[c] = v
    y = np.array(["ckd"] * n_pos + ["notckd"] * (n - n_pos), dtype=object)
    y[0] = "ckd\t"
    df["class"] = y
    return pd.DataFrame(df)


def test_ckd_schema_loader_cleans_and_types(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_fetch_raw", lambda name, spec: _fake_ckd())
    monkeypatch.setattr(D, "preregistration_locked", lambda: True)
    monkeypatch.setattr(D, "PINS_FILE", tmp_path / "pins.json")
    monkeypatch.setattr(D, "DATA_RAW", tmp_path)
    ds = D.load_dataset("kidney_ckd", allow_heldout=True, pin_if_missing=True)
    sch = D.REGISTRY["kidney_ckd"]["schema"]
    assert ds.numeric_cols == sch["numeric"] and ds.binary_cols == list(sch["binary"])
    assert int(ds.y.sum()) == 250 and len(ds.y) == 400
    assert ds.X["age"].dtype == float and ds.X["age"].isna().sum() == 5
    assert (ds.X["wbcc"] == 43.0).sum() >= 2                  # tab-stripped numeric parsed
    assert ds.X["htn"].iloc[0] == "yes" and ds.X["htn"].isna().iloc[1]
    assert set(ds.X["htn"].dropna()) <= {"no", "yes"}


def test_positive_class_count_check(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_fetch_raw", lambda name, spec: _fake_ckd(n_pos=150))  # flipped labels
    monkeypatch.setattr(D, "preregistration_locked", lambda: True)
    monkeypatch.setattr(D, "PINS_FILE", tmp_path / "pins.json")
    monkeypatch.setattr(D, "DATA_RAW", tmp_path)
    with pytest.raises(D.ChecksumError):
        D.load_dataset("kidney_ckd", allow_heldout=True, pin_if_missing=True)
