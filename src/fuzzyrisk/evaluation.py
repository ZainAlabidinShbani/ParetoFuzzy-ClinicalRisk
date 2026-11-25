"""Outer cross-validation runner (leakage-free by construction).

Every model is a fresh estimator whose ``fit`` receives only the outer-training
rows (raw DataFrame); all preprocessing, tuning and evolutionary search happen
inside it. Test rows are used once, for ``predict_proba``.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import StratifiedKFold

from .baselines import TunedModel
from .data import Dataset
from .metrics import all_metrics, auroc, brier, hypervolume, non_dominated_mask
from .models import FuzzyRiskClassifier
from .optim import objective_scales


class PeakRSS:
    """Samples process resident set size in a background thread during a block."""

    def __init__(self, interval=0.02):
        self.interval = interval
        self.proc = psutil.Process()

    def __enter__(self):
        self.start_mb = self.proc.memory_info().rss / 2**20
        self.peak_mb = self.start_mb
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self.proc.memory_info().rss / 2**20)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak_mb = max(self.peak_mb, self.proc.memory_info().rss / 2**20)


# --------------------------------------------------------------------------- #
def model_factory(name: str, ds: Dataset, seed: int, cfg: dict):
    cols = dict(numeric_cols=ds.numeric_cols, binary_cols=ds.binary_cols,
                nominal_cols=ds.nominal_cols, categories=ds.categories)
    ga = dict(pop_size=cfg.get("pop_size", 60), n_gen=cfg.get("n_gen", 60),
              R_max=cfg.get("R_max", 12), L_max=cfg.get("L_max", 4), random_state=seed)
    n_iter = cfg.get("n_iter", 15)
    table = {
        "LogReg": lambda: TunedModel("logreg", random_state=seed, **cols),
        "DecisionTree": lambda: TunedModel("dtree", random_state=seed, **cols),
        "XGBoost": lambda: TunedModel("xgboost", n_iter=n_iter, random_state=seed, **cols),
        "LightGBM": lambda: TunedModel("lightgbm", n_iter=n_iter, random_state=seed, **cols),
        "CatBoost": lambda: TunedModel("catboost", n_iter=n_iter, random_state=seed, **cols),
        "MLP": lambda: TunedModel("mlp", n_iter=n_iter, random_state=seed, **cols),
        "XGBoost+Platt": lambda: TunedModel("xgboost", "sigmoid", n_iter=n_iter, random_state=seed, **cols),
        "XGBoost+Isotonic": lambda: TunedModel("xgboost", "isotonic", n_iter=n_iter, random_state=seed, **cols),
        "SO-GA-Fuzzy(classic)": lambda: FuzzyRiskClassifier(
            "ga", ("accuracy",), "evolved", **ga, **cols),
        "SO-GA-Fuzzy(hybrid)": lambda: FuzzyRiskClassifier(
            "ga", ("auroc",), "hybrid", **ga, **cols),
        "MO-Fuzzy(NSGA-II)": lambda: FuzzyRiskClassifier(
            "nsga2", ("auroc", "brier", "complexity"), "hybrid", **ga, **cols),
    }
    return table[name]()


def _complexity(model) -> dict:
    if isinstance(model, FuzzyRiskClassifier):
        return {"n_rules": model.n_rules_, "mean_antecedents": model.mean_antecedents_,
                "complexity": model.complexity_, "inner_hv": model.inner_hv_, "n_evals": model.n_evals_}
    if isinstance(model, TunedModel) and model.name == "dtree" and model.calibration is None:
        t = model.model_.named_steps["clf"].tree_
        depth = np.zeros(t.node_count, int)
        for i in range(t.node_count):  # parents precede children in sklearn trees
            for c in (t.children_left[i], t.children_right[i]):
                if c != -1:
                    depth[c] = depth[i] + 1
        leaves = t.children_left == -1
        n, L = int(leaves.sum()), float(depth[leaves].mean())
        return {"n_rules": n, "mean_antecedents": L, "complexity": n + L}
    return {"n_rules": np.nan, "mean_antecedents": np.nan, "complexity": np.nan}


def _test_front(model: FuzzyRiskClassifier, X_te, y_te) -> dict:
    P = model.front_predict_proba(X_te)
    pts = []
    for i, p in enumerate(P):
        a = auroc(y_te, p)
        pts.append([1 - (0.5 if np.isnan(a) else a), brier(y_te, p), model.front_metrics_[i]["complexity"]])
    F = np.asarray(pts)
    Fn = F / np.asarray(objective_scales(model.objectives, model.spec_))
    nd = non_dominated_mask(Fn)
    return {"test_hv": hypervolume(Fn[nd], np.ones(3)), "test_front": F.tolist(),
            "inner_front": [[1 - m["auroc"], m["brier"], m["complexity"]] for m in model.front_metrics_],
            "selected_index": model.selected_index_}


def run_cv(ds: Dataset, models: list[str], seed: int = 0, n_folds: int = 5, cfg: dict | None = None,
           log=print, spy=None):
    """Run outer stratified K-fold CV for each model. Returns (records, predictions, fronts, rules)."""
    cfg = cfg or {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    records, preds, fronts, rules = [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(ds.X, ds.y)):
        X_tr, X_te = ds.X.iloc[tr], ds.X.iloc[te]
        y_tr, y_te = ds.y[tr], ds.y[te]
        for name in models:
            model = model_factory(name, ds, seed, cfg)
            if spy is not None:
                spy(name, fold, X_tr, X_te, model)
            with PeakRSS() as mon:
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr)
                fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = model.predict_proba(X_te)[:, 1]
            pred_s = time.perf_counter() - t0
            rec = {"dataset": ds.name, "seed": seed, "fold": fold, "model": name,
                   "n_train": len(tr), "n_test": len(te), **all_metrics(y_te, p), **_complexity(model),
                   "fit_s": fit_s, "predict_s": pred_s, "peak_rss_mb": mon.peak_mb,
                   "rss_delta_mb": mon.peak_mb - mon.start_mb}
            if isinstance(model, FuzzyRiskClassifier) and model.algorithm == "nsga2":
                fr = _test_front(model, X_te, y_te)
                rec["test_hv"] = fr["test_hv"]
                fronts.append({"dataset": ds.name, "seed": seed, "fold": fold, "model": name, **fr,
                               "hv_history": model.hv_history_})
            if isinstance(model, FuzzyRiskClassifier):
                rules.append({"dataset": ds.name, "seed": seed, "fold": fold, "model": name,
                              "rules": model.describe()})
            records.append(rec)
            preds.append(pd.DataFrame({"dataset": ds.name, "seed": seed, "fold": fold, "model": name,
                                       "row": te, "y": y_te, "p": p}))
            log(f"  [{ds.name} f{fold}] {name:<22} AUROC={rec['auroc']:.3f} Brier={rec['brier']:.3f} "
                f"ECE={rec['ece']:.3f} rules={rec['n_rules']} fit={fit_s:.1f}s")
    return records, pd.concat(preds, ignore_index=True), fronts, rules
