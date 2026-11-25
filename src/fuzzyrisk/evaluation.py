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
    ga = dict(pop_size=cfg.get("pop_size", 80), n_gen=cfg.get("n_gen", 80),
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
                "complexity": model.complexity_, "inner_hv": model.inner_hv_, "n_evals": model.n_evals_,
                "lambda": model.lambda_, "archive_size": len(model.front_X_)}
    if isinstance(model, TunedModel) and model.name == "dtree" and model.calibration is None:
        t = model.model_.named_steps["clf"].tree_
        depth = np.zeros(t.node_count, int)
        for i in range(t.node_count):  # parents precede children in sklearn trees
            for c in (t.children_left[i], t.children_right[i]):
                if c != -1:
                    depth[c] = depth[i] + 1
        leaves = t.children_left == -1
        n, L = int(leaves.sum()), float(depth[leaves].mean())
        # each leaf = one conjunctive rule; total literals = sum of leaf depths
        return {"n_rules": n, "mean_antecedents": L, "complexity": int(depth[leaves].sum())}
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


def fit_one(ds: Dataset, name: str, fold: int, tr, te, seed: int, cfg: dict, spy=None) -> dict:
    """Fit one model on one outer fold; return a JSON-serialisable payload."""
    X_tr, X_te = ds.X.iloc[tr], ds.X.iloc[te]
    y_tr, y_te = ds.y[tr], ds.y[te]
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
    front = rules = None
    extra = []
    if isinstance(model, FuzzyRiskClassifier) and model.algorithm == "nsga2":
        # secondary (ablation) operating point from the same run: best inner-OOF Brier
        j = model.best_brier_index_
        rbj = model.member_rulebase(j)
        pj = rbj.predict_proba(model.prep_.transform(X_te))[:, 1]
        rj = {**rec, "model": name.replace("NSGA-II", "best-Brier"), **all_metrics(y_te, pj),
              "n_rules": rbj.n_rules, "mean_antecedents": rbj.mean_antecedents,
              "complexity": rbj.complexity(), "lambda": model.front_metrics_[j]["lam"],
              "ablation_of": name}
        extra.append({"record": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in rj.items()},
                      "pred": {"row": [int(r) for r in te], "y": [int(v) for v in y_te],
                               "p": [float(v) for v in pj]}})
        fr = _test_front(model, X_te, y_te)
        rec["test_hv"] = fr["test_hv"]
        front = {"dataset": ds.name, "seed": seed, "fold": fold, "model": name, **fr,
                 "hv_history": list(model.hv_history_)}
    if isinstance(model, FuzzyRiskClassifier):
        rules = {"dataset": ds.name, "seed": seed, "fold": fold, "model": name, "rules": model.describe()}
    rec = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in rec.items()}
    pred = {"row": [int(r) for r in te], "y": [int(v) for v in y_te], "p": [float(v) for v in p]}
    del model
    return {"record": rec, "pred": pred, "front": front, "rules": rules, "extra": extra}


def run_cv(ds: Dataset, models: list[str], seed: int = 0, n_folds: int = 5, cfg: dict | None = None,
           log=print, spy=None, skip=None, on_result=None, on_fold_end=None):
    """Run outer stratified K-fold CV for each model.

    skip(dataset, fold, model) -> bool : return True to skip an already-completed fit
    on_result(payload)                 : called immediately after each completed fit
    on_fold_end(fold, fold_payloads)   : called after each fold (e.g. gc + memory logging)
    Returns (records, predictions, fronts, rules) for the fits run in this call.
    """
    cfg = cfg or {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    records, preds, fronts, rules = [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(ds.X, ds.y)):
        fold_payloads = []
        for name in models:
            if skip is not None and skip(ds.name, fold, name):
                continue
            pl = fit_one(ds, name, fold, tr, te, seed, cfg, spy)
            if on_result is not None:
                on_result(pl)
            fold_payloads.append(pl)
            rec = pl["record"]
            records.append(rec)
            preds.append(pd.DataFrame({"dataset": ds.name, "seed": seed, "fold": fold, "model": name,
                                       **pl["pred"]}))
            for ex in pl.get("extra", []):
                records.append(ex["record"])
                preds.append(pd.DataFrame({"dataset": ds.name, "seed": seed, "fold": fold,
                                           "model": ex["record"]["model"], **ex["pred"]}))
            if pl["front"] is not None:
                fronts.append(pl["front"])
            if pl["rules"] is not None:
                rules.append(pl["rules"])
            log(f"  [{ds.name} f{fold}] {name:<22} AUROC={rec['auroc']:.3f} Brier={rec['brier']:.3f} "
                f"ECE={rec['ece']:.3f} rules={rec['n_rules']} fit={rec['fit_s']:.1f}s "
                f"peakRSS={rec['peak_rss_mb']:.0f}MB")
        if on_fold_end is not None:
            on_fold_end(fold, fold_payloads)
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return records, pred_df, fronts, rules
