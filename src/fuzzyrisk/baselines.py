"""Tuned baseline models (LR, DT, XGBoost, LightGBM, CatBoost, MLP) + post-hoc calibration.

Every baseline is ``Pipeline([TabularPreprocessor, model])`` tuned by inner
stratified CV on the training rows passed to ``fit`` (log-loss scoring), so
preprocessing is re-fitted in each inner fold. Optional post-hoc calibration
(Platt = 'sigmoid', 'isotonic') re-uses the tuned hyper-parameters inside
``CalibratedClassifierCV`` with its own inner stratified CV.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import TabularPreprocessor

N_JOBS_MODEL = 2  # bounded threading (16 GB RAM laptop)

BASELINES = ("logreg", "dtree", "xgboost", "lightgbm", "catboost", "mlp")


def _estimator(name: str, seed: int):
    if name == "logreg":
        return "linear", LogisticRegression(max_iter=5000), {"clf__C": np.logspace(-3, 2, 11)}, "grid"
    if name == "dtree":
        return "tree", DecisionTreeClassifier(random_state=seed), {
            "clf__max_depth": [2, 3, 4, 5], "clf__min_samples_leaf": [5, 10, 20, 40]}, "grid"
    if name == "xgboost":
        from xgboost import XGBClassifier
        est = XGBClassifier(tree_method="hist", n_jobs=N_JOBS_MODEL, random_state=seed,
                            eval_metric="logloss", verbosity=0)
        return "tree", est, {
            "clf__n_estimators": randint(50, 500), "clf__max_depth": randint(2, 7),
            "clf__learning_rate": loguniform(0.01, 0.3), "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.6, 0.4), "clf__min_child_weight": randint(1, 11),
            "clf__reg_lambda": loguniform(0.1, 10.0)}, "random"
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(n_jobs=N_JOBS_MODEL, random_state=seed, verbose=-1, subsample_freq=1)
        return "tree", est, {
            "clf__n_estimators": randint(50, 500), "clf__num_leaves": randint(4, 32),
            "clf__learning_rate": loguniform(0.01, 0.3), "clf__min_child_samples": randint(5, 41),
            "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__reg_lambda": loguniform(0.1, 10.0)}, "random"
    if name == "catboost":
        from catboost import CatBoostClassifier
        est = CatBoostClassifier(thread_count=N_JOBS_MODEL, random_seed=seed, verbose=0,
                                 allow_writing_files=False)
        return "tree", est, {
            "clf__iterations": randint(100, 600), "clf__depth": randint(2, 7),
            "clf__learning_rate": loguniform(0.01, 0.3), "clf__l2_leaf_reg": loguniform(1.0, 10.0)}, "random"
    if name == "mlp":
        est = MLPClassifier(early_stopping=True, max_iter=1000, random_state=seed)
        return "linear", est, {
            "clf__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
            "clf__alpha": loguniform(1e-5, 1e-1), "clf__learning_rate_init": loguniform(1e-3, 1e-2)}, "random"
    raise KeyError(name)


class TunedModel(BaseEstimator, ClassifierMixin):
    """Inner-CV tuned pipeline with optional post-hoc calibration."""

    def __init__(self, name="logreg", calibration=None, n_iter=15, inner_folds=3, random_state=0,
                 numeric_cols=(), binary_cols=(), nominal_cols=(), categories=None):
        self.name = name
        self.calibration = calibration
        self.n_iter = n_iter
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.numeric_cols = numeric_cols
        self.binary_cols = binary_cols
        self.nominal_cols = nominal_cols
        self.categories = categories

    def fit(self, X, y):
        y = np.asarray(y, int)
        self.classes_ = np.array([0, 1])
        mode, est, space, kind = _estimator(self.name, self.random_state)
        pipe = Pipeline([
            ("prep", TabularPreprocessor(mode, self.numeric_cols, self.binary_cols,
                                         self.nominal_cols, self.categories)),
            ("clf", est),
        ])
        cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.random_state)
        if kind == "grid":
            search = GridSearchCV(pipe, space, scoring="neg_log_loss", cv=cv, n_jobs=1, refit=True)
        else:
            search = RandomizedSearchCV(pipe, space, n_iter=self.n_iter, scoring="neg_log_loss", cv=cv,
                                        n_jobs=1, refit=True, random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)
        self.best_params_ = search.best_params_
        if self.calibration is None:
            self.model_ = search.best_estimator_
        else:
            base = clone(pipe).set_params(**search.best_params_)
            cal_cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.random_state + 1)
            self.model_ = CalibratedClassifierCV(base, method=self.calibration, cv=cal_cv)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(X)
        return np.clip(p, 0.0, 1.0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # complexity for tree-like interpretable baselines
    @property
    def n_rules_(self):
        if self.name == "dtree" and self.calibration is None:
            return int(self.model_.named_steps["clf"].get_n_leaves())
        return np.nan
