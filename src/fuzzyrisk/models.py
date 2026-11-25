"""Scikit-learn compatible GA/NSGA-II fuzzy risk-score classifiers.

``FuzzyRiskClassifier.fit(X_df, y)`` performs, using only the rows it is given:
  1. fit the fuzzy-mode ``TabularPreprocessor`` (percentile scaling, imputation),
  2. evolve the knowledge base (NSGA-II or single-objective GA),
  3. pick one deployable model (preregistered epsilon-compact rule, MO only),
  4. refit hybrid consequents on all given rows with the genome's inner-selected lambda.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .encoding import GenomeSpec
from .optim import (LAMBDAS, FitnessEvaluator, objective_scales, run_ga, run_nsga2, select_best_brier,
                    select_eps_compact)
from .metrics import hypervolume, non_dominated_mask
from .preprocessing import TabularPreprocessor


class FuzzyRiskClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, algorithm="nsga2", objectives=("auroc", "brier", "complexity"),
                 consequents="hybrid", pop_size=80, n_gen=80, R_max=12, L_max=4,
                 inner_folds=3, lambdas=LAMBDAS, select_eps=0.01, random_state=0,
                 numeric_cols=(), binary_cols=(), nominal_cols=(), categories=None):
        self.algorithm = algorithm
        self.objectives = objectives
        self.consequents = consequents
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.R_max = R_max
        self.L_max = L_max
        self.inner_folds = inner_folds
        self.lambdas = lambdas
        self.select_eps = select_eps
        self.random_state = random_state
        self.numeric_cols = numeric_cols
        self.binary_cols = binary_cols
        self.nominal_cols = nominal_cols
        self.categories = categories

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        y = np.asarray(y, int)
        self.classes_ = np.array([0, 1])
        self.prep_ = TabularPreprocessor("fuzzy", self.numeric_cols, self.binary_cols,
                                         self.nominal_cols, self.categories).fit(X)
        Z = self.prep_.transform(X)
        evolve = self.consequents == "evolved"
        self.spec_ = GenomeSpec(self.prep_.feature_kinds_, self.R_max, self.L_max,
                                evolve_consequents=evolve,
                                feature_names=list(self.prep_.feature_names_))
        cont = self.spec_.cont
        mf_init = (np.quantile(Z[:, cont], [0.2, 0.5, 0.8], axis=0).T
                   if len(cont) else np.zeros((0, 3)))
        ev = FitnessEvaluator(self.spec_, Z, y, self.objectives, self.inner_folds,
                              self.lambdas, seed=self.random_state)
        if self.algorithm == "nsga2":
            res = run_nsga2(ev, mf_init, self.pop_size, self.n_gen, seed=self.random_state)
        elif self.algorithm == "ga":
            res = run_ga(ev, mf_init, self.pop_size, self.n_gen, seed=self.random_state)
        else:
            raise ValueError(self.algorithm)

        # inner (training-only) metrics for every returned solution
        self.front_X_ = res["X"]
        self.front_F_ = res["F"]
        self.front_metrics_ = [ev.metrics(x) for x in res["X"]]
        self.hv_history_ = res.get("hv_history", [])
        self.n_evals_ = res["n_evals"]
        if self.algorithm == "nsga2":
            idx = select_eps_compact(self.front_metrics_, self.select_eps)
            self.best_brier_index_ = select_best_brier(self.front_metrics_)
            scales = objective_scales(self.objectives, self.spec_)
            Fn = self.front_F_ / np.asarray(scales)
            self.inner_hv_ = hypervolume(Fn[non_dominated_mask(Fn)], np.ones(Fn.shape[1]))
        else:
            idx = 0
            self.best_brier_index_ = 0
            self.inner_hv_ = float("nan")
        self.selected_index_ = idx
        self.Z_train_, self.y_train_ = Z, y
        self.rulebase_ = self._finalise(idx)
        return self

    def _finalise(self, i):
        """Decode archive member i and refit its consequents on all training rows,
        using the lambda selected for that genome in the inner loop."""
        rb, _ = self.spec_.decode(self.front_X_[i])
        if self.consequents == "hybrid":
            if rb.n_rules:
                rb.fit_consequents(self.Z_train_, self.y_train_, lam=self.front_metrics_[i]["lam"])
            else:
                pbar = np.clip(self.y_train_.mean(), 1e-6, 1 - 1e-6)
                rb.bias, rb.weights = float(np.log(pbar / (1 - pbar))), np.zeros(0)
        return rb

    # ------------------------------------------------------------------ #
    def predict_proba(self, X):
        return self.rulebase_.predict_proba(self.prep_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def front_predict_proba(self, X):
        """P(y=1) on X for every front member (consequents refitted on training rows).
        For Pareto analysis only; never used for model selection."""
        Z = self.prep_.transform(X)
        return np.array([self._finalise(i).predict_proba(Z)[:, 1] for i in range(len(self.front_X_))])

    def member_rulebase(self, i):
        return self._finalise(i)

    # interpretability summary of the deployed model
    @property
    def n_rules_(self):
        return self.rulebase_.n_rules

    @property
    def mean_antecedents_(self):
        return self.rulebase_.mean_antecedents

    @property
    def complexity_(self):
        """Total literals of the deployed rule base."""
        return self.rulebase_.complexity()

    @property
    def lambda_(self):
        return self.front_metrics_[self.selected_index_].get("lam", float("nan"))

    def describe(self):
        return self.rulebase_.describe(self.prep_)
