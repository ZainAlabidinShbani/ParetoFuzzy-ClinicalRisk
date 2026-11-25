"""Leakage-safe tabular preprocessing.

``TabularPreprocessor`` learns every statistic (medians, modes, percentiles,
means/standard deviations) in :meth:`fit` from the rows it is given, and
:meth:`transform` only applies them. It is always used inside a pipeline that
is fitted on the training fold, so no test-fold information can reach a model.

Modes
-----
``linear`` : median/mode imputation, one-hot nominal, z-scored numerics (LR, MLP)
``tree``   : numerics passed through with NaN preserved (native missing handling
             in XGBoost/LightGBM/CatBoost/sklearn trees), one-hot nominal
``fuzzy``  : median/mode imputation, one-hot nominal, numerics mapped to [0, 1]
             by the training 1st/99th percentiles (clipped), so that fuzzy
             partitions live on a common, outlier-robust universe of discourse
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MODES = ("linear", "tree", "fuzzy")


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, mode="linear", numeric_cols=(), binary_cols=(), nominal_cols=(),
                 categories=None, lower_pct=1.0, upper_pct=99.0):
        self.mode = mode
        self.numeric_cols = numeric_cols
        self.binary_cols = binary_cols
        self.nominal_cols = nominal_cols
        self.categories = categories
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    # ------------------------------------------------------------------ #
    def _cats(self, c, col):
        if self.categories and c in self.categories:
            return list(self.categories[c])
        # fallback (schema absent): categories seen in the *training* rows only
        return sorted(pd.Series(col).dropna().astype(str).unique().tolist())

    def fit(self, X, y=None):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        X = pd.DataFrame(X)
        self.n_fit_rows_ = len(X)
        self.medians_, self.means_, self.stds_, self.lo_, self.hi_ = {}, {}, {}, {}, {}
        for c in self.numeric_cols:
            v = X[c].to_numpy(dtype=float)
            med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            self.medians_[c] = float(med)
            vi = np.where(np.isnan(v), med, v)
            self.means_[c] = float(vi.mean())
            sd = float(vi.std())
            self.stds_[c] = sd if sd > 1e-12 else 1.0
            lo, hi = np.percentile(vi, [self.lower_pct, self.upper_pct])
            if hi - lo < 1e-12:
                lo, hi = float(vi.min()), float(vi.max()) + 1e-12
            self.lo_[c], self.hi_[c] = float(lo), float(hi)
        self.cats_, self.modes_ = {}, {}
        for c in list(self.binary_cols) + list(self.nominal_cols):
            cats = [str(k) for k in self._cats(c, X[c])]
            self.cats_[c] = cats
            s = X[c].dropna().astype(str)
            s = s[s.isin(cats)]
            self.modes_[c] = s.mode().iloc[0] if len(s) else cats[0]

        names, kinds = [], []
        for c in self.numeric_cols:
            names.append(c)
            kinds.append("continuous")
        for c in self.binary_cols:
            names.append(f"{c}={self.cats_[c][1]}")
            kinds.append("binary")
        for c in self.nominal_cols:
            for k in self.cats_[c]:
                names.append(f"{c}={k}")
                kinds.append("binary")
        self.feature_names_ = names
        self.feature_kinds_ = np.array(kinds)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        cols = []
        for c in self.numeric_cols:
            v = X[c].to_numpy(dtype=float)
            if self.mode == "tree":
                cols.append(v)
                continue
            v = np.where(np.isnan(v), self.medians_[c], v)
            if self.mode == "linear":
                v = (v - self.means_[c]) / self.stds_[c]
            else:  # fuzzy
                v = np.clip((v - self.lo_[c]) / (self.hi_[c] - self.lo_[c]), 0.0, 1.0)
            cols.append(v)
        impute = self.mode != "tree"
        for c in self.binary_cols:
            s = X[c].astype(object)
            s = s.where(s.notna(), None)
            s = s.map(lambda t: None if t is None else str(t))
            if impute:
                s = s.map(lambda t: self.modes_[c] if (t is None or t not in self.cats_[c]) else t)
                cols.append((s == self.cats_[c][1]).to_numpy(dtype=float))
            else:
                v = np.where(s.isna(), np.nan, (s == self.cats_[c][1]).astype(float))
                cols.append(v.astype(float))
        for c in self.nominal_cols:
            s = X[c].astype(object)
            s = s.map(lambda t: None if (t is None or (isinstance(t, float) and np.isnan(t))) else str(t))
            if impute:
                s = s.map(lambda t: self.modes_[c] if (t is None or t not in self.cats_[c]) else t)
            for k in self.cats_[c]:
                cols.append((s == k).to_numpy(dtype=float))  # unknown/missing -> all zeros (tree)
        out = np.column_stack(cols) if cols else np.empty((len(X), 0))
        return out.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)
