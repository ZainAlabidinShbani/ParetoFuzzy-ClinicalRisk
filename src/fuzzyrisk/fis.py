"""Zero-order Takagi-Sugeno fuzzy *risk score*.

Universe of discourse: every input is in [0, 1] (see ``TabularPreprocessor``
in ``fuzzy`` mode).

Linguistic partitions
---------------------
* continuous feature j: three triangular sets {Low, Medium, High} forming a
  *strong* (Ruspini) partition with shoulders, defined by ordered centres
  c1 < c2 < c3 (minimum gap ``MIN_GAP``). Memberships sum to one everywhere.
* binary feature j: two crisp sets {No, Yes}: mu_No = 1 - x, mu_Yes = x.

Rules and output
----------------
Rule r: IF x_j1 is A_r,j1 AND ... THEN  w_r            (product t-norm)
Score : z(x) = w_0 + sum_r  beta_r(x) * w_r,   p(x) = sigmoid(z(x))

Each rule adds at most w_r log-odds when fully fired, which reads as a fuzzy
point-based clinical risk score. Consequents are either estimated by
L2-penalised logistic regression given the antecedents (hybrid learning, as in
ANFIS-style hybrid schemes) or evolved directly by the GA (classic GA-fuzzy).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import fit_logistic_l2

MIN_GAP = 0.05
CONT_LABELS = ("Low", "Medium", "High")
BIN_LABELS = ("No", "Yes")


def repair_centres(g) -> np.ndarray:
    """Map 3 raw genes in [0,1] to ordered centres with gaps >= MIN_GAP."""
    c = np.sort(np.clip(np.asarray(g, float), 0.0, 1.0))
    c[0] = min(c[0], 1.0 - 2 * MIN_GAP)
    c[1] = np.clip(c[1], c[0] + MIN_GAP, 1.0 - MIN_GAP)
    c[2] = np.clip(c[2], c[1] + MIN_GAP, 1.0)
    return c


def triangular_partition(x, c) -> np.ndarray:
    """Strong 3-set triangular partition with shoulders. Returns (n, 3)."""
    x = np.asarray(x, float)
    c1, c2, c3 = c
    low = np.clip((c2 - x) / (c2 - c1), 0.0, 1.0)
    high = np.clip((x - c2) / (c3 - c2), 0.0, 1.0)
    med = 1.0 - low - high
    return np.stack([low, np.clip(med, 0.0, 1.0), high], axis=1)


def binary_partition(x) -> np.ndarray:
    x = np.asarray(x, float)
    return np.stack([1.0 - x, x], axis=1)


def n_labels(kind: str) -> int:
    return 3 if kind == "continuous" else 2


@dataclass
class FuzzyRuleBase:
    """Decoded knowledge base (data base + rule base + consequents)."""

    kinds: np.ndarray                       # (d,) 'continuous' | 'binary'
    centres: dict                            # j -> (3,) for continuous features
    rules: list                              # list of tuples ((j, label_idx), ...), label_idx >= 1
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias: float = 0.0
    feature_names: list | None = None

    # ------------------------------------------------------------------ #
    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def mean_antecedents(self) -> float:
        return float(np.mean([len(r) for r in self.rules])) if self.rules else 0.0

    @property
    def total_conditions(self) -> int:
        return int(sum(len(r) for r in self.rules))

    @property
    def total_literals(self) -> int:
        return self.total_conditions

    def complexity(self) -> int:
        """Interpretability objective f3 = total literals (antecedent conditions summed over
        distinct active rules). Integer-valued, which limits front saturation."""
        return self.total_conditions

    # ------------------------------------------------------------------ #
    def memberships(self, X) -> dict:
        X = np.asarray(X, float)
        used = sorted({j for r in self.rules for j, _ in r})
        mu = {}
        for j in used:
            if self.kinds[j] == "continuous":
                mu[j] = triangular_partition(X[:, j], self.centres[j])
            else:
                mu[j] = binary_partition(X[:, j])
        return mu

    def firing(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        B = np.ones((X.shape[0], len(self.rules)))
        mu = self.memberships(X)
        for r, rule in enumerate(self.rules):
            for j, lab in rule:
                B[:, r] *= mu[j][:, lab - 1]
        return B

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        if not self.rules:  # intercept-only model
            return np.full(X.shape[0], float(self.bias))
        return self.bias + self.firing(X) @ self.weights

    def predict_proba(self, X) -> np.ndarray:
        z = np.clip(self.decision_function(X), -35, 35)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def fit_consequents(self, X, y, lam: float = 1.0) -> "FuzzyRuleBase":
        B = self.firing(X)
        self.bias, self.weights = fit_logistic_l2(B, y, lam=lam)
        return self

    # ------------------------------------------------------------------ #
    def label_name(self, j: int, lab: int) -> str:
        return (CONT_LABELS if self.kinds[j] == "continuous" else BIN_LABELS)[lab - 1]

    def describe(self, preprocessor=None) -> list[str]:
        """Human-readable rule list; MF centres shown in original units if a
        fitted ``TabularPreprocessor`` (fuzzy mode) is supplied."""
        names = self.feature_names or [f"x{j}" for j in range(len(self.kinds))]
        lines = [f"BASELINE log-odds = {self.bias:+.3f}"]
        for r, rule in enumerate(self.rules):
            conds = " AND ".join(f"{names[j]} is {self.label_name(j, lab)}" for j, lab in rule)
            w = self.weights[r] if len(self.weights) else float("nan")
            lines.append(f"R{r + 1}: IF {conds} THEN {w:+.3f} log-odds")
        used = sorted({j for r in self.rules for j, _ in r if self.kinds[j] == "continuous"})
        for j in used:
            c = self.centres[j]
            if preprocessor is not None and names[j] in getattr(preprocessor, "lo_", {}):
                lo, hi = preprocessor.lo_[names[j]], preprocessor.hi_[names[j]]
                c = lo + np.asarray(c) * (hi - lo)
            lines.append(f"MF {names[j]}: Low/Medium/High centres = " + ", ".join(f"{v:.3g}" for v in c))
        return lines
