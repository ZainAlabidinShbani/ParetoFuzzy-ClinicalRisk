"""Real-coded genome for the fuzzy risk score (Pittsburgh approach).

Layout of x (all genes stored as floats; integer genes are rounded on decode):

  [ MF genes      : 3 per continuous feature, in [0, 1]                 ]
  [ antecedents   : R_max * d integers, a_rj in {0 (don't care), 1..K_j} ]
  [ activity bits : R_max integers in {0, 1}                            ]
  [ consequents   : (R_max + 1) reals in [-W, W]  (only if evolved)      ]

Decoding rules (deterministic):
* MF genes -> ordered centres with minimum gap (``repair_centres``).
* A rule is *active* iff its activity bit is 1 and it has >= 1 antecedent.
* Duplicate antecedent sets among active rules are removed (first kept), so
  the reported rule count is the number of *distinct* active rules.
* Rules longer than ``L_max`` are truncated to their first ``L_max``
  conditions (operators never produce them; this is a safety net).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fis import FuzzyRuleBase, n_labels, repair_centres

W_BOUND = 4.0


@dataclass
class GenomeSpec:
    kinds: np.ndarray
    R_max: int = 12
    L_max: int = 4
    evolve_consequents: bool = False
    feature_names: list | None = None

    def __post_init__(self):
        self.kinds = np.asarray(self.kinds)
        self.d = len(self.kinds)
        self.cont = np.flatnonzero(self.kinds == "continuous")
        self.K = np.array([n_labels(k) for k in self.kinds])
        self.n_mf = 3 * len(self.cont)
        self.s_ante = self.n_mf
        self.s_act = self.s_ante + self.R_max * self.d
        self.s_cons = self.s_act + self.R_max
        self.n_var = self.s_cons + ((self.R_max + 1) if self.evolve_consequents else 0)

    # ------------------------------------------------------------------ #
    def bounds(self):
        xl = np.zeros(self.n_var)
        xu = np.ones(self.n_var)
        xu[self.s_ante:self.s_act] = np.tile(self.K, self.R_max)
        if self.evolve_consequents:
            xl[self.s_cons:] = -W_BOUND
            xu[self.s_cons:] = W_BOUND
        return xl, xu

    def ante_matrix(self, x) -> np.ndarray:
        A = np.rint(x[self.s_ante:self.s_act]).astype(int).reshape(self.R_max, self.d)
        return np.clip(A, 0, self.K[None, :])

    def active_bits(self, x) -> np.ndarray:
        return np.rint(x[self.s_act:self.s_cons]).astype(int).clip(0, 1)

    # ------------------------------------------------------------------ #
    def decode(self, x) -> tuple[FuzzyRuleBase, np.ndarray]:
        """Return the rule base and the genome slot index of each kept rule."""
        x = np.asarray(x, float)
        centres = {j: repair_centres(x[3 * i:3 * i + 3]) for i, j in enumerate(self.cont)}
        A = self.ante_matrix(x)
        act = self.active_bits(x)
        rules, slots, seen = [], [], set()
        for r in range(self.R_max):
            if not act[r]:
                continue
            conds = tuple((int(j), int(A[r, j])) for j in np.flatnonzero(A[r]))[: self.L_max]
            if not conds or conds in seen:
                continue
            seen.add(conds)
            rules.append(conds)
            slots.append(r)
        rb = FuzzyRuleBase(kinds=self.kinds, centres=centres, rules=rules,
                           feature_names=self.feature_names)
        slots = np.asarray(slots, int)
        if self.evolve_consequents:
            cons = x[self.s_cons:]
            rb.bias = float(cons[-1])
            rb.weights = cons[:-1][slots].astype(float) if len(slots) else np.zeros(0)
        return rb, slots

    # ------------------------------------------------------------------ #
    def random_rule(self, rng, max_len=3) -> np.ndarray:
        row = np.zeros(self.d, int)
        L = int(rng.integers(1, min(max_len, self.L_max, self.d) + 1))
        feats = rng.choice(self.d, size=L, replace=False)
        for j in feats:
            row[j] = int(rng.integers(1, self.K[j] + 1))
        return row

    def enforce_length(self, A, rng) -> np.ndarray:
        """Drop random conditions from rules longer than L_max (in place)."""
        for r in range(A.shape[0]):
            nz = np.flatnonzero(A[r])
            if len(nz) > self.L_max:
                drop = rng.choice(nz, size=len(nz) - self.L_max, replace=False)
                A[r, drop] = 0
        return A

    def write_rules(self, x, A, act) -> None:
        x[self.s_ante:self.s_act] = A.reshape(-1)
        x[self.s_act:self.s_cons] = act
