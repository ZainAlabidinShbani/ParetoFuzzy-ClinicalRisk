"""NSGA-II (multi-objective) and GA (single-objective) search over fuzzy risk scores.

Fitness is computed **only on the training data passed to the optimiser**
(the outer training fold). With hybrid consequents, each genome's objectives
are measured on pooled out-of-fold predictions of an inner stratified K-fold:
consequents are fitted on K-1 inner folds and scored on the held-back inner
fold, which keeps the discrimination/calibration objectives honest without ever
touching the outer test fold.
"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from .encoding import W_BOUND, GenomeSpec
from .metrics import auroc, brier, ece, fit_logistic_l2, hypervolume, log_loss, non_dominated_mask

# scale used to normalise each objective to ~[0, 1] for hypervolume (ref point = 1)
OBJ_SCALE = {"auroc": 0.5, "brier": 0.25, "ece": 0.5, "logloss": float(np.log(2)), "accuracy": 1.0}


# --------------------------------------------------------------------------- #
LAMBDAS = (0.01, 0.1, 1.0, 10.0)


class FitnessEvaluator:
    """Objectives of a genome from inner-CV out-of-fold predictions (training rows only).

    With hybrid consequents, the L2 penalty of the logistic consequent fit is selected
    per genome from ``lambdas`` by minimum pooled inner-OOF log-loss (a proper scoring
    rule); the objectives are then computed from that lambda's OOF predictions.
    """

    def __init__(self, spec: GenomeSpec, X, y, objectives=("auroc", "brier", "complexity"),
                 inner_folds: int = 3, lambdas=LAMBDAS, seed: int = 0):
        self.spec = spec
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, int)
        self.objectives = tuple(objectives)
        self.lambdas = tuple(sorted(np.atleast_1d(lambdas).astype(float), reverse=True))
        self.n_evals = 0
        if not spec.evolve_consequents:
            skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
            self.folds = list(skf.split(self.X, self.y))

    def predict_oof(self, x):
        """Returns (oof probabilities, rule base, selected lambda or nan)."""
        rb, _ = self.spec.decode(x)
        if self.spec.evolve_consequents:
            return rb.predict_proba(self.X)[:, 1], rb, float("nan")
        y = self.y
        if rb.n_rules == 0:
            p = np.empty(len(y))
            for tr, va in self.folds:
                p[va] = y[tr].mean()
            return p, rb, float("nan")
        B = rb.firing(self.X)
        best, warm = None, [None] * len(self.folds)
        for lam in self.lambdas:  # descending: warm-start along the regularisation path
            p = np.empty(len(y))
            for k, (tr, va) in enumerate(self.folds):
                b0, w = fit_logistic_l2(B[tr], y[tr], lam=lam, w0=warm[k])
                warm[k] = np.concatenate([[b0], w])
                z = np.clip(b0 + B[va] @ w, -35, 35)
                p[va] = 1.0 / (1.0 + np.exp(-z))
            ll = log_loss(y, p)
            if best is None or ll < best[0] - 1e-12:
                best = (ll, p, lam)
        return best[1], rb, best[2]

    def metrics(self, x) -> dict:
        self.n_evals += 1
        p, rb, lam = self.predict_oof(x)
        y = self.y
        a = auroc(y, p)
        return {
            "auroc": 0.5 if np.isnan(a) else a,
            "brier": brier(y, p),
            "ece": ece(y, p),
            "logloss": log_loss(y, p),
            "accuracy": float(np.mean((p >= 0.5) == y)),
            "complexity": rb.complexity(),          # total literals (integer)
            "n_rules": rb.n_rules,
            "mean_antecedents": rb.mean_antecedents,
            "lam": lam,
        }

    def objective_vector(self, m: dict) -> np.ndarray:
        out = []
        for o in self.objectives:
            if o in ("auroc", "accuracy"):
                out.append(1.0 - m[o])
            else:
                out.append(m[o])
        return np.asarray(out, float)


class FuzzyProblem(Problem):
    def __init__(self, evaluator: FitnessEvaluator):
        self.evaluator = evaluator
        xl, xu = evaluator.spec.bounds()
        super().__init__(n_var=evaluator.spec.n_var, n_obj=len(evaluator.objectives), xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.evaluator.objective_vector(self.evaluator.metrics(x)) for x in X])


# --------------------------------------------------------------------------- #
# Variation operators (all randomness from pymoo's seeded Generator)
# --------------------------------------------------------------------------- #
class FuzzySampling(Sampling):
    def __init__(self, spec: GenomeSpec, mf_init: np.ndarray):
        super().__init__()
        self.spec = spec
        self.mf_init = np.asarray(mf_init, float)

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        s, rng = self.spec, random_state
        X = np.zeros((n_samples, s.n_var))
        for i in range(n_samples):
            mf = self.mf_init.copy().reshape(-1)
            if i > 0:
                mf = mf + rng.normal(0.0, 0.08, size=mf.shape)
            X[i, : s.n_mf] = np.clip(mf, 0.0, 1.0)
            A = np.stack([s.random_rule(rng) for _ in range(s.R_max)])
            p_active = rng.uniform(0.15, 1.0)  # spread initial rule-base sizes
            act = (rng.random(s.R_max) < p_active).astype(int)
            if act.sum() == 0:
                act[rng.integers(s.R_max)] = 1
            s.write_rules(X[i], A, act)
            if s.evolve_consequents:
                X[i, s.s_cons:] = np.clip(rng.normal(0.0, 1.0, s.R_max + 1), -W_BOUND, W_BOUND)
        return X


class FuzzyCrossover(Crossover):
    """BLX-alpha on MF centres/consequent bias; uniform rule-slot exchange on rules."""

    def __init__(self, spec: GenomeSpec, alpha: float = 0.3, prob: float = 0.9):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.spec = spec
        self.alpha = alpha

    def _blx(self, a, b, rng, lo, hi):
        mn, mx = np.minimum(a, b), np.maximum(a, b)
        d = mx - mn
        return np.clip(rng.uniform(mn - self.alpha * d, mx + self.alpha * d), lo, hi)

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        s, rng = self.spec, random_state
        _, n_mat, _ = X.shape
        Y = np.empty_like(X)
        for k in range(n_mat):
            a, b = X[0, k].copy(), X[1, k].copy()
            c1, c2 = a.copy(), b.copy()
            if s.n_mf:
                c1[: s.n_mf] = self._blx(a[: s.n_mf], b[: s.n_mf], rng, 0.0, 1.0)
                c2[: s.n_mf] = self._blx(a[: s.n_mf], b[: s.n_mf], rng, 0.0, 1.0)
            Aa, Ab = s.ante_matrix(a), s.ante_matrix(b)
            acta, actb = s.active_bits(a), s.active_bits(b)
            swap = rng.random(s.R_max) < 0.5
            A1, A2 = np.where(swap[:, None], Ab, Aa), np.where(swap[:, None], Aa, Ab)
            t1, t2 = np.where(swap, actb, acta), np.where(swap, acta, actb)
            s.write_rules(c1, A1, t1)
            s.write_rules(c2, A2, t2)
            if s.evolve_consequents:
                wa, wb = a[s.s_cons:], b[s.s_cons:]
                c1[s.s_cons:-1] = np.where(swap, wb[:-1], wa[:-1])
                c2[s.s_cons:-1] = np.where(swap, wa[:-1], wb[:-1])
                c1[-1] = self._blx(wa[-1:], wb[-1:], rng, -W_BOUND, W_BOUND)[0]
                c2[-1] = self._blx(wa[-1:], wb[-1:], rng, -W_BOUND, W_BOUND)[0]
            Y[0, k], Y[1, k] = c1, c2
        return Y


class FuzzyMutation(Mutation):
    """Gaussian MF jitter + structural rule edits (add/drop/relabel/toggle/new)."""

    OPS = ("add", "drop", "relabel", "toggle", "new")
    OP_P = (0.2, 0.25, 0.2, 0.2, 0.15)

    def __init__(self, spec: GenomeSpec, sigma_mf: float = 0.05, sigma_w: float = 0.5):
        super().__init__(prob=1.0)
        self.spec = spec
        self.sigma_mf = sigma_mf
        self.sigma_w = sigma_w

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        s, rng = self.spec, random_state
        X = X.copy()
        for i in range(X.shape[0]):
            x = X[i]
            if s.n_mf:
                pm = min(0.5, 2.0 / s.n_mf)
                m = rng.random(s.n_mf) < pm
                x[: s.n_mf] = np.clip(x[: s.n_mf] + m * rng.normal(0, self.sigma_mf, s.n_mf), 0, 1)
            A, act = s.ante_matrix(x), s.active_bits(x)
            for _ in range(1 + int(rng.random() < 0.3)):
                op = self.OPS[rng.choice(len(self.OPS), p=self.OP_P)]
                live = np.flatnonzero(act)
                r = int(rng.choice(live)) if (len(live) and op != "toggle") else int(rng.integers(s.R_max))
                nz = np.flatnonzero(A[r])
                if op == "add" and len(nz) < min(s.L_max, s.d):
                    j = int(rng.choice(np.flatnonzero(A[r] == 0)))
                    A[r, j] = int(rng.integers(1, s.K[j] + 1))
                elif op == "drop" and len(nz) > 0:
                    A[r, int(rng.choice(nz))] = 0
                elif op == "relabel" and len(nz) > 0:
                    j = int(rng.choice(nz))
                    if s.K[j] > 1:
                        choices = [v for v in range(1, s.K[j] + 1) if v != A[r, j]]
                        A[r, j] = int(rng.choice(choices))
                elif op == "toggle":
                    act[r] = 1 - act[r]
                elif op == "new":
                    A[r] = s.random_rule(rng)
                    act[r] = 1
            s.enforce_length(A, rng)
            s.write_rules(x, A, act)
            if s.evolve_consequents:
                nw = s.R_max + 1
                m = rng.random(nw) < 1.0 / nw * 2
                x[s.s_cons:] = np.clip(x[s.s_cons:] + m * rng.normal(0, self.sigma_w, nw), -W_BOUND, W_BOUND)
        return X


# --------------------------------------------------------------------------- #
class ArchiveCallback(Callback):
    """External archive of all non-dominated solutions seen during the run
    (deduplicated on objective vectors), plus normalised HV of the archive per generation."""

    def __init__(self, scales, ref=1.0):
        super().__init__()
        self.scales = np.asarray(scales, float)
        self.ref = ref
        self.X = None
        self.F = None
        self.hv = []

    def notify(self, algorithm):
        X, F = algorithm.pop.get("X"), algorithm.pop.get("F")
        if self.X is not None:
            X, F = np.vstack([self.X, X]), np.vstack([self.F, F])
        nd = non_dominated_mask(F)
        X, F = X[nd], F[nd]
        _, keep = np.unique(np.round(F, 12), axis=0, return_index=True)  # one genome per objective vector
        keep = np.sort(keep)
        self.X, self.F = X[keep], F[keep]
        Fn = self.F / self.scales
        self.hv.append(hypervolume(Fn, np.full(Fn.shape[1], self.ref)))


def objective_scales(objectives, spec: GenomeSpec):
    """Normalisation for hypervolume; complexity (total literals) scaled by R_max * L_max."""
    return [OBJ_SCALE.get(o, spec.R_max * spec.L_max) for o in objectives]


def run_nsga2(evaluator: FitnessEvaluator, mf_init, pop_size=80, n_gen=80, seed=0):
    spec = evaluator.spec
    problem = FuzzyProblem(evaluator)
    algo = NSGA2(pop_size=pop_size, sampling=FuzzySampling(spec, mf_init),
                 crossover=FuzzyCrossover(spec), mutation=FuzzyMutation(spec),
                 eliminate_duplicates=True)
    cb = ArchiveCallback(objective_scales(evaluator.objectives, spec))
    minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False, callback=cb)
    return {"X": cb.X, "F": cb.F, "hv_history": cb.hv, "n_evals": evaluator.n_evals}


def run_ga(evaluator: FitnessEvaluator, mf_init, pop_size=80, n_gen=80, seed=0):
    spec = evaluator.spec
    assert len(evaluator.objectives) == 1
    problem = FuzzyProblem(evaluator)
    algo = GA(pop_size=pop_size, sampling=FuzzySampling(spec, mf_init),
              crossover=FuzzyCrossover(spec), mutation=FuzzyMutation(spec),
              eliminate_duplicates=True)
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    X, F = res.pop.get("X"), res.pop.get("F")
    best = int(np.argmin(F[:, 0]))
    return {"X": X[[best]], "F": F[[best]], "n_evals": evaluator.n_evals}


def select_eps_compact(front_metrics: list[dict], eps: float = 0.01) -> int:
    """Preregistered headline operating point.

    Among archive members whose inner-OOF AUROC is within ``eps`` of the best, choose
    the one with the fewest total literals; ties broken by lower inner-OOF Brier.
    """
    aucs = np.array([m["auroc"] for m in front_metrics])
    ok = np.flatnonzero(aucs >= aucs.max() - eps)
    key = sorted(ok, key=lambda i: (front_metrics[i]["complexity"], front_metrics[i]["brier"], i))
    return int(key[0])


def select_best_brier(front_metrics: list[dict]) -> int:
    """Secondary (ablation) operating point: lowest inner-OOF Brier; ties -> fewer literals."""
    return int(min(range(len(front_metrics)),
                   key=lambda i: (front_metrics[i]["brier"], front_metrics[i]["complexity"], i)))
