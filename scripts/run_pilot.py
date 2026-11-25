"""Fast 1-seed pilot on the DEVELOPMENT datasets only, with per-fit checkpointing.

Usage:
  python scripts/run_pilot.py --seed 0 [--pop 60 --gen 80 --n-iter 20 --folds 5]
                              [--time-budget-min 8] [--aggregate-only]

* Every completed (dataset, fold, model) fit is written immediately and atomically to
  results/pilot/checkpoints/<dataset>__f<fold>__<model>.json. A restart skips completed
  fits; checkpoints made with a different configuration abort the run instead of mixing.
* --time-budget-min: stop cleanly (exit code 3) before starting a new fit once the
  budget is used, so long runs can be done in resumable chunks.
* Memory hygiene: all BLAS/OpenMP pools and model thread counts are 1; gc.collect()
  after every fit and every fold; peak RSS logged per fold (fold_memory.csv).
Outputs (results/pilot/): records.csv, summary.csv, summary.md, predictions.csv,
fronts.json, rules.json, fold_memory.csv, sessions.json, run_meta.json, pilot.log
"""
from __future__ import annotations

# --- single-threaded numerics: must be set before numpy / sklearn are imported ---
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import gc
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from fuzzyrisk.baselines import N_JOBS_MODEL
from fuzzyrisk.data import dev_datasets, load_dataset
from fuzzyrisk.evaluation import run_cv

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["LogReg", "DecisionTree", "XGBoost", "LightGBM", "CatBoost", "MLP",
          "XGBoost+Platt", "XGBoost+Isotonic",
          "SO-GA-Fuzzy(classic)", "SO-GA-Fuzzy(hybrid)", "MO-Fuzzy(NSGA-II)"]
SUMMARY_COLS = ["auroc", "pr_auc", "brier", "log_loss", "ece", "ece_adaptive", "cal_slope",
                "cal_intercept", "n_rules", "mean_antecedents", "complexity", "fit_s",
                "peak_rss_mb", "test_hv", "inner_hv"]


class BudgetExceeded(Exception):
    pass


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def ckpt_path(ckdir: Path, dataset: str, fold: int, model: str) -> Path:
    return ckdir / f"{dataset}__f{fold}__{slug(model)}.json"


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def git_commit():
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        d = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT,
                                         text=True).strip())
        return c, d
    except Exception:
        return "unknown", None


# --------------------------------------------------------------------------- #
def aggregate(out: Path, ckdir: Path, models, datasets, folds, seed, log) -> bool:
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ckdir.glob("*.json"))]
    expected = {(d, f, m) for d in datasets for f in range(folds) for m in models}
    have = {(p["record"]["dataset"], p["record"]["fold"], p["record"]["model"]) for p in payloads}
    missing = expected - have
    if missing:
        log(f"aggregate: {len(missing)}/{len(expected)} fits still missing; not aggregating.")
        return False
    payloads = [p for p in payloads if (p["record"]["dataset"], p["record"]["fold"], p["record"]["model"])
                in expected]
    R = pd.DataFrame([p["record"] for p in payloads])
    order = {m: i for i, m in enumerate(models)}
    dorder = {d: i for i, d in enumerate(datasets)}
    R = R.sort_values(["dataset", "fold", "model"], key=lambda s: s.map(dorder) if s.name == "dataset"
                      else (s.map(order) if s.name == "model" else s)).reset_index(drop=True)
    R.to_csv(out / "records.csv", index=False, encoding="utf-8")
    P = pd.concat([pd.DataFrame({"dataset": p["record"]["dataset"], "seed": seed, "fold": p["record"]["fold"],
                                 "model": p["record"]["model"], **p["pred"]}) for p in payloads])
    P.to_csv(out / "predictions.csv", index=False, encoding="utf-8")
    (out / "fronts.json").write_text(json.dumps([p["front"] for p in payloads if p["front"]], indent=1),
                                     encoding="utf-8")
    (out / "rules.json").write_text(json.dumps([p["rules"] for p in payloads if p["rules"]], indent=1,
                                               ensure_ascii=False), encoding="utf-8")

    cols = [c for c in SUMMARY_COLS if c in R.columns]
    S = R.groupby(["dataset", "model"], sort=False)[cols].agg(["mean", "std"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S = S.reset_index()
    S["dataset"] = pd.Categorical(S["dataset"], datasets, ordered=True)
    S["model"] = pd.Categorical(S["model"], models, ordered=True)
    S = S.sort_values(["dataset", "model"]).reset_index(drop=True)
    S.to_csv(out / "summary.csv", index=False, encoding="utf-8")

    def f(r, c, d=3):
        m, s = r.get(f"{c}_mean"), r.get(f"{c}_std")
        if m is None or pd.isna(m):
            return "–"
        return f"{m:.{d}f} ± {s:.{d}f}"

    lines = []
    for dname, g in S.groupby("dataset", sort=False, observed=True):
        lines += [f"\n### {dname} (seed {seed}, {folds}-fold outer CV; mean ± SD over folds)\n",
                  "| Model | AUROC | PR-AUC | Brier | ECE | Cal. slope | Rules | Ante/rule | Fit s | Peak RSS MB |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r['model']} | {f(r, 'auroc')} | {f(r, 'pr_auc')} | {f(r, 'brier')} | "
                         f"{f(r, 'ece')} | {f(r, 'cal_slope', 2)} | {f(r, 'n_rules', 1)} | "
                         f"{f(r, 'mean_antecedents', 2)} | {f(r, 'fit_s', 1)} | "
                         f"{R[(R.dataset == dname) & (R.model == r['model'])].peak_rss_mb.max():.0f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sessions = json.loads((out / "sessions.json").read_text(encoding="utf-8")) \
        if (out / "sessions.json").exists() else []
    commits = sorted({p.get("git_commit") for p in payloads})
    meta = {
        "seed": seed, "folds": folds, "config": payloads[0]["config"], "models": models, "datasets": datasets,
        "n_fits": len(payloads),
        "sum_fit_plus_predict_s": round(float((R.fit_s + R.predict_s).sum()), 1),
        "wall_time_s_all_sessions": round(sum(s["wall_s"] for s in sessions), 1),
        "n_sessions": len(sessions),
        "max_fit_peak_rss_mb": round(float(R.peak_rss_mb.max()), 1),
        "max_session_peak_working_set_mb": max((s.get("peak_working_set_mb") or 0) for s in sessions)
        if sessions else None,
        "git_commits_of_checkpoints": commits,
        "python": sys.version.split()[0], "platform": platform.platform(), "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "xgboost",
                                             "lightgbm", "catboost", "pymoo"]},
        "threads": {"BLAS/OpenMP": 1, "model_n_jobs": N_JOBS_MODEL, "search_n_jobs": 1},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"aggregated {len(payloads)} fits -> {out}")
    log((out / "summary.md").read_text(encoding="utf-8"))
    return True


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot"))
    args = ap.parse_args()
    out = Path(args.out)
    ckdir = out / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    logf = open(out / "pilot.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = {"pop_size": args.pop, "n_gen": args.gen, "n_iter": args.n_iter, "R_max": 12, "L_max": 4}
    run_cfg = {"seed": args.seed, "folds": args.folds, **cfg}
    fingerprint = hashlib.sha256(json.dumps(run_cfg, sort_keys=True).encode()).hexdigest()[:16]
    datasets = dev_datasets()  # held-out datasets are never touched by the pilot
    if args.aggregate_only:
        aggregate(out, ckdir, args.models, datasets, args.folds, args.seed, log)
        return 0

    commit, dirty = git_commit()
    t_start = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log(f"\n=== session {stamp} seed={args.seed} folds={args.folds} cfg={cfg} fp={fingerprint} "
        f"commit={commit[:8]}{'+dirty' if dirty else ''} budget={args.time_budget_min} min")
    n_run = 0

    def skip(dataset, fold, model):
        p = ckpt_path(ckdir, dataset, fold, model)
        if p.exists():
            fp = json.loads(p.read_text(encoding="utf-8")).get("fingerprint")
            if fp != fingerprint:
                raise SystemExit(f"checkpoint {p.name} has fingerprint {fp} != {fingerprint}; "
                                 "use a different --out or remove stale checkpoints")
            return True
        if args.time_budget_min is not None and (time.perf_counter() - t_start) / 60 > args.time_budget_min:
            raise BudgetExceeded
        return False

    def on_result(pl):
        nonlocal n_run
        r = pl["record"]
        pl.update(fingerprint=fingerprint, config=run_cfg, git_commit=commit, git_dirty=dirty,
                  finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        atomic_write_json(ckpt_path(ckdir, r["dataset"], r["fold"], r["model"]), pl)
        n_run += 1
        gc.collect()

    def make_fold_end(dname):
        def on_fold_end(fold, payloads):
            gc.collect()
            if not payloads:
                return
            rss_after = psutil.Process().memory_info().rss / 2**20
            peak = max(p["record"]["peak_rss_mb"] for p in payloads)
            row = {"dataset": dname, "fold": fold, "n_fits": len(payloads), "fold_peak_rss_mb": round(peak, 1),
                   "rss_after_gc_mb": round(rss_after, 1),
                   "fold_fit_s": round(sum(p["record"]["fit_s"] for p in payloads), 1),
                   "system_available_mb": round(psutil.virtual_memory().available / 2**20)}
            mf = out / "fold_memory.csv"
            pd.DataFrame([row]).to_csv(mf, mode="a", header=not mf.exists(), index=False, encoding="utf-8")
            log(f"  -- fold {fold} done: peak RSS {peak:.0f} MB, RSS after gc {rss_after:.0f} MB, "
                f"system available {row['system_available_mb']} MB")
        return on_fold_end

    status = "complete"
    try:
        with threadpool_limits(limits=1):
            for name in datasets:
                ds = load_dataset(name)
                log(f"== {name}: n={len(ds.y)} prevalence={ds.prevalence:.3f} sha256={ds.sha256[:12]}")
                run_cv(ds, args.models, seed=args.seed, n_folds=args.folds, cfg=cfg, log=log,
                       skip=skip, on_result=on_result, on_fold_end=make_fold_end(name))
                del ds
                gc.collect()
    except BudgetExceeded:
        status = "budget"
    wall = time.perf_counter() - t_start
    mi = psutil.Process().memory_info()
    sess = {"start_utc": stamp, "wall_s": round(wall, 1), "fits_run": n_run, "status": status,
            "peak_working_set_mb": round(mi.peak_wset / 2**20, 1) if hasattr(mi, "peak_wset") else None,
            "git_commit": commit, "git_dirty": dirty}
    sf = out / "sessions.json"
    sessions = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else []
    atomic_write_json(sf, sessions + [sess])
    done = len(list(ckdir.glob("*.json")))
    log(f"session end: {status}; ran {n_run} fits in {wall / 60:.1f} min; {done} checkpoints on disk; "
        f"process peak working set {sess['peak_working_set_mb']} MB")
    if status == "budget":
        return 3
    aggregate(out, ckdir, args.models, datasets, args.folds, args.seed, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
