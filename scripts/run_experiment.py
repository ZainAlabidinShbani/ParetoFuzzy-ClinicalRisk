"""Preregistered experiment runner: seeds x datasets x outer folds x models, checkpointed.

Usage:
  python scripts/run_experiment.py --seeds 0 1 2 3 4 [--datasets ...] [--allow-heldout]
         [--pop 80 --gen 80 --n-iter 20 --folds 5] [--time-budget-min 8] [--hard-limit-min 9.3]
         [--aggregate-only] [--out results/final]

* Loop order: seed -> dataset -> fold -> model (complete seeds accumulate first).
* Every completed fit is written atomically to <out>/checkpoints/ with a config
  fingerprint; restarts skip completed fits and refuse to mix configurations.
* --time-budget-min: do not start a new fit after this many minutes (exit code 3).
* --hard-limit-min: never start a fit whose expected duration (max observed fit time
  for that dataset/model x 1.25) would end after this many minutes; an unseen
  (dataset, model) pair is only started at the beginning of a chunk.
* Memory hygiene: BLAS/OpenMP pools and all model thread counts = 1; gc.collect()
  after every fit and fold; per-fold peak RSS in fold_memory.csv.
* MO-Fuzzy fits also yield the preregistered best-Brier ablation record ("extra").
Outputs (--out): records.csv (incl. ablation rows), predictions.csv.gz, fronts.json,
rules.json, fold_memory.csv, sessions.json, run_meta.json, run.log.
Statistics, figures and tables: scripts/analyze.py.
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

import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from fuzzyrisk.baselines import N_JOBS_MODEL
from fuzzyrisk.data import REGISTRY, dev_datasets, heldout_datasets, load_dataset
from fuzzyrisk.evaluation import run_cv

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["LogReg", "DecisionTree", "XGBoost", "LightGBM", "CatBoost", "MLP",
          "XGBoost+Platt", "XGBoost+Isotonic",
          "SO-GA-Fuzzy(classic)", "SO-GA-Fuzzy(hybrid)", "MO-Fuzzy(NSGA-II)"]


class BudgetExceeded(Exception):
    pass


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def ckpt_path(ckdir: Path, seed: int, dataset: str, fold: int, model: str) -> Path:
    return ckdir / f"s{seed}__{dataset}__f{fold}__{slug(model)}.json"


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
def aggregate(out: Path, ckdir: Path, models, datasets, folds, seeds, log) -> bool:
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ckdir.glob("*.json"))]

    def key(r):
        return (r["seed"], r["dataset"], r["fold"], r["model"])

    expected = {(s, d, f, m) for s in seeds for d in datasets for f in range(folds) for m in models}
    have = {key(p["record"]) for p in payloads}
    missing = expected - have
    if missing:
        log(f"aggregate: {len(missing)}/{len(expected)} fits still missing; not aggregating.")
        return False
    payloads = [p for p in payloads if key(p["record"]) in expected]
    recs, preds = [], []
    for p in payloads:
        for item in [p] + p.get("extra", []):
            r = item["record"]
            recs.append(r)
            preds.append(pd.DataFrame({"dataset": r["dataset"], "seed": r["seed"], "fold": r["fold"],
                                       "model": r["model"], **item["pred"]}))
    R = pd.DataFrame(recs).sort_values(["seed", "dataset", "fold", "model"]).reset_index(drop=True)
    R.to_csv(out / "records.csv", index=False, encoding="utf-8")
    pd.concat(preds).to_csv(out / "predictions.csv.gz", index=False, encoding="utf-8", compression="gzip")
    (out / "fronts.json").write_text(json.dumps([p["front"] for p in payloads if p["front"]]),
                                     encoding="utf-8")
    (out / "rules.json").write_text(json.dumps([p["rules"] for p in payloads if p["rules"]], indent=1,
                                               ensure_ascii=False), encoding="utf-8")
    sessions = json.loads((out / "sessions.json").read_text(encoding="utf-8")) \
        if (out / "sessions.json").exists() else []
    pins = json.loads((ROOT / "src" / "fuzzyrisk" / "dataset_pins.json").read_text(encoding="utf-8"))
    meta = {
        "seeds": seeds, "folds": folds, "config": {k: v for k, v in payloads[0]["config"].items()
                                                   if k != "seed"},
        "models": models, "datasets": datasets, "n_fits": len(payloads), "n_records": len(R),
        "sum_fit_plus_predict_s": round(float((R.fit_s + R.predict_s).sum()), 1),
        "wall_time_s_all_sessions": round(sum(s["wall_s"] for s in sessions), 1),
        "n_sessions": len(sessions),
        "max_fit_peak_rss_mb": round(float(R.peak_rss_mb.max()), 1),
        "max_session_peak_working_set_mb": max((s.get("peak_working_set_mb") or 0) for s in sessions)
        if sessions else None,
        "git_commits_of_checkpoints": sorted({p.get("git_commit") for p in payloads}),
        "dataset_sha256": {d: pins.get(d) for d in datasets},
        "python": sys.version.split()[0], "platform": platform.platform(), "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "xgboost",
                                             "lightgbm", "catboost", "pymoo"]},
        "threads": {"BLAS/OpenMP": 1, "model_n_jobs": N_JOBS_MODEL, "search_n_jobs": 1},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"aggregated {len(payloads)} fits ({len(R)} records) -> {out}")
    return True


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--allow-heldout", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--hard-limit-min", type=float, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "final"))
    args = ap.parse_args()
    out = Path(args.out)
    ckdir = out / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    datasets = args.datasets or (dev_datasets() + (heldout_datasets() if args.allow_heldout else []))
    cfg = {"pop_size": args.pop, "n_gen": args.gen, "n_iter": args.n_iter, "R_max": 12, "L_max": 4}
    if args.aggregate_only:
        return 0 if aggregate(out, ckdir, args.models, datasets, args.folds, args.seeds, log) else 4

    commit, dirty = git_commit()
    t_start = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log(f"\n=== session {stamp} seeds={args.seeds} datasets={datasets} cfg={cfg} "
        f"commit={commit[:8]}{'+dirty' if dirty else ''} budget={args.time_budget_min} "
        f"hard={args.hard_limit_min}")
    est = {}  # expected fit duration per (dataset, model), from checkpoints
    for p in ckdir.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))["record"]
        est[(r["dataset"], r["model"])] = max(est.get((r["dataset"], r["model"]), 0), r["fit_s"])
    n_run = 0
    fps = {}

    def fingerprint(seed):
        if seed not in fps:
            rc = {"seed": seed, "folds": args.folds, **cfg}
            fps[seed] = (hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()[:16], rc)
        return fps[seed]

    def make_skip(seed):
        def skip(dataset, fold, model):
            p = ckpt_path(ckdir, seed, dataset, fold, model)
            if p.exists():
                fp = json.loads(p.read_text(encoding="utf-8")).get("fingerprint")
                if fp != fingerprint(seed)[0]:
                    raise SystemExit(f"checkpoint {p.name} fingerprint mismatch; refusing to mix configs")
                return True
            el = (time.perf_counter() - t_start) / 60
            if args.time_budget_min is not None and el > args.time_budget_min:
                raise BudgetExceeded
            if args.hard_limit_min is not None:
                e = est.get((dataset, model))
                if e is None and el > 0.3:
                    raise BudgetExceeded
                if e is not None and el + 1.25 * e / 60 > args.hard_limit_min:
                    raise BudgetExceeded
            return False
        return skip

    def make_on_result(seed):
        def on_result(pl):
            nonlocal n_run
            r = pl["record"]
            fp, rc = fingerprint(seed)
            pl.update(fingerprint=fp, config=rc, git_commit=commit, git_dirty=dirty,
                      finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            atomic_write_json(ckpt_path(ckdir, seed, r["dataset"], r["fold"], r["model"]), pl)
            est[(r["dataset"], r["model"])] = max(est.get((r["dataset"], r["model"]), 0), r["fit_s"])
            n_run += 1
            gc.collect()
        return on_result

    def make_fold_end(seed, dname):
        def on_fold_end(fold, payloads):
            gc.collect()
            if not payloads:
                return
            rss_after = psutil.Process().memory_info().rss / 2**20
            peak = max(p["record"]["peak_rss_mb"] for p in payloads)
            row = {"seed": seed, "dataset": dname, "fold": fold, "n_fits": len(payloads),
                   "fold_peak_rss_mb": round(peak, 1), "rss_after_gc_mb": round(rss_after, 1),
                   "fold_fit_s": round(sum(p["record"]["fit_s"] for p in payloads), 1),
                   "system_available_mb": round(psutil.virtual_memory().available / 2**20)}
            mf = out / "fold_memory.csv"
            pd.DataFrame([row]).to_csv(mf, mode="a", header=not mf.exists(), index=False, encoding="utf-8")
            log(f"  -- s{seed} {dname} fold {fold}: peak RSS {peak:.0f} MB, after gc {rss_after:.0f} MB, "
                f"avail {row['system_available_mb']} MB")
        return on_fold_end

    status = "complete"
    try:
        with threadpool_limits(limits=1):
            for seed in args.seeds:
                for name in datasets:
                    if all(ckpt_path(ckdir, seed, name, f, m).exists()
                           for f in range(args.folds) for m in args.models):
                        continue
                    heldout = REGISTRY[name]["role"] == "heldout"
                    ds = load_dataset(name, allow_heldout=heldout and args.allow_heldout,
                                      pin_if_missing=heldout)
                    run_cv(ds, args.models, seed=seed, n_folds=args.folds, cfg=cfg,
                           log=lambda m, s=seed: log(f"s{s} {m.strip()}"),
                           skip=make_skip(seed), on_result=make_on_result(seed),
                           on_fold_end=make_fold_end(seed, name))
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
    total = len(args.seeds) * len(datasets) * args.folds * len(args.models)
    log(f"session end: {status}; ran {n_run} fits in {wall / 60:.1f} min; {done}/{total} checkpoints; "
        f"peak WS {sess['peak_working_set_mb']} MB")
    if status == "budget":
        return 3
    aggregate(out, ckdir, args.models, datasets, args.folds, args.seeds, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
