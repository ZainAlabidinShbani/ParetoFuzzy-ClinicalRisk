"""Fast 1-seed pilot on the DEVELOPMENT datasets only.

Usage:  python scripts/run_pilot.py --seed 0 [--pop 60 --gen 80 --n-iter 20 --folds 5]
Outputs (results/pilot/): records.csv, summary.csv, summary.md, predictions.csv,
fronts.json, rules.json, run_meta.json, pilot.log
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd
import psutil

from fuzzyrisk.data import dev_datasets, load_dataset
from fuzzyrisk.evaluation import run_cv

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["LogReg", "DecisionTree", "XGBoost", "LightGBM", "CatBoost", "MLP",
          "XGBoost+Platt", "XGBoost+Isotonic",
          "SO-GA-Fuzzy(classic)", "SO-GA-Fuzzy(hybrid)", "MO-Fuzzy(NSGA-II)"]
SUMMARY_COLS = ["auroc", "pr_auc", "brier", "log_loss", "ece", "ece_adaptive", "cal_slope",
                "cal_intercept", "n_rules", "mean_antecedents", "fit_s", "peak_rss_mb", "test_hv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "pilot.log", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = {"pop_size": args.pop, "n_gen": args.gen, "n_iter": args.n_iter, "R_max": 12, "L_max": 4}
    datasets = dev_datasets()  # held-out datasets are never touched by the pilot
    log(f"pilot seed={args.seed} folds={args.folds} cfg={cfg} datasets={datasets}")
    t0 = time.perf_counter()
    all_rec, all_pred, all_front, all_rules = [], [], [], []
    for name in datasets:
        ds = load_dataset(name)
        log(f"== {name}: n={len(ds.y)} prevalence={ds.prevalence:.3f} sha256={ds.sha256[:12]}")
        rec, pred, fronts, rules = run_cv(ds, args.models, seed=args.seed, n_folds=args.folds, cfg=cfg, log=log)
        all_rec += rec
        all_pred.append(pred)
        all_front += fronts
        all_rules += rules
    wall = time.perf_counter() - t0

    R = pd.DataFrame(all_rec)
    R.to_csv(out / "records.csv", index=False, encoding="utf-8")
    pd.concat(all_pred).to_csv(out / "predictions.csv", index=False, encoding="utf-8")
    (out / "fronts.json").write_text(json.dumps(all_front, indent=1), encoding="utf-8")
    (out / "rules.json").write_text(json.dumps(all_rules, indent=1, ensure_ascii=False), encoding="utf-8")

    cols = [c for c in SUMMARY_COLS if c in R.columns]
    S = R.groupby(["dataset", "model"], sort=False)[cols].agg(["mean", "std"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S = S.reset_index()
    S.to_csv(out / "summary.csv", index=False, encoding="utf-8")

    lines = []
    for dname, g in S.groupby("dataset", sort=False):
        lines += [f"\n### {dname} (seed {args.seed}, {args.folds}-fold outer CV; mean ± SD over folds)\n",
                  "| Model | AUROC | PR-AUC | Brier | ECE | Cal. slope | Rules | Ante/rule | Fit s | Peak RSS MB |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            def f(c, d=3):
                m, s = r.get(f"{c}_mean"), r.get(f"{c}_std")
                if m is None or pd.isna(m):
                    return "–"
                return f"{m:.{d}f} ± {s:.{d}f}"
            lines.append(f"| {r['model']} | {f('auroc')} | {f('pr_auc')} | {f('brier')} | {f('ece')} | "
                         f"{f('cal_slope', 2)} | {f('n_rules', 1)} | {f('mean_antecedents', 2)} | "
                         f"{f('fit_s', 1)} | {r['peak_rss_mb_mean']:.0f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT,
                                             text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    meta = {
        "seed": args.seed, "folds": args.folds, "config": cfg, "models": args.models, "datasets": datasets,
        "wall_time_s": round(wall, 1),
        "process_peak_working_set_mb": round(psutil.Process().memory_info().peak_wset / 2**20, 1)
        if hasattr(psutil.Process().memory_info(), "peak_wset") else None,
        "git_commit": commit, "git_dirty_src": dirty,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "xgboost",
                                             "lightgbm", "catboost", "pymoo"]},
        "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "model_n_jobs": 2},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"done in {wall / 60:.1f} min; peak working set {meta['process_peak_working_set_mb']} MB")
    log((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
