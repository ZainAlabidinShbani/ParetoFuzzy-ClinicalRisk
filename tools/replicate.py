"""Re-run stored outer-CV fits with the current code and compare them with their checkpoints.

A fit replicates if its test-fold rows, labels and predicted probabilities equal the stored ones
(max |dp| <= 1e-12) and its metrics and rule-base size agree. Used by tools/audit.py (check `replication`)
on a deterministic sample that covers every (git commit, dirty flag) group of checkpoints.

Usage: python tools/replicate.py s0__heart_cleveland__f3__LogReg [more checkpoint stems ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CKDIR = ROOT / "results" / "final" / "checkpoints"
METRICS = ("auroc", "pr_auc", "brier", "log_loss", "ece", "cal_slope", "cal_intercept", "n_rules", "complexity")


def replicate(stem: str, ckdir: Path = CKDIR) -> dict:
    from fuzzyrisk.data import REGISTRY, load_dataset
    from fuzzyrisk.evaluation import run_cv

    stored = json.loads((ckdir / f"{stem}.json").read_text(encoding="utf-8"))
    rec = stored["record"]
    cfg = {k: stored["config"][k] for k in ("pop_size", "n_gen", "n_iter", "R_max", "L_max")}
    heldout = REGISTRY[rec["dataset"]]["role"] == "heldout"
    ds = load_dataset(rec["dataset"], allow_heldout=heldout)
    got = []
    with threadpool_limits(limits=1):
        run_cv(ds, [rec["model"]], seed=rec["seed"], n_folds=stored["config"]["folds"], cfg=cfg,
               log=lambda m: None, skip=lambda d, f, m: f != rec["fold"], on_result=got.append)
    new = got[0]
    same_rows = new["pred"]["row"] == stored["pred"]["row"] and new["pred"]["y"] == stored["pred"]["y"]
    dp = float(np.max(np.abs(np.asarray(new["pred"]["p"]) - np.asarray(stored["pred"]["p"]))))
    # metrics: the new record must equal the metrics of the stored predictions under the current metrics
    # code (fit-time metrics of pre-D3 checkpoints used the old calibration-slope solver), plus the
    # stored structural fields (rules, literals)
    from fuzzyrisk.metrics import all_metrics
    ref = {**rec, **all_metrics(stored["pred"]["y"], stored["pred"]["p"])}
    dm = {m: (new["record"].get(m), ref.get(m)) for m in METRICS
          if not _eq(new["record"].get(m), ref.get(m))}
    return {"checkpoint": stem, "git_commit": stored.get("git_commit", "")[:10],
            "git_dirty": stored.get("git_dirty"), "same_rows": same_rows, "max_abs_dp": dp,
            "metric_mismatches": dm, "ok": bool(same_rows and dp <= 1e-12 and not dm)}


def _eq(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-12
    return a == b


def main(argv):
    out = [replicate(s) for s in argv]
    print(json.dumps(out, indent=1))
    return 0 if all(r["ok"] for r in out) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
