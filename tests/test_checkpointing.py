"""run_cv hooks used for per-fit checkpointing and resume."""
import json

from conftest import FAST_CFG
from fuzzyrisk.evaluation import run_cv

MODELS = ["LogReg", "MO-Fuzzy(NSGA-II)"]


def test_on_result_called_per_fit_and_payload_serialisable(synth):
    got, folds_ended = [], []
    run_cv(synth, MODELS, seed=0, n_folds=3, cfg=FAST_CFG, log=lambda *a: None,
           on_result=got.append, on_fold_end=lambda f, pls: folds_ended.append((f, len(pls))))
    assert len(got) == 3 * len(MODELS)
    assert folds_ended == [(0, 2), (1, 2), (2, 2)]
    for pl in got:
        json.dumps(pl)  # must be JSON-serialisable for checkpoint files
        assert len(pl["pred"]["p"]) == pl["record"]["n_test"]


def test_skip_resumes_without_refitting(synth):
    first = []
    run_cv(synth, MODELS, seed=0, n_folds=3, cfg=FAST_CFG, log=lambda *a: None, on_result=first.append)
    done = {(p["record"]["fold"], p["record"]["model"]) for p in first[:3]}  # pretend 3 fits survived
    second = []
    run_cv(synth, MODELS, seed=0, n_folds=3, cfg=FAST_CFG, log=lambda *a: None,
           skip=lambda d, f, m: (f, m) in done, on_result=second.append)
    redone = {(p["record"]["fold"], p["record"]["model"]) for p in second}
    assert redone.isdisjoint(done) and len(redone) == 6 - 3
    # resumed fits reproduce the uninterrupted run exactly (deterministic seeding)
    ref = {(p["record"]["fold"], p["record"]["model"]): p["pred"]["p"] for p in first}
    for p in second:
        assert p["pred"]["p"] == ref[(p["record"]["fold"], p["record"]["model"])]
