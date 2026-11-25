# FuzzyGA-HealthAI

**Pareto-optimal calibrated and interpretable fuzzy risk scoring via
multi-objective evolutionary optimisation (NSGA-II).**

A zero-order Takagi–Sugeno fuzzy *risk score* whose linguistic partitions and
rule base are evolved by NSGA-II. It optimises three objectives:
discrimination (1 − AUROC), calibration (Brier score) and interpretability
(rules + mean antecedents). It is benchmarked under a nested, leakage-free
protocol against logistic regression, decision trees, XGBoost, LightGBM,
CatBoost, MLP, single-objective GA-fuzzy models, and post-hoc Platt/isotonic
calibration.

> Research software. Not a medical device; must not be used for diagnosis or
> treatment decisions.

## Layout

```text
src/fuzzyrisk/   Python package: data, preprocessing, FIS, encoding, NSGA-II/GA, baselines, metrics, evaluation
scripts/         Entry points (fetch_data.py, run_pilot.py)
tests/           pytest suite (leakage, probabilities, determinism, Pareto validity, interpretability)
tools/           verify_references.py (+ reference_candidates.json)
paper/           elsarticle manuscript, references.bib (auto-generated, verified), literature_review.md
results/         references_verification.json, pilot outputs
legacy/          archived MATLAB scripts, notebooks and derived data (superseded; see legacy/README.md)
RESEARCH_PLAN.md formulation, encoding, operators, protocol, datasets
PROGRESS.md      phase log and pilot results
```

## Setup (Windows 11, Python 3.11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # exact lock: requirements-lock.txt
$env:PYTHONUTF8 = "1"
```

## Usage

```powershell
.venv\Scripts\python scripts\fetch_data.py                # dev datasets only, checksum-verified
.venv\Scripts\python -m pytest -q                         # unit tests
.venv\Scripts\python scripts\run_pilot.py --seed 0        # 1-seed pilot on the 2 dev datasets
.venv\Scripts\python tools\verify_references.py           # re-verify citations against Crossref/arXiv/S2
cd paper; latexmk -pdf main.tex                           # build manuscript
```

## Data policy

Datasets are fetched on demand from OpenML/UCI into `data/` (git-ignored) and
verified against the OpenML md5 and a pinned sha256 of the canonical CSV. The
four **held-out** datasets cannot be loaded until `PREREGISTRATION.md` is
locked (see `RESEARCH_PLAN.md` §7).
