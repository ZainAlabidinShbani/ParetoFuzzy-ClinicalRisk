# Progress log

## 2026-11-25 — Phase 1 (complete)

* Repo audited. Legacy MATLAB toy scripts (synthetic data), stroke-dataset
  notebooks (scikit-fuzzy Mamdani, accuracy-fitness GA) and early docs were
  archived under `legacy/` with `git mv`; see `legacy/README.md`.
  `data/raw/` is untracked and git-ignored.
* Python 3.11 `.venv`; `requirements.txt` (direct pins) + `requirements-lock.txt`.
* Literature: 50/50 references verified live (Crossref/arXiv/Semantic Scholar),
  35 of them core. `paper/references.bib` is generated from the fetched
  metadata; log in `results/references_verification.json`; review in
  `paper/literature_review.md`.
* `paper/main.tex` (elsarticle): pdflatex + bibtex build verified, with no warnings.
* `RESEARCH_PLAN.md`: formulation, encoding, operators, protocol, datasets.
* Commit `9106c1d`.

## 2026-11-25 — Phase 2 (implementation + tests complete; pilot NOT complete)

* `src/fuzzyrisk/`: data registry (dev/held-out, held-out guard, md5 +
  sha256 checks), fold-local preprocessing, TSK fuzzy risk score, genome,
  NSGA-II/GA with custom seeded operators, tuned baselines, post-hoc
  calibration, and the CV runner. Commit `0335de5`.
* Dev datasets fetched and pinned: `pima_diabetes` (n=768, 268 positive) and
  `heart_cleveland` (n=303, 138 positive). Held-out datasets have **not** been
  downloaded.
* Tests: 58/58 pass (`pytest -q`, ~46 s). Commit `3b0c402`.
* Measured cost: ~6 ms per fitness evaluation on Pima, from a smoke test on
  a single split, not a pilot result.
* First pilot attempt (2026-11-25) was stopped by the Claude Code host under
  system memory pressure after 9/110 fits. RAM was ~92% used by other
  applications (Chrome ~4.8 GB, a `crosvm` VM ~3.1 GB); the pilot process
  itself stays below 300 MB. Its partial log was discarded (no checkpoints).

## 2026-11-25 — Phase 2b: pilot runner hardening (commits `e8aa89b`, `1a82eff`)

* Per-(dataset, fold, model) atomic checkpoints with a config fingerprint,
  resume by skipping completed fits, `--time-budget-min` resumable chunks.
* All BLAS/OpenMP pools, XGBoost/LightGBM/CatBoost threads and searches
  set to 1; `gc.collect()` after every fit and fold; per-fold RSS logging.
* Tests: 60/60 pass, including resume reproducibility.

## 2026-11-25 — Phase 2c: 1-seed dev pilot (complete)

**Setup.** Seed 0; outer stratified 5-fold CV; dev datasets only
(`pima_diabetes`, `heart_cleveland`). GA-fuzzy variants: pop 60 × 80 gen
(equal budget), R_max 12, L_max 4. Baselines: inner 3-fold, log-loss
scoring, 20 random-search iterations (grids for LR/DT). All 110 fits come from
commit `1a82eff` (4 resumable sessions of ≤ 8 min). Raw outputs are in
`results/pilot/` (`records.csv` per fold, `predictions.csv`, `fronts.json`,
`rules.json`, `run_meta.json`).

**Rules** = distinct active rules (fuzzy models) or leaves (decision tree).
**Fit s** = mean wall time of one outer-fold fit, including inner tuning or
evolution, single-threaded. **Peak RSS** = max process RSS sampled during fits.

### pima_diabetes (seed 0, 5-fold outer CV; mean ± SD over folds)

| Model | AUROC | PR-AUC | Brier | ECE | Cal. slope | Rules | Ante/rule | Fit s | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.836 ± 0.034 | 0.729 ± 0.052 | 0.156 ± 0.017 | 0.067 ± 0.014 | 1.04 ± 0.25 | – | – | 0.7 ± 0.1 | 266 |
| DecisionTree | 0.777 ± 0.052 | 0.610 ± 0.071 | 0.172 ± 0.020 | 0.045 ± 0.011 | 0.87 ± 0.27 | 5.2 ± 1.6 | 2.34 ± 0.47 | 0.8 ± 0.0 | 266 |
| XGBoost | 0.827 ± 0.047 | 0.714 ± 0.067 | 0.161 ± 0.019 | 0.086 ± 0.026 | 1.05 ± 0.41 | – | – | 9.9 ± 1.0 | 267 |
| LightGBM | 0.833 ± 0.040 | 0.713 ± 0.066 | 0.158 ± 0.020 | 0.094 ± 0.015 | 0.92 ± 0.26 | – | – | 8.5 ± 0.5 | 271 |
| CatBoost | 0.839 ± 0.034 | 0.726 ± 0.053 | 0.155 ± 0.016 | 0.059 ± 0.023 | 1.06 ± 0.26 | – | – | 65.8 ± 2.3 | 285 |
| MLP | 0.810 ± 0.044 | 0.685 ± 0.078 | 0.170 ± 0.022 | 0.081 ± 0.015 | 1.19 ± 0.33 | – | – | 9.0 ± 2.5 | 247 |
| XGBoost+Platt | 0.837 ± 0.043 | 0.720 ± 0.070 | 0.158 ± 0.017 | 0.077 ± 0.019 | 1.18 ± 0.35 | – | – | 10.5 ± 0.7 | 264 |
| XGBoost+Isotonic | 0.837 ± 0.043 | 0.723 ± 0.070 | 0.156 ± 0.018 | 0.091 ± 0.019 | 0.98 ± 0.52 | – | – | 10.2 ± 0.7 | 265 |
| SO-GA-Fuzzy(classic) | 0.771 ± 0.069 | 0.668 ± 0.080 | 0.187 ± 0.033 | 0.128 ± 0.043 | 1.00 ± 0.29 | 7.6 ± 1.1 | 1.91 ± 0.37 | 18.0 ± 0.9 | 265 |
| SO-GA-Fuzzy(hybrid) | 0.830 ± 0.037 | 0.708 ± 0.058 | 0.159 ± 0.017 | 0.089 ± 0.030 | 0.98 ± 0.28 | 10.8 ± 1.3 | 1.96 ± 0.38 | 41.9 ± 2.0 | 265 |
| MO-Fuzzy(NSGA-II) | 0.844 ± 0.045 | 0.735 ± 0.081 | 0.153 ± 0.020 | 0.079 ± 0.015 | 1.17 ± 0.46 | 3.4 ± 0.5 | 1.00 ± 0.00 | 26.5 ± 1.8 | 266 |

### heart_cleveland (seed 0, 5-fold outer CV; mean ± SD over folds)

| Model | AUROC | PR-AUC | Brier | ECE | Cal. slope | Rules | Ante/rule | Fit s | Peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.915 ± 0.023 | 0.909 ± 0.015 | 0.116 ± 0.011 | 0.086 ± 0.027 | 1.10 ± 0.18 | – | – | 2.0 ± 0.2 | 264 |
| DecisionTree | 0.777 ± 0.054 | 0.707 ± 0.092 | 0.193 ± 0.037 | 0.097 ± 0.032 | 0.66 ± 0.19 | 4.0 ± 0.0 | 2.00 ± 0.00 | 2.2 ± 0.1 | 264 |
| XGBoost | 0.903 ± 0.022 | 0.900 ± 0.018 | 0.127 ± 0.017 | 0.100 ± 0.020 | 0.96 ± 0.21 | – | – | 7.3 ± 0.3 | 266 |
| LightGBM | 0.902 ± 0.014 | 0.899 ± 0.019 | 0.122 ± 0.012 | 0.103 ± 0.023 | 1.01 ± 0.17 | – | – | 6.2 ± 0.3 | 269 |
| CatBoost | 0.913 ± 0.011 | 0.909 ± 0.017 | 0.121 ± 0.006 | 0.112 ± 0.032 | 1.20 ± 0.27 | – | – | 55.2 ± 2.6 | 285 |
| MLP | 0.897 ± 0.053 | 0.903 ± 0.035 | 0.131 ± 0.027 | 0.122 ± 0.031 | 1.67 ± 0.53 | – | – | 7.8 ± 0.5 | 285 |
| XGBoost+Platt | 0.900 ± 0.023 | 0.898 ± 0.023 | 0.127 ± 0.018 | 0.127 ± 0.029 | 1.21 ± 0.26 | – | – | 7.7 ± 0.4 | 286 |
| XGBoost+Isotonic | 0.897 ± 0.028 | 0.890 ± 0.029 | 0.130 ± 0.024 | 0.090 ± 0.031 | 0.79 ± 0.46 | – | – | 7.6 ± 0.6 | 286 |
| SO-GA-Fuzzy(classic) | 0.901 ± 0.026 | 0.894 ± 0.023 | 0.128 ± 0.018 | 0.123 ± 0.019 | 0.87 ± 0.29 | 9.4 ± 0.5 | 1.98 ± 0.16 | 16.3 ± 0.5 | 286 |
| SO-GA-Fuzzy(hybrid) | 0.881 ± 0.036 | 0.869 ± 0.049 | 0.139 ± 0.016 | 0.115 ± 0.015 | 0.81 ± 0.11 | 11.4 ± 0.5 | 2.35 ± 0.15 | 34.6 ± 2.3 | 286 |
| MO-Fuzzy(NSGA-II) | 0.893 ± 0.051 | 0.877 ± 0.059 | 0.126 ± 0.029 | 0.120 ± 0.028 | 0.97 ± 0.19 | 5.0 ± 0.0 | 1.20 ± 0.24 | 27.5 ± 1.9 | 275 |

**Runtime / memory.** Total fit + predict time 31.4 min (wall 31.7 min over 4
sessions); 2.7–3.5 min per fold for all 11 models. Process peak RSS ≤ 286 MB
in every fold (max per fold, from `records.csv`: Pima 284/194/266/285/181 MB;
Heart 286/250/264/285/199 MB). System available memory during the run was
1.6–2.6 GB, so the run is light, but the machine was near its limit because of
other applications. The slowest baseline is CatBoost (55–66 s per fold;
single-threaded, 20 configs × 3 inner folds).

**Additional means** (from `records.csv`): MO-Fuzzy complexity f3 = 4.4
(Pima) / 6.2 (Heart) vs 12.8 / 13.7 for SO-GA-Fuzzy(hybrid). Normalised test
hypervolume of the MO front is 0.236 (Pima) / 0.362 (Heart). SO-GA-Fuzzy(classic)
calibration intercept on Pima is −0.31 (others within ±0.15).

**Preliminary observations.** Single seed, 5 folds, no significance tests.
Fold SDs of AUROC are 0.01–0.07, so most between-model differences below are
within noise.

* Pima: MO-Fuzzy has the highest mean AUROC (0.844) and lowest Brier
  (0.153), using 3.4 rules with 1 antecedent each. The top 5 models are
  within 0.01 AUROC. Its ECE (0.079) is mid-pack, and its calibration slope of
  1.17 indicates slightly under-confident probabilities.
* Heart: LogReg is best (AUROC 0.915, Brier 0.116). MO-Fuzzy reaches 0.893 /
  0.126 with 5 rules (1.2 antecedents per rule), 0.020 AUROC below CatBoost.
  Its ECE (0.120) is among the worse values, and binned ECE on ~60 test
  patients per fold is very noisy.
* SO-GA-Fuzzy(classic) (accuracy fitness, evolved consequents) is clearly
  worse on Pima (AUROC 0.771, Brier 0.187, intercept −0.31) but competitive on
  Heart (0.901 / 0.128). SO-GA-Fuzzy(hybrid) matches MO discrimination on Pima
  with about 3× the complexity and is worse on Heart.
* Post-hoc Platt/isotonic calibration did not improve XGBoost's Brier on
  either dataset in this pilot.
* Example deployed MO rule base (Heart, fold 0): baseline +0.70 log-odds;
  `cp=asympt is No` −1.49; `ca is High` +2.32; `exang Yes AND slope=up No`
  +1.77; `sex=male is No` −1.06; `thal=reversable_defect is No` −1.56. Full
  rule bases for every fold are in `rules.json`.

**Issues to decide before Phase 3**

1. **Front saturation:** in every fold, all 60 final individuals are
   non-dominated, so NSGA-II selection relies on crowding alone. Options: a
   larger population (100–120), an external archive, or coarser f3
   (integer rules + antecedents).
2. **ε-compact selection drives MO-Fuzzy to 1-antecedent rules** (an additive
   fuzzy GAM-like score). That suits interpretability, but whether ε = 0.01
   is the right preregistered value should be decided now.
3. **Calibration slope > 1 for MO-Fuzzy on Pima** (under-confidence) is
   probably due to the L2 penalty λ = 1 on consequents. λ could be tuned in the
   inner loop or the penalty lowered.
4. `fold_memory.csv` rows for folds split across two sessions cover only that
   session's fits (column `n_fits`); per-fold peaks above come from
   `records.csv`.


Review the pilot, then Phase 3 (decide items 1–3, write and lock
`PREREGISTRATION.md`, unlock the held-out datasets).
## 2026-11-25 — Pre-Phase-3 refinements (commits `d234092`, `4d08947`)

* f3 = total literals (integer). NSGA-II population 80 × 80 generations
  with an external non-dominated archive (archives of 9–34 solutions in a
  one-fold benchmark, vs a saturated 60/60 front in the pilot).
* λ selected per genome from {0.01, 0.1, 1, 10} by inner-OOF log-loss.
* ε = 0.01 compact rule kept as the headline; best-Brier archive member
  logged as the `MO-Fuzzy(best-Brier)` ablation.
* Registered n/positive counts per dataset, explicit CKD schema,
  `scripts/run_experiment.py`. 64 tests pass.

## 2026-11-25 — Phase 3: preregistration locked (commit `b1a6d27`, tag `prereg-v1`)

* `PREREGISTRATION.md` (`status: locked`): datasets, positive classes
  confirmed from official documentation, frozen configuration, H1–H4,
  analysis plan with the N = 6 power caveat, and SHA-256 of 22 locked files.
* Held-out unlock: all four datasets pass md5 (OpenML) and n/positive checks
  (WDBC 569/212, ILPD 583/416, Thyroid 3772/231, CKD 400/250). sha256 pins
  were added to `dataset_pins.json` at first fetch.

### Deviations from preregistration

Machine-readable copy: `PREREGISTRATION_DEVIATIONS.json`; verified by `tools/audit.py`.

* **D1** (`src/fuzzyrisk/data.py`, new sha256 `9316f24902656f47…`): At unlock, sklearn 1.9's pandas ARFF parser raised TypeError under pandas 3 on integer-like nominal levels (WDBC Class {1,2}). Added a fallback to sklearn's LIAC-ARFF parser for the same md5-verified file. Parsing only; no change to method, data, splits or metrics. Dev datasets still parse with the pandas parser and match their pinned sha256.
* **D2** (`tests/test_metrics_data.py`, new sha256 `a7343a97c02b494e…`): test_heldout_not_present_on_disk encodes a pre-unlock invariant that becomes false by design once held-out data are fetched; it now skips when PREREGISTRATION.md is locked.
* **D3** (`src/fuzzyrisk/metrics.py`, new sha256 `a3c0451779b1af13…`; new test `tests/test_solver_regression.py`, sha256 `3bf4e1bdc6606e47…`), 2026-11-26: while checking the first two complete seeds (held-out results visible), every deployed MO-Fuzzy and SO-GA-Fuzzy(hybrid) model on Thyroid (15/15 each) turned out to have diverged consequents: intercept ≈ −1e12, constant test predictions, AUROC 0.5. Their inner-CV objectives were good. The cause was the plain-Newton penalised-logistic solver: the first step overshoots on imbalanced data with a rule that fires on a small high-risk subgroup, then explodes on saturated logits. The fix adds backtracking (step-halving) on the exact penalised objective. Full steps that do not increase the objective are accepted unchanged, so the estimator is the same and only the solver now converges; it matches sklearn's `LogisticRegression` to 1e-4 on the reproduced Thyroid case. All 300 MO-Fuzzy and SO-GA-Fuzzy(hybrid) fits are re-run with the fixed code. The 174 superseded checkpoints are archived in `results/final/superseded_D3/`, summarised in `results/final/superseded_D3_records.csv`. The same solver computes every model's calibration slope and intercept (old slopes diverged to ~1e12 on 44 Thyroid baseline fits in seeds 0–1), so `scripts/analyze.py` now recomputes all outcome metrics from the stored test-fold predictions and reports the changes in `stats.json`. Design, hyperparameters, data, splits, hypotheses and the analysis plan are unchanged.

## 2026-11-27 — Phase 3 sweep complete; Phases 4–6 (analysis, manuscript, audit, submission pack)

**Sweep.** 1650/1650 preregistered fits (5 seeds × 5 folds × 6 datasets × 11 models) plus the
best-Brier ablation, all single-threaded via `scripts/run_experiment.py` in checkpointed chunks.
Peak RSS per fit ≤ 302 MB. All 300 MO-Fuzzy / SO-GA-Fuzzy(hybrid) fits come from post-D3 commits
(`f98b39e`, `cc126e3`). The last chunks ran in the foreground (`--hard-limit-min 9.3`) after Claude
Code reaped a background chunk under system memory pressure. No fits were lost (atomic checkpoints).

**Headline results** (mean AUROC / Brier over 25 outer test folds; generated from
`results/final/summary_by_dataset.csv` and `stats.json`):

| Model | Pima | Cleveland | WDBC | ILPD | Thyroid | CKD | mean |
|---|---|---|---|---|---|---|---|
| MO-Fuzzy(NSGA-II) | 0.841 / 0.155 | 0.892 / 0.132 | 0.988 / 0.036 | 0.710 / 0.183 | 0.976 / 0.023 | 0.993 / 0.021 | 0.900 / 0.092 |
| MO-Fuzzy(best-Brier) | 0.837 / 0.156 | 0.896 / 0.129 | 0.988 / 0.035 | 0.712 / 0.184 | 0.960 / 0.017 | 0.999 / 0.011 | 0.899 / 0.089 |
| SO-GA-Fuzzy(hybrid) | 0.820 / 0.167 | 0.878 / 0.151 | 0.992 / 0.033 | 0.690 / 0.197 | 0.987 / 0.017 | 0.998 / 0.011 | 0.894 / 0.096 |
| SO-GA-Fuzzy(classic) | 0.793 / 0.178 | 0.890 / 0.133 | 0.987 / 0.048 | 0.636 / 0.202 | 0.845 / 0.080 | 0.997 / 0.067 | 0.858 / 0.118 |
| CatBoost | 0.839 / 0.156 | 0.907 / 0.122 | 0.994 / 0.023 | 0.743 / 0.174 | 0.998 / 0.007 | 1.000 / 0.005 | 0.913 / 0.081 |
| XGBoost | 0.837 / 0.156 | 0.904 / 0.124 | 0.993 / 0.025 | 0.726 / 0.180 | 0.997 / 0.009 | 1.000 / 0.006 | 0.909 / 0.083 |
| LightGBM | 0.836 / 0.157 | 0.903 / 0.123 | 0.994 / 0.024 | 0.721 / 0.182 | 0.998 / 0.008 | 1.000 / 0.002 | 0.909 / 0.083 |
| LogReg | 0.834 / 0.158 | 0.912 / 0.115 | 0.996 / 0.020 | 0.751 / 0.172 | 0.953 / 0.027 | 1.000 / 0.005 | 0.907 / 0.083 |
| XGBoost+Platt | 0.839 / 0.156 | 0.906 / 0.122 | 0.992 / 0.026 | 0.728 / 0.180 | 0.996 / 0.012 | 1.000 / 0.008 | 0.910 / 0.084 |
| XGBoost+Isotonic | 0.837 / 0.156 | 0.905 / 0.122 | 0.991 / 0.026 | 0.725 / 0.180 | 0.994 / 0.011 | 0.999 / 0.008 | 0.909 / 0.084 |
| MLP | 0.827 / 0.163 | 0.889 / 0.137 | 0.991 / 0.032 | 0.686 / 0.185 | 0.973 / 0.021 | 0.996 / 0.048 | 0.894 / 0.098 |
| DecisionTree | 0.758 / 0.181 | 0.798 / 0.179 | 0.953 / 0.061 | 0.695 / 0.185 | 0.974 / 0.015 | 0.975 / 0.036 | 0.859 / 0.110 |

* auroc: Friedman chi2=40.54, p=1.4e-05; Nemenyi CD=6.16; avg rank MO=6.83, CatBoost=2.08; min Holm-Wilcoxon p=0.312
* brier: Friedman chi2=40.85, p=1.2e-05; Nemenyi CD=6.16; avg rank MO=6.83, CatBoost=2.17; min Holm-Wilcoxon p=0.312
* H1 SUPPORTED (6/6, one-sided Wilcoxon p=0.016)
* H2 SUPPORTED (4/6; fails: liver_ilpd, thyroid_sick)
* H3 SUPPORTED (6/6, p=0.016)
* H4 SUPPORTED (literal ratio 0.11-0.25 on 6/6; AUROC loss <= 0.01 on 5/6)

The MO-Fuzzy models had on average 3.3 rules and 3.9 literals (SO-hybrid: 10.5 rules / 21.9 literals).
Recomputing the metrics from stored predictions (D3) changed 70 stored calibration slopes and intercepts, all from fits computed with the pre-D3 solver; no other metric changed.

**Outputs.** Statistics: `results/final/stats.json`. Tables: `paper/tables/*.tex`. Macros:
`paper/generated/macros.tex`. Figures: `paper/figures/*.pdf`. Manuscript: `paper/main.tex` and
`paper/sections/*`. Audit: `tools/audit.py` → `results/final/audit_report.json`; all checks pass,
including the replication of 14 sampled fits (max |Δp| = 0) and byte-identical regeneration.
Submission pack: `submission_pack/`.
