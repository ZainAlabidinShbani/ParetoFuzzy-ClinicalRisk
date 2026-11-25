# Preregistration

status: locked

* Locked: 2026-11-25, before any held-out data file was downloaded.
* Code state: the commit that adds this file (tagged `prereg-v1`). The SHA-256
  hashes in §8 are verified by `tools/audit.py`.
* Study: *Pareto-optimal calibrated and interpretable fuzzy risk scoring via
  multi-objective evolutionary optimisation (NSGA-II)*.

## 1. Datasets and roles

| Role | Dataset | Source | n | Positives | Positive class (confirmed from official documentation) |
|---|---|---|---|---|---|
| dev | `pima_diabetes` | OpenML 37, md5 `3cbaa3e54586aa88cf6aacb4033e4470` | 768 | 268 | `tested_positive` |
| dev | `heart_cleveland` | OpenML 49, md5 `f255fbfe272738f3da960f63a451ca0a` | 303 | 138 | `num` ∈ {>50_1..>50_4} (angiographic narrowing > 50%) |
| held-out | `breast_wdbc` | OpenML 1510, md5 `7aa183d3657e364911ced0cbd6b272bd` | 569 | 212 | `2` = malignant (OpenML text: "benign (1) or malignant (2)") |
| held-out | `liver_ilpd` | OpenML 1480, md5 `155295549a459387fb71312caf1b8360` | 583 | 416 | `1` = liver patient (416 patients / 167 non-patients) |
| held-out | `thyroid_sick` | OpenML 38, md5 `ee5cb4b7f41a5f44b0cd234f675ab492` | 3772 | 231 | `sick` (vs `negative`) |
| held-out | `kidney_ckd` | UCI 336 `data.csv` (no published md5; sha256 pinned at first fetch) | 400 | 250 | `ckd` (vs `notckd`) |

The registered n and positive counts are checked on load (`ChecksumError`
otherwise), which also verifies the positive-class codes. Dev canonical-CSV
sha256 pins at lock time: `pima_diabetes`
`c1b530cb22b468b5bc6da61d1467eec08882409d8edf89b0c7ae7bf6596b86f2`,
`heart_cleveland`
`8148e64304223f8ec9adf98dba07419772ffc87188c3ca242f70d7a3e3ccd451`.
The CKD schema is declared in the registry (`sg`, `al`, `su` are ordinal codes
treated as numeric).

**Known ceiling risk, declared in advance:** WDBC and CKD are known to be
nearly separable. AUROC differences there are expected to be compressed, so
conclusions on these datasets rest mainly on calibration and complexity.

## 2. Models (11) and the ablation

LogReg, DecisionTree, XGBoost, LightGBM, CatBoost, MLP, XGBoost+Platt,
XGBoost+Isotonic, SO-GA-Fuzzy(classic), SO-GA-Fuzzy(hybrid),
**MO-Fuzzy(NSGA-II)** (proposed). Ablation from the same MO runs:
MO-Fuzzy(best-Brier). The ablation is excluded from the Friedman/Nemenyi analyses.

## 3. Frozen method configuration

* TSK zero-order additive fuzzy risk score. Strong 3-set triangular partitions
  (min gap 0.05) for continuous inputs, crisp {No, Yes} sets for
  binary/one-hot inputs, product t-norm.
* R_max = 12 rule slots, L_max = 4 antecedents per rule.
* Consequents (hybrid): L2 logistic regression. λ is selected per genome from
  {0.01, 0.1, 1, 10} by pooled inner 3-fold OOF log-loss.
* NSGA-II objectives (minimised, inner-OOF on the outer-training fold only):
  1 − AUROC, Brier, total literals. Population 80, 80 generations, external
  non-dominated archive.
* Headline operating point: ε-compact (ε = 0.01). Secondary: best-Brier.
* SO-GA-Fuzzy(classic): evolved consequents, fitness = training accuracy.
  SO-GA-Fuzzy(hybrid): fitness = 1 − inner-OOF AUROC. Both use the same
  80 × 80 budget and operators.
* Baselines: inner stratified 3-fold CV with log-loss scoring. Grids for LR and
  DT; 20-iteration random search for XGBoost, LightGBM, CatBoost and MLP.
  Platt/isotonic calibration via inner 3-fold `CalibratedClassifierCV`.
* Compute: all model threads = 1, BLAS/OpenMP = 1.

## 4. Evaluation protocol

5 seeds (0–4) × stratified 5-fold outer CV = 25 test folds per dataset and
model. All preprocessing, tuning, evolution and selection happen inside the
outer-training fold. Outer test metrics: AUROC, PR-AUC, Brier, log-loss, ECE
(10 equal-width bins), adaptive ECE, calibration slope/intercept, rules, mean
antecedents, total literals, fit time, peak RSS.

## 5. Hypotheses (one-sided where stated; dataset-level means over the 25 folds)

* **H1 (calibration vs prior-art GA-fuzzy).** MO-Fuzzy has lower mean Brier than
  SO-GA-Fuzzy(classic). Supported if this holds on ≥ 5/6 datasets; formal test:
  one-sided Wilcoxon signed-rank on the 6 paired dataset means, α = 0.05.
* **H2 (discrimination non-inferiority vs GBDTs).** On each dataset, MO-Fuzzy
  mean AUROC ≥ max(XGBoost, LightGBM, CatBoost mean AUROC) − 0.02. Supported if
  this holds on ≥ 4/6 datasets.
* **H3 (calibration slope).** |slope − 1| (the median over folds of
  |slope_fold − 1|) is smaller for MO-Fuzzy than for SO-GA-Fuzzy(classic).
  Supported on ≥ 4/6 datasets; one-sided Wilcoxon signed-rank as for H1.
* **H4 (parsimony).** MO-Fuzzy mean total literals ≤ 0.5 × SO-GA-Fuzzy(hybrid)
  mean total literals on all 6 datasets, while the AUROC loss
  (hybrid − MO) is ≤ 0.01 on ≥ 4/6 datasets.

## 6. Statistical analysis plan

1. Per dataset: mean ± SD over the 25 folds, reported separately for dev
   (2), held-out (4) and all (6) datasets.
2. Across datasets (N = 6, k = 11 models), for AUROC and Brier: Friedman
   test on per-dataset ranks; Nemenyi critical difference
   (α = 0.05, q_α from the studentized range distribution) shown as CD diagrams.
   Pairwise: two-sided Wilcoxon signed-rank MO-Fuzzy vs each of the 10 other
   models, Holm-corrected. **Power caveat (stated before results):** with
   N = 6 the smallest attainable two-sided exact Wilcoxon p is 0.03125, so no
   pairwise test can survive Holm correction over 10 comparisons. These
   tests are reported for completeness, and conclusions rely on effect sizes
   and the per-dataset analysis below.
3. Per dataset (secondary): Nadeau–Bengio corrected resampled t-test over
   the 25 paired folds (correction n_test/n_train for 5-fold CV),
   MO-Fuzzy vs each model, Holm-corrected within dataset and metric.
4. Calibration plots: pooled reliability diagrams over all 25 test folds.

## 7. Deviations policy

Any change to locked files after this commit, or any departure from §1–§6,
is recorded with its reason in `PROGRESS.md` under *Deviations from
preregistration*, together with the new file hashes. Analysis, figure and
table scripts written after the lock (`scripts/analyze.py`, `tools/audit.py`)
implement §6 and are hashed separately in the audit report.

## 8. SHA-256 of locked files

| File | SHA-256 (LF-normalised) |
|---|---|
| `pyproject.toml` | `934ea8b9e941588efd1a8c0bb95af0683bcaa983070d472a89dcb96b13298906` |
| `requirements-lock.txt` | `1cfb77a16e90b4fee340cb82f2aeb8dab1b31c825f4d1603873047e28d2ff3f4` |
| `requirements.txt` | `33260c5dd2eaa720ebc0069ce7c234c0a9c3e20c002b83abc1a35d3cbe33a85f` |
| `scripts/fetch_data.py` | `4fdcebd6731fc06245553982e711be97f83bbe1f057dd753a7c0bc7c5cef43d4` |
| `scripts/run_experiment.py` | `58e88503901dbd1de5445cf12e004755212b8b52c65fbe26e02ee89ea72cefe1` |
| `scripts/run_pilot.py` | `6ad7aeb85bdb3a038004cfbc0442e69bcaf4b40e09e6dd54af1bb9df4bf282f1` |
| `src/fuzzyrisk/__init__.py` | `41475636ead751e2cf7b8b528e7ab53c629d76c6b5adbed5391fb13fa9ab319a` |
| `src/fuzzyrisk/baselines.py` | `6fd1be38e8384e1ce4ba44913ceb25c7d06eb463cfbc2e2ce5b48d4f739d50b1` |
| `src/fuzzyrisk/data.py` | `09246ff577a3b975af1ceeef2dcb3d263521af1da46949ac5d67b80064928850` |
| `src/fuzzyrisk/encoding.py` | `76e390b84f9c8741c1cd30a52a1f8a08c241df5820dc81369927c724f7a62fa7` |
| `src/fuzzyrisk/evaluation.py` | `50df1ba95f54b6b27731634cf2c6b2f1453c8d3f86f6bf4d44bd90054e94128e` |
| `src/fuzzyrisk/fis.py` | `67ec5c0fece40fc9ae62ee16658578e09c3d57ca39bbc99647416c67058f5fcd` |
| `src/fuzzyrisk/metrics.py` | `5122904d2139805c72c41d057ef899d7a4a93f27c6bb4e783bb3d78fdd7b2aa1` |
| `src/fuzzyrisk/models.py` | `7db7507437ba48328c760ca9f549320908d7c29cbf8f59f3239d9d5b842d4480` |
| `src/fuzzyrisk/optim.py` | `a6d60e6230773a99e623d47f154038aa504e332be4dc47f84df694ae067ca808` |
| `src/fuzzyrisk/preprocessing.py` | `2137f239e0e2d05c8546aeccfc4e19f749cc07f89cb23808ad3fc6ec06c2049c` |
| `tests/conftest.py` | `8e8771adbc3b660ca35e177b99cc1c03561ecb26ffe73d12abad9c9935beb22a` |
| `tests/test_checkpointing.py` | `94e3378f8e46e126c9830b85d8f9a010355224f68da37c2806bb90b864b71bdb` |
| `tests/test_leakage.py` | `d19d86db9b5367f2a5b0847020e0f7b690ae7e4d53a8f9dc0be1e9f61912d8cf` |
| `tests/test_metrics_data.py` | `fe4800459636ecb13f40adfca75fed042d8f22926b925a580f52f929fc2c9cef` |
| `tests/test_pareto_interpretability.py` | `f91a86ad6daa2c99ffb77c8ef8dea15c26bbde491bd35f6ec948df51bf6a40b0` |
| `tests/test_probabilities_determinism.py` | `d3c00f2921439a70440b4930c87d40679efd5ab335d54625bd4937ca374226a9` |
