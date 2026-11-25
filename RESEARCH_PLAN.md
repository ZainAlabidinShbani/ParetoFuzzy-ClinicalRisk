# Research Plan

**Working title:** Pareto-Optimal Calibrated and Interpretable Fuzzy Risk Scoring
via Multi-Objective Evolutionary Optimization (NSGA-II)

**Target venue:** Applied Soft Computing (Elsevier), `elsarticle`, subscription track.

**Status:** Phase 1 complete; Phase 2 implementation and tests complete, dev pilot pending (see PROGRESS.md). Held-out datasets
are locked until Phase 3 preregistration (`PREREGISTRATION.md`, `status: locked`).

---

## 1. Research questions and hypotheses

* **RQ1 (calibration).** Does adding a proper calibration objective to a
  multi-objective genetic fuzzy system yield risk scores whose calibration
  (Brier, ECE, calibration slope) is comparable to tuned GBDTs and better than
  accuracy-driven GA-fuzzy models?
* **RQ2 (interpretability–performance trade-off).** How much discrimination
  (AUROC, PR-AUC) is lost, relative to tuned XGBoost/LightGBM/CatBoost, by a
  fuzzy risk score with at most ~5–8 short rules?
* **RQ3 (post-hoc calibration).** Is the calibration of the multi-objective
  fuzzy score comparable to black-box models plus post-hoc Platt/isotonic
  calibration?

Hypotheses (to be preregistered with exact tests in Phase 3):
* H1: MO-Fuzzy Brier ≤ SO-GA-Fuzzy(classic) Brier on most datasets.
* H2: MO-Fuzzy AUROC is within a practical margin (Δ ≤ 0.02) of the best tuned
  GBDT on most datasets, with ≤ 10 rules.
* H3: MO-Fuzzy calibration slope is closer to 1 than SO-GA-Fuzzy(classic).

## 2. Model: zero-order TSK fuzzy risk score

### 2.1 Universe of discourse

Inside each training fold, a `TabularPreprocessor` (mode `fuzzy`) is fitted:
median/mode imputation, one-hot nominal features, and for each numeric feature
$x_j$ the map $\tilde x_j = \mathrm{clip}\big((x_j - q^{(1)}_j)/(q^{(99)}_j - q^{(1)}_j), 0, 1\big)$
where $q^{(1)}_j, q^{(99)}_j$ are training-fold percentiles. All fuzzy inputs
are in $[0,1]$.

### 2.2 Linguistic partitions (data base)

* Continuous feature $j$: three triangular sets {Low, Medium, High} with
  centres $c_{j1} < c_{j2} < c_{j3}$ forming a **strong (Ruspini) partition**
  with shoulders:
  $\mu_{L}(x) = \mathrm{clip}\!\big(\tfrac{c_2 - x}{c_2 - c_1},0,1\big)$,
  $\mu_{H}(x) = \mathrm{clip}\!\big(\tfrac{x - c_2}{c_3 - c_2},0,1\big)$,
  $\mu_{M} = 1 - \mu_L - \mu_H$, so $\sum_k \mu_k(x) = 1$ for all $x$
  (coverage and distinguishability by construction). A minimum gap
  $c_{k+1} - c_k \ge \delta = 0.05$ is enforced by repair.
* Binary/one-hot feature: crisp sets {No, Yes}: $\mu_{No} = 1 - x$, $\mu_{Yes} = x$.

### 2.3 Rule base and inference

Rule $r$: IF $x_{j_1}$ is $A_{r j_1}$ AND … AND $x_{j_L}$ is $A_{r j_L}$ THEN $w_r$, with $1 \le L \le L_{\max}$.
Firing strength (product t-norm): $\beta_r(x) = \prod_{j \in S_r} \mu_{j,A_{rj}}(x_j)$.

Output (additive fuzzy risk score):
$$ z(x) = w_0 + \sum_{r=1}^{R} w_r\, \beta_r(x), \qquad \hat p(x) = \sigma(z(x)). $$
Each rule contributes at most $w_r$ log-odds when fully fired: a soft,
rule-based analogue of point-based clinical scores (`ustun2016`).

### 2.4 Consequent estimation (hybrid learning)

Given antecedents and partitions, $z$ is linear in $(w_0, \mathbf w)$, so
consequents are estimated by **L2-penalised logistic regression** (Newton–Raphson,
$\lambda = 1$, intercept unpenalised):
$$ \min_{w_0,\mathbf w} \sum_i \ell_{\log}\big(y_i, \sigma(w_0 + \mathbf B_i \mathbf w)\big) + \tfrac{\lambda}{2}\|\mathbf w\|^2 , \quad B_{ir} = \beta_r(x_i). $$
The GA searches the discrete/continuous structure (antecedents, partitions),
and a convex solver fits the consequents (cf. hybrid learning in ANFIS, `jang1993`).
Ablation: `evolved` consequents ($w_r \in [-4, 4]$ as genes), the classic
GA-fuzzy setting.

## 3. Encoding (Pittsburgh, real-coded with integer genes)

For $d$ preprocessed features ($d_c$ continuous), $R_{\max}$ rule slots
(default 12) and $L_{\max}$ (default 4):

| Segment | Size | Domain |
| --- | --- | --- |
| MF genes | $3 d_c$ | $[0,1]$ → sorted + min-gap repair |
| Antecedents $a_{rj}$ | $R_{\max} d$ | $\{0,\dots,K_j\}$, 0 = don't care |
| Activity bits | $R_{\max}$ | $\{0,1\}$ |
| Consequents (ablation only) | $R_{\max}+1$ | $[-4,4]$ |

Decoding: a rule is active iff its bit is 1 and it has ≥ 1 antecedent;
duplicate antecedent sets are removed, so the reported rule count is the
number of **distinct active rules**.

## 4. NSGA-II

### 4.1 Objectives (all minimised; computed on the outer-training fold only)

For each genome, pooled **out-of-fold** predictions $\hat p^{\text{oof}}$ are
obtained by inner stratified 3-fold CV on the outer-training fold (consequents
fitted on 2/3, predicted on 1/3):

1. Discrimination: $f_1 = 1 - \mathrm{AUROC}(y, \hat p^{\text{oof}})$
2. Calibration (proper score): $f_2 = \mathrm{Brier}(y, \hat p^{\text{oof}})$
   (ablations: ECE, log-loss)
3. Interpretability: $f_3 = R_{\text{act}} + \bar L$, where $R_{\text{act}}$ is the number of
   distinct active rules and $\bar L$ the mean number of antecedents per rule.

Brier is used as the calibration objective because it is a strictly proper
scoring rule (`gneiting2007`) and is far less noisy than binned ECE on
folds of 100–600 samples. ECE, adaptive ECE and the calibration slope and
intercept are reported as **outcomes**, not optimised.

### 4.2 Operators (custom pymoo operators; all randomness from the seeded generator)

* **Initialisation:** MF centres at training quantiles (0.2, 0.5, 0.8) plus
  $\mathcal N(0, 0.08)$ jitter (first individual exact). Each rule slot gets
  a random rule with 1–3 antecedents. Per-individual activity probability is
  drawn from $U(0.15, 1)$ to spread the initial complexity.
* **Crossover (p = 0.9):** BLX-α (α = 0.3) on MF genes. Uniform **rule-slot**
  exchange on antecedents, activity bits and slot consequents.
* **Mutation:** Gaussian jitter of MF genes (σ = 0.05, per-gene p = min(0.5, 2/#MF)),
  plus 1–2 structural edits per individual from {add condition (0.2), drop
  condition (0.25), relabel (0.2), toggle rule (0.2), new random rule (0.15)}.
  Length repair to $L_{\max}$.
* Survival: standard NSGA-II (non-dominated sorting + crowding distance,
  `deb2002`); duplicate elimination on.

### 4.3 Single deployable model: preregistered ε-compact rule

From the final non-dominated set, take every solution whose inner-OOF AUROC is
within ε = 0.01 of the front's best. Choose the one with the lowest $f_3$,
breaking ties by the lowest inner-OOF Brier. Consequents are then refitted on
the full outer-training fold. The whole front is also refitted and scored on
the outer test fold, for Pareto analysis only; those scores are never used
for selection.

## 5. Baselines

All baselines are `Pipeline(TabularPreprocessor, model)`, so preprocessing is
re-fitted inside every inner and outer fold. Hyperparameters are tuned by
inner stratified 3-fold CV on the outer-training fold with **log-loss** scoring
(a proper scoring rule, so tuning does not favour discrimination over
calibration).

| Model | Search |
| --- | --- |
| Logistic Regression (L2) | grid C ∈ 10^[-3, 2] (11 values) |
| Decision Tree | grid depth ∈ {2,3,4,5} × min_leaf ∈ {5,10,20,40} |
| XGBoost | random search: n_estimators, max_depth 2–6, η log-U[0.01,0.3], subsample, colsample, min_child_weight, λ |
| LightGBM | random search: num_leaves 4–31, n_estimators, η, min_child_samples, subsample, colsample, λ |
| CatBoost | random search: depth 2–6, iterations, η, l2_leaf_reg |
| MLP | random search: hidden sizes, α, initial LR; early stopping |
| SO-GA-Fuzzy (classic) | same encoding, **evolved consequents**, fitness = training accuracy (the prior-art setting) |
| SO-GA-Fuzzy (hybrid) | same encoding + hybrid consequents, fitness = 1 − inner-OOF AUROC (isolates the effect of multi-objective search) |
| Post-hoc calibration | tuned XGBoost (and others in Phase 4) + Platt (sigmoid) or isotonic via inner 3-fold `CalibratedClassifierCV` |

Equal search budgets: the GA-fuzzy variants (SO and MO) use the same
population size and number of generations.

## 6. Evaluation protocol (leakage-free)

* **Outer loop:** repeated stratified 5-fold CV. Pilot: 1 repeat (seed 0).
  Final: 5 repeats (seeds 0–4).
* **Inner loop:** all tuning, GA fitness and Pareto selection use only the outer-training fold.
* **Preprocessing:** fitted inside every (inner and outer) training fold.
  Column types come from the declared schema (ARFF nominal/numeric), never
  from observed values.
* **Row-wise cleaning** (e.g. Pima physiologically impossible zeros → NaN)
  is a fixed rule with no fitted parameters, so it is applied before splitting.
* **Unit tests** (`tests/`) assert disjoint, exhaustive splits; train-only
  fitting (a spy estimator records every row it is fitted on); invariance of
  fitted preprocessing to test-fold perturbations; and a locked held-out guard.

### Metrics (outer test folds)

Discrimination: AUROC, PR-AUC. Calibration: Brier, log-loss, ECE (10
equal-width bins), adaptive ECE (10 equal-mass bins, `nixon2019`), calibration
intercept and slope (logistic recalibration, `vancalster2016`), and
calibration-in-the-large. Interpretability: distinct active rules, mean and
total antecedents, $f_3$. Pareto quality: normalised hypervolume with
objective scales (0.5, 0.25, $R_{\max}+L_{\max}$) and reference point (1,1,1),
`zitzler1999`, on both inner-OOF and outer-test fronts. Cost: fit and predict
wall-time, peak process working set (RSS).

### Statistics (Phase 4, preregistered in Phase 3)

Per-dataset mean ± SD over the 25 outer folds; Friedman test plus Nemenyi or
Holm post-hoc across datasets (`demsar2006`); Bayesian signed-rank with ROPE
(`benavoli2017`) for MO-Fuzzy vs each baseline. With 6 datasets, power for
across-dataset tests is low; per-dataset corrected resampled t-tests are
reported as secondary.

## 7. Datasets and strict dev / held-out separation

| Role | Name | Source | n | Positive class |
| --- | --- | --- | --- | --- |
| **dev** | `pima_diabetes` | OpenML 37 (md5 3cbaa3e5…) | 768 | tested_positive |
| **dev** | `heart_cleveland` | OpenML 49 (md5 f255fbfe…) | 303 | num > 50% narrowing |
| held-out | `breast_wdbc` | OpenML 1510 (md5 7aa183d3…) | 569 | malignant |
| held-out | `liver_ilpd` | OpenML 1480 (md5 15529554…) | 583 | liver patient |
| held-out | `thyroid_sick` | OpenML 38 (md5 ee5cb4b7…) | 3772 | sick |
| held-out | `kidney_ckd` | UCI 336 / OpenML 42972 | 400 | ckd |

Held-out availability and binary framing were verified **from metadata only**
(OpenML dataset descriptions, qualities and feature lists; UCI API). No
held-out data file has been downloaded or inspected. `load_dataset` refuses
held-out names unless `allow_heldout=True` **and** `PREREGISTRATION.md`
declares `status: locked`.

Known risks to address at preregistration:
* `breast_wdbc` and `kidney_ckd` are near-separable (ceiling AUROC), which
  compresses discrimination differences. The analysis will emphasise
  calibration and complexity on these; `hepatitis` (OpenML 55, n=155) is a
  documented alternative.
* `kidney_ckd` has no OpenML md5; its sha256 is pinned at first fetch, and
  its categorical schema must be declared explicitly in the registry before
  unlocking.
* Positive-class codes for `breast_wdbc`/`liver_ilpd` ('2'/'1') are taken from
  OpenML metadata conventions and must be confirmed against the dataset
  documentation at unlock, before any model is run.

## 8. Phases

1. **Audit, environment, literature, plan.** Done.
2. **Implementation, unit tests, 1-seed dev pilot.** Code + 58 tests done; pilot pending.
3. Preregistration: freeze code (git tag), hyperparameters, selection rule,
   metrics, tests, and held-out unlock.
4. Full runs: 5×5 CV on all 6 datasets; ablations (objective sets, consequent
   mode, $R_{\max}$, ε); statistics.
5. Figures (Pareto fronts, reliability diagrams, rule bases, HV convergence)
   and manuscript.
6. Reproducibility package, internal review, submission.

## 9. Threats to validity

Small datasets (n = 303–768 in dev), meaning noisy folds and inner-OOF
objectives. The fixed λ = 1 consequent penalty has not been tuned. Fuzzy
input scaling is based on percentiles. All public benchmarks are old and
convenience-sampled, so clinical transportability is not claimed. This is
research software, not a medical device.
