# Literature review (Phase 1)

All 50 cited works were resolved against live bibliographic APIs (Crossref,
arXiv, Semantic Scholar) by `tools/verify_references.py`; the log is
`results/references_verification.json` and `paper/references.bib` is generated
from the fetched metadata only. **35** are core literature for the review
below; 15 are method, software, or dataset citations.

> Caveat on content: verification establishes that each paper exists with the
> stated title, authors, venue and year. The one-line summaries below describe
> each work's well-known contribution at a high level; specific numerical
> claims from any paper must be re-checked against its full text before they
> are quoted in the manuscript.

## 1. Genetic and multi-objective evolutionary fuzzy systems (8)

| Key | Contribution relevant to this study |
| --- | --- |
| `ishibuchi1997` | Early formulation of fuzzy rule selection as single- vs two-objective GA (accuracy vs number of rules): the origin of the accuracy–complexity Pareto view. |
| `ishibuchi2007` | Multiobjective fuzzy GBML to analyse the interpretability–accuracy trade-off; shows training vs test behaviour along the front. |
| `herrera2008` | Taxonomy of genetic fuzzy systems (tuning vs learning of data base / rule base). Positions our method as *simultaneous learning of DB + RB*. |
| `alcala2009` | MOEA that concurrently learns rule and data bases of linguistic FRBSs: closest design ancestor of our encoding. |
| `antonelli2011` | Three-objective MOEFS: accuracy, complexity, and *partition integrity*. Motivates our ordered strong partitions with minimum gap. |
| `gacto2011` | Survey of interpretability measures (complexity- vs semantics-based, RB vs DB level); grounds our interpretability objective and reported measures. |
| `fazzolari2013` | Review of MOEFS applications; documents that objectives are almost always accuracy + complexity. |
| `fernandez2019` | Evolutionary fuzzy systems as XAI; argues for transparent, accurate models in high-stakes domains. |

## 2. Fuzzy / fuzzy-genetic models in medicine (7)

| Key | Contribution |
| --- | --- |
| `penareyes1999` | Fuzzy-genetic system for breast-cancer diagnosis (WBCD) with few, short rules; canonical medical GA-fuzzy reference. |
| `nauck1999` | Neuro-fuzzy learning of interpretable classification rules from medical data. |
| `kahramanli2008` | Hybrid neural–fuzzy system evaluated on diabetes and heart-disease data (dev-set domains here). |
| `anooj2012` | Weighted fuzzy rules for heart-disease risk-level prediction. |
| `seera2014` | Hybrid intelligent system (fuzzy min–max NN + CART + RF) for medical data classification. |
| `gorzalczany2017` | Multi-objective genetic-fuzzy approach for *interpretable and accurate* medical classification: the closest prior MO work in medicine. |
| `pota2017` | Methodology for designing rule-based fuzzy classifiers in medicine with attention to interpretability. |

**Gap observed in groups 1–2:** objectives are accuracy (or error/F1) and
complexity. Probability *calibration*, the property a risk score needs, is
neither optimised nor reported. GBDT baselines are rarely tuned with nested
protocols.

## 3. Probability calibration and scoring rules (7)

| Key | Contribution |
| --- | --- |
| `brier1950` | Brier score. |
| `gneiting2007` | Strictly proper scoring rules (Brier, log score) and the calibration/sharpness view. |
| `zadrozny2002` | Isotonic regression (and related) post-hoc calibration. |
| `niculescu2005` | Empirical study of which learners give calibrated probabilities; Platt vs isotonic post-hoc calibration. |
| `naeini2015` | Bayesian binning calibration; binned calibration error estimators (ECE/MCE). |
| `guo2017` | Modern NNs are miscalibrated; temperature scaling; popularised ECE reliability diagrams. |
| `nixon2019` | Pitfalls of ECE; adaptive (equal-mass) binning. We report both. |

## 4. Clinical prediction-model methodology (6)

| Key | Contribution |
| --- | --- |
| `steyerberg2010` | Framework of performance measures (discrimination, calibration, reclassification, clinical usefulness). |
| `vancalster2016` | Calibration hierarchy: mean, weak (intercept/slope), moderate, strong. |
| `vancalster2019` | Calibration as the "Achilles heel" of predictive analytics. |
| `huang2020` | Tutorial on calibration measurement and recalibration for clinical models. |
| `collins2015` | TRIPOD reporting guideline (our reporting checklist). |
| `christodoulou2019` | Systematic review: no performance benefit of ML over LR in clinical prediction; motivates a strong LR baseline. |

## 5. Interpretable models and tabular benchmarks (4)

| Key | Contribution |
| --- | --- |
| `rudin2019` | Use inherently interpretable models for high-stakes decisions instead of post-hoc explanations. |
| `ustun2016` | SLIM: sparse integer medical scoring systems. Our fuzzy risk score is a soft, rule-based analogue of point scores. |
| `grinsztajn2022` | Tree-based models remain state of the art on medium-sized tabular data. |
| `shwartzziv2022` | Deep tabular models do not beat XGBoost in general; GBDTs are the baseline to beat. |

## 6. Evaluation methodology (3)

| Key | Contribution |
| --- | --- |
| `demsar2006` | Statistical comparison of classifiers over multiple datasets (Friedman/Nemenyi, Wilcoxon). |
| `cawley2010` | Model-selection overfitting; nested CV for unbiased performance estimates. |
| `kaufman2012` | Data leakage formulation, detection and avoidance. |

## Method, software and dataset citations (15)

Foundations: `zadeh1965`, `mamdani1975`, `takagi1985`, `jang1993`, `deb2002`,
`zitzler1999` (hypervolume / S-metric). Evaluation methods: `benavoli2017`,
`saito2015`. Software: `blank2020` (pymoo), `pedregosa2011` (scikit-learn),
`chen2016` (XGBoost), `ke2017` (LightGBM), `prokhorenkova2018` (CatBoost).
Data: `vanschoren2014` (OpenML), `detrano1989` (Cleveland heart disease).

## Research gap (as carried into RESEARCH_PLAN.md)

1. GA-fuzzy health-risk models optimise accuracy/F1 (± rule count), never
   calibration, although calibration is a first-class clinical requirement
   (`vancalster2019`, `huang2020`).
2. Multi-objective EFS work (`ishibuchi2007`, `alcala2009`, `antonelli2011`,
   `gorzalczany2017`) uses accuracy + complexity; none of these targets a
   *probabilistic risk score* with a proper scoring rule as an objective.
3. Comparisons rarely include tuned GBDTs (`grinsztajn2022`,
   `shwartzziv2022`) or post-hoc calibrated baselines (`niculescu2005`) under
   nested, leakage-free protocols (`cawley2010`, `kaufman2012`).
