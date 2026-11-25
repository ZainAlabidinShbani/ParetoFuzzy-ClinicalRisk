# FuzzyGA-HealthAI notebooks

The notebooks are intended to be run as a reproducible analysis pipeline.
Execute them in order so that feature definitions and preprocessing decisions
are established before model comparison.

| Order | Notebook | Responsibility |
| --- | --- | --- |
| 1 | `01_data_exploration.ipynb` | Dataset overview, quality checks, and initial insights |
| 2 | `02_feature_analysis.ipynb` | Cleaning, encoding, scaling, and feature selection |
| 3 | `03_model_training.ipynb` | Fuzzy baseline, genetic optimization, and neural-network baseline |
| 4 | `04_results_visualization.ipynb` | Metrics, confusion matrices, ROC curves, and comparisons |

## Preprocessing decisions to document

- Missing-value treatment, including BMI and unknown smoking status
- Categorical encoding and any fuzzy-category mapping
- Numeric scaling for age, glucose, and BMI
- Target representation used by each model
- Dataset split and class-balance strategy

Keep notebook outputs and generated figures under `results/` when they are
intended for reporting. Do not include private or identifiable health data.
