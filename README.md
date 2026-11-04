# FuzzyGA-HealthAI

FuzzyGA-HealthAI is an academic research project investigating hybrid fuzzy
logic and genetic-algorithm methods for health-risk analysis. The repository
combines exploratory notebooks, MATLAB experiments, data preparation, results,
and research writing.

## Research workflow

1. Explore and clean the health dataset.
2. Encode clinical features for fuzzy and statistical models.
3. Establish a fuzzy-logic baseline.
4. Optimize membership functions or parameters with a genetic algorithm.
5. Compare the optimized system with a neural-network baseline.
6. Export metrics and visualizations for analysis and reporting.

The project is experimental and must not be used for medical diagnosis or
individual treatment decisions.

## Repository structure

```text
FuzzyGA-HealthAI/
├── code/
│   ├── matlab/             # MATLAB fuzzy and optimization experiments
│   └── python/             # Python data and modeling code, when present
├── data/
│   ├── raw/                # Source datasets; keep unmodified
│   └── processed/          # Derived, analysis-ready data
├── notebooks/              # Exploration, training, and visualization
├── results/                # Generated metrics and figures
├── scripts/                # Reproducibility and utility scripts
├── docs/                   # Research notes and paper sources
├── requirements.txt        # Python dependencies (currently project-managed)
└── README.md
```

## Notebook sequence

See [`notebooks/README.md`](./notebooks/README.md) for the intended sequence:
exploration, feature analysis, model training, and results visualization.

## Running experiments

The repository currently contains MATLAB source and notebooks but does not
declare Python dependencies in `requirements.txt`. Before running an
experiment:

1. Check the notebook imports and the MATLAB release/toolboxes required by the
   selected script.
2. Place source data under `data/raw/` and keep derived data under
   `data/processed/`.
3. Run notebooks in order and record dataset versions, preprocessing choices,
   random seeds, and evaluation splits.
4. Save generated figures and metrics under `results/`.

Do not commit private, identifiable, or licensed health data.

## Reporting expectations

When comparing models, report class balance, missing-value handling,
normalization, train/test separation, random seeds, and metrics such as
precision, recall, F1, ROC-AUC, and calibration where appropriate.

## Documentation

Research material is maintained in [`docs/`](./docs/), while implementation
details are documented in [`code/README.md`](./code/README.md).
