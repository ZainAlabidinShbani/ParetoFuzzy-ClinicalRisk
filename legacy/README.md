# Legacy material (archived, superseded)

This folder preserves the project's earlier exploratory work unchanged. It is
superseded by the Python package in `src/fuzzyrisk/` but kept for provenance.

| Path | Content | Status |
| --- | --- | --- |
| `matlab/` | `fuzzy.m`, `fuzzy2.m`, `fuzzy3.m`: Mamdani FIS toy experiments (old `newfis` API) on **synthetic** heart-rate/temperature data | Superseded; not used for any reported result |
| `notebooks/` | EDA, feature analysis, expert Mamdani baseline (scikit-fuzzy) and single-objective GA tuning of membership functions (accuracy fitness) on the Kaggle stroke dataset | Superseded; ideas reused (MF-parameter chromosome, GA tuning) |
| `data_processed/` | Derived artefacts of the stroke dataset produced by the notebooks | Kept for reproducibility of the legacy notebooks only |
| `docs/` | Early notes, review paragraphs (`.docx`), outline, and empty paper/bib stubs | Superseded by `paper/` and `RESEARCH_PLAN.md` |

The raw stroke CSV (`data/raw/healthcare-dataset-stroke-data.csv`) is no longer
tracked (`data/raw/` is git-ignored); it remains in earlier git history. It is the
Kaggle "Stroke Prediction Dataset" (fedesoriano), whose original provenance is
undocumented, which is one reason the new study uses OpenML/UCI datasets with
documented sources and checksums.

What was reused conceptually:
- Normalised MF-parameter chromosome (`nb4.py`), now generalised to ordered
  strong fuzzy partitions with a minimum-gap repair.
- GA tuning of a fuzzy rule base, now the **single-objective GA-Fuzzy baseline**
  (`src/fuzzyrisk/optim.py`).

What was not reused: the scikit-fuzzy control-system simulation (too slow for
population-based search: one `ControlSystemSimulation` call per sample) and the
accuracy-only fitness, which the study identifies as the gap.
