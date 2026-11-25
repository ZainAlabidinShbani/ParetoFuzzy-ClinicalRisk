# FuzzyGA-HealthAI code

This directory contains implementation experiments for the FuzzyGA-HealthAI
research workflow.

## Subdirectories

- [`python/`](./python/) — Python data preparation and modeling code, when
  present
- [`matlab/`](./matlab/) — MATLAB fuzzy-logic and genetic-algorithm scripts

## MATLAB scripts

The current MATLAB scripts (`fuzzy.m`, `fuzzy2.m`, and `fuzzy3.m`) represent
related experiment variants. Review each script's input assumptions before
running it; outputs and figures should be saved under the repository's
`results/` directory rather than mixed with source files.

## Reproducibility

Record the MATLAB release, required toolboxes, input dataset version,
preprocessing choices, and random seeds for every reported experiment.
