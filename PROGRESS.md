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
* **Pilot (seed 0, 5-fold, pop 60 × 80 gen, 20 random-search iterations):**
  stopped by the Claude Code host because the system was critically low on
  memory. It completed 9/110 model×fold fits (Pima fold 0 only), so no pilot
  table is reported. The partial log is in `results/pilot/pilot.log`
  (untracked). Re-run when memory allows:
  `.venv\Scripts\python scripts\run_pilot.py --seed 0`

## Next

1. Re-run the pilot, preferably from a plain terminal with other applications closed.
2. Review the pilot, then Phase 3 (preregistration, held-out unlock).
