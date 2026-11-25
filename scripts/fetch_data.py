"""Fetch and checksum-verify the DEVELOPMENT datasets only.

Held-out datasets are never fetched here; they are unlocked in Phase 3.
Usage:  python scripts/fetch_data.py [--pin]
  --pin   record the canonical sha256 on first fetch (afterwards mismatches fail)
"""
import argparse
import json

from fuzzyrisk.data import dev_datasets, load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--pin", action="store_true")
args = ap.parse_args()
for name in dev_datasets():
    ds = load_dataset(name, pin_if_missing=args.pin)
    print(json.dumps({"dataset": name, "n": len(ds.y), "n_features_raw": ds.X.shape[1],
                      "prevalence": round(ds.prevalence, 4), "numeric": ds.numeric_cols,
                      "binary": ds.binary_cols, "nominal": ds.nominal_cols,
                      "missing_cells": int(ds.X.isna().sum().sum()), "sha256": ds.sha256}))
