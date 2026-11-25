"""Write results/final/dataset_info.json (n, prevalence, feature types, source) for the paper tables."""
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
warnings.simplefilter("ignore")

from fuzzyrisk.data import REGISTRY, load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
out = {}
for name, spec in REGISTRY.items():
    ds = load_dataset(name, allow_heldout=spec["role"] == "heldout")
    src = (f"OpenML {spec['openml_id']}" if spec["source"] == "openml" else f"UCI {spec['uci_id']}")
    out[name] = {"role": spec["role"], "n": int(len(ds.y)), "n_pos": int(ds.y.sum()),
                 "prevalence": round(float(ds.prevalence), 6), "n_num": len(ds.numeric_cols),
                 "n_bin": len(ds.binary_cols), "n_nom": len(ds.nominal_cols),
                 "missing_cells": int(ds.X.isna().sum().sum()), "source": src, "sha256": ds.sha256}
p = ROOT / "results" / "final" / "dataset_info.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(out, indent=1))
