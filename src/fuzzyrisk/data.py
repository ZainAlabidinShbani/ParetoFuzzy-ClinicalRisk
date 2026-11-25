"""Dataset registry and checksum-verified loaders.

Design rules
------------
* Every dataset has a fixed *role*: ``dev`` (design, tuning, pilot) or
  ``heldout`` (untouched until the protocol is preregistered and locked).
* Held-out datasets cannot be loaded unless ``allow_heldout=True`` **and** the
  repository contains a locked ``PREREGISTRATION.md`` (line ``status: locked``).
* Integrity is checked twice: (1) the OpenML ARFF file md5 published by
  OpenML is verified by :func:`sklearn.datasets.fetch_openml`; (2) the sha256
  of the canonical CSV serialisation is compared to a pinned value.
* Column types come from the *declared schema* (ARFF nominal vs numeric), never
  from observed values, so typing cannot leak information across folds.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
PREREG_FILE = ROOT / "PREREGISTRATION.md"
PINS_FILE = Path(__file__).with_name("dataset_pins.json")


class HeldOutAccessError(PermissionError):
    """Raised when a held-out dataset is requested before preregistration."""


class ChecksumError(RuntimeError):
    """Raised when a dataset does not match its pinned checksum."""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, dict] = {
    # ---------------- development datasets ----------------
    "pima_diabetes": {
        "role": "dev",
        "source": "openml", "openml_id": 37, "openml_md5": "3cbaa3e54586aa88cf6aacb4033e4470",
        "target": "class", "positive": ["tested_positive"],
        # physiologically impossible zeros are missing values (row-wise rule, no fitting)
        "zero_as_missing": ["plas", "pres", "skin", "insu", "mass"],
        "expected_n": 768, "expected_pos": 268,
        "description": "Pima Indians Diabetes (onset of diabetes within 5 years), n=768.",
        "citation_keys": ["vanschoren2014"],
    },
    "heart_cleveland": {
        "role": "dev",
        "source": "openml", "openml_id": 49, "openml_md5": "f255fbfe272738f3da960f63a451ca0a",
        "target": "num", "positive": [">50_1", ">50_2", ">50_3", ">50_4"],
        "zero_as_missing": [],
        "expected_n": 303, "expected_pos": 138,
        "description": "Cleveland heart disease (angiographic >50% narrowing), n=303.",
        "citation_keys": ["detrano1989", "vanschoren2014"],
    },
    # ---------------- held-out datasets (locked) ----------------
    "breast_wdbc": {
        "role": "heldout",
        "source": "openml", "openml_id": 1510, "openml_md5": "7aa183d3657e364911ced0cbd6b272bd",
        # OpenML 1510 description: "prognosis (benign (1) or malignant (2))"
        "target": "Class", "positive": ["2"],
        "zero_as_missing": [],
        "expected_n": 569, "expected_pos": 212,
        "description": "Breast Cancer Wisconsin (Diagnostic), malignant vs benign, n=569.",
    },
    "liver_ilpd": {
        "role": "heldout",
        "source": "openml", "openml_id": 1480, "openml_md5": "155295549a459387fb71312caf1b8360",
        # OpenML 1480 / UCI ILPD: 416 liver patients, 167 non-patients; selector 1 = patient
        "target": "Class", "positive": ["1"],
        "zero_as_missing": [],
        "expected_n": 583, "expected_pos": 416,
        "description": "Indian Liver Patient Dataset, liver disease vs not, n=583.",
    },
    "thyroid_sick": {
        "role": "heldout",
        "source": "openml", "openml_id": 38, "openml_md5": "ee5cb4b7f41a5f44b0cd234f675ab492",
        "target": "Class", "positive": ["sick"],
        "zero_as_missing": [],
        "expected_n": 3772, "expected_pos": 231,
        "description": "Thyroid 'sick' (Garvan), sick vs negative, n=3772.",
    },
    "kidney_ckd": {
        "role": "heldout",
        "source": "uci", "uci_id": 336, "url": "https://archive.ics.uci.edu/static/public/336/data.csv",
        "target": "class", "positive": ["ckd"],
        "zero_as_missing": [],
        "expected_n": 400, "expected_pos": 250,
        # Declared schema (UCI variable table). sg/al/su are ordinal codes -> numeric.
        "schema": {
            "numeric": ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv",
                        "wbcc", "rbcc"],
            "binary": {"rbc": ["normal", "abnormal"], "pc": ["normal", "abnormal"],
                       "pcc": ["notpresent", "present"], "ba": ["notpresent", "present"],
                       "htn": ["no", "yes"], "dm": ["no", "yes"], "cad": ["no", "yes"],
                       "appet": ["good", "poor"], "pe": ["no", "yes"], "ane": ["no", "yes"]},
            "aliases": {"wc": "wbcc", "rc": "rbcc"},
        },
        "description": "Chronic Kidney Disease (UCI 336), ckd vs notckd, n=400.",
    },
}


def dev_datasets() -> list[str]:
    return [k for k, v in REGISTRY.items() if v["role"] == "dev"]


def heldout_datasets() -> list[str]:
    return [k for k, v in REGISTRY.items() if v["role"] == "heldout"]


def preregistration_locked() -> bool:
    if not PREREG_FILE.exists():
        return False
    text = PREREG_FILE.read_text(encoding="utf-8").lower()
    return "status: locked" in text


# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    numeric_cols: list[str]
    binary_cols: list[str]
    nominal_cols: list[str]
    categories: dict[str, list] = field(default_factory=dict)  # declared categories
    sha256: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def prevalence(self) -> float:
        return float(np.mean(self.y))


def _canonical_sha256(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n", float_format="%.10g")
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def _load_pins() -> dict:
    if PINS_FILE.exists():
        return json.loads(PINS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_pin(name: str, sha: str) -> None:
    pins = _load_pins()
    pins[name] = sha
    PINS_FILE.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fetch_raw(name: str, spec: dict) -> pd.DataFrame:
    if spec["source"] == "openml":
        from sklearn.datasets import fetch_openml

        DATA_CACHE.mkdir(parents=True, exist_ok=True)
        try:
            bunch = fetch_openml(data_id=spec["openml_id"], as_frame=True, data_home=str(DATA_CACHE),
                                 parser="pandas")
        except TypeError:
            # Deviation D1 (PROGRESS.md): sklearn 1.9's pandas ARFF parser fails under pandas 3
            # on integer-like nominal levels (e.g. WDBC Class {1,2}); parse the same
            # md5-verified file with sklearn's LIAC-ARFF parser instead.
            bunch = fetch_openml(data_id=spec["openml_id"], as_frame=True, data_home=str(DATA_CACHE),
                                 parser="liac-arff")
        frame = bunch.frame.copy()
        # (1) the md5 OpenML serves in the description must equal the registered md5
        served = bunch.details.get("md5_checksum")
        if served != spec["openml_md5"]:
            raise ChecksumError(f"{name}: OpenML md5 {served} != registered {spec['openml_md5']}")
        # (2) the md5 of the ARFF file actually downloaded must equal it as well
        file_id = bunch.details.get("file_id")
        arffs = list((DATA_CACHE / "openml" / "openml.org" / "data" / "v1" / "download" / str(file_id))
                     .glob("*.arff.gz"))
        if len(arffs) != 1:
            raise ChecksumError(f"{name}: cached ARFF for file_id {file_id} not found")
        md5 = hashlib.md5(gzip.decompress(arffs[0].read_bytes())).hexdigest()
        if md5 != spec["openml_md5"]:
            raise ChecksumError(f"{name}: downloaded ARFF md5 {md5} != registered {spec['openml_md5']}")
        return frame
    if spec["source"] == "uci":
        import requests

        r = requests.get(spec["url"], timeout=60, headers={"User-Agent": "dataset-fetch/0.1"})
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), na_values=["?", "\t?"], skipinitialspace=True)
    raise ValueError(spec["source"])


def load_dataset(name: str, allow_heldout: bool = False, pin_if_missing: bool = False) -> Dataset:
    """Load a registered dataset as a typed frame with a binary {0,1} target.

    Parameters
    ----------
    allow_heldout : must be True to load a held-out dataset, and additionally
        ``PREREGISTRATION.md`` must declare ``status: locked``.
    pin_if_missing : record the canonical sha256 if no pin exists yet (used
        once per dataset on first fetch; afterwards mismatches raise).
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    if spec["role"] == "heldout":
        if not allow_heldout:
            raise HeldOutAccessError(f"{name} is a held-out dataset; it is locked until preregistration.")
        if not preregistration_locked():
            raise HeldOutAccessError(f"{name}: PREREGISTRATION.md missing or not 'status: locked'.")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw_csv = DATA_RAW / f"{name}.csv"
    frame = _fetch_raw(name, spec)

    sha = _canonical_sha256(frame)
    pins = _load_pins()
    if name in pins:
        if pins[name] != sha:
            raise ChecksumError(f"{name}: canonical sha256 {sha} != pinned {pins[name]}")
    elif pin_if_missing:
        _save_pin(name, sha)
    else:
        raise ChecksumError(f"{name}: no pinned sha256; run scripts/fetch_data.py --pin first.")
    if not raw_csv.exists():
        frame.to_csv(raw_csv, index=False, lineterminator="\n", encoding="utf-8")

    meta_clean = {}
    if "schema" in spec:  # non-ARFF source: row-wise cleaning against the declared schema
        frame = frame.rename(columns=spec["schema"].get("aliases", {}))
        for c in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[c]):  # object or pandas 'str' dtype
                s_ = frame[c].astype(object).where(frame[c].notna(), None)
                s_ = s_.map(lambda t: None if t is None else str(t).strip().lower())
                frame[c] = s_.where(~s_.isin(["?", "", "nan", "none"]) & s_.notna(), np.nan).astype(object)
    target = spec["target"]
    y_raw = frame[target].astype(str).str.strip()
    y = y_raw.isin(spec["positive"]).astype(int).to_numpy()
    if "expected_n" in spec and (len(y) != spec["expected_n"] or int(y.sum()) != spec["expected_pos"]):
        raise ChecksumError(f"{name}: n={len(y)}, positives={int(y.sum())} != registered "
                            f"{spec['expected_n']}/{spec['expected_pos']} (positive-class code check)")
    X = frame.drop(columns=[target]).copy()

    for c in spec.get("zero_as_missing", []):
        X[c] = X[c].where(X[c] != 0, np.nan)

    numeric, binary, nominal, cats = [], [], [], {}
    if "schema" in spec:
        sch = spec["schema"]
        declared = set(sch["numeric"]) | set(sch["binary"]) | set(sch.get("nominal", {}))
        if set(X.columns) != declared:
            raise ChecksumError(f"{name}: columns {sorted(X.columns)} != declared schema {sorted(declared)}")
        for c in sch["numeric"]:
            v = pd.to_numeric(X[c], errors="coerce")
            meta_clean[c] = int(v.isna().sum() - X[c].isna().sum())  # non-numeric tokens -> NaN
            X[c] = v.astype(float)
            numeric.append(c)
        for c, levels in sch["binary"].items():
            bad = X[c].notna() & ~X[c].isin(levels)
            meta_clean[c] = int(bad.sum())
            X[c] = X[c].where(~bad, np.nan).astype(object)
            cats[c] = list(levels)
            binary.append(c)
        for c, levels in sch.get("nominal", {}).items():
            bad = X[c].notna() & ~X[c].isin(levels)
            meta_clean[c] = int(bad.sum())
            X[c] = X[c].where(~bad, np.nan).astype(object)
            cats[c] = list(levels)
            nominal.append(c)
    else:
        for c in X.columns:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                declared = list(X[c].cat.categories)
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
                cats[c] = declared
                (binary if len(declared) == 2 else nominal).append(c)
            elif not pd.api.types.is_numeric_dtype(X[c]):
                raise ChecksumError(f"{name}: column {c} has no declared schema")
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
                numeric.append(c)

    return Dataset(
        name=name, X=X.reset_index(drop=True), y=y, numeric_cols=numeric, binary_cols=binary,
        nominal_cols=nominal, categories=cats, sha256=sha,
        meta={**{k: v for k, v in spec.items() if k not in ("zero_as_missing",)},
              "coerced_to_missing": meta_clean},
    )
