"""SHA-256 manifest of the preregistered code base.

Hashes are computed on line-ending-normalised content (CRLF -> LF), so a git
checkout with autocrlf does not change them. ``dataset_pins.json`` is excluded
because held-out sha256 pins are appended at first fetch (after the lock);
the dev pins it contains at lock time are listed separately in PREREGISTRATION.md.

Usage:
  python tools/hash_manifest.py            # print JSON manifest
  python tools/hash_manifest.py --markdown # print a Markdown table
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKED_GLOBS = ["src/fuzzyrisk/*.py", "scripts/fetch_data.py", "scripts/run_pilot.py",
                "scripts/run_experiment.py", "tests/*.py", "pyproject.toml", "requirements.txt",
                "requirements-lock.txt"]


def sha256_normalised(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def manifest() -> dict[str, str]:
    files = sorted({p for g in LOCKED_GLOBS for p in ROOT.glob(g) if p.is_file()})
    return {p.relative_to(ROOT).as_posix(): sha256_normalised(p) for p in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    m = manifest()
    if a.markdown:
        print("| File | SHA-256 (LF-normalised) |\n|---|---|")
        for k, v in m.items():
            print(f"| `{k}` | `{v}` |")
    else:
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
