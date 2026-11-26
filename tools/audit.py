"""Automated reproducibility audit.

Checks
  1. prereg_hashes      : every locked file's LF-normalised SHA-256 equals PREREGISTRATION.md §8, or the
                          new hash of a documented deviation (PREREGISTRATION_DEVIATIONS.json).
  2. run_provenance     : every checkpoint was produced at a git commit whose locked files satisfy (1).
  2b. replication     : a deterministic sample of stored fits, covering every (git commit, dirty flag)
                          group of checkpoints, is re-run with the current code (tools/replicate.py) and
                          must reproduce the stored test-fold probabilities exactly.
  3. dev_pins           : dev dataset sha256 pins equal the values written in PREREGISTRATION.md.
  4. results_complete   : results/final holds every preregistered fit (seeds x datasets x folds x models).
  5. regeneration       : scripts/analyze.py --no-figures on a copy of the inputs reproduces
                          paper/tables/*.tex, paper/generated/macros.tex, stats.json and the summaries
                          byte-for-byte.
  6. citations          : every \\cite key in the manuscript exists in references.bib and is verified in
                          results/references_verification.json.
  7. latex_build        : pdflatex + bibtex + pdflatex x2 on a copy of paper/ -> no errors, no undefined
                          citations or references.
  8. pytest             : the full test suite passes.
Writes results/final/audit_report.json; exit code 1 if any check fails.
Usage: python tools/audit.py [--skip-pytest] [--skip-latex]
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from hash_manifest import LOCKED_GLOBS, manifest, sha256_normalised  # noqa: E402

RES = ROOT / "results" / "final"
PAPER = ROOT / "paper"
POST_LOCK = ["scripts/analyze.py", "scripts/figures.py", "scripts/dataset_info.py", "tools/audit.py",
             "tools/hash_manifest.py", "tools/build_submission.py", "tools/replicate.py"]


def prereg_hashes() -> dict:
    text = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    return dict(re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", text))


def allowed_hashes() -> dict[str, set]:
    allowed = {k: {v} for k, v in prereg_hashes().items()}
    devs = json.loads((ROOT / "PREREGISTRATION_DEVIATIONS.json").read_text(encoding="utf-8"))
    for d in devs:
        allowed.setdefault(d["file"], set()).add(d["new_sha256"])
    return allowed


def check_prereg():
    allowed = allowed_hashes()
    cur = manifest()
    bad = {f: h for f, h in cur.items() if f not in allowed or h not in allowed[f]}
    missing = sorted(set(allowed) - set(cur))
    return {"ok": not bad and not missing, "n_files": len(cur), "mismatched": bad, "missing": missing,
            "deviations_applied": sorted({f for f, h in cur.items() if f in allowed and h != prereg_hashes().get(f)})}


def check_provenance():
    allowed = allowed_hashes()
    commits = set()
    for p in (RES / "checkpoints").glob("*.json"):
        commits.add(json.loads(p.read_text(encoding="utf-8")).get("git_commit"))
    out, ok = {}, True
    for c in sorted(commits):
        files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", c], cwd=ROOT, text=True).split()
        locked = [f for f in files if any(Path(f).match(g) for g in LOCKED_GLOBS)]
        bad = []
        for f in locked:
            blob = subprocess.check_output(["git", "show", f"{c}:{f}"], cwd=ROOT)
            h = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
            if f not in allowed or h not in allowed[f]:
                bad.append(f)
        dirty = any(json.loads(p.read_text(encoding="utf-8")).get("git_dirty")
                    for p in (RES / "checkpoints").glob("*.json")
                    if json.loads(p.read_text(encoding="utf-8")).get("git_commit") == c)
        # A dirty working tree is accepted only if no locked file differs between that commit and the next
        # commit, i.e. the uncommitted changes present while those fits ran were untracked or post-lock files
        # (here: analysis scripts committed in the next commit). The `replication` check additionally
        # re-runs fits from every (commit, dirty) group with the current code.
        changed, nxt = [], None
        if dirty:
            later = subprocess.check_output(["git", "rev-list", "--reverse", "--ancestry-path", f"{c}..HEAD"],
                                            cwd=ROOT, text=True).split()
            nxt = later[0] if later else None
            diff = subprocess.check_output(["git", "diff", "--name-only", c, nxt or "HEAD"], cwd=ROOT,
                                           text=True).split()
            changed = [f for f in diff if any(Path(f).match(g) for g in LOCKED_GLOBS)]
        out[c[:10]] = {"locked_files": len(locked), "mismatched": bad, "any_dirty_checkpoint": dirty,
                       "next_commit": nxt[:10] if nxt else None,
                       "locked_files_changed_in_next_commit": changed}
        ok &= not bad and not changed
    return {"ok": ok and bool(commits), "commits": out}


SIZE_ORDER = ["heart_cleveland", "kidney_ckd", "breast_wdbc", "liver_ilpd", "pima_diabetes", "thyroid_sick"]
REPL_MODELS = ["LogReg", "DecisionTree", "XGBoost", "MO-Fuzzy(NSGA-II)"]


def replication_sample() -> list[str]:
    groups: dict = {}
    for p in sorted((RES / "checkpoints").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        groups.setdefault((d.get("git_commit"), d.get("git_dirty")), []).append(
            (SIZE_ORDER.index(d["record"]["dataset"]), d["record"]["model"], p.stem))
    stems = []
    for _, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for m in REPL_MODELS:
            cand = sorted(i for i in items if i[1] == m)
            if cand:
                stems.append(cand[0][2])
    return stems


def check_replication():
    from replicate import replicate
    res = [replicate(s) for s in replication_sample()]
    groups = sorted({(r["git_commit"], r["git_dirty"]) for r in res})
    return {"ok": bool(res) and all(r["ok"] for r in res), "n_replicated": len(res),
            "groups_covered": [f"{c}{'+dirty' if d else ''}" for c, d in groups], "fits": res}


def check_dev_pins():
    text = (ROOT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    pins = json.loads((ROOT / "src" / "fuzzyrisk" / "dataset_pins.json").read_text(encoding="utf-8"))
    res = {d: (pins.get(d) in text) for d in ("pima_diabetes", "heart_cleveland")}
    return {"ok": all(res.values()), "dev_pins_match_prereg": res}


def check_complete():
    meta = json.loads((RES / "run_meta.json").read_text(encoding="utf-8"))
    exp = len(meta["seeds"]) * len(meta["datasets"]) * meta["folds"] * len(meta["models"])
    return {"ok": meta["n_fits"] == exp == 1650, "n_fits": meta["n_fits"], "expected": exp,
            "git_commits": meta["git_commits_of_checkpoints"]}


def check_regeneration():
    tmp = Path(tempfile.mkdtemp(prefix="audit_"))
    try:
        r, p = tmp / "results", tmp / "paper"
        r.mkdir()
        for f in ("records.csv", "predictions.csv.gz", "fronts.json", "rules.json", "run_meta.json",
                  "dataset_info.json"):
            shutil.copy2(RES / f, r / f)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze.py"), "--results", str(r),
                        "--paper", str(p), "--no-figures"], check=True, capture_output=True, cwd=ROOT)
        diffs = []
        pairs = [(p / "generated" / "macros.tex", PAPER / "generated" / "macros.tex")]
        pairs += [(f, PAPER / "tables" / f.name) for f in sorted((p / "tables").glob("*.tex"))]
        pairs += [(r / f, RES / f) for f in ("stats.json", "summary_by_dataset.csv", "summary_groups.csv")]
        for a, b in pairs:
            if not b.exists() or not filecmp.cmp(a, b, shallow=False):
                diffs.append(str(b.relative_to(ROOT)))
        return {"ok": not diffs, "n_compared": len(pairs), "differences": diffs}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def tex_sources():
    files = [PAPER / "main.tex"] + sorted((PAPER / "sections").glob("*.tex"))
    return files


def check_citations():
    keys = set()
    for f in tex_sources():
        for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", f.read_text(encoding="utf-8")):
            keys |= {k.strip() for k in m.group(1).split(",")}
    bib = set(re.findall(r"^@\w+\{([^,]+),", (PAPER / "references.bib").read_text(encoding="utf-8"), re.M))
    ver = json.loads((ROOT / "results" / "references_verification.json").read_text(encoding="utf-8"))
    verified = {e["key"] for e in ver["entries"] if e["verified"]}
    return {"ok": keys <= bib and keys <= verified, "n_cited": len(keys), "n_bib": len(bib),
            "n_verified": len(verified), "missing_in_bib": sorted(keys - bib),
            "unverified": sorted(keys - verified), "uncited_bib_entries": sorted(bib - keys)}


def check_latex():
    tmp = Path(tempfile.mkdtemp(prefix="audit_tex_"))
    try:
        shutil.copytree(PAPER, tmp / "paper", ignore=shutil.ignore_patterns("*.aux", "*.bbl", "*.blg",
                                                                            "*.log", "*.out", "*.pdf"))
        d = tmp / "paper"
        for f in (PAPER / "figures").glob("*.pdf"):
            shutil.copy2(f, d / "figures" / f.name)
        steps = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], ["bibtex", "main"],
                 ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"],
                 ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"]]
        rcs = [subprocess.run(s, cwd=d, capture_output=True, text=True, encoding="utf-8",
                              errors="replace").returncode for s in steps]
        log = (d / "main.log").read_text(encoding="utf-8", errors="replace")
        blg = (d / "main.blg").read_text(encoding="utf-8", errors="replace") if (d / "main.blg").exists() else ""
        undefined = re.findall(r"(?:Citation|Reference) `([^']+)' .*undefined", log)
        errors = re.findall(r"^! .*", log, re.M)
        bib_err = re.findall(r"error message", blg)
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
        overfull = len(re.findall(r"Overfull \\hbox", log))
        return {"ok": all(rc == 0 for rc in rcs[:1] + rcs[2:]) and not undefined and not errors and not bib_err,
                "return_codes": rcs, "undefined": sorted(set(undefined)), "errors": errors[:5],
                "bibtex_errors": len(bib_err), "pages": int(pages.group(1)) if pages else None,
                "overfull_hbox_warnings": overfull}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_pytest():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    return {"ok": r.returncode == 0, "summary": tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--skip-latex", action="store_true")
    a = ap.parse_args()
    report = {"prereg_hashes": check_prereg(), "run_provenance": check_provenance(),
              "replication": check_replication(),
              "dev_pins": check_dev_pins(), "results_complete": check_complete(),
              "regeneration": check_regeneration(), "citations": check_citations()}
    if not a.skip_latex:
        report["latex_build"] = check_latex()
    if not a.skip_pytest:
        report["pytest"] = check_pytest()
    report["post_lock_code_sha256"] = {f: sha256_normalised(ROOT / f) for f in POST_LOCK if (ROOT / f).exists()}
    report["all_ok"] = all(v["ok"] for k, v in report.items() if isinstance(v, dict) and "ok" in v)
    (RES / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for k, v in report.items():
        if isinstance(v, dict) and "ok" in v:
            print(f"[{'PASS' if v['ok'] else 'FAIL'}] {k}")
    print("ALL OK" if report["all_ok"] else "AUDIT FAILED")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
