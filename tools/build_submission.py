"""Build the manuscript and assemble submission_pack/.

Outputs
  submission_pack/manuscript.pdf
  submission_pack/highlights.txt                 (3-5 bullets, <= 85 characters each; checked)
  submission_pack/latex_source.zip               (main.tex, sections, tables, generated, figures, .bib, .bbl)
  submission_pack/editorial_manager_metadata.txt (title, abstract with macros expanded, keywords, ...)
Usage: python tools/build_submission.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "submission_pack"


def build_pdf():
    for s in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], ["bibtex", "main"],
              ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"],
              ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"]):
        r = subprocess.run(s, cwd=PAPER, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 and s[0] != "bibtex":
            raise SystemExit(f"{' '.join(s)} failed:\n{r.stdout[-2000:]}")


def macros() -> dict:
    txt = (PAPER / "generated" / "macros.tex").read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", txt, re.M)}


def detex(s: str, mac: dict) -> str:
    for k in sorted(mac, key=len, reverse=True):
        s = s.replace("\\" + k + "{}", mac[k]).replace("\\" + k, mac[k])
    s = s.replace("{,}", ",").replace("$\\times$", "x").replace("\\times", "x")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.replace("--", "-").replace("~", " ").replace("\\%", "%").replace("{}", "")
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = s.replace("\\", "")
    return re.sub(r"\s+", " ", s).strip()


def main():
    build_pdf()
    OUT.mkdir(exist_ok=True)
    shutil.copy2(PAPER / "main.pdf", OUT / "manuscript.pdf")
    mac = macros()

    hl = [detex(m.group(1), mac) for m in re.finditer(r"\\item\s+(.*)",
                                                       (PAPER / "sections" / "highlights.tex").read_text(encoding="utf-8"))]
    too_long = [h for h in hl if len(h) > 85]
    if not 3 <= len(hl) <= 5 or too_long:
        raise SystemExit(f"highlights invalid: n={len(hl)}, too long: {too_long}")
    (OUT / "highlights.txt").write_text("\n".join(f"- {h}" for h in hl) + "\n", encoding="utf-8")

    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    title = detex(re.search(r"\\title\{(.*?)\}", main_tex, re.S).group(1), mac)
    raw = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.S).group(1)
    abstract = detex(re.sub(r"^\s*%.*$", "", raw, flags=re.M), mac)
    kw = [detex(k, mac) for k in re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", main_tex, re.S)
          .group(1).split("\\sep")]
    meta = f"""EDITORIAL MANAGER SUBMISSION METADATA (Applied Soft Computing)
=================================================================

Article type: Full Length Article

Title:
{title}

Abstract ({len(abstract.split())} words):
{abstract}

Keywords:
{'; '.join(kw)}

Suggested classifications / subject areas:
- Fuzzy systems and fuzzy logic (genetic fuzzy systems, Takagi-Sugeno models)
- Evolutionary computation (multi-objective evolutionary algorithms, NSGA-II)
- Interpretable / explainable machine learning
- Applications in medicine and healthcare (clinical risk prediction, calibration)

Suggested reviewers:
The authors must choose suggested reviewers themselves and check their
availability, conflicts of interest and current contact details. No names or
emails were generated automatically. The expertise the paper needs:
- multi-objective evolutionary fuzzy systems and interpretability measures
  (research area of refs. ishibuchi2007, alcala2009, antonelli2011, fazzolari2013, gacto2011);
- calibration of clinical prediction models (research area of refs. vancalster2016,
  vancalster2019, huang2020);
- benchmarking of tabular machine-learning models (research area of refs. grinsztajn2022, demsar2006).

Highlights: see highlights.txt
Manuscript: manuscript.pdf (elsarticle, preprint 12pt); LaTeX source: latex_source.zip
Data availability: public datasets (OpenML, UCI); checksums verified by the loaders.
Declaration of generative AI use: included in the manuscript (authors to review).
Funding / competing interests / CRediT: placeholders in the manuscript, to be completed by the authors.
"""
    (OUT / "editorial_manager_metadata.txt").write_text(meta, encoding="utf-8")

    with zipfile.ZipFile(OUT / "latex_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in ["main.tex", "references.bib", "main.bbl"]:
            z.write(PAPER / p, p)
        for d in ("sections", "tables", "generated", "figures"):
            for f in sorted((PAPER / d).glob("*")):
                if f.is_file():
                    z.write(f, f"{d}/{f.name}")
    print("submission pack:", sorted(p.name for p in OUT.iterdir()))
    for h in hl:
        print(f"  ({len(h)} chars) {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
