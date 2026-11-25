"""Verify candidate references against live bibliographic APIs and emit BibTeX.

Every entry in ``tools/reference_candidates.json`` is resolved through
Crossref (DOI), the arXiv API (arXiv id) or Semantic Scholar (title match).
An entry is accepted only if the fetched title matches the expected title
(normalised similarity >= 0.90) AND the fetched first-author surname matches.
BibTeX is generated exclusively from the *fetched* metadata, never from the
candidate list, so no bibliographic field is typed by hand.

Outputs
-------
results/references_verification.json : full log (accepted and rejected)
paper/references.bib                  : accepted entries only

Usage:  python tools/verify_references.py
"""
from __future__ import annotations

import datetime as _dt
import difflib
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "tools" / "reference_candidates.json"
OUT_JSON = ROOT / "results" / "references_verification.json"
OUT_BIB = ROOT / "paper" / "references.bib"
HEADERS = {"User-Agent": "refcheck-script/0.1 (python-requests)"}
TITLE_THRESHOLD = 0.90


def norm(s: str) -> str:
    s = (s or "").translate(str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "ß": "ss"}))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"<[^>]+>", " ", s)  # strip JATS/MathML tags
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def get(url: str, params=None, tries: int = 5) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=40)
            if r.status_code == 429 or r.status_code >= 500:
                last = r
                time.sleep(3 * (i + 1))
                continue
            return r
        except requests.RequestException as e:  # pragma: no cover - network
            last = e
            time.sleep(3 * (i + 1))
    if isinstance(last, requests.Response):
        return last
    raise RuntimeError(f"request failed: {url}: {last}")


# --------------------------------------------------------------------------- #
# Resolvers -> normalised record dict
# --------------------------------------------------------------------------- #
def resolve_crossref(doi: str) -> dict:
    r = get(f"https://api.crossref.org/works/{requests.utils.quote(doi, safe='')}")
    if r.status_code != 200:
        return {"ok": False, "error": f"crossref HTTP {r.status_code}"}
    m = r.json()["message"]
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        dp = (m.get(k) or {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            year = int(dp[0][0])
            break
    authors = [
        {"family": a.get("family") or a.get("name", ""), "given": a.get("given", "")}
        for a in m.get("author", [])
    ]
    return {
        "ok": True,
        "source": "crossref",
        "type": m.get("type"),
        "title": ": ".join([" ".join(m.get("title", [""]))] + [t for t in m.get("subtitle", []) if t]),
        "authors": authors,
        "year": year,
        "container": " ".join(m.get("container-title", [""])),
        "volume": m.get("volume"),
        "issue": m.get("issue"),
        "pages": m.get("page") or m.get("article-number"),
        "publisher": m.get("publisher"),
        "doi": m.get("DOI"),
        "url": f"https://doi.org/{m.get('DOI')}",
    }


def s2_lookup(path: str, params: dict | None = None) -> dict | None:
    fields = "title,year,authors,venue,externalIds,journal,publicationVenue"
    p = {"fields": fields}
    if params:
        p.update(params)
    r = get(f"https://api.semanticscholar.org/graph/v1/{path}", params=p)
    time.sleep(1.2)
    if r.status_code != 200:
        return None
    j = r.json()
    if "data" in j:
        return j["data"][0] if j["data"] else None
    return j


def resolve_arxiv(aid: str) -> dict:
    r = get("http://export.arxiv.org/api/query", params={"id_list": aid})
    time.sleep(3.1)  # arXiv API etiquette
    if r.status_code != 200:
        return {"ok": False, "error": f"arxiv HTTP {r.status_code}"}
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    e = root.find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return {"ok": False, "error": "arxiv: no entry"}
    title = " ".join(e.find("a:title", ns).text.split())
    authors = []
    for a in e.findall("a:author", ns):
        full = a.find("a:name", ns).text.strip()
        parts = full.split()
        authors.append({"family": parts[-1], "given": " ".join(parts[:-1])})
    year = int(e.find("a:published", ns).text[:4])
    rec = {
        "ok": True, "source": "arxiv", "type": "preprint", "title": title,
        "authors": authors, "year": year, "container": None, "arxiv": aid,
        "url": f"https://arxiv.org/abs/{aid}", "doi": None,
    }
    # Semantic Scholar venue is recorded as a note only: its year/volume fields for
    # arXiv-mapped papers proved unreliable (e.g. CoRR volumes), so they are not used.
    s2 = s2_lookup(f"paper/arXiv:{aid}")
    if s2:
        venue = (s2.get("publicationVenue") or {}).get("name") or s2.get("venue")
        if venue and "arxiv" not in venue.lower():
            rec["venue_note"] = venue
    return rec


def resolve_s2_title(title: str) -> dict:
    s2 = None
    for attempt in range(4):  # the public endpoint intermittently returns 429/empty
        s2 = s2_lookup("paper/search/match", {"query": title})
        if s2:
            break
        time.sleep(5 * (attempt + 1))
    if not s2:
        return {"ok": False, "error": "semantic scholar: no match"}
    authors = []
    for a in s2.get("authors", []):
        parts = a["name"].split()
        authors.append({"family": parts[-1], "given": " ".join(parts[:-1])})
    ext = s2.get("externalIds") or {}
    jn = s2.get("journal") or {}
    venue = (s2.get("publicationVenue") or {}).get("name") or s2.get("venue")
    return {
        "ok": True, "source": "semanticscholar", "type": "s2", "title": s2["title"],
        "authors": authors, "year": s2.get("year"), "container": venue,
        "volume": jn.get("volume"), "pages": (jn.get("pages") or "").strip() or None,
        "doi": ext.get("DOI"), "s2_paper_id": s2.get("paperId"),
        "url": f"https://www.semanticscholar.org/paper/{s2.get('paperId')}",
    }


# --------------------------------------------------------------------------- #
# BibTeX
# --------------------------------------------------------------------------- #
def tex_escape(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    s = " ".join(s.split())
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


def to_bibtex(key: str, rec: dict) -> str:
    t = rec.get("type")
    container = rec.get("container") or ""
    if t == "journal-article":
        etype, cfield = "article", "journal"
    elif t in ("proceedings-article", "book-chapter"):
        etype, cfield = ("inproceedings", "booktitle") if t == "proceedings-article" else ("incollection", "booktitle")
    elif t == "book":
        etype, cfield = "book", None
    elif t in ("preprint", "s2"):
        conf_words = ("conference", "proceedings", "neural information", "international conference",
                      "icml", "neurips", "nips", "cvpr", "workshop", "kdd", "aaai")
        if container and any(w in container.lower() for w in conf_words):
            etype, cfield = "inproceedings", "booktitle"
        elif container:
            etype, cfield = "article", "journal"
        else:
            etype, cfield = "misc", None
    else:
        etype, cfield = "misc", "howpublished"
    authors = " and ".join(
        (f"{tex_escape(a['family'])}, {tex_escape(a['given'])}" if a.get("given")
         else "{" + tex_escape(a["family"]) + "}")  # group/consortium author: keep verbatim
        for a in rec["authors"]
    ) or "Anonymous"
    fields = [("author", authors), ("title", "{" + tex_escape(rec["title"]) + "}"), ("year", str(rec["year"]))]
    if cfield and container:
        fields.append((cfield, tex_escape(container)))
    for f in ("volume", "issue", "pages", "publisher"):
        v = rec.get(f)
        if v and not (f == "publisher" and etype in ("article",)):
            fields.append(("number" if f == "issue" else f, tex_escape(str(v)).replace("-", "--") if f == "pages" else tex_escape(str(v))))
    if rec.get("doi"):
        fields.append(("doi", rec["doi"]))
    if rec.get("arxiv"):
        fields.append(("eprint", rec["arxiv"]))
        fields.append(("archivePrefix", "arXiv"))
    if etype == "misc" and rec.get("arxiv"):
        fields.append(("howpublished", f"arXiv:{rec['arxiv']}"))
    if rec.get("venue_note"):
        fields.append(("note", f"Peer-reviewed version: {tex_escape(rec['venue_note'])}"))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)  # title is double-braced (case kept)
    return f"@{etype}{{{key},\n{body}\n}}\n"


def main() -> int:
    cands = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    log, bib = [], []
    for c in cands:
        try:
            if c.get("doi"):
                rec = resolve_crossref(c["doi"])
            elif c.get("arxiv"):
                rec = resolve_arxiv(c["arxiv"])
            else:
                rec = resolve_s2_title(c["s2_title"])
        except Exception as e:  # pragma: no cover - network
            rec = {"ok": False, "error": repr(e)}
        entry = {k: c[k] for k in ("key", "group", "role")}
        entry["query"] = {k: c[k] for k in ("doi", "arxiv", "s2_title") if k in c}
        entry["expected_title"] = c["title"]
        if rec.get("ok"):
            ts = sim(c["title"], rec["title"])
            # first *person* author: skip leading group/consortium authors (no given name)
            persons = [a for a in rec["authors"] if a.get("given")]
            fa = (persons or rec["authors"] or [{"family": ""}])[0]["family"]
            am = bool(fa) and (norm(c["author"]) in norm(fa) or norm(fa) in norm(c["author"]))
            entry.update(
                fetched_title=rec["title"], fetched_first_author=fa, fetched_year=rec["year"],
                fetched_venue=rec.get("container") or rec.get("venue_note"), source=rec["source"], doi=rec.get("doi"),
                url=rec.get("url"), title_similarity=round(ts, 3), first_author_match=am,
                verified=bool(ts >= TITLE_THRESHOLD and am),
            )
            if entry["verified"]:
                bib.append(to_bibtex(c["key"], rec))
        else:
            entry.update(verified=False, error=rec.get("error"))
        log.append(entry)
        print(f"[{'OK ' if entry['verified'] else 'XX '}] {c['key']:<20} {entry.get('title_similarity', '-')}"
              f" {entry.get('fetched_year', '')} {entry.get('error', '')}", flush=True)
        time.sleep(0.3)

    n_ok = sum(e["verified"] for e in log)
    summary = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "method": "Crossref /works/{doi}; arXiv Atom API; Semantic Scholar graph API (search/match). "
                  f"Accepted iff normalised title similarity >= {TITLE_THRESHOLD} and first-author surname match. "
                  "BibTeX generated only from fetched metadata.",
        "n_candidates": len(log), "n_verified": n_ok,
        "n_verified_core": sum(e["verified"] and e["role"] == "core" for e in log),
        "entries": log,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_BIB.parent.mkdir(parents=True, exist_ok=True)
    header = ("% AUTO-GENERATED by tools/verify_references.py from live API metadata.\n"
              "% Do not edit by hand; edit tools/reference_candidates.json and re-run.\n\n")
    OUT_BIB.write_text(header + "\n".join(bib), encoding="utf-8")
    print(f"verified {n_ok}/{len(log)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
