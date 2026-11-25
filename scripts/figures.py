"""Publication figures (PDF) for the analysis; called from scripts/analyze.py.

Colour use follows a validated categorical order (blue, orange, aqua, yellow; adjacent-pair CVD
dE >= 11.8, normal-vision dE >= 23.8). Identity is never carried by colour alone: every series also
has a distinct marker and a legend or direct label. Grey is reserved for context (archive points,
reference lines). PDFs are written without creation-date metadata so that they are reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import (ABLATION, C_AQUA, C_BLUE, C_GRAY, C_GRID, C_ORANGE, C_TEXT, C_TEXT2,  # noqa: E402
                     C_YELLOW, CLASSIC, DATASETS, DNAME, GBDTS, MNAME, MO, MODELS, RULE_DATASETS,
                     parse_rules, representative_fold)

PDF_META = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": C_TEXT2, "axes.linewidth": 0.6, "xtick.color": C_TEXT2, "ytick.color": C_TEXT2,
    "axes.labelcolor": C_TEXT, "text.color": C_TEXT, "axes.grid": True, "grid.color": C_GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "svg.hashsalt": "fixed",
})


def _save(fig, path: Path):
    fig.savefig(path, metadata=PDF_META, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def pareto_figure(fronts, S, figdir: Path):
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.3), sharex=False)
    for j, ds in enumerate(DATASETS):
        fr = [f for f in fronts if f["dataset"] == ds and f["model"] == MO]
        pts, sel, bb = [], [], []
        for f in fr:
            T = np.asarray(f["test_front"], float)
            I = np.asarray(f["inner_front"], float)
            pts.append(T)
            sel.append(T[f["selected_index"]])
            k = min(range(len(I)), key=lambda i: (I[i, 1], I[i, 2], i))  # best inner Brier
            bb.append(T[k])
        P = np.vstack(pts)
        sel, bb = np.asarray(sel), np.asarray(bb)
        rng = np.random.default_rng(0)
        jit = rng.uniform(-0.2, 0.2, len(P))
        gb_auc = max(float(S[(S.dataset == ds) & (S.model == g)].auroc_mean.iloc[0]) for g in GBDTS)
        gb_bri = min(float(S[(S.dataset == ds) & (S.model == g)].brier_mean.iloc[0]) for g in GBDTS)
        for row, (col, ylab, ref) in enumerate([(0, "Test AUROC", gb_auc), (1, "Test Brier", gb_bri)]):
            ax = axes[row, j]
            y = 1 - P[:, 0] if col == 0 else P[:, 1]
            ax.scatter(P[:, 2] + jit, y, s=2, color=C_GRAY, alpha=0.25, linewidths=0, rasterized=True,
                       label="archive members (all folds)")
            Ls = np.unique(P[:, 2])
            med = [(L, np.median(y[P[:, 2] == L])) for L in Ls if (P[:, 2] == L).sum() >= 10]
            if med:
                mx, my = zip(*med)
                ax.plot(mx, my, color=C_BLUE, lw=1.2, label="median per literal count")
            ys = 1 - sel[:, 0] if col == 0 else sel[:, 1]
            yb = 1 - bb[:, 0] if col == 0 else bb[:, 1]
            ax.scatter([sel[:, 2].mean()], [ys.mean()], marker="*", s=70, color=C_BLUE, edgecolors="white",
                       linewidths=0.6, zorder=5, label=r"$\varepsilon$-compact (mean)")
            ax.scatter([bb[:, 2].mean()], [yb.mean()], marker="D", s=24, color=C_ORANGE, edgecolors="white",
                       linewidths=0.6, zorder=5, label="best-Brier (mean)")
            ax.axhline(ref, color=C_TEXT2, lw=0.8, ls="--", label="best GBDT (mean)")
            hi = np.percentile(P[:, 2], 99)
            ax.set_xlim(-0.5, max(6, hi + 1))
            if row == 0:
                ax.set_title(DNAME[ds])
                lo = np.percentile(y, 2)
                ax.set_ylim(max(0.45, lo - 0.02), min(1.0, max(y.max(), ref) + 0.01))
            else:
                ax.set_xlabel("Total literals")
                ax.set_ylim(max(0.0, min(y.min(), ref) - 0.005), np.percentile(y, 98) + 0.005)
            if j == 0:
                ax.set_ylabel(ylab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04),
               markerscale=1.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.6, h_pad=0.6)
    _save(fig, figdir / "pareto_fronts.pdf")


# --------------------------------------------------------------------------- #
def reliability_figure(res: Path, figdir: Path):
    P = pd.read_csv(res / "predictions.csv.gz")
    show = [(MO, C_BLUE, "o"), (CLASSIC, C_ORANGE, "s"), ("XGBoost", C_AQUA, "^"), ("CatBoost", C_YELLOW, "D")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    edges = np.linspace(0, 1, 11)
    for ax, ds in zip(axes.ravel(), DATASETS):
        ax.plot([0, 1], [0, 1], color=C_GRAY, lw=0.8, ls="--", label="perfect calibration")
        for m, c, mk in show:
            sub = P[(P.dataset == ds) & (P.model == m)]
            b = np.clip(np.digitize(sub.p.values, edges) - 1, 0, 9)
            xs, ys = [], []
            for k in range(10):
                sel = b == k
                if sel.sum() >= 10:
                    xs.append(sub.p.values[sel].mean())
                    ys.append(sub.y.values[sel].mean())
            ax.plot(xs, ys, color=c, marker=mk, ms=3.5, lw=1.2, mec="white", mew=0.5, label=MNAME[m])
        ax.set_title(DNAME[ds])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed frequency")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, figdir / "reliability.pdf")


# --------------------------------------------------------------------------- #
def cd_diagram(avg_ranks: dict, cd: float, title: str, path: Path):
    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [n for n, _ in items]
    r = np.array([v for _, v in items])
    k = len(r)
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.axis("off")
    y0 = 0.80
    ax.plot([1, k], [y0, y0], color=C_TEXT, lw=0.8)
    for t in range(1, k + 1):
        ax.plot([t, t], [y0, y0 + 0.03], color=C_TEXT, lw=0.8)
        ax.text(t, y0 + 0.05, str(t), ha="center", va="bottom", fontsize=6.5)
    ax.plot([1, 1 + cd], [1.0, 1.0], color=C_TEXT, lw=1.2)
    for t in (1, 1 + cd):
        ax.plot([t, t], [0.98, 1.02], color=C_TEXT, lw=1.2)
    ax.text(1 + cd + 0.1, 1.0, f"CD = {cd:.2f}", ha="left", va="center", fontsize=6.5)
    half = (k + 1) // 2
    step = 0.72 / max(half, k - half)
    for i, (n, v) in enumerate(zip(names, r)):
        left = i < half
        yy = y0 - 0.12 - step * (i if left else k - 1 - i)
        xt = 0.55 if left else k + 0.45
        ax.plot([v, v, xt], [y0, yy, yy], color=C_TEXT2, lw=0.6)
        col, wt = (C_BLUE, "bold") if n == MO else (C_TEXT, "normal")
        ax.text(xt + (-0.03 if left else 0.03), yy, f"{MNAME[n]} ({v:.2f})", ha="right" if left else "left",
                va="center", fontsize=6.5, color=col, fontweight=wt)
    # cliques: maximal runs of models whose rank range is < CD
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c_i, (a, b) in enumerate(cliques):
        yy = y0 - 0.05 - 0.035 * c_i
        ax.plot([r[a] - 0.04, r[b] + 0.04], [yy, yy], color=C_TEXT, lw=2.2, solid_capstyle="round")
    ax.set_title(title, fontsize=8, pad=2)
    _save(fig, path)


# --------------------------------------------------------------------------- #
def mf_figure(R, res: Path, figdir: Path):
    rules = json.loads((res / "rules.json").read_text(encoding="utf-8"))
    for ds in RULE_DATASETS:
        s, f = representative_fold(R, ds)
        entry = next(r for r in rules if r["dataset"] == ds and r["seed"] == s and r["fold"] == f
                     and r["model"] == MO)
        _, _, mfs = parse_rules(entry["rules"])
        if not mfs:
            continue
        n = len(mfs)
        fig, axes = plt.subplots(1, n, figsize=(min(7.2, 1.9 * n + 0.4), 1.5), squeeze=False)
        for ax, (name, c) in zip(axes[0], mfs):
            c1, c2, c3 = c
            span = c3 - c1
            x = np.linspace(c1 - 0.25 * span, c3 + 0.25 * span, 400)
            low = np.clip((c2 - x) / (c2 - c1), 0, 1)
            high = np.clip((x - c2) / (c3 - c2), 0, 1)
            med = np.clip(1 - low - high, 0, 1)
            for yv, col, ls, lab, xc in ((low, C_BLUE, "-", "Low", c1), (med, C_ORANGE, "--", "Medium", c2),
                                         (high, C_AQUA, "-.", "High", c3)):
                ax.plot(x, yv, color=col, ls=ls, lw=1.4)
                ax.text(xc, 1.06, lab, ha="center", va="bottom", fontsize=6, color=C_TEXT)
            ax.set_ylim(0, 1.25)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel(name)
            ax.grid(False)
        axes[0, 0].set_ylabel(r"$\mu$")
        fig.tight_layout(w_pad=0.8)
        _save(fig, figdir / f"mf_{ds}.pdf")


def make_all(R, S, fronts, res: Path, figdir: Path, mac_items=None):
    st = json.loads((res / "stats.json").read_text(encoding="utf-8"))
    pareto_figure(fronts, S, figdir)
    reliability_figure(res, figdir)
    for metric, lab in (("auroc", "AUROC"), ("brier", "Brier score")):
        cd_diagram(st[metric]["avg_ranks"], st[metric]["nemenyi_cd"],
                   f"Average ranks by {lab} over 6 datasets (Nemenyi, $\\alpha=0.05$)",
                   figdir / f"cd_{metric}.pdf")
    mf_figure(R, res, figdir)
