#!/usr/bin/env python3
"""Composed LuminaST main-claim figure.

This is a full figure-level claim chain, not a collection of isolated panels.
It is clean-room inspired by the supplied STPAINTER paper's figure grammar but
uses LuminaST-specific labels, claim gates, and explicit evidence-tier notes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "results" / "figures" / "lumina_composed_main_claim.png"

C = {
    "blue": "#2F6FA7",
    "teal": "#1AA39A",
    "green": "#5BA85B",
    "orange": "#E7903C",
    "red": "#C94C4C",
    "purple": "#7D5FB2",
    "gray": "#65717E",
    "light": "#F6F8FA",
    "ink": "#1F2328",
}


def _load_metrics():
    path = PROJECT_ROOT / "results" / "benchmark" / "lumina_sweep_latest.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            rows = payload.get("sweep") or payload.get("metrics") or payload.get("rows") or []
            xs, pcc, ssim, rmse = [], [], [], []
            for row in rows:
                if isinstance(row, dict):
                    xs.append(int(row.get("n_hvg", row.get("n_genes", len(xs) + 1))))
                    pcc.append(float(row.get("pcc", row.get("pearson", np.nan))))
                    ssim.append(float(row.get("ssim", np.nan)))
                    rmse.append(float(row.get("rmse", np.nan)))
            if xs and np.isfinite(pcc).any():
                return (
                    xs,
                    {"PCC↑": pcc, "SSIM↑": ssim, "RMSE↓": rmse},
                    f"local-small: {path.relative_to(PROJECT_ROOT)}",
                )
        except Exception:
            pass
    return (
        [10, 20, 50, 100, 200, 300],
        {
            "PCC↑": [0.52, 0.61, 0.70, 0.76, 0.79, 0.80],
            "SSIM↑": [0.46, 0.55, 0.63, 0.69, 0.73, 0.75],
            "RMSE↓": [0.42, 0.35, 0.29, 0.24, 0.22, 0.21],
        },
        "demo/planning: replace with benchmark JSON before paper claims",
    )


def _label(ax, letter):
    ax.text(-0.06, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def _badge(ax, text, xy=(0.98, 0.02)):
    ax.text(
        *xy,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=C["gray"],
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#D0D7DE", lw=0.7),
    )


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            fc=color + "22",
            ec=color,
            lw=1.25,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=C["ink"])


def _arrow(ax, a, b, text=None):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=C["gray"], lw=1.15)
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + 0.035,
            text,
            ha="center",
            fontsize=7,
            color=C["gray"],
        )


def draw_claim_spine(ax):
    ax.set_axis_off()
    steps = [
        ("1 Sparse ST", C["blue"]),
        ("2 Enhance", C["purple"]),
        ("3 Quantify", C["orange"]),
        ("4 Spatial proof", C["teal"]),
        ("5 Biology", C["green"]),
        ("6 Claim gate", C["red"]),
    ]
    x0, gap, w = 0.02, 0.018, 0.145
    for i, (txt, col) in enumerate(steps):
        x = x0 + i * (w + gap)
        _box(ax, x, 0.27, w, 0.46, txt, col)
        if i < len(steps) - 1:
            _arrow(ax, (x + w, 0.50), (x + w + gap, 0.50), "therefore" if i in (1, 3) else None)
    ax.text(
        0.02,
        0.86,
        "Figure-level logic chain: every panel answers a claim-gating question",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.08,
        "Evidence tier: demo/planning unless metric JSON and real validation artifacts are present. Clean-room: visual grammar only, no copied paper artwork.",
        fontsize=8,
        color=C["gray"],
    )


def panel_input(ax):
    ax.set_axis_off()
    _label(ax, "A")
    ax.set_title("Input problem + reference context", loc="left", fontsize=11)
    rng = np.random.default_rng(2)
    for i, (x, y, col) in enumerate(
        [(0.12, 0.45, C["blue"]), (0.28, 0.58, C["orange"]), (0.20, 0.24, C["teal"])]
    ):
        pts = rng.normal([x, y], [0.035, 0.045], size=(70, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=col, alpha=0.55, linewidths=0)
    _box(ax, 0.50, 0.52, 0.38, 0.22, "optional atlas\nlabels + genes", C["teal"])
    _box(ax, 0.50, 0.20, 0.38, 0.22, "target ST\nsparse counts", C["blue"])
    _arrow(ax, (0.40, 0.45), (0.50, 0.62))
    _arrow(ax, (0.40, 0.36), (0.50, 0.31))
    ax.text(
        0.02,
        0.04,
        "Claim question: what information enters the enhancement?",
        fontsize=7.5,
        color=C["gray"],
    )


def panel_workflow(ax):
    ax.set_axis_off()
    _label(ax, "B")
    ax.set_title("LuminaST workflow with claim gates", loc="left", fontsize=11)
    boxes = [
        (0.04, "schema\nvalidate", C["blue"]),
        (0.28, "latent\nenhance", C["purple"]),
        (0.52, "impute\n+ uncertainty", C["orange"]),
        (0.76, "claim\nledger", C["red"]),
    ]
    for x, text, col in boxes:
        _box(ax, x, 0.46, 0.18, 0.25, text, col)
    for x in [0.22, 0.46, 0.70]:
        _arrow(ax, (x, 0.585), (x + 0.06, 0.585))
    _box(
        ax,
        0.30,
        0.13,
        0.38,
        0.20,
        "downstream biology checks\nmarkers · pathways · LR · protein",
        C["green"],
    )
    _arrow(ax, (0.61, 0.46), (0.52, 0.33), "validate")
    _badge(ax, "method + gate")


def panel_metrics(ax):
    _label(ax, "C")
    xs, curves, source = _load_metrics()
    for (name, vals), col in zip(curves.items(), [C["blue"], C["green"], C["red"]]):
        ax.plot(xs, vals, marker="o", lw=2.0, color=col, label=name)
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs], fontsize=7)
    ax.set_title("Quantitative recovery trend", fontsize=11)
    ax.set_xlabel("gene panel / HVG size")
    ax.set_ylabel("metric")
    ax.grid(ls=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    _badge(ax, source)


def panel_spatial(ax):
    _label(ax, "D")
    rng = np.random.default_rng(5)
    centers = np.array([[0.25, 0.25], [0.68, 0.32], [0.45, 0.72]])
    pts = np.vstack([rng.normal(c, 0.075, size=(170, 2)) for c in centers])
    pts = np.clip(pts, 0, 1)
    gene = np.exp(-((pts[:, 0] - 0.68) ** 2 + (pts[:, 1] - 0.32) ** 2) / 0.022)
    raw = gene * rng.binomial(1, 0.36, size=gene.size)
    imp = 0.78 * gene + 0.15 * rng.random(gene.size)
    ax.scatter(pts[:, 0], pts[:, 1], c=raw, s=7, cmap="magma", vmin=0, vmax=1, linewidths=0)
    ax.scatter(pts[:, 0] + 1.15, pts[:, 1], c=imp, s=7, cmap="magma", vmin=0, vmax=1, linewidths=0)
    ax.text(0.5, 1.03, "raw", ha="center", fontsize=8)
    ax.text(1.65, 1.03, "enhanced", ha="center", fontsize=8)
    ax.add_patch(Rectangle((-0.03, -0.03), 1.06, 1.06, fill=False, ec="#D0D7DE"))
    ax.add_patch(Rectangle((1.12, -0.03), 1.06, 1.06, fill=False, ec="#D0D7DE"))
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 2.25)
    ax.set_ylim(-0.08, 1.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Spatial marker restoration", fontsize=11)
    _badge(ax, "spatial proof")


def panel_concordance(ax):
    _label(ax, "E")
    labels = ["Epi", "T", "Myeloid", "Fib", "Endo"]
    raw = np.array([0.50, 0.16, 0.14, 0.13, 0.07])
    enh = np.array([0.48, 0.18, 0.15, 0.12, 0.07])
    y = np.arange(len(labels))
    ax.barh(y + 0.18, raw, 0.32, color="#B7C7D9", label="raw/transfer")
    ax.barh(y - 0.18, enh, 0.32, color=C["teal"], label="enhanced")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("fraction")
    ax.grid(axis="x", ls=":", alpha=0.35)
    ax.set_title("Annotation concordance gate", fontsize=11)
    ax.legend(fontsize=7, frameon=False)
    _badge(ax, "replace with Sankey when labels exist")


def panel_biology(ax):
    _label(ax, "F")
    genes = ["IL7R", "GZMB", "FOXP3", "CD68", "COL1A1"]
    groups = ["CD4 Tn", "CD8 Teff", "Treg", "Macro", "Fib"]
    rng = np.random.default_rng(9)
    mean = np.eye(5) * 0.8 + rng.random((5, 5)) * 0.2
    pct = np.clip(mean + rng.normal(0.05, 0.10, mean.shape), 0.05, 1)
    for i in range(5):
        for j in range(5):
            ax.scatter(
                j,
                i,
                s=30 + 230 * pct[i, j],
                c=[mean[i, j]],
                cmap="viridis",
                vmin=0,
                vmax=1,
                ec="#24292F",
                lw=0.3,
            )
    ax.set_xticks(range(5), genes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(5), groups, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Biology validation gate", fontsize=11)
    _badge(ax, "size=fraction · color=mean")


def render(out_path: Path = DEFAULT_OUT):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 6, height_ratios=[0.48, 1, 1, 1])
    draw_claim_spine(fig.add_subplot(gs[0, :]))
    panel_input(fig.add_subplot(gs[1, 0:2]))
    panel_workflow(fig.add_subplot(gs[1, 2:4]))
    panel_metrics(fig.add_subplot(gs[1, 4:6]))
    panel_spatial(fig.add_subplot(gs[2, 0:4]))
    panel_concordance(fig.add_subplot(gs[2, 4:6]))
    panel_biology(fig.add_subplot(gs[3, 0:2]))
    ax_note = fig.add_subplot(gs[3, 2:6])
    ax_note.set_axis_off()
    _label(ax_note, "G")
    ax_note.set_title("Integrated claim statement", loc="left", fontsize=11)
    text = (
        "Sparse ST + optional atlas context → LuminaST enhancement → quantitative recovery → spatial marker coherence → "
        "annotation and biological validation → claim ledger.\n\n"
        "Interpretation: this composed figure is a planning/demo main-claim assembly. It shows what each panel must prove "
        "before manuscript claims are upgraded; it does not assert broad pan-cancer or baseline-superiority results without real benchmark evidence."
    )
    ax_note.text(0.02, 0.70, text, fontsize=10, va="top", wrap=True)
    _box(ax_note, 0.02, 0.12, 0.28, 0.18, "Evidence tier\ndemo/planning", C["orange"])
    _box(ax_note, 0.36, 0.12, 0.28, 0.18, "Clean-room\nvisual grammar only", C["blue"])
    _box(ax_note, 0.70, 0.12, 0.25, 0.18, "Next gate\nreal benchmark JSON", C["green"])
    fig.suptitle(
        "LuminaST composed main-claim figure: from sparse ST to claim-gated biological validation",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    render(args.out)
    try:
        display_path = args.out.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = args.out
    print(f"wrote {display_path}")


if __name__ == "__main__":
    main()
