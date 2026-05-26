#!/usr/bin/env python3
"""Clean-room LuminaST manuscript visualization storyboard.

This figure is informed by reading the supplied STPAINTER PDF, but it does not
copy source artwork, captions, logos, or panel wording.  The goal is to encode
similar *visual logic* for this repository: workflow schematic, quality trends,
spatial marker recovery, cluster-to-label concordance, and sub-lineage checks.

The script is deterministic and uses local benchmark JSON when available;
otherwise it renders clearly labelled planning/demo values so unsupported paper
claims are not implied.
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
DEFAULT_OUT = PROJECT_ROOT / "results" / "figures" / "lumina_reference_visual_story.png"

PALETTE = {
    "blue": "#356AA0",
    "teal": "#20A39E",
    "green": "#5BA85B",
    "orange": "#E7903C",
    "red": "#C94C4C",
    "purple": "#7D5FB2",
    "gray": "#6E7781",
    "light": "#F6F8FA",
}


def _load_metric_series() -> tuple[list[int], dict[str, list[float]], str]:
    """Return local benchmark-like curves and provenance label."""
    candidates = [
        PROJECT_ROOT / "results" / "benchmark" / "lumina_sweep_latest.json",
        PROJECT_ROOT.parent / "results" / "paper_metrics" / "l-f2_metrics_stub.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            # Accept a few plausible local schemas without making the script brittle.
            for key in ("sweep", "metrics", "rows", "results"):
                rows = payload.get(key) if isinstance(payload, dict) else None
                if isinstance(rows, list) and rows:
                    xs, pcc, ssim, rmse = [], [], [], []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        xs.append(int(row.get("n_hvg", row.get("n_genes", len(xs) + 1))))
                        pcc.append(float(row.get("pcc", row.get("pearson", np.nan))))
                        ssim.append(float(row.get("ssim", row.get("spatial_ssim", np.nan))))
                        rmse.append(float(row.get("rmse", np.nan)))
                    if xs and np.isfinite(pcc).any():
                        return (
                            xs,
                            {"PCC": pcc, "SSIM": ssim, "RMSE": rmse},
                            f"local metrics: {path.relative_to(PROJECT_ROOT)}",
                        )
    xs = [10, 20, 50, 100, 200, 300]
    curves = {
        "PCC": [0.52, 0.61, 0.70, 0.76, 0.79, 0.80],
        "SSIM": [0.46, 0.55, 0.63, 0.69, 0.73, 0.75],
        "RMSE": [0.42, 0.35, 0.29, 0.24, 0.22, 0.21],
    }
    return xs, curves, "planning/demo values — replace with benchmark JSON before paper claims"


def _panel_label(ax, label: str) -> None:
    ax.text(-0.04, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def _draw_workflow(ax) -> None:
    ax.set_axis_off()
    _panel_label(ax, "A")
    ax.set_title("LuminaST clean-room workflow", loc="left", fontsize=11, pad=8)
    boxes = [
        (0.03, 0.58, 0.18, 0.25, "Target ST slice\nraw sparse counts", PALETTE["blue"]),
        (0.29, 0.58, 0.18, 0.25, "Reference atlas\noptional labels", PALETTE["teal"]),
        (0.55, 0.58, 0.18, 0.25, "Schema + latent\nenhancement", PALETTE["purple"]),
        (0.78, 0.58, 0.18, 0.25, "Imputed matrix\n+ uncertainty", PALETTE["orange"]),
        (0.55, 0.18, 0.18, 0.22, "Biology checks\nmarkers / LR / paths", PALETTE["green"]),
        (0.78, 0.18, 0.18, 0.22, "Claim ledger\nfigure package", PALETTE["red"]),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.025",
                linewidth=1.2,
                edgecolor=color,
                facecolor=color + "22",
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color="#1F2328")
    arrows = [
        ((0.21, 0.705), (0.29, 0.705)),
        ((0.47, 0.705), (0.55, 0.705)),
        ((0.73, 0.705), (0.78, 0.705)),
        ((0.64, 0.58), (0.64, 0.40)),
        ((0.73, 0.29), (0.78, 0.29)),
    ]
    for a, b in arrows:
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=12, color=PALETTE["gray"], linewidth=1.2
            )
        )
    ax.text(
        0.03,
        0.08,
        "Adapted logic: method overview + downstream panels; artwork and labels are repository-specific.",
        fontsize=7,
        color=PALETTE["gray"],
    )


def _draw_metric_sweep(ax) -> None:
    _panel_label(ax, "B")
    xs, curves, source = _load_metric_series()
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["red"]]
    for (name, vals), color in zip(curves.items(), colors):
        ax.plot(xs, vals, marker="o", linewidth=2, label=name, color=color)
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs], fontsize=7)
    ax.set_xlabel("HVG / held-out panel size")
    ax.set_ylabel("Metric value")
    ax.set_title("Quality trends across gene panels", fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    ax.text(0.0, -0.27, source, transform=ax.transAxes, fontsize=6.5, color=PALETTE["gray"])


def _draw_spatial_marker_grid(ax) -> None:
    _panel_label(ax, "C")
    rng = np.random.default_rng(7)
    centers = np.array([[0.25, 0.28], [0.70, 0.32], [0.48, 0.72]])
    pts = np.vstack([rng.normal(c, 0.075, size=(180, 2)) for c in centers])
    pts = np.clip(pts, 0, 1)
    gene = np.exp(-((pts[:, 0] - 0.70) ** 2 + (pts[:, 1] - 0.32) ** 2) / 0.025)
    raw = gene * rng.binomial(1, 0.42, size=gene.size)
    imputed = 0.75 * gene + 0.18 * rng.random(gene.size)
    ax.scatter(pts[:, 0], pts[:, 1], c=raw, s=8, cmap="magma", alpha=0.85, linewidths=0)
    ax.scatter(pts[:, 0] + 1.12, pts[:, 1], c=imputed, s=8, cmap="magma", alpha=0.85, linewidths=0)
    ax.add_patch(Rectangle((-0.03, -0.03), 1.06, 1.06, fill=False, edgecolor="#D0D7DE"))
    ax.add_patch(Rectangle((1.09, -0.03), 1.06, 1.06, fill=False, edgecolor="#D0D7DE"))
    ax.text(0.5, 1.06, "raw", ha="center", fontsize=8)
    ax.text(1.62, 1.06, "LuminaST imputed", ha="center", fontsize=8)
    ax.set_xlim(-0.08, 2.22)
    ax.set_ylim(-0.08, 1.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Spatial marker recovery pattern", fontsize=10)


def _draw_cluster_concordance(ax) -> None:
    _panel_label(ax, "D")
    labels = ["Epi", "T", "Myeloid", "Fib", "Endo"]
    raw = np.array([0.50, 0.16, 0.14, 0.13, 0.07])
    enhanced = np.array([0.48, 0.18, 0.15, 0.12, 0.07])
    y = np.arange(len(labels))
    ax.barh(y + 0.18, raw, height=0.32, color="#B7C7D9", label="raw/transfer")
    ax.barh(y - 0.18, enhanced, height=0.32, color=PALETTE["teal"], label="enhanced")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("fraction")
    ax.set_xlim(0, 0.6)
    ax.set_title("Cluster/annotation balance check", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.legend(fontsize=7, frameon=False)


def _draw_sublineage_dotplot(ax) -> None:
    _panel_label(ax, "E")
    groups = ["CD4 Tn", "CD8 Teff", "Treg", "Macro", "Fib"]
    genes = ["IL7R", "GZMB", "FOXP3", "CD68", "COL1A1"]
    rng = np.random.default_rng(11)
    mean = np.eye(len(groups)) * 0.8 + rng.random((len(groups), len(genes))) * 0.25
    pct = np.clip(mean + rng.normal(0.05, 0.12, mean.shape), 0.05, 1.0)
    for i in range(len(groups)):
        for j in range(len(genes)):
            ax.scatter(
                j,
                i,
                s=35 + 260 * pct[i, j],
                c=[mean[i, j]],
                cmap="viridis",
                vmin=0,
                vmax=1,
                edgecolor="#24292F",
                linewidth=0.35,
            )
    ax.set_xticks(range(len(genes)), genes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups)), groups, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Sub-lineage marker dot plot", fontsize=10)
    ax.text(
        0.0,
        -0.28,
        "size=fraction expressing; color=mean expression",
        transform=ax.transAxes,
        fontsize=6.5,
        color=PALETTE["gray"],
    )


def _draw_heatmap(ax) -> None:
    _panel_label(ax, "F")
    rng = np.random.default_rng(13)
    mat = rng.normal(0, 0.25, size=(7, 7))
    mat += np.eye(7) * 0.8
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(
        range(7), [f"patch {i + 1}" for i in range(7)], rotation=45, ha="right", fontsize=7
    )
    ax.set_yticks(range(7), ["CD4", "CD8A", "CD68", "EPCAM", "ENG", "POSTN", "LR"], fontsize=7)
    ax.set_title("External spatial/protein validation matrix", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)


def render(out_path: Path = DEFAULT_OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1, 1])
    _draw_workflow(fig.add_subplot(gs[0, :2]))
    _draw_metric_sweep(fig.add_subplot(gs[0, 2]))
    _draw_spatial_marker_grid(fig.add_subplot(gs[1, :2]))
    _draw_cluster_concordance(fig.add_subplot(gs[1, 2]))
    _draw_sublineage_dotplot(fig.add_subplot(gs[2, 0]))
    _draw_heatmap(fig.add_subplot(gs[2, 1:]))
    fig.suptitle(
        "LuminaST visualization storyboard derived from reference-paper panel logic (clean-room)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    render(args.out)
    try:
        display_path = args.out.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = args.out
    print(f"wrote {display_path}")


if __name__ == "__main__":
    main()
