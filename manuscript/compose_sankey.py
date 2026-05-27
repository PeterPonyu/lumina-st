"""Sankey / alluvial composer skeleton for cluster-mapping figures.

Round 13 W005 — closes the stPainter §2.3 (Sankey cluster mapping)
composer skeleton gap. The composer is data-format-agnostic: it accepts
a contingency dict ``{truth_label: {pred_label: count}}`` and renders
an alluvial flow between truth labels (left) and predicted labels (right).

The synthetic-mode demo here uses a deterministically generated
contingency table so the rendered PNG is part of CI artifacts; the
real-data demo (Day 5 cluster labels) plugs into the same entry point.

Run as a script: ``python compose_sankey.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from compose_palette import color_for


def render_sankey(
    contingency: dict[str, dict[str, int]],
    out_path: Path,
    title: str = "Cluster mapping (synthetic placeholder)",
) -> Path:
    """Render an alluvial flow from truth labels (left) to predicted labels (right).

    Args:
        contingency: ``{truth_label: {pred_label: count}}`` cell-count table.
        out_path: PNG output path.
        title: figure title.

    Returns:
        ``out_path`` after writing.
    """
    if not contingency:
        raise ValueError("contingency must contain at least one truth label")
    for truth_label, pred_counts in contingency.items():
        if not pred_counts:
            raise ValueError(f"contingency row {truth_label!r} must not be empty")
        for pred_label, count in pred_counts.items():
            if count < 0:
                raise ValueError(
                    f"contingency count for {truth_label!r}->{pred_label!r} must be >= 0"
                )

    truth_labels = list(contingency.keys())
    pred_labels = sorted({p for d in contingency.values() for p in d.keys()})

    truth_totals = {t: sum(contingency[t].values()) for t in truth_labels}
    pred_totals = {p: sum(contingency[t].get(p, 0) for t in truth_labels)
                   for p in pred_labels}
    total = sum(truth_totals.values())
    if total <= 0:
        raise ValueError("contingency must contain at least one positive count")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    # Place truth bars on the left, pred bars on the right.
    gap = 0.02
    y_truth: dict[str, tuple[float, float]] = {}
    cursor = 1.0
    for t in truth_labels:
        h = truth_totals[t] / total - gap
        y_truth[t] = (cursor - h, cursor)
        ax.add_patch(
            Rectangle((0.05, cursor - h), 0.05, h,
                      color=color_for(t), alpha=0.95)
        )
        ax.text(0.04, cursor - h / 2, t, ha="right", va="center", fontsize=8)
        cursor -= h + gap

    y_pred: dict[str, tuple[float, float]] = {}
    cursor = 1.0
    for p in pred_labels:
        h = pred_totals[p] / total - gap
        y_pred[p] = (cursor - h, cursor)
        ax.add_patch(
            Rectangle((0.9, cursor - h), 0.05, h,
                      color=color_for(p), alpha=0.95)
        )
        ax.text(0.96, cursor - h / 2, p, ha="left", va="center", fontsize=8)
        cursor -= h + gap

    # Draw alluvial ribbons between aligned blocks (no curve fitting; a
    # simple shaded polygon is sufficient for a skeleton renderer).
    truth_offset = {t: y_truth[t][1] for t in truth_labels}
    pred_offset = {p: y_pred[p][1] for p in pred_labels}
    for t in truth_labels:
        for p in pred_labels:
            n = contingency[t].get(p, 0)
            if n <= 0:
                continue
            h = n / total
            ytop_l = truth_offset[t]
            ybot_l = ytop_l - h
            ytop_r = pred_offset[p]
            ybot_r = ytop_r - h
            xs = [0.10, 0.10, 0.90, 0.90]
            ys = [ybot_l, ytop_l, ytop_r, ybot_r]
            ax.fill(xs, ys, color=color_for(t), alpha=0.25, linewidth=0)
            truth_offset[t] = ybot_l
            pred_offset[p] = ybot_r

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _synthetic_contingency(seed: int = 0) -> dict[str, dict[str, int]]:
    """Deterministic synthetic contingency table for skeleton rendering."""
    rng = np.random.default_rng(seed)
    truth = ["T_cell", "B_cell", "Myeloid", "Epithelial", "Stromal"]
    pred = ["c0", "c1", "c2", "c3", "c4", "c5"]
    out: dict[str, dict[str, int]] = {}
    for t in truth:
        diag = rng.integers(60, 100)
        row = {p: int(rng.integers(0, 20)) for p in pred}
        # Strong diagonal: roughly map each truth label to one pred cluster.
        row[pred[truth.index(t) % len(pred)]] = int(diag)
        out[t] = row
    return out


def main() -> None:
    """Skeleton entrypoint: render a synthetic Sankey + manifest sidecar."""
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    contingency = _synthetic_contingency(seed=0)
    png = render_sankey(contingency, out_dir / "fig_sankey_skeleton.png")
    manifest = {
        "title": "Cluster mapping (synthetic placeholder)",
        "source": "compose_sankey._synthetic_contingency",
        "real_data_contract": (
            "Replace _synthetic_contingency() with a loader that returns "
            "{truth_label: {pred_label: count}} from real Day 5 cluster "
            "labels (per docs/DATA_PREP_CALENDAR.md)."
        ),
        "rendered_path": str(png.name),
    }
    (out_dir / "fig_sankey_skeleton.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
