"""
Wave 1 — Sankey diagram of LuminaST Leiden clusters → in-data label.

Source rep: `obs['leiden_bio']` (computed by `_plot_utils.ensure_leiden`).
Target label: first available of `cancer_type`, `spatial_cluster`, `annotation`.

Outputs:
  <out_dir>/leiden_to_label_sankey.png  (matplotlib alluvial-style)
  <out_dir>/leiden_to_label_sankey.html (plotly Sankey when available)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from scripts.visualize._plot_utils import (
    CATEGORICAL_PALETTE,
    ensure_leiden,
    pick_label_key,
    stable_categorical_colors,
)


def _flow_matrix(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"L": left.astype(str).to_numpy(), "R": right.astype(str).to_numpy()})
    return df.groupby(["L", "R"]).size().unstack(fill_value=0)


def render_sankey(adata: AnnData, png_path: Path, html_path: Optional[Path] = None) -> None:
    ensure_leiden(adata, use_rep="latent_enhanced", key="leiden_bio")
    if "leiden_bio" not in adata.obs:
        return
    label_key = pick_label_key(adata, ["cancer_type", "spatial_cluster", "annotation"])
    if label_key is None:
        return
    flow = _flow_matrix(adata.obs["leiden_bio"], adata.obs[label_key])
    if flow.empty:
        return

    left_labels = list(flow.index)
    right_labels = list(flow.columns)
    left_totals = flow.sum(axis=1).to_numpy()
    right_totals = flow.sum(axis=0).to_numpy()

    left_palette = stable_categorical_colors(pd.Series(left_labels))
    right_palette = stable_categorical_colors(pd.Series(right_labels))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    h_gap = 0.5
    left_y_top = 0.0
    left_blocks = {}
    for lab, total in zip(left_labels, left_totals):
        height = total
        left_blocks[lab] = (left_y_top, left_y_top + height)
        ax.add_patch(plt.Rectangle((0, -left_y_top - height), 0.05, height, color=left_palette[lab]))
        ax.text(-0.02, -left_y_top - height / 2, f"L{lab}", ha="right", va="center", fontsize=7)
        left_y_top += height + h_gap

    right_y_top = 0.0
    right_blocks = {}
    for lab, total in zip(right_labels, right_totals):
        height = total
        right_blocks[lab] = (right_y_top, right_y_top + height)
        ax.add_patch(plt.Rectangle((1, -right_y_top - height), 0.05, height, color=right_palette[lab]))
        ax.text(1.07, -right_y_top - height / 2, str(lab), ha="left", va="center", fontsize=7)
        right_y_top += height + h_gap

    left_cursor = {lab: left_blocks[lab][0] for lab in left_labels}
    right_cursor = {lab: right_blocks[lab][0] for lab in right_labels}
    for li, lab_l in enumerate(left_labels):
        for ri, lab_r in enumerate(right_labels):
            flow_lr = flow.iloc[li, ri]
            if flow_lr <= 0:
                continue
            y_l_top = -left_cursor[lab_l]
            y_l_bot = -(left_cursor[lab_l] + flow_lr)
            y_r_top = -right_cursor[lab_r]
            y_r_bot = -(right_cursor[lab_r] + flow_lr)
            xs = np.linspace(0.05, 1.0, 50)
            t = (xs - 0.05) / 0.95
            ease = 0.5 - 0.5 * np.cos(np.pi * t)
            top = y_l_top + (y_r_top - y_l_top) * ease
            bot = y_l_bot + (y_r_bot - y_l_bot) * ease
            ax.fill_between(xs, bot, top, color=left_palette[lab_l], alpha=0.35, linewidth=0)
            left_cursor[lab_l] += flow_lr
            right_cursor[lab_r] += flow_lr

    total_h = max(left_y_top, right_y_top)
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-total_h, 0.5)
    ax.set_axis_off()
    ax.set_title(f"Leiden (latent_enhanced)  →  {label_key}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    if html_path is not None:
        try:
            import plotly.graph_objects as go

            nodes = [f"L{lab}" for lab in left_labels] + [str(lab) for lab in right_labels]
            node_colors = [left_palette[lab] for lab in left_labels] + [right_palette[lab] for lab in right_labels]
            src, tgt, val, link_color = [], [], [], []
            n_left = len(left_labels)
            for li, lab_l in enumerate(left_labels):
                for ri, lab_r in enumerate(right_labels):
                    v = int(flow.iloc[li, ri])
                    if v <= 0:
                        continue
                    src.append(li)
                    tgt.append(n_left + ri)
                    val.append(v)
                    # plotly Sankey only accepts named/rgb(a) strings, not hex+alpha
                    hex_color = left_palette[lab_l].lstrip("#")
                    r_, g_, b_ = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                    link_color.append(f"rgba({r_},{g_},{b_},0.5)")
            fig_p = go.Figure(go.Sankey(
                node=dict(label=nodes, color=node_colors, pad=15, thickness=12),
                link=dict(source=src, target=tgt, value=val, color=link_color),
            ))
            fig_p.update_layout(title=f"Leiden → {label_key} (interactive)", width=800, height=520)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            fig_p.write_html(html_path, include_plotlyjs="cdn")
        except Exception as exc:
            print(f"[warn] plotly Sankey HTML failed: {exc}")
