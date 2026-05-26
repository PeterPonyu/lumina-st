"""
Wave 1 — Sankey diagram linking Leiden clusters to true annotation labels.

Uses plotly Sankey when available (exports PNG); falls back to matplotlib
stacked-bar chart otherwise.

Outputs:
  <out_path> (PNG)
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

from scripts.visualize._plot_utils import ensure_leiden, pick_label_key


def _write_fallback_html(cross: pd.DataFrame, html_path: Path, true_key: str) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'>",
                f"<title>Leiden clusters → {true_key}</title>",
                "</head><body>",
                f"<h1>Leiden clusters → {true_key}</h1>",
                "<p>Plotly Sankey export was unavailable; showing contingency table.</p>",
                cross.to_html(),
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def render_sankey(
    adata: AnnData,
    out_path: Path,
    html_path: Optional[Path] = None,
    label_key: str = "cancer_type",
) -> None:
    ensure_leiden(adata)
    if "leiden_bio" not in adata.obs:
        return

    true_key = pick_label_key(
        adata,
        [
            label_key,
            "cancer_type",
            "annotation",
            "cell_type",
            "celltype",
            "label",
            "spatial_cluster",
        ],
    )
    if true_key is None:
        return

    leiden = adata.obs["leiden_bio"].astype(str)
    true_labels = adata.obs[true_key].astype(str)
    cross = pd.crosstab(leiden, true_labels)
    if cross.empty:
        return

    # -------- plotly Sankey (primary) --------
    try:
        import plotly.graph_objects as go  # noqa: F401

        left_labels = list(cross.index)
        right_labels = list(cross.columns)
        n_left = len(left_labels)

        src: list[int] = []
        tgt: list[int] = []
        val: list[int] = []
        for li in range(n_left):
            for ri in range(len(right_labels)):
                v = int(cross.iloc[li, ri])
                if v > 0:
                    src.append(li)
                    tgt.append(n_left + ri)
                    val.append(v)

        all_labels = left_labels + right_labels
        left_colors = [f"hsl({int(360 * i / max(1, n_left))}, 60%, 55%)" for i in range(n_left)]
        right_colors = [
            f"hsl({int(360 * i / max(1, len(right_labels)))}, 40%, 50%)"
            for i in range(len(right_labels))
        ]
        node_colors = left_colors + right_colors
        link_colors = [left_colors[s].replace("hsl", "hsla").replace(")", ", 0.4)") for s in src]

        fig = go.Figure(
            go.Sankey(
                node=dict(label=all_labels, color=node_colors, pad=15, thickness=12),
                link=dict(source=src, target=tgt, value=val, color=link_colors),
            )
        )
        fig.update_layout(title=f"Leiden clusters → {true_key}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if html_path is not None:
            try:
                html_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(html_path, include_plotlyjs="cdn")
            except Exception:
                _write_fallback_html(cross, html_path, true_key)
        try:
            fig.write_image(str(out_path))
            return
        except Exception:
            pass
    except Exception:
        pass

    # -------- matplotlib stacked-bar fallback --------
    cross_norm = cross.div(cross.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(max(10, len(cross.index) * 0.7), 6))
    bottom = np.zeros(len(cross_norm))
    for col in cross_norm.columns:
        ax.bar(cross_norm.index, cross_norm[col], bottom=bottom, label=col)
        bottom += cross_norm[col].values

    ax.set_xlabel("Leiden cluster")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Leiden clusters → {true_key} (stacked bar)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if html_path is not None and not html_path.exists():
        _write_fallback_html(cross, html_path, true_key)
