"""Stable {method → color} map for the LuminaST manuscript figures.

Round 10 W002 — so a reader can track an adapter across all four figures
without re-learning the color in each one. Indices are pinned by method
name; new methods added in the future hash into the tab10/tab20 palette
deterministically.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt

# Method-name → tab10 index. Always-available adapters first so their
# colors stay distinct from the external competitor block; external
# adapters get the later indices so a reader visually parses
# "available vs unavailable" by color region.
_METHOD_TAB10: dict[str, int] = {
    # Always-available baselines + LuminaST
    "mean": 0,           # blue
    "knn": 1,            # orange
    "lumina_st": 2,      # green
    # External competitor block (marker color signals competitor)
    "spaim": 3,          # red
    "tissue": 4,         # purple
    "stmcdi": 5,         # brown
    "cellt": 6,          # pink
}

_TAB10 = plt.colormaps["tab10"]
_TAB20 = plt.colormaps["tab20"]

# Color reserved for "unavailable" status indicators (grey-out).
UNAVAILABLE_COLOR = (0.6, 0.6, 0.6, 0.7)


def color_for(method: str) -> tuple[float, float, float, float]:
    """Return an RGBA color for a method name.

    Unknown method names fall through to tab20 hashed by name so
    the palette never raises; the hash is stable across processes
    so repeated runs reproduce the same color.
    """
    key = method.lower()
    if key in _METHOD_TAB10:
        return tuple(_TAB10(_METHOD_TAB10[key]))  # type: ignore[return-value]
    # Stable fallback: hash the name modulo 20.
    idx = sum(ord(c) for c in key) % 20
    return tuple(_TAB20(idx))  # type: ignore[return-value]


def palette_for(methods: Iterable[str]) -> list[tuple[float, float, float, float]]:
    """Return colors aligned with the given method iterable."""
    return [color_for(m) for m in methods]
