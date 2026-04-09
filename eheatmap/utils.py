"""
eheatmap - Utility functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap


_R_TO_MPL_COLORS = {
    "grey60": "#999999",
    "grey30": "#4D4D4D",
    "grey50": "#7F7F7F",
    "grey40": "#666666",
    "grey80": "#CCCCCC",
    "grey90": "#E5E5E5",
    "grey10": "#1A1A1A",
    "grey20": "#333333",
    "grey70": "#B3B3B3",
    "firebrick3": "#CD2626",
    "firebrick": "#B22222",
    "navy": "#000080",
}


def _to_mpl_color(c):
    """Convert R-style color names to matplotlib-compatible hex."""
    if isinstance(c, str) and c in _R_TO_MPL_COLORS:
        return _R_TO_MPL_COLORS[c]
    return c


def _despine(ax):
    """Remove all spines and ticks from an axis."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _get_annotation_colors(values, palette=None):
    """Map discrete values to colors."""
    unique = list(pd.Series(values).dropna().unique())
    n = len(unique)
    if palette is not None:
        if isinstance(palette, dict):
            return {v: _to_mpl_color(palette.get(v, "gray")) for v in unique}
        elif isinstance(palette, (list, tuple)):
            colors = list(palette)
            while len(colors) < n:
                colors += colors
            return dict(zip(unique, colors[:n]))
    base = plt.colormaps.get_cmap("tab10").colors
    colors = [base[i % 10] for i in range(n)]
    return dict(zip(unique, colors))


def _get_continuous_colors(values, cmap="Blues"):
    """Normalize continuous values to colormap."""
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return ["gray"] * len(values)
    vmin, vmax = valid.min(), valid.max()
    if vmin == vmax:
        return [plt.colormaps.get_cmap(cmap)(0.5)] * len(values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.colormaps.get_cmap(cmap)
    return [cm(norm(v)) if not np.isnan(v) else "gray" for v in arr]


def _is_continuous(values):
    """Check if annotation values are continuous (numeric with > 5 unique)."""
    try:
        arr = np.array(values, dtype=float)
        return len(np.unique(arr[~np.isnan(arr)])) > 5
    except (ValueError, TypeError):
        return False
