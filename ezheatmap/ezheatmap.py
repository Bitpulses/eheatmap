"""
ezheatmap 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from typing import Optional, List, Union, Dict, Any, Tuple
import warnings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _dist_euclidean(x):
    return pdist(x, metric="euclidean")


def _dist_correlation(x):
    """1 - Pearson correlation as distance."""
    if x.shape[0] <= 1:
        return np.array([0.0])
    corr = np.corrcoef(x)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)
    dist = 1 - corr
    n = dist.shape[0]
    idx = np.triu_indices(n, k=1)
    return dist[idx]


def _compute_distance(data, measure):
    """Return condensed distance array."""
    if measure == "euclidean":
        return _dist_euclidean(data)
    elif measure == "correlation":
        return _dist_correlation(data)
    elif measure == "none":
        return None
    elif isinstance(measure, np.ndarray):
        return measure
    else:
        return pdist(data, metric=measure)


def _cluster(data, method="complete", distance="euclidean"):
    """Perform hierarchical clustering, return (order, linkage_matrix)."""
    if data.shape[0] <= 1:
        return list(range(data.shape[0])), None
    dist = _compute_distance(data, distance)
    if dist is None:
        return list(range(data.shape[0])), None
    if not np.all(np.isfinite(dist)):
        return list(range(data.shape[0])), None
    Z = hierarchy.linkage(dist, method=method)
    order = hierarchy.leaves_list(Z)
    return order, Z


def _cutree(Z, k):
    """Cut tree into k clusters, return list of index arrays in dendrogram leaf order."""
    labels = hierarchy.fcluster(Z, k, criterion="maxclust")
    groups = []
    for i in range(1, k + 1):
        idx = np.where(labels == i)[0]
        if len(idx) > 0:
            groups.append(idx.tolist())
    return groups


def _cutree_by_leaves(Z, k):
    """Cut tree into k clusters, return list of index arrays in dendrogram leaf order."""
    dendro = hierarchy.dendrogram(Z, no_plot=True)
    leaves = dendro["leaves"]
    labels = hierarchy.fcluster(Z, k, criterion="maxclust")
    # Group leaves by cluster label, in leaf order
    leaf_labels = {leaf: labels[leaf] for leaf in leaves}
    groups = []
    current_label = None
    current_group = []
    for leaf in leaves:
        lbl = leaf_labels[leaf]
        if lbl != current_label:
            if current_group:
                groups.append(current_group)
            current_label = lbl
            current_group = [leaf]
        else:
            current_group.append(leaf)
    if current_group:
        groups.append(current_group)
    return groups


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


def _draw_sub_heatmap(ax, sub_df, cmap_obj, norm, border_color, use_border,
                      display_numbers, number_format, number_color, fontsize_number,
                      has_numbers):
    """Draw a single heatmap sub-block (used for split rendering)."""
    ax.pcolormesh(
        sub_df.values,
        cmap=cmap_obj,
        norm=norm,
        edgecolors=border_color if use_border else "none",
        linewidths=0.3 if use_border else 0,
    )
    ax.set_xlim(0, sub_df.shape[1])
    ax.set_ylim(0, sub_df.shape[0])
    ax.invert_yaxis()
    _despine(ax)

    if has_numbers:
        for ri2 in range(sub_df.shape[0]):
            for ci2 in range(sub_df.shape[1]):
                val = sub_df.values[ri2, ci2]
                if np.isnan(val):
                    continue
                txt = number_format % val if isinstance(display_numbers, bool) else str(val)
                ax.text(
                    ci2 + 0.5, ri2 + 0.5, txt,
                    ha="center", va="center",
                    fontsize=fontsize_number, color=number_color,
                )




# ---------------------------------------------------------------------------
# Main function – mirrors R pheatmap() signature
# ---------------------------------------------------------------------------


def ezheatmap(
    mat,
    color=None,
    kmeans_k=None,
    breaks=None,
    border_color="grey60",
    cellwidth=None,
    cellheight=None,
    scale="none",
    cluster_rows=True,
    cluster_cols=True,
    clustering_distance_rows="euclidean",
    clustering_distance_cols="euclidean",
    clustering_method="complete",
    cutree_rows=None,
    cutree_cols=None,
    treeheight_row=None,
    treeheight_col=None,
    dendrogram_linewidth=None,
    dendrogram_colors=None,
    legend=True,
    legend_breaks=None,
    legend_labels=None,
    legend_title=None,
    annotation_row=None,
    annotation_col=None,
    annotation_colors=None,
    annotation_legend=True,
    annotation_names_row=True,
    annotation_names_col=True,
    annotation_top=None,
    annotation_bottom=None,
    annotation_left=None,
    annotation_right=None,
    annotation_title_pos=None,
    drop_categories=True,
    show_rownames=True,
    show_colnames=True,
    row_names_side='left',
    col_names_side='bottom',
    main=None,
    fontsize=10,
    fontsize_row=None,
    fontsize_col=None,
    angle_col=270,
    display_numbers=False,
    number_format="%.2f",
    number_color="grey30",
    fontsize_number=None,
    gaps_row=None,
    gaps_col=None,
    row_split_gap=0.5,
    col_split_gap=0.2,
    labels_row=None,
    labels_col=None,
    filename=None,
    width=None,
    height=None,
    silent=False,
    na_col="#DDDDDD",
    figsize=None,
    cmap=None,
    vmin=None,
    vmax=None,
    center=None,
    row_split=None,
    col_split=None,
    row_annotation=None,
    col_annotation=None,
    row_dendrogram=True,
    col_dendrogram=True,
    method=None,
    metric=None,
):
    """
    Draw a clustered heatmap
    """

    # ------------------------------------------------------------------
    # 0. Normalise inputs
    # ------------------------------------------------------------------
    if not isinstance(mat, pd.DataFrame):
        mat = pd.DataFrame(mat)

    df = mat.copy()

    if cmap is not None and color is None:
        color = cmap
    if color is None:
        color = "RdYlBu_r"

    if method is not None:
        clustering_method = method
    if metric is not None:
        if clustering_distance_rows == "euclidean":
            clustering_distance_rows = metric
        if clustering_distance_cols == "euclidean":
            clustering_distance_cols = metric

    if row_annotation is not None and annotation_row is None:
        annotation_row = row_annotation
    if col_annotation is not None and annotation_col is None:
        annotation_col = col_annotation

    # Map old params to new directional params (backward compatibility)
    if (
        annotation_row is not None
        and annotation_left is None
        and annotation_right is None
    ):
        annotation_right = annotation_row
    if (
        annotation_col is not None
        and annotation_top is None
        and annotation_bottom is None
    ):
        annotation_bottom = annotation_col

    # annotation_title_pos: [left, top, right, bottom]
    if annotation_title_pos is None:
        annotation_title_pos = ["bottom", "left", "top", "right"]
    title_pos = {
        "left": annotation_title_pos[0],
        "top": annotation_title_pos[1],
        "right": annotation_title_pos[2],
        "bottom": annotation_title_pos[3],
    }

    if row_split is not None and cutree_rows is None:
        if isinstance(row_split, int):
            cutree_rows = row_split
    if col_split is not None and cutree_cols is None:
        if isinstance(col_split, int):
            cutree_cols = col_split

    if fontsize_row is None:
        fontsize_row = fontsize
    if fontsize_col is None:
        fontsize_col = fontsize
    if fontsize_number is None:
        fontsize_number = int(0.8 * fontsize)
    
    # Set default dendrogram linewidth
    if dendrogram_linewidth is None:
        dendrogram_linewidth = 0.5
    
    # Set default dendrogram colors
    if dendrogram_colors is None:
        dendrogram_colors = 'colorful'  # Default to colorful

    if not row_dendrogram:
        cluster_rows = False
    if not col_dendrogram:
        cluster_cols = False

    border_color = _to_mpl_color(border_color)
    number_color = _to_mpl_color(number_color)
    na_col = _to_mpl_color(na_col)

    # ------------------------------------------------------------------
    # Handle NA
    # ------------------------------------------------------------------
    # has_na = df.isna().any().any()
    df_for_cluster = df.fillna(0)

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------
    if scale == "row":
        df = df.apply(lambda r: (r - r.mean()) / r.std() if r.std() > 0 else 0, axis=1)
        df = df.fillna(0)
    elif scale == "column":
        df = (df - df.mean()) / df.std()
        df = df.fillna(0)

    df_for_cluster = df.fillna(0)

    # ------------------------------------------------------------------
    # K-means aggregation
    # ------------------------------------------------------------------
    if kmeans_k is not None and kmeans_k > 1:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=kmeans_k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(df.values)
        centers = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=df.columns,
            index=[f"Cluster_{i + 1}" for i in range(kmeans_k)],
        )
        df = centers
        show_rownames = True
        df_for_cluster = df.fillna(0)
        annotation_row = None
        row_annotation = None
        cutree_rows = None
        gaps_row = None

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    if isinstance(cluster_rows, bool) and cluster_rows:
        row_order, row_Z = _cluster(
            df_for_cluster.values, clustering_method, clustering_distance_rows
        )
    elif hasattr(cluster_rows, "linkage"):
        row_Z = cluster_rows.linkage
        row_order = hierarchy.leaves_list(row_Z)
    else:
        row_order = list(range(df.shape[0]))
        row_Z = None

    if isinstance(cluster_cols, bool) and cluster_cols:
        col_order, col_Z = _cluster(
            df_for_cluster.values.T, clustering_method, clustering_distance_cols
        )
    elif hasattr(cluster_cols, "linkage"):
        col_Z = cluster_cols.linkage
        col_order = hierarchy.leaves_list(col_Z)
    else:
        col_order = list(range(df.shape[1]))
        col_Z = None

    df = df.iloc[row_order, col_order]

    # ------------------------------------------------------------------
    # Labels override
    # ------------------------------------------------------------------
    if labels_row is not None:
        df.index = labels_row
    if labels_col is not None:
        df.columns = labels_col

    # ------------------------------------------------------------------
    # Annotation processing – directional
    # ------------------------------------------------------------------
    anno_data = {}
    anno_colors = {}
    anno_continuous = {}

    def _process_anno(anno_input, index):
        if anno_input is None:
            return None, {}, {}
        if isinstance(anno_input, pd.DataFrame):
            data = anno_input.loc[index]
        elif isinstance(anno_input, pd.Series):
            data = anno_input.loc[index].to_frame()
        else:
            return None, {}, {}
        colors = {}
        continuous = {}
        for col in data.columns:
            vals = data[col]
            if annotation_colors is not None and col in annotation_colors:
                colors[col] = _get_annotation_colors(vals, annotation_colors[col])
            else:
                if _is_continuous(vals):
                    continuous[col] = vals
                else:
                    colors[col] = _get_annotation_colors(vals)
        return data, colors, continuous

    for direction in ("top", "bottom", "left", "right"):
        param_name = f"annotation_{direction}"
        anno_input = locals().get(param_name)
        if direction in ("left", "right"):
            index = df.index
        else:
            index = df.columns
        data, colors, continuous = _process_anno(anno_input, index)
        if data is not None:
            anno_data[direction] = data
            anno_colors[direction] = colors
            anno_continuous[direction] = continuous

    # ------------------------------------------------------------------
    # Gaps / splits
    # ------------------------------------------------------------------
    if cutree_rows is not None and row_Z is not None:
        row_groups = _cutree_by_leaves(row_Z, cutree_rows)
    elif gaps_row is not None:
        sorted_gaps = sorted(gaps_row)
        row_groups = []
        prev = 0
        for g in sorted_gaps:
            row_groups.append(list(range(prev, g)))
            prev = g
        row_groups.append(list(range(prev, df.shape[0])))
    else:
        row_groups = [list(range(df.shape[0]))]

    if cutree_cols is not None and col_Z is not None:
        col_groups = _cutree_by_leaves(col_Z, cutree_cols)
    elif gaps_col is not None:
        sorted_gaps = sorted(gaps_col)
        col_groups = []
        prev = 0
        for g in sorted_gaps:
            col_groups.append(list(range(prev, g)))
            prev = g
        col_groups.append(list(range(prev, df.shape[1])))
    else:
        col_groups = [list(range(df.shape[1]))]

    n_row_groups = len(row_groups)
    n_col_groups = len(col_groups)

    row_cum = [0]
    for rg in row_groups:
        row_cum.append(row_cum[-1] + len(rg))
    col_cum = [0]
    for cg in col_groups:
        col_cum.append(col_cum[-1] + len(cg))

    n_row_split = n_row_groups
    n_col_split = n_col_groups

    # ------------------------------------------------------------------
    # Color mapping
    # ------------------------------------------------------------------
    if breaks is not None:
        breaks = np.asarray(breaks)
        if vmin is None:
            vmin = breaks[0]
        if vmax is None:
            vmax = breaks[-1]
    else:
        if vmin is None:
            vmin = df.values.min()
        if vmax is None:
            vmax = df.values.max()

    if center is not None:
        half = max(abs(vmax - center), abs(vmin - center))
        vmin = center - half
        vmax = center + half

    if isinstance(color, str):
        try:
            cmap_obj = plt.colormaps.get_cmap(color)
        except Exception:
            cmap_obj = plt.colormaps.get_cmap("viridis")
    elif isinstance(color, (list, tuple)):
        converted = [_to_mpl_color(c) for c in color]
        cmap_obj = ListedColormap(converted)
    else:
        cmap_obj = plt.colormaps.get_cmap("RdYlBu_r")

    norm = Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------------
    # Collect legend items
    # ------------------------------------------------------------------
    has_any_legends = annotation_legend and any(
        len(anno_colors.get(d, {})) > 0 or len(anno_continuous.get(d, {})) > 0
        for d in ("top", "bottom", "left", "right")
    )

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    dendro_ratio = 0.6
    anno_bar = 0.15
    label_space = 1.0
    legend_area_w = 5.0 if (legend or has_any_legends) else 0

    left_w = dendro_ratio if (row_Z is not None) else 0
    if "left" in anno_data:
        left_w += anno_bar * anno_data["left"].shape[1]
    right_w = anno_bar * anno_data["right"].shape[1] if "right" in anno_data else 0
    top_h = dendro_ratio if (col_Z is not None) else 0
    if "top" in anno_data:
        top_h += anno_bar * anno_data["top"].shape[1]
    bottom_h = anno_bar * anno_data["bottom"].shape[1] if "bottom" in anno_data else 0

    if cellwidth is not None:
        hm_w = df.shape[1] * cellwidth / 72
    else:
        hm_w = max(df.shape[1] * 0.12, 3)

    if cellheight is not None:
        hm_h = df.shape[0] * cellheight / 72
    else:
        hm_h = max(df.shape[0] * 0.12, 3)

    total_w = left_w + hm_w + right_w + label_space + legend_area_w
    total_h = top_h + hm_h + bottom_h + label_space

    if figsize is not None:
        fig_w, fig_h = figsize
    elif width is not None and height is not None:
        fig_w, fig_h = width, height
    else:
        fig_w = min(total_w, 28)
        fig_h = min(total_h, 28)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ------------------------------------------------------------------
    # GridSpec (directional layout)
    # ------------------------------------------------------------------
    n_cols = 0
    ci = {}

    # Dendrogram always outermost: left of left-annotation
    if left_w > 0:
        ci["dendro_row"] = n_cols
        n_cols += 1
    if "left" in anno_data:
        ci["left"] = n_cols
        n_cols += 1
    ci["heatmap"] = n_cols
    n_cols += 1
    if "right" in anno_data:
        ci["right"] = n_cols
        n_cols += 1

    n_rows = 0
    ri = {}

    # Dendrogram always outermost: above top-annotation
    if top_h > 0:
        ri["dendro_col"] = n_rows
        n_rows += 1
    if "top" in anno_data:
        ri["top"] = n_rows
        n_rows += 1
    ri["heatmap"] = n_rows
    n_rows += 1
    if "bottom" in anno_data:
        ri["bottom"] = n_rows
        n_rows += 1

    wr = []
    if left_w > 0:
        # treeheight_row controls the width of row dendrogram area (left side)
        # Using relative scale: value is proportional to dendro_ratio
        if treeheight_row is not None and treeheight_row > 0:
            wr.append(treeheight_row * dendro_ratio / 50.0)
        else:
            wr.append(dendro_ratio)
    if "left" in anno_data:
        wr.append(anno_bar * anno_data["left"].shape[1])
    wr.append(hm_w)
    if "right" in anno_data:
        wr.append(anno_bar * anno_data["right"].shape[1])

    hr = []
    if top_h > 0:
        # treeheight_col controls the height of column dendrogram area (top)
        # Using relative scale: value is proportional to dendro_ratio
        if treeheight_col is not None and treeheight_col > 0:
            hr.append(treeheight_col * dendro_ratio / 50.0)
        else:
            hr.append(dendro_ratio)
    if "top" in anno_data:
        hr.append(anno_bar * anno_data["top"].shape[1])
    hr.append(hm_h)
    if "bottom" in anno_data:
        hr.append(anno_bar * anno_data["bottom"].shape[1])

    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        width_ratios=wr,
        height_ratios=hr,
        wspace=0.01,
        hspace=0.01,
        left=0.05,
        right=0.85,
        top=0.95,
        bottom=0.05,
    )

    # ------------------------------------------------------------------
    # Compute split gap (hspace / wspace) — PyComplexHeatmap approach
    # ------------------------------------------------------------------
    has_numbers = display_numbers is not False and display_numbers is not None
    use_border = border_color not in ("NA", "none", None)

    hm_col = ci["heatmap"]
    hm_row = ri["heatmap"]
    main_ax = fig.add_subplot(gs[hm_row, hm_col])

    mm2inch = 1.0 / 25.4
    dpi = fig.dpi

    if n_row_groups > 1 or n_col_groups > 1:
        main_ax.set_axis_off()
        fig.canvas.draw()
        hm_extent = main_ax.get_window_extent()

        # Normalize gap by cell size so the same mm value produces the same
        # visual appearance even when cells are not square.
        cell_w_px = hm_extent.width / max(df.shape[1], 1)
        cell_h_px = hm_extent.height / max(df.shape[0], 1)
        ref_cell_px = min(cell_w_px, cell_h_px)

        col_split_gap_pixel = col_split_gap * mm2inch * dpi * (cell_w_px / ref_cell_px)
        row_split_gap_pixel = row_split_gap * mm2inch * dpi * (cell_h_px / ref_cell_px)
        wspace = (col_split_gap_pixel * n_col_split) / (
            hm_extent.width + col_split_gap_pixel - col_split_gap_pixel * n_col_split
        )
        hspace = (row_split_gap_pixel * n_row_split) / (
            hm_extent.height + row_split_gap_pixel - row_split_gap_pixel * n_row_split
        )
    else:
        wspace = 0.01
        hspace = 0.01

    # ------------------------------------------------------------------
    # Draw heatmap cells (split-aware, PyComplexHeatmap approach)
    # ------------------------------------------------------------------
    if n_row_groups > 1 or n_col_groups > 1:
        # Create inner GridSpec with gap
        inner = gridspec.GridSpecFromSubplotSpec(
            n_row_split,
            n_col_split,
            subplot_spec=gs[hm_row, hm_col],
            height_ratios=[len(g) for g in row_groups],
            width_ratios=[len(g) for g in col_groups],
            wspace=wspace,
            hspace=hspace,
        )
        for i in range(n_row_split):
            for j in range(n_col_split):
                ax = fig.add_subplot(inner[i, j])
                r_start, r_end = row_cum[i], row_cum[i + 1]
                c_start, c_end = col_cum[j], col_cum[j + 1]
                sub = df.iloc[r_start:r_end, c_start:c_end]

                _draw_sub_heatmap(
                    ax, sub, cmap_obj, norm, border_color, use_border,
                    display_numbers, number_format, number_color,
                    fontsize_number, has_numbers,
                )

                # Row labels: only on the leftmost or rightmost column
                show_row_at_j = (j == 0) if row_names_side == 'left' else (j == n_col_split - 1)
                if show_row_at_j and show_rownames:
                    ax.set_yticks(np.arange(sub.shape[0]) + 0.5)
                    ax.set_yticklabels(sub.index, fontsize=fontsize_row, ha="left" if row_names_side == 'right' else "right")
                    ax.tick_params(axis="y", length=0)
                    if row_names_side == 'right':
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position('right')

                # Column labels: only on the bottom or top row
                show_col_at_i = (i == n_row_split - 1) if col_names_side == 'bottom' else (i == 0)
                if show_col_at_i and show_colnames:
                    ax.set_xticks(np.arange(sub.shape[1]) + 0.5)
                    ax.set_xticklabels(
                        sub.columns,
                        rotation=angle_col,
                        ha="center" if angle_col in (0, 270) else "right",
                        fontsize=fontsize_col,
                    )
                    ax.tick_params(axis="x", length=0)
                    if col_names_side == 'top':
                        ax.xaxis.tick_top()
                        ax.xaxis.set_label_position('top')
    else:
        # No split — single heatmap
        main_ax.pcolormesh(
            df.values,
            cmap=cmap_obj,
            norm=norm,
            edgecolors=border_color if use_border else "none",
            linewidths=0.3 if use_border else 0,
        )
        main_ax.set_xlim(0, df.shape[1])
        main_ax.set_ylim(0, df.shape[0])
        main_ax.invert_yaxis()
        _despine(main_ax)

        if show_colnames:
            main_ax.set_xticks(np.arange(df.shape[1]) + 0.5)
            main_ax.set_xticklabels(
                df.columns,
                rotation=angle_col,
                ha="center" if angle_col in (0, 270) else "right",
                fontsize=fontsize_col,
            )
            main_ax.tick_params(axis="x", length=0)
            if col_names_side == 'top':
                main_ax.xaxis.tick_top()
                main_ax.xaxis.set_label_position('top')
        else:
            main_ax.set_xticks([])

        if show_rownames:
            main_ax.set_yticks(np.arange(df.shape[0]) + 0.5)
            main_ax.set_yticklabels(df.index, fontsize=fontsize_row, ha="left" if row_names_side == 'right' else "right")
            main_ax.tick_params(axis="y", length=0)
            if row_names_side == 'right':
                main_ax.yaxis.tick_right()
                main_ax.yaxis.set_label_position('right')
        else:
            main_ax.set_yticks([])

        if has_numbers:
            for ri2 in range(df.shape[0]):
                for ci2 in range(df.shape[1]):
                    val = df.values[ri2, ci2]
                    if np.isnan(val):
                        continue
                    txt = number_format % val if isinstance(display_numbers, bool) else str(val)
                    main_ax.text(
                        ci2 + 0.5, ri2 + 0.5, txt,
                        ha="center", va="center",
                        fontsize=fontsize_number, color=number_color,
                    )

    # ------------------------------------------------------------------
    # Dendrograms — single complete tree with remapped coordinates
    #     Following PyComplexHeatmap's DenTree + get_coords approach
    # ------------------------------------------------------------------
    fig.canvas.draw()

    # Helper classes from PyComplexHeatmap
    class Branch():
        """Represents a single branch in the dendrogram."""
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None
            self.left = None
            self.right = None
            self.create_branch()

        def create_branch(self):
            self.left_pos = (self.x[0], self.y[0])
            self.right_pos = (self.x[-1], self.y[-1])
            self.root_pos = (np.mean(self.x[1:-1]), np.mean(self.y[1:-1]))
            # A position is a parent if its y != 0 (not a leaf)
            self.left_is_parent = self.left_pos[1] != 0
            self.right_is_parent = self.right_pos[1] != 0

    class DenTree:
        """Tree structure for coordinate remapping."""
        def __init__(self, icoord, dcoord):
            self.icoord = icoord
            self.dcoord = dcoord
            self.create_tree()

        def create_tree(self):
            self.branches = {}  # keys are root_pos, values are Branch objects
            self.parents = []
            self.leaves = []
            
            for x, y in zip(self.icoord, self.dcoord):
                branch = Branch(x, y)
                self.branches[branch.root_pos] = branch
            
            for root_pos in self.branches:
                branch = self.branches[root_pos]
                if branch.left_is_parent or branch.right_is_parent:
                    self.parents.append(branch.root_pos)
                    if branch.left_is_parent:
                        branch.left = self.branches[branch.left_pos]
                        branch.left.parent = branch
                    if branch.right_is_parent:
                        branch.right = self.branches[branch.right_pos]
                        branch.right.parent = branch
                
                if not branch.left_is_parent:
                    self.leaves.append(branch.left_pos)
                if not branch.right_is_parent:
                    self.leaves.append(branch.right_pos)
            
            self.leaves = sorted(self.leaves, key=lambda x: x[0])
            self.parents = sorted(self.parents, key=lambda x: x[1])
            
            for key in self.branches:
                if self.branches[key].parent is None:
                    self.root = self.branches[key]

    # Helper: remap dendrogram coordinates to match split heatmap
    def _get_remapped_coords(Z, orientation, groups, inner_gs, gap_pixel, dendrogram_colors='colorful'):
        """
        Remap dendrogram coordinates so leaves align with split heatmap cells.
        Following PyComplexHeatmap's get_coords method.
        """
        n_leaves = Z.shape[0] + 1
        
        # Get dendrogram structure with colors
        # If colorful, use scipy default multi-color; otherwise use single color
        if dendrogram_colors == 'colorful':
            dendro = hierarchy.dendrogram(Z, no_plot=True)
        else:
            dendro = hierarchy.dendrogram(Z, no_plot=True, link_color_func=lambda k: dendrogram_colors)
        icoord = np.array(dendro['icoord']) / 10.0  # Normalize (PyComplexHeatmap uses /10)
        dcoord = np.array(dendro['dcoord'])
        color_list = dendro.get('color_list', ['black'] * len(Z))
        
        # Build tree structure
        tree = DenTree(icoord, dcoord)
        
        # Calculate sizes for each leaf position
        # sizes[i] = number of rows/cols in the i-th group (in dendrogram leaf order)
        leaves = dendro['leaves']
        sizes = []
        for leaf_idx in leaves:
            # Find which group this leaf belongs to
            for g_idx, grp in enumerate(groups):
                if leaf_idx in grp:
                    local_idx = grp.index(leaf_idx)
                    sizes.append(1)  # Each leaf represents 1 row/col
                    break
        
        icoord_max = len(leaves)
        ratio = 1.0
        x_gap = 0.0
        
        # Calculate scaling ratio based on gap
        if gap_pixel is not None and gap_pixel > 0:
            if orientation == 'top':
                # Get first subplot to measure real width
                ax_temp = fig.add_subplot(inner_gs[0, 0])
                ax_pos = ax_temp.get_window_extent()
                real_width = ax_pos.width - (len(groups) - 1) * gap_pixel
                ratio = real_width / ax_pos.width
                x_gap = (gap_pixel / ax_pos.width) * icoord_max
                ax_temp.remove()
            else:  # 'left'
                ax_temp = fig.add_subplot(inner_gs[0, 0])
                ax_pos = ax_temp.get_window_extent()
                real_height = ax_pos.height - (len(groups) - 1) * gap_pixel
                ratio = real_height / ax_pos.height
                x_gap = (gap_pixel / ax_pos.height) * icoord_max
                ax_temp.remove()
        
        # Map old x coordinates to new coordinates
        xcoord_mapping = {}
        
        # Calculate cumulative sizes
        cum_sizes = np.cumsum(sizes)
        
        # For each leaf, calculate new x coordinate
        for x_leaf in tree.leaves:
            idx = int(x_leaf[0])  # Leaf index in dendrogram order
            frac = x_leaf[0] % 1  # Fractional part (usually 0 or 0.5)
            
            # New x = fraction * size * ratio + cumulative offset + gap
            new_x = frac * sizes[idx] * ratio
            if idx > 0:
                new_x += cum_sizes[idx - 1] * ratio + x_gap * idx
            
            xcoord_mapping[x_leaf[0]] = new_x
        
        # Update leaf positions
        for key in tree.branches:
            branch = tree.branches[key]
            if not branch.left_is_parent:  # left is leaf
                x1 = xcoord_mapping[branch.left_pos[0]]
                y1 = branch.left_pos[1]
                branch.left_pos = (x1, y1)
            if not branch.right_is_parent:  # right is leaf
                x2 = xcoord_mapping[branch.right_pos[0]]
                y2 = branch.right_pos[1]
                branch.right_pos = (x2, y2)
            
            # Update root position
            y_root = branch.root_pos[1]
            x_root = (branch.left_pos[0] + branch.right_pos[0]) / 2
            branch.root_pos = (x_root, y_root)
        
        # Update parent node positions (bottom-up)
        for parent_pos in tree.parents:
            branch = tree.branches[parent_pos]
            if branch.left_is_parent:
                x1 = branch.left.root_pos[0]
                xcoord_mapping[branch.left_pos[0]] = x1
            else:
                x1 = branch.left_pos[0]
            
            if branch.right_is_parent:
                x2 = branch.right.root_pos[0]
                xcoord_mapping[branch.right_pos[0]] = x2
            else:
                x2 = branch.right_pos[0]
            
            xcoord_mapping[branch.root_pos[0]] = (x1 + x2) / 2
            y_root = branch.root_pos[1]
            x_root = (x1 + x2) / 2
            branch.root_pos = (x_root, y_root)
        
        # Remap icoord
        remapped_icoord = np.array([
            [xcoord_mapping[i] for i in a] for a in icoord
        ])
        
        return remapped_icoord, dcoord, dendro, color_list

    # --- Row dendrogram ---
    if row_Z is not None and "dendro_row" in ci:
        if n_row_split > 1:
            # Create GridSpec for row dendrogram (single tree spanning all splits)
            row_dendro_gs = gridspec.GridSpecFromSubplotSpec(
                1, 1,
                subplot_spec=gs[ri["heatmap"], ci["dendro_row"]],
            )

            ax_d = fig.add_subplot(row_dendro_gs[0, 0])

            # Get remapped coordinates WITH colors
            row_gap_pixel = row_split_gap * (1.0 / 25.4) * fig.dpi
            remapped_icoord, dcoord, dendro, color_list = _get_remapped_coords(
                row_Z, 'left', row_groups, inner, row_gap_pixel, dendrogram_colors
            )

            # Draw using the remapped coordinates WITH colors
            # For 'left' orientation: x=distance, y=position (inverted)
            for i in range(len(row_Z)):
                x = dcoord[i]
                y = remapped_icoord[i]
                color = color_list[i]
                ax_d.plot(x, y, color=color, linewidth=dendrogram_linewidth)

            # Set limits
            all_y = remapped_icoord.flatten()
            max_dist = np.max(dcoord) if len(dcoord) > 0 else 1
            ax_d.set_xlim(max_dist, 0)  # Invert: root at left
            ax_d.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
            _despine(ax_d)
        else:
            # No split — single dendrogram WITH colors
            ax_d = fig.add_subplot(gs[ri["heatmap"], ci["dendro_row"]])
            if dendrogram_colors == 'colorful':
                hierarchy.dendrogram(row_Z, ax=ax_d, orientation="left",
                                     no_labels=True)
            else:
                hierarchy.dendrogram(row_Z, ax=ax_d, orientation="left",
                                     no_labels=True,
                                     link_color_func=lambda k: dendrogram_colors)
            # Set linewidth for all lines in dendrogram
            for line in ax_d.get_lines():
                line.set_linewidth(dendrogram_linewidth)
            ax_d.invert_yaxis()
            _despine(ax_d)

    # --- Col dendrogram ---
    if col_Z is not None and "dendro_col" in ri:
        if n_col_split > 1:
            # Create GridSpec for col dendrogram (single tree spanning all splits)
            col_dendro_gs = gridspec.GridSpecFromSubplotSpec(
                1, 1,
                subplot_spec=gs[ri["dendro_col"], ci["heatmap"]],
            )

            ax_d = fig.add_subplot(col_dendro_gs[0, 0])

            # Get remapped coordinates WITH colors
            col_gap_pixel = col_split_gap * (1.0 / 25.4) * fig.dpi
            remapped_icoord, dcoord, dendro, color_list = _get_remapped_coords(
                col_Z, 'top', col_groups, inner, col_gap_pixel, dendrogram_colors
            )

            # Draw using the remapped coordinates WITH colors
            # For 'top' orientation: x=position, y=distance
            for i in range(len(col_Z)):
                x = remapped_icoord[i]
                y = dcoord[i]
                color = color_list[i]
                ax_d.plot(x, y, color=color, linewidth=dendrogram_linewidth)

            # Set limits
            all_x = remapped_icoord.flatten()
            max_dist = np.max(dcoord) if len(dcoord) > 0 else 1
            ax_d.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
            ax_d.set_ylim(0, max_dist * 1.05)
            _despine(ax_d)
        else:
            # No split — single dendrogram WITH colors
            ax_d = fig.add_subplot(gs[ri["dendro_col"], ci["heatmap"]])
            if dendrogram_colors == 'colorful':
                hierarchy.dendrogram(col_Z, ax=ax_d, orientation="top",
                                     no_labels=True)
            else:
                hierarchy.dendrogram(col_Z, ax=ax_d, orientation="top",
                                     no_labels=True,
                                     link_color_func=lambda k: dendrogram_colors)
            # Set linewidth for all lines in dendrogram
            for line in ax_d.get_lines():
                line.set_linewidth(dendrogram_linewidth)
            _despine(ax_d)

    # ------------------------------------------------------------------
    # Annotation bars (split-aware, matching heatmap GridSpec)
    # ------------------------------------------------------------------
    def _draw_anno_bars(direction, data, colors_map, continuous_data):
        # Determine if we should show annotation names based on direction
        if direction in ("left", "right"):
            show_names = annotation_names_row
        else:  # top, bottom
            show_names = annotation_names_col

        if direction in ("left", "right"):
            gs_key = ci.get(direction)
            if gs_key is None:
                return
            hm_row_idx = ri["heatmap"]
            n_a = data.shape[1]
            anno_gs = gridspec.GridSpecFromSubplotSpec(
                n_row_split,
                n_a,
                subplot_spec=gs[hm_row_idx, gs_key],
                height_ratios=[len(g) for g in row_groups],
                hspace=hspace,
                wspace=0,
            )
            for ai, acol in enumerate(data.columns):
                for gi in range(n_row_split):
                    ax_a = fig.add_subplot(anno_gs[gi, ai])
                    r_start, r_end = row_cum[gi], row_cum[gi + 1]
                    vals = data.iloc[r_start:r_end][acol].values
                    if acol in colors_map:
                        cm = colors_map[acol]
                        for ri2, v in enumerate(vals):
                            c = cm.get(v, "gray")
                            rect = Rectangle(
                                (0, ri2), 1, 1,
                                facecolor=c, edgecolor="white", linewidth=0.3,
                            )
                            ax_a.add_patch(rect)
                    else:
                        cvals = _get_continuous_colors(vals)
                        for ri2, c in enumerate(cvals):
                            rect = Rectangle(
                                (0, ri2), 1, 1,
                                facecolor=c, edgecolor="white", linewidth=0.3,
                            )
                            ax_a.add_patch(rect)
                    ax_a.set_xlim(0, 1)
                    ax_a.set_ylim(0, len(vals))
                    ax_a.invert_yaxis()
                    _despine(ax_a)

                    if gi == 0 and show_names:
                        pos = title_pos[direction]
                        if pos == "top":
                            ax_a.set_title(
                                acol, fontsize=fontsize - 2,
                                rotation=90, ha="center", va="bottom", pad=4.0,
                            )
                        else:
                            ax_a.set_xlabel(
                                acol, fontsize=fontsize - 2,
                                rotation=90, ha="center", va="top", labelpad=4.0,
                            )
        else:
            gs_key = ri.get(direction)
            if gs_key is None:
                return
            hm_col_idx = ci["heatmap"]
            n_a = data.shape[1]
            anno_gs = gridspec.GridSpecFromSubplotSpec(
                n_a,
                n_col_split,
                subplot_spec=gs[gs_key, hm_col_idx],
                width_ratios=[len(g) for g in col_groups],
                wspace=wspace,
                hspace=0,
            )
            for ai, acol in enumerate(data.columns):
                for gj in range(n_col_split):
                    ax_a = fig.add_subplot(anno_gs[ai, gj])
                    c_start, c_end = col_cum[gj], col_cum[gj + 1]
                    vals = data.iloc[c_start:c_end][acol].values
                    if acol in colors_map:
                        cm = colors_map[acol]
                        for ci2, v in enumerate(vals):
                            c = cm.get(v, "gray")
                            rect = Rectangle(
                                (ci2, 0), 1, 1,
                                facecolor=c, edgecolor="white", linewidth=0.3,
                            )
                            ax_a.add_patch(rect)
                    else:
                        cvals = _get_continuous_colors(vals)
                        for ci2, c in enumerate(cvals):
                            rect = Rectangle(
                                (ci2, 0), 1, 1,
                                facecolor=c, edgecolor="white", linewidth=0.3,
                            )
                            ax_a.add_patch(rect)
                    ax_a.set_xlim(0, len(vals))
                    ax_a.set_ylim(0, 1)
                    _despine(ax_a)

                    if gj == 0 and show_names:
                        pos = title_pos[direction]
                        if pos == "left":
                            ax_a.set_ylabel(
                                acol, fontsize=fontsize - 2,
                                rotation=0, ha="right", va="center", labelpad=4.0,
                            )
                        else:
                            ax_a.yaxis.set_label_position("right")
                            ax_a.set_ylabel(
                                acol, fontsize=fontsize - 2,
                                rotation=0, ha="left", va="center", labelpad=4.0,
                            )

    for direction in ("left", "right", "top", "bottom"):
        if direction in anno_data:
            _draw_anno_bars(
                direction,
                anno_data[direction],
                anno_colors.get(direction, {}),
                anno_continuous.get(direction, {}),
            )

    # ------------------------------------------------------------------
    #  Legend area – GridSpec-backed, zero-overlap layout
    # ------------------------------------------------------------------
    mm2inch = 1.0 / 25.4

    def _plot_color_dict_legend_fig(D, title, kws=None):
        lgd_kws = kws.copy() if kws else {}
        lgd_kws.setdefault("frameon", False)
        lgd_kws.setdefault("ncol", 1)
        lgd_kws["loc"] = "upper left"
        lgd_kws["bbox_transform"] = fig.transFigure
        lgd_kws.setdefault("borderpad", 0.1 * mm2inch * 72)
        lgd_kws.setdefault("markerscale", 1.5)
        lgd_kws.setdefault("handleheight", 1.0)
        lgd_kws.setdefault("handlelength", 1.5)
        lgd_kws.setdefault("borderaxespad", 0.1)
        lgd_kws.setdefault("handletextpad", 0.3)
        lgd_kws.setdefault("labelspacing", 0.1)
        lgd_kws.setdefault("columnspacing", 0.4)
        lgd_kws.setdefault("title", title)
        lgd_kws.setdefault("markerfirst", True)
        patches = [
            Patch(facecolor=c, edgecolor="black", linewidth=0.5) for c in D.values()
        ]
        L = fig.legend(handles=patches, labels=list(D.keys()), **lgd_kws)
        fig.canvas.draw()
        return L

    def _plot_cmap_legend(cax, cmap, label, kws=None):
        cbar_kws = kws.copy() if kws else {}
        ticklabels = cbar_kws.pop("ticklabels", None)
        fs = cbar_kws.pop("fontsize", None)
        cbar_kws.setdefault("label", label)
        cbar_kws.setdefault("orientation", "vertical")
        cbar_kws.setdefault("fraction", 1)
        cbar_kws.setdefault("shrink", 1)
        cbar_kws.setdefault("pad", 0)
        vmax = cbar_kws.pop("vmax", 1)
        vmin = cbar_kws.pop("vmin", 0)
        cax.set_ylim([vmin, vmax])
        center = cbar_kws.pop("center", None)
        if center is None:
            m = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        else:
            from matplotlib.colors import TwoSlopeNorm

            m = ScalarMappable(
                cmap=cmap, norm=TwoSlopeNorm(center, vmin=vmin, vmax=vmax)
            )
        cbar_kws.setdefault("ticks", [vmin, (vmin + vmax) / 2, vmax])
        cax.yaxis.set_label_position("right")
        cax.yaxis.set_ticks_position("right")
        cbar = fig.colorbar(m, cax=cax, **cbar_kws)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
        if fs is not None:
            cbar.ax.tick_params(labelsize=fs)
        return cbar

    if legend or has_any_legends:
        fig.canvas.draw()

        legend_list = []

        if legend:
            legend_kws = {}
            if legend_breaks is not None:
                legend_kws["ticks"] = legend_breaks
                if legend_labels is not None:
                    legend_kws["ticklabels"] = legend_labels
            legend_list.append(
                (cmap_obj, legend_title or "Value", legend_kws, 4, "cmap")
            )

        if annotation_legend:
            for direction in ("left", "right", "top", "bottom"):
                for acol, cmap_m in anno_colors.get(direction, {}).items():
                    # drop_categories=True (default): only show categories present in data
                    # drop_categories=False: show all categories from annotation_colors if provided
                    legend_list.append(
                        (
                            cmap_m,
                            acol,
                            {"fontsize": fontsize - 3},
                            len(cmap_m),
                            "color_dict",
                        )
                    )
                for acol, vals in anno_continuous.get(direction, {}).items():
                    arr = np.array(vals, dtype=float)
                    valid = arr[~np.isnan(arr)]
                    if len(valid) > 0:
                        lkws = {
                            "vmin": round(valid.min(), 2),
                            "vmax": round(valid.max(), 2),
                            "fontsize": fontsize - 4,
                        }
                        legend_list.append(("Blues", acol, lkws, 4, "cmap"))

        if len(legend_list) > 0:
            type_order = {"cmap": 0, "color_dict": 1, "markers": 2}
            legend_list = sorted(
                legend_list, key=lambda x: (type_order.get(x[4], 9), x[3])
            )

            fig.canvas.draw()
            hm_pos = main_ax.get_position()
            dpi = fig.dpi
            fig_w_px = fig.get_window_extent().width
            fig_h_px = fig.get_window_extent().height

            rightmost_x = hm_pos.x1
            for ax in fig.axes:
                p = ax.get_position()
                if p.x1 > rightmost_x:
                    rightmost_x = p.x1

            v_gap = 2 * mm2inch * dpi / fig_h_px
            legend_vpad = 5 * mm2inch * dpi / fig_h_px
            cmap_width_mm = 4.5
            cmap_width_norm = cmap_width_mm * mm2inch * dpi / fig_w_px
            h_gap_norm = 2 * mm2inch * dpi / fig_w_px

            leg_start_x = rightmost_x + 0.02
            leg_start_y = hm_pos.y1 - legend_vpad
            leg_bottom = hm_pos.y0

            cbars = []
            y = leg_start_y
            x = leg_start_x
            lgd_col_max_width = 0

            i = 0
            while i < len(legend_list):
                obj, title, legend_kws, n, lgd_t = legend_list[i]

                if lgd_t == "cmap":
                    f = 15 * mm2inch * dpi / fig_h_px
                    if y - f < leg_bottom:
                        x = leg_start_x + lgd_col_max_width / fig_w_px + h_gap_norm
                        y = leg_start_y
                        lgd_col_max_width = 0

                    cax = fig.add_axes(
                        [x, y - f, cmap_width_norm, f],
                        xmargin=0,
                        ymargin=0,
                    )
                    cax.figure.subplots_adjust(bottom=0)
                    cbar = _plot_cmap_legend(cax, cmap=obj, label=title, kws=legend_kws)
                    cbar.ax.tick_params(labelsize=fontsize - 4)
                    fig.canvas.draw()
                    cbar_actual_h = cbar.ax.get_window_extent().height / fig_h_px
                    cbar_width = cbar.ax.get_window_extent().width
                    cbars.append(cbar)
                    if cbar_width > lgd_col_max_width:
                        lgd_col_max_width = cbar_width
                    f = cbar_actual_h

                elif lgd_t == "color_dict":
                    legend_kws["bbox_to_anchor"] = (x, y)
                    L = _plot_color_dict_legend_fig(D=obj, title=title, kws=legend_kws)
                    if (
                        L is None
                        or L.get_window_extent().height > (y - leg_bottom) * fig_h_px
                    ):
                        if L is not None:
                            L.remove()
                        x = leg_start_x + lgd_col_max_width / fig_w_px + h_gap_norm
                        y = leg_start_y
                        lgd_col_max_width = 0
                        legend_kws["bbox_to_anchor"] = (x, y)
                        L = _plot_color_dict_legend_fig(
                            D=obj, title=title, kws=legend_kws
                        )

                    if L is not None:
                        L_width = L.get_window_extent().width
                        if L_width > lgd_col_max_width:
                            lgd_col_max_width = L_width
                        f = L.get_window_extent().height / fig_h_px
                        cbars.append(L)
                    else:
                        f = 0

                y = y - f - v_gap
                i += 1

            for cbar in cbars:
                if hasattr(cbar, "ax") and hasattr(cbar.ax, "yaxis"):
                    cbar.ax.yaxis.label.set_fontsize(fontsize - 2)
                    cbar.ax.yaxis.label.set_fontweight("bold")

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    if main is not None:
        fig.suptitle(main, fontsize=fontsize + 2, y=1)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    if silent:
        plt.close(fig)

    return fig

