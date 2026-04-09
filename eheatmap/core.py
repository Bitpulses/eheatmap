"""
Effortless heatmap generation in Python: Focus on your data, not the plotting code. 
It provides extensive control over clustering, annotations, color mapping, and layout,
making it ideal for scientific data visualization and publication-ready figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable

from .utils import _to_mpl_color, _despine, _get_annotation_colors, _is_continuous
from .clustering import (
    compute_row_clusters,
    compute_col_clusters,
    compute_row_groups,
    compute_col_groups,
    compute_cum_indices,
)
from .annotations import (
    process_annotations,
    draw_heatmap_cells,
    draw_cell_labels,
    draw_cell_numbers,
    draw_annotation_bars,
)
from .dendrogram import draw_dendrogram
from .legend import build_legend_list, draw_legends


def eheatmap(
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
    dendrogram_linewidth=0.5,
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
    row_names_side='left',
    annotation_top=None,
    annotation_bottom=None,
    annotation_left=None,
    annotation_right=None,
    annotation_title_pos= ["bottom", "left", "top", "right"],
    drop_categories=True,
    show_rownames=True,
    show_colnames=True,
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
    col_split_gap=0.4,
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
    split_border_color=None,
    split_border_width=0.5,
    row_annotation=None,
    col_annotation=None,
    row_dendrogram=True,
    col_dendrogram=True,
    method=None,
    metric=None,
):
    """
    Draw a clustered heatmap – Python equivalent of R's pheatmap().
    """

    # ------------------------------------------------------------
    # 0. Normalise inputs
    # ------------------------------------------------------------
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

    if dendrogram_linewidth is None:
        dendrogram_linewidth = 0.5

    if dendrogram_colors is None:
        dendrogram_colors = "colorful"

    if not row_dendrogram:
        cluster_rows = False
    if not col_dendrogram:
        cluster_cols = False

    border_color = _to_mpl_color(border_color)
    number_color = _to_mpl_color(number_color)
    na_col = _to_mpl_color(na_col)

    # ------------------------------------------------------------
    # 1. Handle NA
    # ------------------------------------------------------------
    df_for_cluster = df.fillna(0)

    # ------------------------------------------------------------
    # 2. Scale
    # ------------------------------------------------------------
    if scale == "row":
        df = df.apply(lambda r: (r - r.mean()) / r.std() if r.std() > 0 else 0, axis=1)
        df = df.fillna(0)
    elif scale == "column":
        df = (df - df.mean()) / df.std()
        df = df.fillna(0)

    df_for_cluster = df.fillna(0)

    # ------------------------------------------------------------
    # 3. K-means aggregation
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 4. Clustering
    # ------------------------------------------------------------
    row_order, row_Z = compute_row_clusters(
        df_for_cluster, cluster_rows, clustering_method, clustering_distance_rows
    )
    col_order, col_Z = compute_col_clusters(
        df_for_cluster, cluster_cols, clustering_method, clustering_distance_cols
    )

    df = df.iloc[row_order, col_order]

    # ------------------------------------------------------------
    # 5. Labels override
    # ------------------------------------------------------------
    if labels_row is not None:
        df.index = labels_row
    if labels_col is not None:
        df.columns = labels_col

    # ------------------------------------------------------------
    # 6. Annotation processing
    # ------------------------------------------------------------
    anno_data, anno_colors, anno_continuous = process_annotations(
        df,
        annotation_top,
        annotation_bottom,
        annotation_left,
        annotation_right,
        annotation_colors,
    )

    # ------------------------------------------------------------
    # 7. Gaps / splits
    # ------------------------------------------------------------
    row_groups = compute_row_groups(df, cutree_rows, gaps_row, row_Z)
    col_groups = compute_col_groups(df, cutree_cols, gaps_col, col_Z)

    n_row_groups = len(row_groups)
    n_col_groups = len(col_groups)

    row_cum = compute_cum_indices(row_groups)
    col_cum = compute_cum_indices(col_groups)

    n_row_split = n_row_groups
    n_col_split = n_col_groups

    has_splits = n_row_split > 1 or n_col_split > 1
    draw_outer_border = not has_splits and split_border_color is not None
    show_anno_border = has_splits or draw_outer_border

    # ------------------------------------------------------------
    # 8. Color mapping
    # ------------------------------------------------------------
    if breaks is not None:
        breaks = np.asarray(breaks)
        if vmin is None:
            vmin = breaks[0]
        if vmax is None:
            vmax = breaks[-1]
    else:
        if vmin is None:
            vmin = df.values[~np.isnan(df.values)].min() if np.any(~np.isnan(df.values)) else -1
        if vmax is None:
            vmax = df.values[~np.isnan(df.values)].max() if np.any(~np.isnan(df.values)) else 1

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

    # ------------------------------------------------------------
    # 9. Collect legend items
    # ------------------------------------------------------------
    has_any_legends = annotation_legend and any(
        len(anno_colors.get(d, {})) > 0 or len(anno_continuous.get(d, {})) > 0
        for d in ("top", "bottom", "left", "right")
    )

    # ------------------------------------------------------------
    # 10. Figure layout
    # ------------------------------------------------------------
    dendro_ratio = 0.6
    anno_bar = 0.1
    label_space = 1.0
    legend_area_w = 5.0 if (legend or has_any_legends) else 0

    if cellwidth is not None:
        hm_w = df.shape[1] * cellwidth / 72
    else:
        hm_w = max(df.shape[1] * 0.12, 3)

    if cellheight is not None:
        hm_h = df.shape[0] * cellheight / 72
    else:
        hm_h = max(df.shape[0] * 0.12, 3)

    left_w = dendro_ratio if (row_Z is not None) else 0
    if "left" in anno_data:
        left_w += anno_bar * anno_data["left"].shape[1]
    right_w = anno_bar * anno_data["right"].shape[1] if "right" in anno_data else 0
    top_h_est = dendro_ratio if (col_Z is not None) else 0
    if "top" in anno_data:
        top_h_est += anno_bar * anno_data["top"].shape[1]
    bottom_h_est = (
        anno_bar * anno_data["bottom"].shape[1] if "bottom" in anno_data else 0
    )

    est_total_w = left_w + hm_w + right_w + label_space + legend_area_w
    est_total_h = top_h_est + hm_h + bottom_h_est + label_space

    if figsize is not None:
        fig_w, fig_h = figsize
    elif width is not None and height is not None:
        fig_w, fig_h = width, height
    else:
        fig_w = min(est_total_w, 28)
        fig_h = min(est_total_h, 28)

    fig = plt.figure(figsize=(fig_w, fig_h))

    margin_w = 0.80
    margin_h = 0.90

    sum_wr = (dendro_ratio if (row_Z is not None) else 0) + hm_w
    if "left" in anno_data:
        sum_wr += anno_bar * anno_data["left"].shape[1]
    if "right" in anno_data:
        sum_wr += anno_bar * anno_data["right"].shape[1]

    S = (anno_bar / sum_wr) * (fig_w * margin_w) if sum_wr > 0 else anno_bar

    sum_hr_fixed = (dendro_ratio if (col_Z is not None) else 0) + hm_h

    N_tracks_h = 0
    if "top" in anno_data:
        N_tracks_h += anno_data["top"].shape[1]
    if "bottom" in anno_data:
        N_tracks_h += anno_data["bottom"].shape[1]

    Available_H = fig_h * margin_h
    denom = Available_H - (S * N_tracks_h)

    if denom > 0 and N_tracks_h > 0:
        col_anno_unit_ratio = (S * sum_hr_fixed) / denom
    else:
        col_anno_unit_ratio = anno_bar

    top_h = dendro_ratio if (col_Z is not None) else 0
    if "top" in anno_data:
        top_h += col_anno_unit_ratio * anno_data["top"].shape[1]
    bottom_h = 0
    if "bottom" in anno_data:
        bottom_h = col_anno_unit_ratio * anno_data["bottom"].shape[1]

    n_cols = 0
    ci = {}

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
        if treeheight_col is not None and treeheight_col > 0:
            hr.append(treeheight_col * dendro_ratio / 50.0)
        else:
            hr.append(dendro_ratio)
    if "top" in anno_data:
        hr.append(col_anno_unit_ratio * anno_data["top"].shape[1])
    hr.append(hm_h)
    if "bottom" in anno_data:
        hr.append(col_anno_unit_ratio * anno_data["bottom"].shape[1])

    sum_wr = sum(wr)
    sum_hr = sum(hr)

    hm_ratio_w = hm_w / sum_wr
    hm_ratio_h = hm_h / sum_hr

    fig_w, fig_h = fig.get_size_inches()

    target_hspace = 0.04

    if fig_w > 0 and hm_ratio_w > 0:
        wspace = (target_hspace * fig_h * hm_ratio_h) / (fig_w * hm_ratio_w)
    else:
        wspace = target_hspace

    hspace = target_hspace

    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        width_ratios=wr,
        height_ratios=hr,
        wspace=wspace,
        hspace=hspace,
        left=0.05,
        right=0.85,
        top=0.95,
        bottom=0.05,
    )

    # ------------------------------------------------------------
    # 11. Draw heatmap cells
    # ------------------------------------------------------------
    use_border = border_color not in ("NA", "none", None)
    main_ax, wspace, hspace, inner_gs = draw_heatmap_cells(
        fig,
        gs,
        df,
        cmap_obj,
        norm,
        row_groups,
        col_groups,
        row_cum,
        col_cum,
        border_color,
        use_border,
        display_numbers,
        number_format,
        number_color,
        fontsize_number,
        split_border_color,
        split_border_width,
        ci,
        ri,
    )

    # Draw cell labels for non-split heatmap
    if n_row_split == 1 and n_col_split == 1:
        draw_cell_labels(
            main_ax,
            df,
            show_rownames,
            show_colnames,
            fontsize_row,
            fontsize_col,
            angle_col,
        )
        draw_cell_numbers(
            main_ax, df, display_numbers, number_format, number_color, fontsize_number
        )

        if draw_outer_border:
            for spine in main_ax.spines.values():
                spine.set_visible(True)
                spine.set_color(split_border_color)
                spine.set_linewidth(split_border_width)

    # ------------------------------------------------------------
    # 12. Dendrograms
    # ------------------------------------------------------------
    fig.canvas.draw()
    mm2inch = 1.0 / 25.4

    if row_Z is not None and "dendro_row" in ci:
        row_gap_pixel = row_split_gap * mm2inch * fig.dpi if n_row_split > 1 else None
        draw_dendrogram(
            fig,
            row_Z,
            "left",
            row_groups,
            gs[ri["heatmap"], ci["dendro_row"]],
            n_row_split,
            dendrogram_linewidth,
            dendrogram_colors,
            row_gap_pixel,
            inner_gs,
        )

    if col_Z is not None and "dendro_col" in ri:
        col_gap_pixel = col_split_gap * mm2inch * fig.dpi if n_col_split > 1 else None
        draw_dendrogram(
            fig,
            col_Z,
            "top",
            col_groups,
            gs[ri["dendro_col"], ci["heatmap"]],
            n_col_split,
            dendrogram_linewidth,
            dendrogram_colors,
            col_gap_pixel,
            inner_gs,
        )

    # ------------------------------------------------------------
    # 13. Annotation bars
    # ------------------------------------------------------------
    for direction in ("left", "right", "top", "bottom"):
        if direction in anno_data:
            draw_annotation_bars(
                fig,
                direction,
                anno_data[direction],
                anno_colors.get(direction, {}),
                anno_continuous.get(direction, {}),
                gs,
                ci,
                ri,
                row_groups,
                col_groups,
                row_cum,
                col_cum,
                n_row_split,
                n_col_split,
                wspace,
                hspace,
                title_pos,
                annotation_names_row,
                annotation_names_col,
                fontsize,
                show_anno_border,
                split_border_color,
                split_border_width,
            )

    # ------------------------------------------------------------
    # 14. Legends
    # ------------------------------------------------------------
    if legend or has_any_legends:
        # 优化：如果没有指定 legend_breaks，但指定了 breaks，则使用 breaks 作为刻度
        final_legend_breaks = legend_breaks if legend_breaks is not None else breaks
        
        legend_list = build_legend_list(
            legend,
            legend_title,
            final_legend_breaks,
            legend_labels,
            cmap_obj,
            annotation_legend,
            anno_colors,
            anno_continuous,
            vmin=vmin,
            vmax=vmax,
            center=center,
        )
        draw_legends(fig, main_ax, legend_list, fontsize)

    # ------------------------------------------------------------
    # 15. Title
    # ------------------------------------------------------------
    if main is not None:
        fig.suptitle(main, fontsize=fontsize + 2, y=1)

    # ------------------------------------------------------------
    # 16. Save
    # ------------------------------------------------------------
    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    if silent:
        plt.close(fig)

    return fig

# Backward compatibility alias
pheatmap = eheatmap
