"""
eheatmap - Annotations module
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from .utils import _get_annotation_colors, _get_continuous_colors, _to_mpl_color


def _draw_sub_heatmap(
    ax,
    sub_df,
    cmap_obj,
    norm,
    border_color,
    use_border,
    display_numbers,
    number_format,
    number_color,
    fontsize_number,
    has_numbers,
    is_split=False,
    split_border_color=None,
    split_border_width=None,
):
    """Draw a single heatmap sub-block (used for split rendering)."""
    ax.pcolormesh(
        sub_df.values,
        cmap=cmap_obj,
        norm=norm,
        edgecolors=border_color if use_border else "none",
        linewidths=0.02 if use_border else 0,
    )
    ax.set_xlim(0, sub_df.shape[1])
    ax.set_ylim(0, sub_df.shape[0])
    ax.invert_yaxis()

    ax.set_xticks([])
    ax.set_yticks([])

    if is_split and split_border_color:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(split_border_color)
            spine.set_linewidth(split_border_width)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if has_numbers:
        for ri2 in range(sub_df.shape[0]):
            for ci2 in range(sub_df.shape[1]):
                val = sub_df.values[ri2, ci2]
                if np.isnan(val):
                    continue
                txt = (
                    number_format % val
                    if isinstance(display_numbers, bool)
                    else str(val)
                )
                ax.text(
                    ci2 + 0.5,
                    ri2 + 0.5,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=fontsize_number,
                    color=number_color,
                )


def draw_heatmap_cells(
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
):
    """Draw heatmap cells with optional split support."""
    has_numbers = display_numbers is not False and display_numbers is not None
    n_row_split = len(row_groups)
    n_col_split = len(col_groups)
    main_ax = fig.add_subplot(gs[ri["heatmap"], ci["heatmap"]])

    if n_row_split > 1 or n_col_split > 1:
        main_ax.set_axis_off()
        fig.canvas.draw()
        hm_extent = main_ax.get_window_extent()

        mm2inch = 1.0 / 25.4
        dpi = fig.dpi
        border_comp_px = split_border_width * dpi / 72.0

        if n_row_split > 1:
            target_row_gap_px = 0.4 * mm2inch * dpi + border_comp_px
            n = n_row_split
            num = n * target_row_gap_px
            den = hm_extent.height - (n - 1) * target_row_gap_px
            hspace = (num / den) if den > 0 else 0.01
        else:
            hspace = 0.01

        if n_col_split > 1:
            target_col_gap_px = 0.4 * mm2inch * dpi + border_comp_px
            n = n_col_split
            num = n * target_col_gap_px
            den = hm_extent.width - (n - 1) * target_col_gap_px
            wspace = (num / den) if den > 0 else 0.01
        else:
            wspace = 0.01
            hspace = 0.01

        inner = gridspec.GridSpecFromSubplotSpec(
            n_row_split,
            n_col_split,
            subplot_spec=gs[ri["heatmap"], ci["heatmap"]],
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
                    ax,
                    sub,
                    cmap_obj,
                    norm,
                    border_color,
                    use_border,
                    display_numbers,
                    number_format,
                    number_color,
                    fontsize_number,
                    has_numbers,
                    is_split=True,
                    split_border_color=split_border_color,
                    split_border_width=split_border_width,
                )
    else:
        main_ax.pcolormesh(
            df.values,
            cmap=cmap_obj,
            norm=norm,
            edgecolors=border_color if use_border else "none",
            linewidths=0.02 if use_border else 0,
        )
        main_ax.set_xlim(0, df.shape[1])
        main_ax.set_ylim(0, df.shape[0])
        main_ax.invert_yaxis()

        for spine in main_ax.spines.values():
            spine.set_visible(False)

    return (
        main_ax,
        wspace if n_row_split > 1 or n_col_split > 1 else 0.01,
        hspace if n_row_split > 1 or n_col_split > 1 else 0.01,
        inner if n_row_split > 1 or n_col_split > 1 else None,
    )


def draw_cell_labels(
    ax, df, show_rownames, show_colnames, fontsize_row, fontsize_col, angle_col
):
    """Draw row and column labels."""
    if show_colnames:
        ax.set_xticks(np.arange(df.shape[1]) + 0.5)
        ax.set_xticklabels(
            df.columns,
            rotation=angle_col,
            ha="center" if angle_col in (0, 270) else "right",
            fontsize=fontsize_col,
        )
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks([])

    if show_rownames:
        ax.set_yticks(np.arange(df.shape[0]) + 0.5)
        ax.set_yticklabels(df.index, fontsize=fontsize_row, ha="right")
        ax.tick_params(axis="y", length=0)
    else:
        ax.set_yticks([])


def draw_cell_numbers(
    ax, df, display_numbers, number_format, number_color, fontsize_number
):
    """Draw cell numbers if requested."""
    has_numbers = display_numbers is not False and display_numbers is not None
    if has_numbers:
        for ri2 in range(df.shape[0]):
            for ci2 in range(df.shape[1]):
                val = df.values[ri2, ci2]
                if np.isnan(val):
                    continue
                txt = (
                    number_format % val
                    if isinstance(display_numbers, bool)
                    else str(val)
                )
                ax.text(
                    ci2 + 0.5,
                    ri2 + 0.5,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=fontsize_number,
                    color=number_color,
                )


def draw_annotation_bars(
    fig,
    direction,
    data,
    colors_map,
    continuous_data,
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
    border_color,
    border_width,
):
    """Draw annotation bars for a given direction."""
    if direction in ("left", "right"):
        show_names = annotation_names_row
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
                            (0, ri2),
                            1,
                            1,
                            facecolor=c,
                            edgecolor="none",
                            linewidth=0,
                        )
                        ax_a.add_patch(rect)
                else:
                    cvals = _get_continuous_colors(vals)
                    for ri2, c in enumerate(cvals):
                        rect = Rectangle(
                            (0, ri2),
                            1,
                            1,
                            facecolor=c,
                            edgecolor="none",
                            linewidth=0,
                        )
                        ax_a.add_patch(rect)
                ax_a.set_xlim(0, 1)
                ax_a.set_ylim(0, len(vals))
                ax_a.invert_yaxis()
                ax_a.set_xticks([])
                ax_a.set_yticks([])

                if show_anno_border:
                    for spine in ax_a.spines.values():
                        spine.set_visible(True)
                        spine.set_color(border_color)
                        spine.set_linewidth(border_width)
                else:
                    for spine in ax_a.spines.values():
                        spine.set_visible(False)

                if gi == 0 and show_names:
                    pos = title_pos[direction]
                    if pos == "top":
                        ax_a.set_title(
                            acol,
                            fontsize=fontsize - 2,
                            rotation=90,
                            ha="center",
                            va="bottom",
                            pad=4.0,
                        )
                    else:
                        ax_a.set_xlabel(
                            acol,
                            fontsize=fontsize - 2,
                            rotation=90,
                            ha="center",
                            va="top",
                            labelpad=4.0,
                        )
    else:
        show_names = annotation_names_col
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
                            (ci2, 0),
                            1,
                            1,
                            facecolor=c,
                            edgecolor="none",
                            linewidth=0,
                        )
                        ax_a.add_patch(rect)
                else:
                    cvals = _get_continuous_colors(vals)
                    for ci2, c in enumerate(cvals):
                        rect = Rectangle(
                            (ci2, 0),
                            1,
                            1,
                            facecolor=c,
                            edgecolor="none",
                            linewidth=0,
                        )
                        ax_a.add_patch(rect)
                ax_a.set_xlim(0, len(vals))
                ax_a.set_ylim(0, 1)
                ax_a.set_xticks([])
                ax_a.set_yticks([])

                if show_anno_border:
                    for spine in ax_a.spines.values():
                        spine.set_visible(True)
                        spine.set_color(border_color)
                        spine.set_linewidth(border_width)
                else:
                    for spine in ax_a.spines.values():
                        spine.set_visible(False)

                # 对于底部/顶部注释（列注释）：
                # - 没有split时：在每个注释行的第一个split块（gj == 0）显示title
                # - 有split时：在每个注释行的最后一个split块显示title，这样title会出现在注释条的右侧
                if show_names:
                    should_show_title = False
                    if n_col_split == 1:
                        # 没有split，在第一个（也是唯一一个）split块显示title
                        should_show_title = (gj == 0)
                    else:
                        # 有split，在最后一个split块显示title
                        should_show_title = (gj == n_col_split - 1)
                    
                    if should_show_title:
                        pos = title_pos[direction]
                        if pos == "left":
                            ax_a.set_ylabel(
                                acol,
                                fontsize=fontsize - 2,
                                rotation=0,
                                ha="right",
                                va="center",
                                labelpad=4.0,
                            )
                        else:
                            ax_a.yaxis.set_label_position("right")
                            ax_a.set_ylabel(
                                acol,
                                fontsize=fontsize - 2,
                                rotation=0,
                                ha="left",
                                va="center",
                                labelpad=4.0,
                            )


def process_annotations(
    df,
    annotation_top,
    annotation_bottom,
    annotation_left,
    annotation_right,
    annotation_colors,
):
    """Process annotation data and colors."""
    import pandas as pd
    from .utils import _is_continuous

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

    return anno_data, anno_colors, anno_continuous
