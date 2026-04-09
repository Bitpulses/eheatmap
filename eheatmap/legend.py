"""
eheatmap - Legend module
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch


def _plot_color_dict_legend_fig(fig, D, title, kws=None, mm2inch=1.0 / 25.4):
    """Plot legend for color dictionary (categorical)."""
    lgd_kws = kws.copy() if kws else {}
    lgd_kws.setdefault("frameon", False)
    lgd_kws.setdefault("ncol", 1)
    lgd_kws["loc"] = "upper left"
    lgd_kws["bbox_transform"] = fig.transFigure
    lgd_kws.setdefault("borderpad", 0.1 * mm2inch * 72)
    lgd_kws.setdefault("markerscale", 1.5)
    lgd_kws.setdefault("handlelength", 1.5)
    lgd_kws.setdefault("borderaxespad", 0.1)
    lgd_kws.setdefault("handletextpad", 0.3)
    lgd_kws.setdefault("labelspacing", 0.1)
    lgd_kws.setdefault("columnspacing", 0.4)
    lgd_kws.setdefault("title", title)
    lgd_kws.setdefault("markerfirst", True)
    # Ensure title and markers are left-aligned
    lgd_kws.setdefault("alignment", "left")
    lgd_kws.setdefault("handleheight", 1.0)
    patches = [Patch(facecolor=c, edgecolor="black", linewidth=0.5) for c in D.values()]
    L = fig.legend(handles=patches, labels=list(D.keys()), **lgd_kws)
    fig.canvas.draw()
    return L


def _plot_cmap_legend(fig, cax, cmap, label, kws=None):
    """Plot colormap legend."""
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
        m = ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(center, vmin=vmin, vmax=vmax))
    
    # Generate default ticks if not specified
    default_ticklabels = None
    if "ticks" not in cbar_kws:
        # 强制显示三个刻度：最小值，中间值，最大值
        # 不需要取整数，直接使用原始浮点数值
        mid_val = (vmin + vmax) / 2.0
        ticks = [vmin, mid_val, vmax]
        # 预先计算好格式化后的字符串，用于后续设置
        default_ticklabels = [f"{v:.1f}" for v in ticks]
        cbar_kws["ticks"] = ticks

    cax.yaxis.set_label_position("right")
    cax.yaxis.set_ticks_position("right")
    cbar = fig.colorbar(m, cax=cax, **cbar_kws)
    
    # 如果是我们自动生成的刻度，应用1位小数格式化
    if default_ticklabels is not None:
        cbar.set_ticklabels(default_ticklabels)
        
    if fs is not None:
        cbar.ax.tick_params(labelsize=fs)
    return cbar


def build_legend_list(
    legend,
    legend_title,
    legend_breaks,
    legend_labels,
    cmap_obj,
    annotation_legend,
    anno_colors,
    anno_continuous,
    vmin=None,
    vmax=None,
    center=None,
):
    """Build legend list from all legend items."""
    legend_list = []

    if legend:
        legend_kws = {}
        if vmin is not None:
            legend_kws["vmin"] = vmin
        if vmax is not None:
            legend_kws["vmax"] = vmax
        if center is not None:
            legend_kws["center"] = center
        if legend_breaks is not None:
            legend_kws["ticks"] = legend_breaks
            if legend_labels is not None:
                legend_kws["ticklabels"] = legend_labels
        legend_list.append((cmap_obj, legend_title or "Value", legend_kws, 4, "cmap"))

    if annotation_legend:
        for direction in ("left", "right", "top", "bottom"):
            for acol, cmap_m in anno_colors.get(direction, {}).items():
                legend_list.append(
                    (cmap_m, acol, {"fontsize": 7}, len(cmap_m), "color_dict")
                )
            for acol, vals in anno_continuous.get(direction, {}).items():
                arr = np.array(vals, dtype=float)
                valid = arr[~np.isnan(arr)]
                if len(valid) > 0:
                    lkws = {
                        "vmin": round(valid.min(), 2),
                        "vmax": round(valid.max(), 2),
                        "fontsize": 6,
                    }
                    legend_list.append(("Blues", acol, lkws, 4, "cmap"))

    if len(legend_list) > 0:
        type_order = {"cmap": 0, "color_dict": 1, "markers": 2}
        legend_list = sorted(legend_list, key=lambda x: (type_order.get(x[4], 9), x[3]))

    return legend_list


def draw_legends(fig, main_ax, legend_list, fontsize):
    """Draw all legends on the figure."""
    mm2inch = 1.0 / 25.4
    fig.canvas.draw()

    if len(legend_list) == 0:
        return

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
            cbar = _plot_cmap_legend(
                fig, cax=cax, cmap=obj, label=title, kws=legend_kws
            )
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
            L = _plot_color_dict_legend_fig(
                fig=fig, D=obj, title=title, kws=legend_kws, mm2inch=mm2inch
            )
            if L is None or L.get_window_extent().height > (y - leg_bottom) * fig_h_px:
                if L is not None:
                    L.remove()
                x = leg_start_x + lgd_col_max_width / fig_w_px + h_gap_norm
                y = leg_start_y
                lgd_col_max_width = 0
                legend_kws["bbox_to_anchor"] = (x, y)
                L = _plot_color_dict_legend_fig(
                    fig=fig, D=obj, title=title, kws=legend_kws, mm2inch=mm2inch
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
