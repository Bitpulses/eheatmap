"""
eheatmap - Dendrogram module
"""

import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from .utils import _despine


class Branch:
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
        self.left_is_parent = self.left_pos[1] != 0
        self.right_is_parent = self.right_pos[1] != 0


class DenTree:
    """Tree structure for coordinate remapping."""

    def __init__(self, icoord, dcoord):
        self.icoord = icoord
        self.dcoord = dcoord
        self.create_tree()

    def create_tree(self):
        self.branches = {}
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


def _get_remapped_coords(
    Z, orientation, groups, fig, inner_gs, gap_pixel, dendrogram_colors="colorful"
):
    """
    Remap dendrogram coordinates so leaves align with split heatmap cells.
    """
    n_leaves = Z.shape[0] + 1

    if dendrogram_colors == "colorful":
        dendro = hierarchy.dendrogram(Z, no_plot=True)
    else:
        dendro = hierarchy.dendrogram(
            Z, no_plot=True, link_color_func=lambda k: dendrogram_colors
        )

    icoord = np.array(dendro["icoord"]) / 10.0
    dcoord = np.array(dendro["dcoord"])
    color_list = dendro.get("color_list", ["black"] * len(Z))

    tree = DenTree(icoord, dcoord)

    leaves = dendro["leaves"]
    sizes = []
    for leaf_idx in leaves:
        for g_idx, grp in enumerate(groups):
            if leaf_idx in grp:
                sizes.append(1)
                break

    icoord_max = len(leaves)
    ratio = 1.0
    x_gap = 0.0

    if gap_pixel is not None and gap_pixel > 0:
        if orientation == "top":
            ax_temp = fig.add_subplot(inner_gs[0, 0])
            ax_pos = ax_temp.get_window_extent()
            real_width = ax_pos.width - (len(groups) - 1) * gap_pixel
            ratio = real_width / ax_pos.width
            x_gap = (gap_pixel / ax_pos.width) * icoord_max
            ax_temp.remove()
        else:
            ax_temp = fig.add_subplot(inner_gs[0, 0])
            ax_pos = ax_temp.get_window_extent()
            real_height = ax_pos.height - (len(groups) - 1) * gap_pixel
            ratio = real_height / ax_pos.height
            x_gap = (gap_pixel / ax_pos.height) * icoord_max
            ax_temp.remove()

    xcoord_mapping = {}
    cum_sizes = np.cumsum(sizes)

    for x_leaf in tree.leaves:
        idx = int(x_leaf[0])
        frac = x_leaf[0] % 1
        new_x = frac * sizes[idx] * ratio
        if idx > 0:
            new_x += cum_sizes[idx - 1] * ratio + x_gap * idx
        xcoord_mapping[x_leaf[0]] = new_x

    for key in tree.branches:
        branch = tree.branches[key]
        if not branch.left_is_parent:
            x1 = xcoord_mapping[branch.left_pos[0]]
            y1 = branch.left_pos[1]
            branch.left_pos = (x1, y1)
        if not branch.right_is_parent:
            x2 = xcoord_mapping[branch.right_pos[0]]
            y2 = branch.right_pos[1]
            branch.right_pos = (x2, y2)

        y_root = branch.root_pos[1]
        x_root = (branch.left_pos[0] + branch.right_pos[0]) / 2
        branch.root_pos = (x_root, y_root)

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

    remapped_icoord = np.array([[xcoord_mapping[i] for i in a] for a in icoord])

    return remapped_icoord, dcoord, dendro, color_list


def draw_dendrogram(
    fig,
    Z,
    orientation,
    groups,
    gs_position,
    n_splits,
    dendrogram_linewidth,
    dendrogram_colors,
    gap_pixel=None,
    inner_gs=None,
):
    """Draw dendrogram with optional split support."""
    from matplotlib import gridspec

    if orientation == "left":
        dendro_gs = gridspec.GridSpecFromSubplotSpec(
            1,
            1,
            subplot_spec=gs_position,
        )
    else:
        dendro_gs = gridspec.GridSpecFromSubplotSpec(
            1,
            1,
            subplot_spec=gs_position,
        )

    ax_d = fig.add_subplot(dendro_gs[0, 0])

    if n_splits > 1:
        remapped_icoord, dcoord, dendro, color_list = _get_remapped_coords(
            Z, orientation, groups, fig, inner_gs, gap_pixel, dendrogram_colors
        )

        if orientation == "left":
            for i in range(len(Z)):
                x = dcoord[i]
                y = remapped_icoord[i]
                color = color_list[i]
                ax_d.plot(x, y, color=color, linewidth=dendrogram_linewidth)

            all_y = remapped_icoord.flatten()
            max_dist = np.max(dcoord) if len(dcoord) > 0 else 1
            ax_d.set_xlim(max_dist, 0)
            ax_d.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
        else:
            for i in range(len(Z)):
                x = remapped_icoord[i]
                y = dcoord[i]
                color = color_list[i]
                ax_d.plot(x, y, color=color, linewidth=dendrogram_linewidth)

            all_x = remapped_icoord.flatten()
            max_dist = np.max(dcoord) if len(dcoord) > 0 else 1
            ax_d.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
            ax_d.set_ylim(0, max_dist * 1.05)
    else:
        if dendrogram_colors == "colorful":
            hierarchy.dendrogram(Z, ax=ax_d, orientation=orientation, no_labels=True)
        else:
            hierarchy.dendrogram(
                Z,
                ax=ax_d,
                orientation=orientation,
                no_labels=True,
                link_color_func=lambda k: dendrogram_colors,
            )

        for line in ax_d.get_lines():
            line.set_linewidth(dendrogram_linewidth)
        for coll in ax_d.collections:
            if hasattr(coll, "set_linewidths"):
                coll.set_linewidths([dendrogram_linewidth])

        if orientation == "left":
            ax_d.invert_yaxis()

    _despine(ax_d)
    return ax_d
