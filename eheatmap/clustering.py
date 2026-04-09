"""
eheatmap - Clustering module
"""

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from typing import Optional, Tuple, List, Union


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


def compute_row_clusters(df, cluster_rows, clustering_method, clustering_distance_rows):
    """Compute row clustering."""
    if isinstance(cluster_rows, bool) and cluster_rows:
        row_order, row_Z = _cluster(
            df.values, clustering_method, clustering_distance_rows
        )
    elif hasattr(cluster_rows, "linkage"):
        row_Z = cluster_rows.linkage
        row_order = hierarchy.leaves_list(row_Z)
    else:
        row_order = list(range(df.shape[0]))
        row_Z = None
    return row_order, row_Z


def compute_col_clusters(df, cluster_cols, clustering_method, clustering_distance_cols):
    """Compute column clustering."""
    if isinstance(cluster_cols, bool) and cluster_cols:
        col_order, col_Z = _cluster(
            df.values.T, clustering_method, clustering_distance_cols
        )
    elif hasattr(cluster_cols, "linkage"):
        col_Z = cluster_cols.linkage
        col_order = hierarchy.leaves_list(col_Z)
    else:
        col_order = list(range(df.shape[1]))
        col_Z = None
    return col_order, col_Z


def compute_row_groups(df, cutree_rows, gaps_row, row_Z):
    """Compute row groups/splits."""
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
    return row_groups


def compute_col_groups(df, cutree_cols, gaps_col, col_Z):
    """Compute column groups/splits."""
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
    return col_groups


def compute_cum_indices(groups):
    """Compute cumulative indices for groups."""
    cum = [0]
    for g in groups:
        cum.append(cum[-1] + len(g))
    return cum
