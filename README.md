# eheatmap
Effortless heatmap generation in Python: Focus on your data, not the plotting code. It provides extensive control over clustering, annotations, color mapping, and layout, making it ideal for scientific data visualization and publication-ready figures.

The development of this tool was inspired by PyComplexHeatmap and the R package pheatmap. We would like to express our gratitude to all the developers and maintainers of these two projects.

## ✨ Features

* **Hierarchical Clustering**: Support for row and column clustering with various linkage methods (e.g., Ward, Complete) and distance metrics (Euclidean, Correlation).
* **Split & Gap Customization**: Split heatmaps into clusters with customizable gaps and split borders.
* **Comprehensive Annotations**: Add discrete or continuous annotation bars on all four sides (Top, Bottom, Left, Right).
* **Dendrogram Customization**: Adjustable tree height, line width, and color schemes.
* **Data Preprocessing**: Built-in row/column scaling and K-means clustering integration.

![](./figures/Annotation_Heatmap.png)

## Install
```
# Clone or navigate to the project directory
git clone https://github.com/Bitpulses/eheatmap.git

cd overlapviz

# Install
pip install .
```
## Basic Usage
```
import matplotlib.pyplot as plt
from eheatmap import eheatmap

fig = eheatmap(
    df,
    scale="row",                    # Row Z-score normalization
    cutree_rows=3,                  # Split rows into 3 clusters
    cutree_cols=3,                  # Split columns into 3 groups
    annotation_row=row_anno,        # Row annotations
    annotation_col=col_anno,        # Column annotations
    annotation_colors=annotation_colors,  # Custom color mapping
    color="viridis",                # CNS-recommended colormap
    center=0,                       # Color center point
    border_color="grey80",          # Cell border color
    fontsize=9,
    dendrogram_linewidth=1,
    treeheight_col=20,
    treeheight_row=20,
    dendrogram_colors="colorful",
    split_border_color="black",     # Split line color
    split_border_width=1.0,
    legend=True,
    annotation_legend=True,
    main="Annotation Heatmap"
)
```
