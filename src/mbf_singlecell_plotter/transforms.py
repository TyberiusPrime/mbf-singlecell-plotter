"""Layer 2: plot-data transforms. Pure functions — DataFrame in, DataFrame out."""

import collections

import numpy as np
import pandas as pd

from .util import map_to_integers, unmap


def prepare_scatter_df(
    data,
    gene: str,
    clip_quantile: float = 0.95,
    zero_value: float = 0.0,
) -> pd.DataFrame:
    """Prepare a DataFrame for scatter plot visualization.

    Returns df with columns: x, y, expression, is_zero (and is_clipped for numerical).
    """
    from natsort import natsorted

    coords = data.coordinates()
    expr, expr_name = data.get_column(gene)
    is_numerical = (
        (expr.dtype != "object")
        and (expr.dtype != "category")
        and (expr.dtype != "bool")
    )
    if not is_numerical and expr.dtype != "category":
        expr = expr.astype("category")

    df = coords.copy()
    df["expression"] = expr

    if is_numerical:
        df["is_zero"] = df["expression"] == zero_value
        clip_val = df.loc[~df["is_zero"], "expression"].quantile(clip_quantile)
        df["is_clipped"] = df["expression"] > clip_val
    else:
        df["is_zero"] = False
        df["is_clipped"] = False

    return df


def prepare_density_df(data, bins: int = 200) -> pd.DataFrame:
    """2D histogram → long-form DataFrame with x, y, density, x_width, y_width columns."""
    coords = data.coordinates()
    H, xedges, yedges = np.histogram2d(coords["x"], coords["y"], bins=bins)
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    x_width = xedges[1] - xedges[0]
    y_width = yedges[1] - yedges[0]
    X, Y = np.meshgrid(x_centers, y_centers)
    df = pd.DataFrame(
        {
            "x": X.flatten(),
            "y": Y.flatten(),
            "density": H.T.flatten(),
            "x_width": x_width,
            "y_width": y_width,
        }
    )
    return df[df["density"] > 0].copy()


def compute_boundaries(
    data,
    cell_type_column: str,
    colors: list | None = None,
    resolution: int = 200,
    blur: float = 1.1,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Compute boundary scatter points for cell-type regions.

    Returns DataFrame with x, y, color (hex string) columns.
    Requires scikit-image.
    """
    import skimage
    import matplotlib.colors as mcolors
    import matplotlib
    from natsort import natsorted
    from .theme import DEFAULT_COLORS_BORDERS

    if colors is None:
        colors = DEFAULT_COLORS_BORDERS

    cmap = matplotlib.colors.ListedColormap(colors)
    coords = data.coordinates()
    cell_types, _ = data.get_column(cell_type_column)

    if cell_types.dtype == "category":
        cats = list(cell_types.cat.categories)
    else:
        cats = natsorted(cell_types.unique())

    img = np.zeros((resolution, resolution), dtype=np.uint8)
    color_img = np.zeros((resolution, resolution), dtype=object)
    problematic: dict = {}

    pdf = coords.copy()
    pdf["cell_type"] = cell_types

    for cat_no, cat in enumerate(cats):
        sdf = pdf[pdf.cell_type == cat]
        mapped_x = map_to_integers(
            sdf["x"], resolution, pdf["x"].min(), pdf["x"].max()
        )
        mapped_y = map_to_integers(
            sdf["y"], resolution, pdf["y"].min(), pdf["y"].max()
        )
        color = cmap(cat_no)
        for x, y in zip(mapped_x, mapped_y):
            img[x][y] = 255
            if (x, y) in problematic:
                problematic[x, y][color] += 1
            else:
                if color_img[x, y] == 0 or color_img[x, y] == color:
                    color_img[x, y] = color
                else:
                    problematic[x, y] = collections.Counter()
                    problematic[x, y][color] += 1

    for (x, y), counts in problematic.items():
        color_img[x, y] = counts.most_common(1)[0][0]

    flooded = skimage.segmentation.flood(img, (0, 0))
    flooded = skimage.filters.gaussian(flooded, blur)
    bounds = flooded < threshold
    bounds = skimage.segmentation.find_boundaries(bounds)

    def search_color(x, y, dist):
        for xi in range(max(0, x - dist), min(x + dist + 1, resolution - 1)):
            for yi in range(max(y - dist, 0), min(y + dist + 1, resolution - 1)):
                col = color_img[xi, yi]
                if col != 0:
                    return col
        return 0

    boundary_points: dict = collections.defaultdict(list)
    for x in range(resolution):
        for y in range(resolution):
            if bounds[x][y]:
                col = color_img[x, y]
                if col == 0:
                    rdist = 1
                    while rdist < 100 and col == 0:
                        col = search_color(x, y, rdist)
                        rdist += 1
                if col != 0:
                    boundary_points["x"].append(x)
                    boundary_points["y"].append(y)
                    boundary_points["color"].append(col)
                    for offset in [1]:
                        boundary_points["x"].append(x + offset)
                        boundary_points["y"].append(y)
                        boundary_points["color"].append(col)
                        boundary_points["x"].append(x - offset)
                        boundary_points["y"].append(y)
                        boundary_points["color"].append(col)
                        boundary_points["x"].append(x)
                        boundary_points["y"].append(y + offset)
                        boundary_points["color"].append(col)
                        boundary_points["x"].append(x)
                        boundary_points["y"].append(y - offset)
                        boundary_points["color"].append(col)
                        boundary_points["x"].append(x + offset)
                        boundary_points["y"].append(y + offset)
                        boundary_points["color"].append(col)
                        boundary_points["x"].append(x - offset)
                        boundary_points["y"].append(y - offset)
                        boundary_points["color"].append(col)
                else:
                    raise ValueError(
                        "Color was still 0 after looking at 100 cells in each direction"
                    )

    bdf = pd.DataFrame(
        {
            "x": unmap(pd.Series(boundary_points["x"]), pdf["x"], resolution),
            "y": unmap(pd.Series(boundary_points["y"]), pdf["y"], resolution),
            "color": boundary_points["color"],
        }
    )
    # Convert RGBA tuples to hex strings for plotnine
    bdf["color"] = bdf["color"].apply(
        lambda c: mcolors.to_hex(c) if not isinstance(c, str) else c
    )
    return bdf
