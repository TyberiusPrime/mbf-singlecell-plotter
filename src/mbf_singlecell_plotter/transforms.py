"""Layer 2: plot-data transforms. Pure functions — DataFrame in, DataFrame out."""

import collections

import numpy as np
import pandas as pd

from .util import map_to_integers, unmap

_EMBEDDING_COLOR_DEFAULTS = ("#FF4444", "#4444FF", "#FFCC00", "#44BB44")


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


def _corner_to_bounds(corner, ref_data):
    """Return ``(xl, xr, yb, yt)`` for a region corner in reference embedding space.

    *corner* is either:

    - a **string** grid label (e.g. ``"A1"``) — returns the full cell edges using
      the reference embedding's grid settings.
    - a **2-tuple of floats** ``(x, y)`` — treated as an exact data point
      (``xl == xr``, ``yb == yt``).
    """
    if isinstance(corner, str):
        from .data import _parse_grid_label

        gs = ref_data._grid_size
        glv = ref_data._grid_letters_on_vertical
        col, row = _parse_grid_label(corner, gs, glv)
        x_min_d, x_max_d, y_min_d, y_max_d = ref_data.full_bounds()
        cell_w = (x_max_d - x_min_d) / gs
        cell_h = (y_max_d - y_min_d) / gs
        return (
            x_min_d + col * cell_w,         # xl: left edge of this column
            x_min_d + (col + 1) * cell_w,   # xr: right edge of this column
            y_max_d - (row + 1) * cell_h,   # yb: bottom edge of this row
            y_max_d - row * cell_h,          # yt: top edge of this row
        )
    else:
        x, y = float(corner[0]), float(corner[1])
        return (x, x, y, y)


def prepare_embedding_color_df(
    current_data,
    reference_data,
    corner_colors=_EMBEDDING_COLOR_DEFAULTS,
    region=None,
    outside_color: str = "#C0C0C0",
) -> pd.DataFrame:
    """Assign 2D gradient colors to cells based on their position in reference_data.

    Each cell is colored by bilinear interpolation between four corner colors at
    its normalized (x, y) position in the reference embedding.  The returned
    DataFrame carries x, y coordinates from current_data and a ``color`` column
    of hex strings ready for ``scale_color_identity()``.

    Args:
        current_data:   EmbeddingData supplying x, y plot coordinates.
        reference_data: EmbeddingData whose coordinates drive the color mapping.
        corner_colors:  4-tuple ``(top_left, top_right, bottom_left, bottom_right)``.
        region:         Optional 2-tuple ``(corner1, corner2)`` that restricts which
                        cells receive the gradient.  Each corner is either a grid
                        label string (e.g. ``"A1"``) or an ``(x, y)`` float tuple
                        in reference-embedding data coordinates.  Cells outside the
                        resulting bounding box receive *outside_color* instead.
        outside_color:  Hex color for cells outside *region* (default ``"#C0C0C0"``).

    Returns:
        DataFrame with columns: x, y, color (hex string).
    """
    import matplotlib.colors as mcolors

    current_coords = current_data.coordinates()
    ref_coords = reference_data.coordinates().loc[current_coords.index]

    rx, ry = ref_coords["x"], ref_coords["y"]

    # Determine the normalisation bounds — use region extents when given so the
    # full colour spectrum maps to the region, not the whole embedding.
    if region is not None:
        c1 = _corner_to_bounds(region[0], reference_data)
        c2 = _corner_to_bounds(region[1], reference_data)
        x_min = min(c1[0], c1[1], c2[0], c2[1])
        x_max = max(c1[0], c1[1], c2[0], c2[1])
        y_min = min(c1[2], c1[3], c2[2], c2[3])
        y_max = max(c1[2], c1[3], c2[2], c2[3])
        in_region = (
            (rx >= x_min) & (rx <= x_max)
            & (ry >= y_min) & (ry <= y_max)
        )
    else:
        x_min, x_max = rx.min(), rx.max()
        y_min, y_max = ry.min(), ry.max()
        in_region = None

    t = (rx - x_min) / (x_max - x_min + 1e-12)  # [0,1] left → right
    s = (ry - y_min) / (y_max - y_min + 1e-12)  # [0,1] bottom → top

    tl = np.array(mcolors.to_rgb(corner_colors[0]))
    tr = np.array(mcolors.to_rgb(corner_colors[1]))
    bl = np.array(mcolors.to_rgb(corner_colors[2]))
    br = np.array(mcolors.to_rgb(corner_colors[3]))

    t_arr = t.values[:, None]
    s_arr = s.values[:, None]
    rgb = (
        (1 - t_arr) * s_arr * tl
        + t_arr * s_arr * tr
        + (1 - t_arr) * (1 - s_arr) * bl
        + t_arr * (1 - s_arr) * br
    )
    rgb = np.clip(rgb, 0, 1)

    hex_colors = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in rgb
    ]

    if in_region is not None:
        hex_colors = [
            hc if flag else outside_color
            for hc, flag in zip(hex_colors, in_region)
        ]

    df = current_coords.copy()
    df["color"] = hex_colors
    return df
