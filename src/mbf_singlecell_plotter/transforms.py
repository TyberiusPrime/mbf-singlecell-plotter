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


def _cross2d(a, b):
    """2D cross product ax*by - ay*bx. Broadcasts over leading batch dims."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inverse_bilinear(pts, p00, p10, p01, p11):
    """Inverse bilinear: compute (lr, bt) ∈ [0,1]² for each point in *pts*.

    Args:
        pts:  (N, 2) ndarray of query points.
        p00:  (2,) bottom-left  corner  (lr=0, bt=0).
        p10:  (2,) bottom-right corner  (lr=1, bt=0).
        p01:  (2,) top-left     corner  (lr=0, bt=1).
        p11:  (2,) top-right    corner  (lr=1, bt=1).

    Returns:
        lr, bt — (N,) arrays.  Points inside the quad have lr, bt ∈ [0, 1].
    """
    E = p10 - p00        # (2,)
    F = p01 - p00        # (2,)
    G = p00 - p10 - p01 + p11   # (2,)  zero for rectangles
    H = pts - p00        # (N, 2)

    EcG = _cross2d(E, G)   # scalar
    EcF = _cross2d(E, F)   # scalar
    HcG = _cross2d(H, G)   # (N,)
    HcF = _cross2d(H, F)   # (N,)

    a = -EcG              # scalar
    b = HcG - EcF         # (N,)
    c = HcF               # (N,)

    def _bt_from_lr(lr):
        dx = F[0] + lr * G[0]
        dy = F[1] + lr * G[1]
        bt_x = np.where(dx != 0, (H[:, 0] - lr * E[0]) / np.where(dx != 0, dx, 1.0), 0.0)
        bt_y = np.where(dy != 0, (H[:, 1] - lr * E[1]) / np.where(dy != 0, dy, 1.0), 0.0)
        return np.where(np.abs(dx) >= np.abs(dy), bt_x, bt_y)

    def _penalty(s, t):
        return (np.maximum(0, np.maximum(-s, s - 1))
                + np.maximum(0, np.maximum(-t, t - 1)))

    if abs(a) < 1e-10:
        # Degenerate / rectangular — linear equation
        lr = np.where(np.abs(b) > 1e-10, -c / b, 0.0)
    else:
        disc = np.maximum(b ** 2 - 4 * a * c, 0.0)
        sd = np.sqrt(disc)
        lr0 = (-b + sd) / (2 * a)
        lr1 = (-b - sd) / (2 * a)
        bt0 = _bt_from_lr(lr0)
        bt1 = _bt_from_lr(lr1)
        lr = np.where(_penalty(lr0, bt0) <= _penalty(lr1, bt1), lr0, lr1)

    bt = _bt_from_lr(lr)
    return lr, bt


def prepare_embedding_color_df(
    current_data,
    reference_data,
    corner_colors=_EMBEDDING_COLOR_DEFAULTS,
    region=None,
    outside_color: str = "#C0C0C0",
) -> pd.DataFrame:
    """Assign 2D gradient colors to cells based on their position in reference_data.

    Each cell is colored by bilinear interpolation between four corner colors at
    its (lr, bt) position in the reference embedding.  The returned DataFrame
    carries x, y coordinates from current_data and a ``color`` column of hex
    strings ready for ``scale_color_identity()``.

    Args:
        current_data:   EmbeddingData supplying x, y plot coordinates.
        reference_data: EmbeddingData whose coordinates drive the color mapping.
        corner_colors:  4-tuple ``(top_left, top_right, bottom_left, bottom_right)``.
        region:         Restricts which cells receive the gradient.  Two forms:

                        * **2-tuple** ``((x0, y0), (x1, y1))`` — axis-aligned
                          bounding box (any two opposite corners).
                        * **4-tuple** ``(top_left, top_right, bottom_left,
                          bottom_right)`` — arbitrary (possibly non-rectangular)
                          quadrilateral.

                        In both cases coordinates are ``(x, y)`` float pairs in
                        reference-embedding space.  The full colour spectrum maps
                        to the region interior; cells outside receive *outside_color*.
        outside_color:  Hex color for cells outside *region* (default ``"#C0C0C0"``).

    Returns:
        DataFrame with columns: x, y, color (hex string).
    """
    import matplotlib.colors as mcolors

    current_coords = current_data.coordinates()
    ref_coords = reference_data.coordinates().loc[current_coords.index]

    rx, ry = ref_coords["x"], ref_coords["y"]

    if region is not None:
        if len(region) == 2:
            # 2-corner bounding box shorthand: (top_left, bottom_right)
            (x0, y0), (x1, y1) = region
            xlo, xhi = min(x0, x1), max(x0, x1)
            ylo, yhi = min(y0, y1), max(y0, y1)
            region = (
                (xlo, yhi),  # top_left
                (xhi, yhi),  # top_right
                (xlo, ylo),  # bottom_left
                (xhi, ylo),  # bottom_right
            )
        # 4-corner quad: (top_left, top_right, bottom_left, bottom_right)
        # Bilinear basis: p00=bottom_left(lr=0,bt=0), p10=bottom_right(lr=1,bt=0),
        #                 p01=top_left(lr=0,bt=1),     p11=top_right(lr=1,bt=1)
        tl, tr, bl, br = [np.array(c, dtype=float) for c in region]
        pts = np.column_stack([rx.values, ry.values])
        lr_arr, bt_arr = _inverse_bilinear(pts, p00=bl, p10=br, p01=tl, p11=tr)
        in_region = (lr_arr >= 0) & (lr_arr <= 1) & (bt_arr >= 0) & (bt_arr <= 1)
        t = pd.Series(lr_arr, index=rx.index)   # left → right
        s = pd.Series(bt_arr, index=rx.index)   # bottom → top
    else:
        x_min, x_max = rx.min(), rx.max()
        y_min, y_max = ry.min(), ry.max()
        in_region = None
        t = (rx - x_min) / (x_max - x_min + 1e-12)   # [0,1] left → right
        s = (ry - y_min) / (y_max - y_min + 1e-12)   # [0,1] bottom → top

    tl_c = np.array(mcolors.to_rgb(corner_colors[0]))
    tr_c = np.array(mcolors.to_rgb(corner_colors[1]))
    bl_c = np.array(mcolors.to_rgb(corner_colors[2]))
    br_c = np.array(mcolors.to_rgb(corner_colors[3]))

    t_arr = t.values[:, None]
    s_arr = s.values[:, None]
    rgb = (
        (1 - t_arr) * s_arr * tl_c
        + t_arr * s_arr * tr_c
        + (1 - t_arr) * (1 - s_arr) * bl_c
        + t_arr * (1 - s_arr) * br_c
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
