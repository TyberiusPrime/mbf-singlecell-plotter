"""Layer 2: plot-data transforms. Pure functions — DataFrame in, DataFrame out."""

import collections

import numpy as np
import pandas as pd

from .util import map_to_integers, unmap

_EMBEDDING_COLOR_DEFAULTS = ("#FF4444", "#4444FF", "#FFCC00", "#44BB44")


def prepare_density_df(
    data, bins: int = 200, facet=None, facet_row=None, facet_col=None
) -> pd.DataFrame:
    """2D histogram → long-form DataFrame with x, y, density, x_width, y_width columns.

    When *facet* is a :class:`pandas.Series` aligned to the embedding's cell
    index, a separate histogram is computed per facet group over shared global
    bin edges (so every panel shares the same coordinate frame and colour
    scale) and an ordered-Categorical ``facet`` column is added — ready for
    ``plotnine.facet_wrap("~facet")``.

    Alternatively, *facet_row* and *facet_col* (both Series) produce a 2-D grid
    of histograms with ``facet_row`` / ``facet_col`` Categorical columns, ready
    for ``plotnine.facet_grid(rows="facet_row", cols="facet_col")``.
    """
    coords = data.coordinates()
    # Shared global edges so every facet panel uses the same coordinate frame.
    x_edges = np.linspace(float(coords["x"].min()), float(coords["x"].max()), bins + 1)
    y_edges = np.linspace(float(coords["y"].min()), float(coords["y"].max()), bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_width = x_edges[1] - x_edges[0]
    y_width = y_edges[1] - y_edges[0]
    X, Y = np.meshgrid(x_centers, y_centers)

    def _hist(sub):
        H, _, _ = np.histogram2d(sub["x"], sub["y"], bins=[x_edges, y_edges])
        return H.T.flatten()

    if facet is None and facet_row is None and facet_col is None:
        df = pd.DataFrame(
            {
                "x": X.flatten(),
                "y": Y.flatten(),
                "density": _hist(coords),
                "x_width": x_width,
                "y_width": y_width,
            }
        )
        return df[df["density"] > 0].copy()

    # ── 1-D faceting ──────────────────────────────────────────────────────────
    if facet is not None:
        facet = facet.reindex(coords.index)
        cat_order = _facet_categories(facet)
        frames = []
        for value in cat_order:
            sub = coords[facet == value]
            if len(sub) == 0:
                continue
            fdf = pd.DataFrame(
                {
                    "x": X.flatten(),
                    "y": Y.flatten(),
                    "density": _hist(sub),
                    "x_width": x_width,
                    "y_width": y_width,
                    "facet": value,
                }
            )
            frames.append(fdf[fdf["density"] > 0])
        if not frames:
            return pd.DataFrame(
                columns=["x", "y", "density", "x_width", "y_width", "facet"]
            )
        df = pd.concat(frames, ignore_index=True)
        df["facet"] = pd.Categorical(df["facet"], categories=cat_order)
        return df

    # ── 2-D faceting (facet_grid) ─────────────────────────────────────────────

    facet_row = facet_row.reindex(coords.index) if facet_row is not None else None
    facet_col = facet_col.reindex(coords.index) if facet_col is not None else None
    row_cats = _facet_categories(facet_row) if facet_row is not None else [None]
    col_cats = _facet_categories(facet_col) if facet_col is not None else [None]

    frames = []
    for rv in row_cats:
        for cv in col_cats:
            mask = np.ones(len(coords), dtype=bool)
            if facet_row is not None:
                mask &= facet_row.values == rv
            if facet_col is not None:
                mask &= facet_col.values == cv
            sub = coords[mask]
            if len(sub) == 0:
                continue
            row = {
                "x": X.flatten(),
                "y": Y.flatten(),
                "density": _hist(sub),
                "x_width": x_width,
                "y_width": y_width,
            }
            if facet_row is not None:
                row["facet_row"] = rv
            if facet_col is not None:
                row["facet_col"] = cv
            fdf = pd.DataFrame(row)
            frames.append(fdf[fdf["density"] > 0])
    if not frames:
        cols = ["x", "y", "density", "x_width", "y_width"]
        if facet_row is not None:
            cols.append("facet_row")
        if facet_col is not None:
            cols.append("facet_col")
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    if facet_row is not None:
        df["facet_row"] = pd.Categorical(df["facet_row"], categories=row_cats)
    if facet_col is not None:
        df["facet_col"] = pd.Categorical(df["facet_col"], categories=col_cats)
    return df


def _facet_categories(series: pd.Series) -> list:
    """Return the ordered category list for a facet Series."""
    from natsort import natsorted

    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return list(natsorted(pd.unique(series.dropna())))


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
        mapped_x = map_to_integers(sdf["x"], resolution, pdf["x"].min(), pdf["x"].max())
        mapped_y = map_to_integers(sdf["y"], resolution, pdf["y"].min(), pdf["y"].max())
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


def compute_grid_moran(
    data,
    n_bins: int = 40,
    min_cells: int = 3,
    var_score_column: str | None = None,
) -> pd.DataFrame:
    """Compute Moran's I spatial autocorrelation for every gene over a binned UMAP grid.

    Each gene is also assigned a *top bin* — the bin with the highest weighted
    mean expression score ``mean_expr * log1p(cell_count)``, which balances
    expression level against bin reliability.

    Args:
        data:             EmbeddingData instance.
        n_bins:           Number of equal-width bins per axis (default 40).
        min_cells:        Minimum cells required for a bin to be included (default 3).
        var_score_column: If given, use ``adata.var[var_score_column]`` as the gene
                          score instead of computing Moran's I on the fly.  The
                          column must be numeric; higher values mean more spatially
                          informative.  Top-bin spatial assignment is still derived
                          from the embedding.

    Returns:
        DataFrame with columns:

        * ``gene``          — gene name
        * ``moran_i``       — score (Moran's I or the var column values)
        * ``top_bin``       — ``(xi, yi)`` integer bin-index tuple of the top bin
        * ``top_bin_score`` — weighted score of the top bin
        * ``top_bin_x``     — x coordinate (embedding space) of the top bin centre
        * ``top_bin_y``     — y coordinate (embedding space) of the top bin centre
    """
    from scipy import sparse as sp

    coords = data.coordinates()
    x = coords["x"].values
    y = coords["y"].values

    x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    y_edges = np.linspace(y.min(), y.max(), n_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # 0-indexed bin assignments; clip to [0, n_bins-1]
    x_bin = np.clip(np.searchsorted(x_edges[1:-1], x), 0, n_bins - 1)
    y_bin = np.clip(np.searchsorted(y_edges[1:-1], y), 0, n_bins - 1)

    # ── aggregate per occupied bin ────────────────────────────────────────────
    ad = data.ad
    # Bulk-load X once as CSR (row-major) so per-bin row slicing is cheap.
    # For H5adFacade this is a single ``export matrix_csr`` call; for AnnData it
    # just forwards/converts the in-memory X.
    X = data.get_X_csr()  # (n_cells, n_genes) sparse CSR

    # coords may be a filtered subset of the cells; map its (obs_names) index
    # back to positional rows in the full X matrix so the mean is taken over
    # the right gene-expression rows.
    ci = ad.obs_names.get_indexer(coords.index)
    cell_df = pd.DataFrame({"xi": x_bin, "yi": y_bin, "ci": ci})
    groups = cell_df.groupby(["xi", "yi"])

    bins_xy, counts, grid_expr_rows = [], [], []
    for (xi, yi), grp in groups:
        if len(grp) < min_cells:
            continue
        bins_xy.append((int(xi), int(yi)))
        counts.append(len(grp))
        block = X[grp["ci"].values]
        row_mean = np.asarray(block.mean(axis=0)).ravel()
        grid_expr_rows.append(row_mean)

    if len(bins_xy) < 4:
        raise ValueError(
            f"Only {len(bins_xy)} occupied bins with min_cells={min_cells}. "
            "Reduce min_cells or increase n_bins."
        )

    grid_expr = np.vstack(grid_expr_rows)  # (B, G)
    counts = np.array(counts)
    B = len(bins_xy)

    if var_score_column is not None:
        # Use pre-computed scores from adata.var — skip Moran's I computation
        moran_i = np.asarray(ad.var[var_score_column].values, dtype=float)
    else:
        bin_index = {b: i for i, b in enumerate(bins_xy)}

        # ── queen-contiguity weights among occupied bins ──────────────────────
        rows_w, cols_w = [], []
        for i, (xi, yi) in enumerate(bins_xy):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    j = bin_index.get((xi + dx, yi + dy))
                    if j is not None:
                        rows_w.append(i)
                        cols_w.append(j)

        W = sp.csr_matrix((np.ones(len(rows_w)), (rows_w, cols_w)), shape=(B, B))
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        W = W.multiply(1.0 / row_sums[:, None])
        S0 = float(W.sum())

        # ── Moran's I, all genes simultaneously ──────────────────────────────
        Z = grid_expr - grid_expr.mean(axis=0)  # (B, G)
        WZ = W @ Z  # (B, G)
        numerator = (Z * WZ).sum(axis=0)
        denominator = (Z**2).sum(axis=0)
        moran_i = (B / S0) * numerator / np.maximum(denominator, 1e-12)

    # ── top bin per gene (weighted by log1p cell count) ───────────────────────
    score = grid_expr * np.log1p(counts)[:, None]  # (B, G)
    top_idx = score.argmax(axis=0)  # (G,)

    gene_names = list(ad.var_names)
    G = len(gene_names)
    top_bins = [bins_xy[i] for i in top_idx]
    top_bin_xs = [x_centers[b[0]] for b in top_bins]
    top_bin_ys = [y_centers[b[1]] for b in top_bins]

    return pd.DataFrame(
        {
            "gene": gene_names,
            "moran_i": moran_i,
            "top_bin": top_bins,
            "top_bin_score": score[top_idx, np.arange(G)],
            "top_bin_x": top_bin_xs,
            "top_bin_y": top_bin_ys,
        }
    )


def marker_genes_by_region(
    gene_df: pd.DataFrame,
    k: int = 20,
    min_moran: float = 0.2,
) -> dict:
    """Group genes by their top bin and return the top-k by Moran's I per region.

    Args:
        gene_df:   Output of :func:`compute_grid_moran`.
        k:         Maximum number of marker genes per region.
        min_moran: Minimum Moran's I threshold (genes below this are excluded).

    Returns:
        Dict mapping ``(xi, yi)`` bin-index tuple → list of gene names (descending
        Moran's I order, up to *k* entries).
    """
    filtered = gene_df[gene_df["moran_i"] >= min_moran]
    result = {}
    for bin_key, grp in filtered.groupby("top_bin"):
        result[bin_key] = grp.nlargest(k, "moran_i")["gene"].tolist()
    return result


def compute_cluster_markers(
    data,
    column: str,
    *,
    layer: str | None = None,
    min_cells_per_group: int = 10,
) -> pd.DataFrame:
    """Rank marker genes per category of a categorical column via pseudobulk.

    Each category (cluster) of *column* is aggregated into a **pseudobulk mean**
    expression vector over its cells (the same ``X[rows].mean(axis=0)`` primitive
    used by :func:`compute_grid_moran`).  Every gene then gets, per category:

    * ``delta``     — ``mean_in_cluster - mean_in_rest``, where the baseline is
                      the **category-weighted** mean of the *other* categories'
                      pseudobulk means (so a few large clusters cannot dominate
                      the baseline).  On log-normalized expression this is a log
                      fold-change; on scaled/z-scored data it is a z-score
                      difference (effect size).  Robust to negative values.
    * ``mean_expr`` — the category's pseudobulk mean expression.
    * ``score``     — ``max(delta, 0) * log1p(max(mean_expr, 0))``: the effect
                      gated by expression level so lowly-expressed (or, on scaled
                      data, sub-average) genes cannot top the ranking.

    Args:
        data:                :class:`~mbf_singlecell_plotter.data.EmbeddingData`.
        column:              Categorical (or bool) obs column with cluster labels.
        layer:               Expression layer to aggregate.  ``None`` (default)
                             uses the source's configured layer; pass a layer key
                             (or ``"X"``) to compute markers on raw / log-normalized
                             counts instead of, e.g., a scaled matrix.
        min_cells_per_group: Categories with fewer cells are skipped (default 10).

    Returns:
        Long-form DataFrame with columns
        ``category, gene, delta, mean_expr, score`` (one row per category × gene).

    Raises:
        ValueError: if *column* is numeric (not categorical), or fewer than two
            categories clear *min_cells_per_group*.
    """
    col_data, _col_name = data.get_column(column)

    is_cat = isinstance(col_data.dtype, pd.CategoricalDtype)
    is_bool = pd.api.types.is_bool_dtype(col_data)
    if not (is_cat or is_bool) and pd.api.types.is_numeric_dtype(col_data):
        raise ValueError(
            f"Column '{column}' contains numeric data. "
            "Cluster markers require a categorical column."
        )
    if is_bool:
        # Coerce bool → str so True/False become plain category labels
        # (mirrors ScatterPlotter.plot / grid_local_histogram).
        col_data = col_data.astype(str)

    ad = data.ad
    # Bulk-load X once as CSR (row-major) so per-category row slicing is cheap.
    # A layer override reads that layer instead of the configured one.
    expr_data = data if layer is None else data._replace(layer=layer)
    X = expr_data.get_X_csr()  # (n_cells, n_genes) sparse CSR

    # col_data may be a filtered subset; map its obs_names back to positional
    # rows in the full X matrix so means are taken over the right cells.
    cats = col_data.dropna()
    ci = ad.obs_names.get_indexer(cats.index)
    valid = ci >= 0
    cell_df = pd.DataFrame({"cat": np.asarray(cats.values)[valid], "ci": ci[valid]})

    means, labels, counts = [], [], []
    for cat, grp in cell_df.groupby("cat", observed=True):
        if len(grp) < min_cells_per_group:
            continue
        block = X[grp["ci"].values]
        means.append(np.asarray(block.mean(axis=0)).ravel())
        labels.append(cat)
        counts.append(len(grp))

    if len(labels) < 2:
        raise ValueError(
            f"Need at least 2 categories with >= {min_cells_per_group} cells "
            f"in '{column}', found {len(labels)}. Reduce min_cells_per_group."
        )

    mean_by_cat = np.vstack(means)  # (C, G)
    n_cats, n_genes = mean_by_cat.shape

    # Category-weighted "rest": mean of the *other* categories' pseudobulk means.
    total = mean_by_cat.sum(axis=0)  # (G,)
    mean_rest = (total[None, :] - mean_by_cat) / (n_cats - 1)  # (C, G)

    # Difference of means — a log fold-change on log-normalized data, a z-score
    # delta on scaled data.  Robust to negatives (no log of the raw values).
    delta = mean_by_cat - mean_rest
    # Gate by (non-negative) expression so lowly-expressed noise can't top the list.
    score = np.maximum(delta, 0.0) * np.log1p(np.clip(mean_by_cat, 0.0, None))

    gene_names = np.asarray(ad.var_names)
    return pd.DataFrame(
        {
            "category": np.repeat(np.asarray(labels, dtype=object), n_genes),
            "gene": np.tile(gene_names, n_cats),
            "delta": delta.ravel(),
            "mean_expr": mean_by_cat.ravel(),
            "score": score.ravel(),
        }
    )


def marker_genes_by_category(
    marker_df: pd.DataFrame,
    k: int = 20,
    min_score: float = 0.0,
) -> dict:
    """Group genes by category and return the top-*k* markers per category.

    Args:
        marker_df: Output of :func:`compute_cluster_markers`.
        k:         Maximum number of marker genes per category.
        min_score: Minimum combined score threshold (genes with ``score`` at or
                   below this are excluded; default 0.0 keeps genes up-regulated
                   versus the rest).

    Returns:
        Dict mapping category label → list of ``{"gene", "delta", "score"}``
        dicts, in descending ``score`` order (up to *k* entries).
    """
    filtered = marker_df[marker_df["score"] > min_score]
    result = {}
    for cat, grp in filtered.groupby("category", observed=True):
        top = grp.nlargest(k, "score")
        result[cat] = [
            {
                "gene": row.gene,
                "delta": float(row.delta),
                "score": float(row.score),
            }
            for row in top.itertuples(index=False)
        ]
    return result


def _corner_to_bounds(corner, ref_data):
    """Return ``(xl, xr, yb, yt)`` for a region corner in reference embedding space.

    *corner* is either a grid label string (e.g. ``"A1"``) or an ``(x, y)`` float tuple.
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
            x_min_d + col * cell_w,
            x_min_d + (col + 1) * cell_w,
            y_max_d - (row + 1) * cell_h,
            y_max_d - row * cell_h,
        )
    else:
        x, y = float(corner[0]), float(corner[1])
        return (x, x, y, y)


def _region_to_bbox(region, reference_data):
    """Resolve a 2-corner region ``(corner1, corner2)`` into ``(xlo, xhi, ylo, yhi)``.

    Each corner is either a grid-label string (e.g. ``"A1"``) or an ``(x, y)``
    float pair. Shared by ``gradient_region`` and ``cell_filter`` handling.
    """
    corner1, corner2 = region
    if isinstance(corner1, str) or isinstance(corner2, str):
        b1 = _corner_to_bounds(corner1, reference_data)
        b2 = _corner_to_bounds(corner2, reference_data)
        xlo = min(b1[0], b1[1], b2[0], b2[1])
        xhi = max(b1[0], b1[1], b2[0], b2[1])
        ylo = min(b1[2], b1[3], b2[2], b2[3])
        yhi = max(b1[2], b1[3], b2[2], b2[3])
    else:
        (x0, y0), (x1, y1) = corner1, corner2
        xlo, xhi = min(x0, x1), max(x0, x1)
        ylo, yhi = min(y0, y1), max(y0, y1)
    return xlo, xhi, ylo, yhi


def _bbox_membership(rx, ry, bbox):
    """Boolean membership mask for points ``(rx, ry)`` in axis-aligned ``bbox``."""
    xlo, xhi, ylo, yhi = bbox
    return (xlo <= rx.values) & (rx.values <= xhi) & (ylo <= ry.values) & (ry.values <= yhi)


def _is_region_corner(c):
    """True if *c* is a single region corner (grid label or (x, y) pair), not a region itself."""
    if isinstance(c, str):
        return True
    return (
        isinstance(c, (tuple, list))
        and len(c) == 2
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in c)
    )


def _normalize_regions(region_or_regions):
    """Normalize a single ``(corner1, corner2)`` region or a list of such regions.

    Returns a list of 2-corner regions, so callers can always iterate and OR
    together the resulting membership masks.
    """
    if (
        len(region_or_regions) == 2
        and _is_region_corner(region_or_regions[0])
        and _is_region_corner(region_or_regions[1])
    ):
        return [region_or_regions]
    return list(region_or_regions)


def _regions_to_mask_fn(spec):
    """Normalize a filter *spec* into a callable ``fn(data) -> pd.Series[bool]``.

    *spec* is either an existing callable (returned unchanged) or a region /
    list of regions (each a ``(corner1, corner2)`` box of grid labels or
    ``(x, y)`` pairs).  For region specs the returned closure ORs
    :func:`_bbox_membership` over ``data.coordinates()`` for every region and
    returns a boolean Series indexed by obs — the shape expected by
    :meth:`EmbeddingData._filter_mask`.  Shared by ``set_filter`` /
    ``hard_filter`` (data.py) and ``cell_filter`` handling below.
    """
    if callable(spec):
        return spec

    regions = _normalize_regions(spec)

    def _fn(data):
        coords = data.coordinates()
        rx, ry = coords["x"], coords["y"]
        mask = np.zeros(len(coords), dtype=bool)
        for region in regions:
            mask |= _bbox_membership(rx, ry, _region_to_bbox(region, data))
        return pd.Series(mask, index=coords.index)

    return _fn


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
    E = p10 - p00  # (2,)
    F = p01 - p00  # (2,)
    G = p00 - p10 - p01 + p11  # (2,)  zero for rectangles
    H = pts - p00  # (N, 2)

    EcG = _cross2d(E, G)  # scalar
    EcF = _cross2d(E, F)  # scalar
    HcG = _cross2d(H, G)  # (N,)
    HcF = _cross2d(H, F)  # (N,)

    a = -EcG  # scalar
    b = HcG - EcF  # (N,)
    c = HcF  # (N,)

    def _bt_from_lr(lr):
        dx = F[0] + lr * G[0]
        dy = F[1] + lr * G[1]
        bt_x = np.where(
            dx != 0, (H[:, 0] - lr * E[0]) / np.where(dx != 0, dx, 1.0), 0.0
        )
        bt_y = np.where(
            dy != 0, (H[:, 1] - lr * E[1]) / np.where(dy != 0, dy, 1.0), 0.0
        )
        return np.where(np.abs(dx) >= np.abs(dy), bt_x, bt_y)

    def _penalty(s, t):
        return np.maximum(0, np.maximum(-s, s - 1)) + np.maximum(
            0, np.maximum(-t, t - 1)
        )

    if abs(a) < 1e-10:
        # Degenerate / rectangular — linear equation
        lr = np.where(np.abs(b) > 1e-10, -c / b, 0.0)
    else:
        disc = np.maximum(b**2 - 4 * a * c, 0.0)
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
    gradient_region=None,
    cell_filter=None,
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
        gradient_region: Restricts which grid cells ar part of the gradient.  Two forms:

                        * **2-tuple** ``(corner1, corner2)`` — axis-aligned bounding
                          box.  Each corner is either a grid-label string
                          (e.g. ``"A1"``) or an ``(x, y)`` float pair.
                        * **4-tuple** ``(top_left, top_right, bottom_left,
                          bottom_right)`` — arbitrary (possibly non-rectangular)
                          quadrilateral; each element is an ``(x, y)`` float pair.

                        The full colour spectrum maps to the gradient_region interior;
                        cells outside receive *outside_color*.
        cell_filter: Optional, restricts which cells receive color. Others get
                        *outside_color*. Accepts the same spec grammar as
                        ``EmbeddingData.set_filter``: a single region ``(corner1,
                        corner2)`` (grid-label strings or ``(x, y)`` float pairs,
                        same as the 2-corner form of *gradient_region*), a list of
                        such regions (a cell is colored if it falls inside *any* of
                        them), or a callable ``fn(reference_data) -> bool mask``.
                        Regions/callables are resolved in the *reference* embedding.
        outside_color:  Hex color for cells outside *gradient_region*/*cell_filter* (default ``"#C0C0C0"``).

    Returns:
        DataFrame with columns: x, y, color (hex string).
    """
    import matplotlib.colors as mcolors

    current_coords = current_data.coordinates()
    ref_coords = reference_data.coordinates().loc[current_coords.index]

    rx, ry = ref_coords["x"], ref_coords["y"]

    if gradient_region is not None:
        if len(gradient_region) == 2:
            # 2-corner bounding box: grid strings or (x, y) float tuples
            xlo, xhi, ylo, yhi = _region_to_bbox(gradient_region, reference_data)
            gradient_region = (
                (xlo, yhi),  # top_left
                (xhi, yhi),  # top_right
                (xlo, ylo),  # bottom_left
                (xhi, ylo),  # bottom_right
            )
        elif len(gradient_region) != 4:
            raise ValueError("Invalid gradient_region")
        # 4-corner quad: (top_left, top_right, bottom_left, bottom_right)
        # Bilinear basis: p00=bottom_left(lr=0,bt=0), p10=bottom_right(lr=1,bt=0),
        #                 p01=top_left(lr=0,bt=1),     p11=top_right(lr=1,bt=1)
        # Accept any ordering: sort into tl/tr/bl/br automatically.
        # Top two = highest y; within each pair, left = lower x.
        pts4 = sorted(
            [np.array(c, dtype=float) for c in gradient_region], key=lambda p: -p[1]
        )
        top_two = sorted(pts4[:2], key=lambda p: p[0])
        bot_two = sorted(pts4[2:], key=lambda p: p[0])
        tl, tr = top_two[0], top_two[1]
        bl, br = bot_two[0], bot_two[1]
        pts = np.column_stack([rx.values, ry.values])
        lr_arr, bt_arr = _inverse_bilinear(pts, p00=bl, p10=br, p01=tl, p11=tr)
        in_gradient_region = (
            (lr_arr >= 0) & (lr_arr <= 1) & (bt_arr >= 0) & (bt_arr <= 1)
        )
        t = pd.Series(lr_arr, index=rx.index)  # left → right
        s = pd.Series(bt_arr, index=rx.index)  # bottom → top
    else:
        x_min, x_max = rx.min(), rx.max()
        y_min, y_max = ry.min(), ry.max()
        in_gradient_region = None
        t = (rx - x_min) / (x_max - x_min + 1e-12)  # [0,1] left → right
        s = (ry - y_min) / (y_max - y_min + 1e-12)  # [0,1] bottom → top

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
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in rgb
    ]

    if in_gradient_region is not None:
        hex_colors = [
            hc if flag else outside_color
            for hc, flag in zip(hex_colors, in_gradient_region)
        ]

    hex_colors = np.array(hex_colors)

    if cell_filter is not None:
        # Resolve regions/callables in the reference embedding, then align the
        # mask onto the cells actually being plotted (current_coords order).
        mask = _regions_to_mask_fn(cell_filter)(reference_data)
        mask = pd.Series(np.asarray(mask), index=reference_data.coordinates().index)
        in_cell_region = mask.reindex(current_coords.index).fillna(False).to_numpy()
        hex_colors[~in_cell_region] = outside_color

    df = current_coords.copy()
    df["color"] = hex_colors

    return df
