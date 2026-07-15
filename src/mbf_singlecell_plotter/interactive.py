"""Layer 4: interactive HTML export with marker-gene tooltips.

Two flavours share the same figure/overlay/panel machinery:

* :func:`save_interactive_moran_grid` — marker genes per *spatial grid cell*,
  ranked by Moran's I spatial autocorrelation (the original view).
* :func:`save_interactive_cluster_markers` — marker genes per *category* of a
  categorical column, ranked by a pseudobulk one-vs-rest score; a hovered grid
  cell is mapped to its predominant category and shows that cluster's markers.
"""

import base64
import io
import json
from typing import Any
from collections.abc import Callable
from pathlib import Path

import numpy as np


# ── shared figure / geometry / binning helpers ───────────────────────────────
def _prepare_figure(plotter, column, dpi):
    """Render *column* to a base64 PNG and return CSS geometry + data→CSS mappers.

    Returns ``(img_b64, css_w, css_h, dx, dy, geom)`` where ``dx``/``dy`` map
    data coordinates to CSS pixels and ``geom`` carries the axes bounding box
    (used by the debug overlay).  The panel defaults to 5×5in unless the plotter
    already has a fixed panel size.
    """
    from .plots import _PlotWithPostDraw
    import matplotlib.pyplot as plt

    _pl = plotter if plotter._fixed_panel_size is not None else plotter.panel_size(5, 5)

    p = _pl.plot(column)
    fig = p.draw()

    # Run post-draw hooks (custom colorbars, panel resize via _apply_fixed_panel)
    if isinstance(p, _PlotWithPostDraw):
        for fn in p._post_draw_fns:
            fn(fig)

    # Freeze the layout engine so fig.savefig() doesn't re-run constrained /
    # tight layout and shift axes positions relative to what we read below.
    le = fig.get_layout_engine()
    if le is not None:
        le.execute(fig)
        fig.set_layout_engine(None)

    # ── Stable axes geometry (read AFTER layout is frozen) ───────────────────
    ax = fig.axes[0]
    ax_pos = ax.get_position()  # fractions of figure in [0, 1]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # CSS display size: always 96 dpi, regardless of PNG resolution
    fig_w_in, fig_h_in = fig.get_size_inches()
    css_w = round(fig_w_in * 96)
    css_h = round(fig_h_in * 96)

    # Axes bounding box in CSS pixels (SVG y=0 is at the top)
    ax_left = ax_pos.x0 * css_w
    ax_right = ax_pos.x1 * css_w
    ax_top = (1.0 - ax_pos.y1) * css_h
    ax_bottom = (1.0 - ax_pos.y0) * css_h

    def _dx(x):
        return ax_left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (ax_right - ax_left)

    def _dy(y):
        # data y increases upward; CSS/SVG y increases downward
        frac = (y - ylim[0]) / (ylim[1] - ylim[0])
        return ax_bottom + frac * (ax_top - ax_bottom)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    geom = {
        "ax_left": ax_left,
        "ax_right": ax_right,
        "ax_top": ax_top,
        "ax_bottom": ax_bottom,
        "xlim": xlim,
        "ylim": ylim,
    }
    return img_b64, css_w, css_h, _dx, _dy, geom


def _grid_binning(data):
    """Bin every cell into the EmbeddingData grid (matching the visible cells).

    Returns a dict of grid geometry plus per-cell ``xi_all``/``yi_all`` bin
    indices (aligned with ``all_coords.index``) and a ``bin_to_label`` mapper.
    Uses the same ``searchsorted`` binning as :func:`compute_grid_moran` so the
    per-bin cell membership is consistent with the marker computation.
    """
    from .data import _LETTERS

    data_full = data.unfocus()
    gs = data_full._grid_size
    glv = data_full._grid_letters_on_vertical
    x_min_d, x_max_d, y_min_d, y_max_d = data_full.full_bounds()
    cell_w = (x_max_d - x_min_d) / gs
    cell_h = (y_max_d - y_min_d) / gs

    # Drop cells with no embedding coordinate (NaN). These come from a
    # source-routed embedding whose source drops some primary cells; otherwise
    # searchsorted bins them all into the last grid cell, skewing the overlay
    # counts and the per-bin cluster membership.
    all_coords = data_full._finite_coordinates()
    x_edges = np.linspace(x_min_d, x_max_d, gs + 1)
    y_edges = np.linspace(y_min_d, y_max_d, gs + 1)
    xi_all = np.clip(np.searchsorted(x_edges[1:-1], all_coords["x"].values), 0, gs - 1)
    yi_all = np.clip(np.searchsorted(y_edges[1:-1], all_coords["y"].values), 0, gs - 1)

    def _bin_to_label(xi: int, yi: int) -> str:
        """Convert (xi, yi) bin indices directly to a grid label."""
        row_from_top = gs - 1 - yi  # yi=0 is bottom → last row from top
        if glv:
            return f"{_LETTERS[row_from_top]}{xi + 1}"
        return f"{_LETTERS[xi]}{row_from_top + 1}"

    return {
        "data_full": data_full,
        "gs": gs,
        "glv": glv,
        "x_min_d": x_min_d,
        "x_max_d": x_max_d,
        "y_min_d": y_min_d,
        "y_max_d": y_max_d,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "all_coords": all_coords,
        "xi_all": xi_all,
        "yi_all": yi_all,
        "bin_to_label": _bin_to_label,
    }


def _cell_geometry(xi: int, yi: int, b: dict, _dx, _dy) -> dict:
    """Return the CSS-pixel ``{x, y, w, h}`` rect for grid bin ``(xi, yi)``."""
    gs = b["gs"]
    row_from_top = gs - 1 - yi
    x0_d = b["x_min_d"] + xi * b["cell_w"]
    x1_d = x0_d + b["cell_w"]
    y1_d = b["y_max_d"] - row_from_top * b["cell_h"]  # top edge in data coords
    y0_d = y1_d - b["cell_h"]  # bottom edge
    svg_x = _dx(x0_d)
    svg_y = _dy(y1_d)
    return {
        "x": round(svg_x, 1),
        "y": round(svg_y, 1),
        "w": round(_dx(x1_d) - svg_x, 1),
        "h": round(_dy(y0_d) - svg_y, 1),
    }


def _build_debug_svg(geom: dict, b: dict, _dx, _dy) -> str:
    """Build the debug overlay: axes bbox + corner coords + all grid outlines."""
    ax_left = geom["ax_left"]
    ax_right = geom["ax_right"]
    ax_top = geom["ax_top"]
    ax_bottom = geom["ax_bottom"]
    xlim = geom["xlim"]
    ylim = geom["ylim"]
    gs = b["gs"]
    x_min_d = b["x_min_d"]
    y_max_d = b["y_max_d"]
    cell_w = b["cell_w"]
    cell_h = b["cell_h"]

    # 1. Red dashed rect: computed axes bounding box
    debug_svg = (
        f"<!-- axes bbox -->"
        f'<rect x="{ax_left:.1f}" y="{ax_top:.1f}"'
        f' width="{ax_right - ax_left:.1f}" height="{ax_bottom - ax_top:.1f}"'
        f' fill="none" stroke="red" stroke-width="2"'
        f' stroke-dasharray="6 3" pointer-events="none"/>'
    )
    # Corner labels: data coords at the four corners of the axes
    corners = [
        (ax_left, ax_top, f"{xlim[0]:.2f},{ylim[1]:.2f}", "start", "hanging"),
        (ax_right, ax_top, f"{xlim[1]:.2f},{ylim[1]:.2f}", "end", "hanging"),
        (ax_left, ax_bottom, f"{xlim[0]:.2f},{ylim[0]:.2f}", "start", "auto"),
        (ax_right, ax_bottom, f"{xlim[1]:.2f},{ylim[0]:.2f}", "end", "auto"),
    ]
    for cx2, cy2, lbl, anchor, baseline in corners:
        debug_svg += (
            f'<text x="{cx2:.1f}" y="{cy2:.1f}" font-size="9"'
            f' fill="red" text-anchor="{anchor}"'
            f' dominant-baseline="{baseline}"'
            f' pointer-events="none">{lbl}</text>'
        )

    # 2. Blue outlines for ALL EmbeddingData grid cells — these should align
    #    exactly with the visible grid lines in the scatter plot.
    for col_idx in range(gs):
        for row_from_top in range(gs):
            x0_d = x_min_d + col_idx * cell_w
            x1_d = x0_d + cell_w
            y1_d = y_max_d - row_from_top * cell_h
            y0_d = y1_d - cell_h
            rx = _dx(x0_d)
            ry = _dy(y1_d)
            rw = _dx(x1_d) - rx
            rh = _dy(y0_d) - ry
            debug_svg += (
                f'<rect x="{rx:.1f}" y="{ry:.1f}"'
                f' width="{rw:.1f}" height="{rh:.1f}"'
                f' fill="none" stroke="rgba(0,80,220,.35)"'
                f' stroke-width="0.6" pointer-events="none"/>'
            )
    return debug_svg


def _resolve_gene_url(gene_url, data):
    """Normalise the ``gene_url`` argument into a per-gene resolver.

    Returns ``(gene_url_template, has_gene_urls, resolve)`` where ``resolve(g)``
    yields a precomputed URL for callables (invoked once per gene with the bare
    ``var_index`` and the alternative id) or ``None`` for string templates
    (substituted client-side) / when no URL strategy was given.
    """
    _cb = gene_url if callable(gene_url) else None
    gene_url_template = gene_url if isinstance(gene_url, str) else ""
    has_gene_urls = gene_url is not None

    def _resolve(g: str):
        if _cb is not None:
            return _cb(g, data.alternative_id_for(g))  # ty: ignore
        return None

    return gene_url_template, has_gene_urls, _resolve


# ── grid view: markers per spatial bin (Moran's I) ───────────────────────────
def save_interactive_moran_grid(
    plotter,
    column: str,
    output_path,
    *,
    min_cells: int = 3,
    k: int = 20,
    min_moran: float = 0.2,
    var_score_column: str | None = None,
    dpi: int = 150,
    debug: bool = False,
    gene_url: str | Callable[[str, str | None], str] | None = None,
    gene_url_inline: bool = False,
    save_tsv: bool = False,
) -> None:
    """Save an interactive HTML scatter plot with Moran's I marker gene tooltips.

    The HTML file embeds the scatter plot as a PNG and adds an invisible grid
    overlay.  Hovering over a cell highlights it and shows the marker genes of
    that spatial bin (ranked by Moran's I) in a panel below.  Clicking locks the
    highlight; clicking the same cell again returns to hover mode; clicking
    another cell switches the selection.

    Args:
        plotter:          Configured :class:`~mbf_singlecell_plotter.ScatterPlotter`.
        column:           Gene or obs column passed to :meth:`~ScatterPlotter.plot`.
        output_path:      Destination ``.html`` path (string or Path).
        min_cells:        Minimum cells per bin (default 3).
        k:                Marker genes stored per region (default 20).
        min_moran:        Minimum score threshold to qualify as a marker (default 0.2).
                          Applied to Moran's I or to *var_score_column* values.
        var_score_column: Column in ``adata.var`` to use as the gene score instead
                          of computing Moran's I on the fly.  Must be numeric;
                          higher values = more informative genes.  When ``None``
                          (default), Moran's I is computed from the embedding.
        dpi:              PNG resolution (default 150).  Display size is always
                          fixed at 96 dpi CSS pixels regardless of this value.
        gene_url:         URL template (a ``str`` with a ``{gene}`` placeholder)
                          **or** a callable ``gene_url(gene_id, alt_gene_id=None)``
                          returning a URL ``str`` (or ``None`` to skip a gene).
                          When ``None`` (default) genes are plain text.
        gene_url_inline:  If ``True`` the linked resource is displayed in an
                          ``<img>`` panel below rather than opened in a new
                          browser tab (default ``False``).
        save_tsv:         If ``True`` also write a tidy ``.tsv`` of the marker
                          genes next to the HTML (same path with a ``.tsv``
                          suffix), one row per (grid cell, gene) with columns
                          ``grid_cell, gene, _display_name, moran_i, rank`` (plus
                          an ``alternative_id`` column when an alternative id
                          column is configured; default ``False``).
    """
    from .transforms import compute_grid_moran, marker_genes_by_region
    from collections import defaultdict, Counter

    data = plotter._data

    img_b64, css_w, css_h, _dx, _dy, geom = _prepare_figure(plotter, column, dpi)
    b = _grid_binning(data)
    gs = b["gs"]

    bin_cell_counts = Counter(zip(b["xi_all"].tolist(), b["yi_all"].tolist()))

    # ── Compute marker genes per bin ──────────────────────────────────────────
    gene_df = compute_grid_moran(
        data, n_bins=gs, min_cells=min_cells, var_score_column=var_score_column
    )
    markers = marker_genes_by_region(gene_df, k=k, min_moran=min_moran)
    gene_moran = dict(zip(gene_df["gene"], gene_df["moran_i"]))

    grid_cell_genes = defaultdict(list)  # (xi,yi) → [(gene, score)]
    for (xi, yi), genes in markers.items():
        for g in genes:
            grid_cell_genes[(int(xi), int(yi))].append(
                (g, float(gene_moran.get(g, 0.0)))
            )

    gene_url_template, has_gene_urls, _gene_url = _resolve_gene_url(gene_url, data)

    # ── Build overlay cells for ALL occupied bins ─────────────────────────────
    cells = []
    for (xi, yi), n_cells in sorted(bin_cell_counts.items()):
        gene_list = grid_cell_genes.get((xi, yi), [])
        seen = set()
        deduped = []
        for gene, mi in sorted(gene_list, key=lambda t: -t[1]):
            if gene not in seen:
                seen.add(gene)
                deduped.append(
                    {
                        "name": plotter._display_name(data, gene),
                        "gene": gene,
                        "url": _gene_url(gene),
                        "mi": round(mi, 3),
                    }
                )
        deduped = deduped[:k]

        cell = _cell_geometry(xi, yi, b, _dx, _dy)
        cell.update(
            {
                "label": b["bin_to_label"](xi, yi),
                "genes": deduped,
                "n_cells": n_cells,
            }
        )
        cells.append(cell)

    debug_svg = _build_debug_svg(geom, b, _dx, _dy) if debug else ""

    html = _build_html(
        img_b64,
        css_w,
        css_h,
        cells,
        column,
        debug_svg,
        gene_url_template=gene_url_template,
        has_gene_urls=has_gene_urls,
        gene_url_inline=gene_url_inline,
        score_label="I",
    )
    Path(output_path).write_text(html, encoding="utf-8")

    if save_tsv:
        alt_ids, has_alt = _alt_id_lookup(data)
        columns = ["grid_cell", "gene", "_display_name"]
        if has_alt:
            columns.append("alternative_id")
        columns += ["moran_i", "rank"]
        rows = []
        for cell in cells:
            for rank, g in enumerate(cell["genes"], start=1):
                row = {
                    "grid_cell": cell["label"],
                    "gene": g["gene"],
                    "_display_name": g["name"],
                    "moran_i": g["mi"],
                    "rank": rank,
                }
                if has_alt:
                    row["alternative_id"] = alt_ids.get(g["gene"])
                rows.append(row)
        _write_marker_tsv(rows, output_path, columns)


def _alt_id_lookup(data) -> tuple[dict, bool]:
    """Return ``(gene → alternative_id map, present)`` for the source's alt-id column.

    When no alternative id column is configured the map is empty and *present* is
    ``False`` so callers can omit the ``alternative_id`` column entirely.
    """
    if data._alternative_id_column is not None:
        return data.ad.var[data._alternative_id_column].to_dict(), True
    return {}, False


def _write_marker_tsv(rows: list[dict], output_path, columns: list[str]) -> None:
    """Write marker *rows* as a tab-separated file next to *output_path*.

    The destination is *output_path* with its suffix replaced by ``.tsv``.
    """
    import pandas as pd

    tsv_path = Path(output_path).with_suffix(".tsv")
    pd.DataFrame(rows, columns=columns).to_csv(tsv_path, sep="\t", index=False)


# ── cluster view: markers per category (pseudobulk one-vs-rest) ───────────────
def save_interactive_cluster_markers(
    plotter,
    column: str,
    output_path,
    *,
    k: int = 20,
    min_score: float = 0.0,
    min_cells_per_group: int = 10,
    min_cluster_cells: int = 1,
    layer: str | None = None,
    dpi: int = 150,
    debug: bool = False,
    gene_url: str | Callable[[str, str | None], str] | None = None,
    gene_url_inline: bool = False,
    save_tsv: bool = False,
) -> None:
    """Save an interactive HTML view of per-cluster pseudobulk marker genes.

    The scatter is coloured by the categorical ``column``.  Each gene is scored
    per category via a pseudobulk one-vs-rest comparison (see
    :func:`~mbf_singlecell_plotter.transforms.compute_cluster_markers`).  Hovering
    a grid cell lists **every** category present in that bin as its own section
    (largest first), each with that cluster's top-*k* markers and their
    mean-difference Δ — so a small cluster sharing a bin with a large one stays
    reachable.  Clicking locks/switches the selection just like the grid view.

    Args:
        plotter:             Configured :class:`~mbf_singlecell_plotter.ScatterPlotter`.
        column:              Categorical obs column (cluster labels) — also the
                             column the scatter is coloured by.
        output_path:         Destination ``.html`` path (string or Path).
        k:                   Marker genes shown per cluster (default 20).
        min_score:           Minimum combined score to qualify as a marker
                             (default 0.0 → keep genes up-regulated vs the rest).
        min_cells_per_group: Categories with fewer cells are skipped (default 10).
        min_cluster_cells:   A category must have at least this many cells *within
                             a bin* to be listed for that bin (default 1 → list
                             every category present; raise to hide stray cells
                             that bleed across bin borders).
        layer:               Expression layer for marker computation (``None`` =
                             the source's configured layer; pass e.g. a raw /
                             log-normalized layer key to score on that instead).
        dpi:                 PNG resolution (default 150).
        gene_url:            URL template with a ``{gene}`` placeholder, or a
                             callable ``gene_url(gene_id, alt_gene_id=None)``.
                             When ``None`` (default) genes are plain text.
        gene_url_inline:     If ``True`` the linked resource is shown inline in an
                             ``<img>`` panel rather than a new tab (default False).
        save_tsv:            If ``True`` also write a tidy ``.tsv`` of the marker
                             genes next to the HTML (same path with a ``.tsv``
                             suffix), one row per (cluster, gene) with columns
                             ``cluster, gene, _display_name, delta, rank`` (plus
                             an ``alternative_id`` column when an alternative id
                             column is configured; default ``False``).
    """
    from .transforms import compute_cluster_markers, marker_genes_by_category
    from collections import Counter
    import pandas as pd

    data = plotter._data

    img_b64, css_w, css_h, _dx, _dy, geom = _prepare_figure(plotter, column, dpi)
    b = _grid_binning(data)

    bin_cell_counts = Counter(zip(b["xi_all"].tolist(), b["yi_all"].tolist()))

    # ── Categories present per occupied bin (largest first) ───────────────────
    cat_series = b["data_full"].get_column(column).series.reindex(b["all_coords"].index)
    bin_df = pd.DataFrame(
        {"xi": b["xi_all"], "yi": b["yi_all"], "cat": cat_series.values}
    )
    bin_df = bin_df[pd.notna(bin_df["cat"])]
    bin_categories: dict[tuple[int, int], list[tuple[Any, int]]] = {}
    for (xi, yi), grp in bin_df.groupby(["xi", "yi"], observed=True):
        vc = grp["cat"].value_counts()  # descending by count
        bin_categories[(int(xi), int(yi))] = [
            (cat, int(n)) for cat, n in vc.items() if n >= min_cluster_cells
        ]

    # ── Marker genes per category ─────────────────────────────────────────────
    marker_df = compute_cluster_markers(
        data, column, layer=layer, min_cells_per_group=min_cells_per_group
    )
    markers = marker_genes_by_category(marker_df, k=k, min_score=min_score)

    gene_url_template, has_gene_urls, _gene_url = _resolve_gene_url(gene_url, data)

    # Build each category's gene chips once (identical across every bin it hits).
    cat_genes: dict[Any, list[dict]] = {}
    for cat, recs in markers.items():
        cat_genes[cat] = [
            {
                "name": plotter._display_name(data, r["gene"]),
                "gene": r["gene"],
                "url": _gene_url(r["gene"]),
                "mi": round(float(r["delta"]), 3),
            }
            for r in recs[:k]
        ]

    # ── Build overlay cells for ALL occupied bins ─────────────────────────────
    cells = []
    for (xi, yi), n_cells in sorted(bin_cell_counts.items()):
        clusters = [
            {
                "name": f"cluster {cat}",
                "n": n,
                "genes": cat_genes.get(cat, []),
            }
            for cat, n in bin_categories.get((xi, yi), [])
        ]

        cell = _cell_geometry(xi, yi, b, _dx, _dy)
        cell.update(
            {
                "label": b["bin_to_label"](xi, yi),
                "clusters": clusters,
                "n_cells": n_cells,
            }
        )
        cells.append(cell)

    debug_svg = _build_debug_svg(geom, b, _dx, _dy) if debug else ""

    html = _build_html(
        img_b64,
        css_w,
        css_h,
        cells,
        column,
        debug_svg,
        gene_url_template=gene_url_template,
        has_gene_urls=has_gene_urls,
        gene_url_inline=gene_url_inline,
        score_label="Δ",
    )
    Path(output_path).write_text(html, encoding="utf-8")

    if save_tsv:
        alt_ids, has_alt = _alt_id_lookup(data)
        columns = ["cluster", "gene", "_display_name"]
        if has_alt:
            columns.append("alternative_id")
        columns += ["delta", "rank"]
        rows = []
        for cat, genes in cat_genes.items():
            for rank, g in enumerate(genes, start=1):
                row = {
                    "cluster": cat,
                    "gene": g["gene"],
                    "_display_name": g["name"],
                    "delta": g["mi"],
                    "rank": rank,
                }
                if has_alt:
                    row["alternative_id"] = alt_ids.get(g["gene"])
                rows.append(row)
        _write_marker_tsv(rows, output_path, columns)


def _build_html(
    img_b64: str,
    css_w: int,
    css_h: int,
    cells: list[Any],
    column: str,
    debug_svg: str = "",
    *,
    gene_url_template: str = "",
    has_gene_urls: bool = False,
    gene_url_inline: bool = False,
    score_label: str = "I",
) -> str:
    cells_json = json.dumps(cells, separators=(",", ":"))
    gene_url_js = json.dumps(gene_url_template)
    gene_url_inline_js = "true" if (has_gene_urls and gene_url_inline) else "false"
    score_label_js = json.dumps(score_label)

    rect_tags = []
    for i, c in enumerate(cells):
        rect_tags.append(
            f'<rect class="gc" data-i="{i}"'
            f' x="{c["x"]}" y="{c["y"]}"'
            f' width="{c["w"]}" height="{c["h"]}"/>'
        )
    overlay_rects = "\n    ".join(rect_tags)

    placeholder = (
        "No cells found in embedding."
        if not cells
        else "Hover over a region to see its cell count and marker genes."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Marker genes — {column}</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: system-ui, -apple-system, sans-serif;
  background: #f4f4f6;
  padding: 20px;
  color: #1a1a1a;
}}
h1 {{
  font-size: 14px;
  font-weight: 600;
  color: #555;
  margin-bottom: 10px;
  letter-spacing: .02em;
}}
/* ── plot wrapper ── */
#wrap {{
  display: inline-block;
  position: relative;
  line-height: 0;
  box-shadow: 0 1px 6px rgba(0,0,0,.12);
  border-radius: 3px;
  overflow: hidden;
}}
#wrap img {{
  display: block;
  width: {css_w}px;
  height: {css_h}px;
}}
/* ── interactive overlay ── */
#overlay {{
  position: absolute;
  top: 0; left: 0;
  width: {css_w}px;
  height: {css_h}px;
  pointer-events: none;
  overflow: visible;
}}
.gc {{
  fill: transparent;
  stroke: none;
  pointer-events: all;
  cursor: pointer;
  transition: fill .08s, stroke .08s;
}}
.gc.hov {{
  fill: rgba(255, 210, 30, .22);
  stroke: rgba(160, 110, 0, .55);
  stroke-width: 1;
}}
.gc.act {{
  fill: rgba(255, 160, 0, .36);
  stroke: rgba(140, 80, 0, .80);
  stroke-width: 1.5;
}}
/* ── gene panel ── */
#panel {{
  margin-top: 10px;
  padding: 12px 14px;
  width: {css_w}px;
  max-width: 100%;
  min-height: 58px;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 13px;
  line-height: 1.55;
}}
#panel .ph {{ color: #bbb; font-style: italic; }}
#panel .hdr {{
  font-weight: 600;
  color: #333;
  margin-bottom: 8px;
  font-size: 13px;
}}
#panel .hdr-row {{
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 8px;
}}
#panel .hdr-row .hdr {{ margin-bottom: 0; }}
.copy-btn {{
  font-size: 11px;
  padding: 1px 7px;
  border: 1px solid #b0b8d8;
  border-radius: 3px;
  background: #eef2ff;
  color: #2244aa;
  cursor: pointer;
  white-space: nowrap;
  line-height: 1.6;
  transition: background .1s;
}}
.copy-btn:hover {{ background: #dde6ff; }}
.copy-btn.ok {{ background: #d4f0d4; border-color: #7bc47b; color: #1a6e1a; }}
#panel .cluster {{
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid #eee;
}}
#panel .cluster:first-of-type {{ margin-top: 6px; }}
#panel .chdr {{
  font-weight: 600;
  color: #444;
  font-size: 12px;
}}
#panel .chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}}
.chip {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: #eef2ff;
  border: 1px solid #c5cfee;
  border-radius: 3px;
  padding: 2px 8px;
  font-size: 12px;
  color: #2244aa;
}}
.chip .mi {{
  font-size: 10px;
  color: #778;
  letter-spacing: -.3px;
}}
.chip.link {{
  cursor: pointer;
  text-decoration: underline;
  text-underline-offset: 2px;
}}
.chip.link:hover {{
  background: #dde6ff;
  border-color: #99b;
}}
#img-wrap {{
  margin-top: 10px;
  display: none;
}}
#img-wrap img {{
  max-width: 100%;
  border: 1px solid #ddd;
  border-radius: 4px;
}}
</style>
</head>
<body>
<h1>Marker genes &mdash; {column}</h1>
<div id="wrap">
  <img src="data:image/png;base64,{img_b64}" alt="scatter plot">
  <svg id="overlay" xmlns="http://www.w3.org/2000/svg"
       viewBox="0 0 {css_w} {css_h}">
    {overlay_rects}
    {debug_svg}
  </svg>
</div>
<div id="panel"><span class="ph">{placeholder}</span></div>
<div id="img-wrap"><img id="img-el" src="" alt="gene image"></div>

<script>
(function () {{
  const CELLS = {cells_json};
  const GENE_URL = {gene_url_js};
  const GENE_URL_INLINE = {gene_url_inline_js};
  const SCORE_LABEL = {score_label_js};
  const panel = document.getElementById('panel');
  const imgWrap = document.getElementById('img-wrap');
  const imgEl  = document.getElementById('img-el');
  const rects = [...document.querySelectorAll('#overlay .gc')];
  let active = null;   // index into rects / CELLS, or null

  function geneUrl(name, precomputed) {{
    return precomputed || (GENE_URL ? GENE_URL.replace('{{gene}}', encodeURIComponent(name)) : null);
  }}

  function flashBtn(btn) {{
    btn.classList.add('ok');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => {{ btn.textContent = orig; btn.classList.remove('ok'); }}, 1200);
  }}

  function chipsHtml(genes) {{
    return `<div class="chips">${{genes.map(g => {{
        const url = geneUrl(g.gene, g.url);
        const cls = url ? 'chip link' : 'chip';
        const da = url ? ` data-gene="${{g.gene}}" data-url="${{url}}"` : '';
        return `<span class="${{cls}}"${{da}}><span>${{g.name}}</span>` +
               `<span class="mi">${{SCORE_LABEL}} = ${{g.mi.toFixed(3)}}</span></span>`;
      }}).join('')}}</div>`;
  }}

  function copyBtnsHtml(scope) {{
    return `<button class="copy-btn" data-scope="${{scope}}" data-copy="newline">Copy ↵</button>` +
           `<button class="copy-btn" data-scope="${{scope}}" data-copy="comma">Copy ,</button>`;
  }}

  function genesLabel(n) {{
    return n > 0
      ? `${{n}} marker gene${{n === 1 ? '' : 's'}}`
      : 'no marker genes above threshold';
  }}

  function renderGenes(idx) {{
    const c = CELLS[idx];
    const nc = c.n_cells || 0;
    const cellPart = nc > 0
      ? `${{nc.toLocaleString()}} cell${{nc === 1 ? '' : 's'}}`
      : '';

    // Resolve the gene list a copy button refers to (whole cell, or one cluster).
    const genesForScope = (scope) =>
      scope === 'all' ? c.genes : c.clusters[+scope].genes;

    if (c.clusters) {{
      // multi-cluster view: one section per category present in the bin
      const cl = c.clusters;
      const clPart = `${{cl.length}} cluster${{cl.length === 1 ? '' : 's'}}`;
      const sep = cellPart ? ' · ' : '';
      let html =
        `<div class="hdr-row">` +
          `<span class="hdr">Region ${{c.label}} — ${{cellPart}}${{sep}}${{clPart}}</span>` +
        `</div>`;
      if (cl.length === 0) {{
        html += `<span class="ph">no clusters in this region</span>`;
      }}
      cl.forEach((g, ci) => {{
        const n = g.genes.length;
        html +=
          `<div class="cluster">` +
            `<div class="hdr-row">` +
              `<span class="chdr">${{g.name}} — ` +
                `${{g.n.toLocaleString()}} cell${{g.n === 1 ? '' : 's'}} · ${{genesLabel(n)}}</span>` +
              (n > 0 ? copyBtnsHtml(ci) : '') +
            `</div>` +
            (n > 0 ? chipsHtml(g.genes) : '') +
          `</div>`;
      }});
      panel.innerHTML = html;
    }} else {{
      // flat view (grid / Moran): a single gene list for the whole cell
      const n = c.genes.length;
      const sep = cellPart && n > 0 ? ' · ' : '';
      panel.innerHTML =
        `<div class="hdr-row">` +
          `<span class="hdr">Region ${{c.label}} — ${{cellPart}}${{sep}}${{genesLabel(n)}}</span>` +
          (n > 0 ? copyBtnsHtml('all') : '') +
        `</div>` +
        (n > 0 ? chipsHtml(c.genes) : '');
    }}

    panel.querySelectorAll('.copy-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        const names = genesForScope(btn.dataset.scope).map(g => g.name);
        const text = btn.dataset.copy === 'newline' ? names.join('\\n') : names.join(', ');
        navigator.clipboard.writeText(text).then(() => flashBtn(btn));
      }});
    }});
  }}

  function clearPanel() {{
    panel.innerHTML =
      '<span class="ph">Hover over a region to see its cell count and marker genes.</span>';
    imgWrap.style.display = 'none';
  }}

  // Gene chip clicks (event delegation on panel)
  panel.addEventListener('click', (e) => {{
    const chip = e.target.closest('.chip.link');
    if (!chip) return;
    const url = chip.dataset.url;
    if (!url) return;
    if (GENE_URL_INLINE) {{
      imgEl.src = url;
      imgWrap.style.display = 'block';
    }} else {{
      window.open(url, '_blank', 'noopener,noreferrer');
    }}
  }});

  rects.forEach((el, i) => {{
    el.addEventListener('mouseenter', () => {{
      if (active === null) {{ el.classList.add('hov'); renderGenes(i); }}
    }});
    el.addEventListener('mouseleave', () => {{
      if (active === null) {{ el.classList.remove('hov'); clearPanel(); }}
    }});
    el.addEventListener('click', () => {{
      if (active === i) {{
        el.classList.remove('act');
        el.classList.add('hov');
        active = null;
        renderGenes(i);
      }} else {{
        if (active !== null) {{ rects[active].classList.remove('act', 'hov'); }}
        el.classList.remove('hov');
        el.classList.add('act');
        active = i;
        renderGenes(i);
      }}
    }});
  }});
}})();
</script>
</body>
</html>
"""
