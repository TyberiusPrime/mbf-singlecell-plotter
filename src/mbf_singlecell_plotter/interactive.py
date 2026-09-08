"""Layer 4: interactive HTML export with marker-gene tooltips.

Two flavours share the same figure/overlay/panel machinery:

* :func:`save_interactive_moran_grid` — marker genes per *spatial grid cell*,
  ranked by Moran's I spatial autocorrelation (the original view).
* :func:`save_interactive_cluster_markers` — marker genes per *category* of a
  categorical column, ranked by a pseudobulk one-vs-rest score; a hovered grid
  cell is mapped to its predominant category and shows that cluster's markers.

Each of the two is really three steps, and they are available separately
because the first two are expensive and the third is not:

1. ``write_figure_cache`` — draw the scatter once, keep the PNG and the CSS
   geometry an overlay needs.
2. ``write_cluster_markers_cache`` / ``write_moran_grid_cache`` — score every
   gene, and record the per-bin counts, *unfiltered*.
3. ``render_interactive_*`` — build the HTML from those files alone.  It opens
   no h5ad and holds no plotter, so every threshold it applies (``k``,
   ``min_score``, ``min_moran``, ``min_cluster_cells``) and everything it
   labels (``gene_url``, ``debug``) is free to change.

``mbf_singlecell_plotter.ppg2`` gives each step its own job; the ``save_*``
functions run all three back to back through a scratch directory, so a direct
call and a pipegraph run cannot drift apart.
"""

import base64
import io
import json
from typing import Any
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


#: A per-gene URL template: a ``str`` carrying a ``{gene}`` placeholder, filled
#: in by the viewer.  Every ``gene_url=`` takes one, a sequence of them, or a
#: ``gene_url(gene_id, alt_gene_id) -> str | None`` callable for the one case a
#: template cannot express.
GeneUrlTemplate = str


# ── shared figure / geometry / binning helpers ───────────────────────────────
def _prepare_figure(plotter, column, dpi, *, legend_boxes: bool = False):
    """Render *column* to a PNG and return it with its CSS geometry.

    Returns ``(png_bytes, css_w, css_h, geom)``.  ``geom`` carries the axes
    bounding box and the data limits -- everything :func:`_mappers` needs to
    rebuild the data→CSS mappers, and everything the debug overlay reads -- so
    the whole return value is JSON-serializable apart from the PNG itself.
    That is what lets the expensive half of an export be cached: nothing here
    is a closure over the figure or over the data.

    The panel defaults to 5×5in unless the plotter already has a fixed panel
    size.  With *legend_boxes* the figure is drawn once more up front so
    ``geom["legend_blocks"]`` can carry the on-screen box of every legend key.
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

    if legend_boxes:
        # Legend offsetboxes only get their offsets during a draw, so place
        # every artist before reading the entry boxes below.
        fig.draw_without_rendering()

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

    legend_blocks = _legend_entry_boxes(fig) if legend_boxes else []

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)

    geom = {
        "legend_blocks": legend_blocks,
        "ax_left": ax_left,
        "ax_right": ax_right,
        "ax_top": ax_top,
        "ax_bottom": ax_bottom,
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
    }
    return buf.getvalue(), css_w, css_h, geom


def _mappers(geom: dict):
    """The ``(dx, dy)`` data→CSS pixel mappers of a *geom* dict.

    Both are affine in the axes bbox and the data limits, so they can be
    rebuilt from :func:`_prepare_figure`'s serializable output alone -- the
    rendering half of an export never has to hold on to the figure.
    """
    ax_left = geom["ax_left"]
    ax_right = geom["ax_right"]
    ax_top = geom["ax_top"]
    ax_bottom = geom["ax_bottom"]
    xlim = geom["xlim"]
    ylim = geom["ylim"]

    def _dx(x):
        return ax_left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (ax_right - ax_left)

    def _dy(y):
        # data y increases upward; CSS/SVG y increases downward
        frac = (y - ylim[0]) / (ylim[1] - ylim[0])
        return ax_bottom + frac * (ax_top - ax_bottom)

    return _dx, _dy


# Legend keys end tight against their label text; a little breathing room on
# the right makes the hotspot comfortable to hit without covering the plot.
_LEGEND_PAD_RIGHT = 6.0


def _legend_entry_boxes(fig) -> list[list[dict]]:
    """Read the per-key boxes of every legend in *fig*, in CSS pixels.

    plotnine draws a discrete guide as an ``AnchoredOffsetbox`` whose leaves are
    one ``HPacker`` per key — a swatch (``DrawingArea``) next to its label
    (``TextArea``).  Their window extents give us the on-screen rectangle of each
    legend entry, so no OCR of the rendered PNG is needed.

    Returns one list of ``{label, x, y, w, h}`` dicts per legend found (a figure
    can carry several, e.g. a colour guide plus a border guide); empty when the
    plot has no discrete legend.  Must be called after the layout is frozen and
    the figure has been drawn once.
    """
    import matplotlib.offsetbox as ob

    scale = 96.0 / fig.dpi  # display pixels → CSS pixels
    fig_h_px = fig.get_size_inches()[1] * fig.dpi

    def _keys(art, out):
        for ch in art.get_children():
            if isinstance(ch, ob.HPacker):
                kids = ch.get_children()
                swatch = any(isinstance(k, ob.DrawingArea) for k in kids)
                texts = [
                    t.get_children()[0].get_text()
                    for t in kids
                    if isinstance(t, ob.TextArea) and t.get_children()
                ]
                if swatch and texts:
                    out.append((ch, texts[0]))
                    continue
            _keys(ch, out)
        return out

    blocks = []
    for art in fig.get_children():
        if not isinstance(art, ob.AnchoredOffsetbox):
            continue
        entries = []
        for box, label in _keys(art, []):
            bb = box.get_window_extent()
            entries.append(
                {
                    "label": label,
                    "x": round(bb.x0 * scale, 1),
                    "y": round((fig_h_px - bb.y1) * scale, 1),
                    "w": round((bb.x1 - bb.x0) * scale, 1),
                    "h": round((bb.y1 - bb.y0) * scale, 1),
                }
            )
        if entries:
            blocks.append(entries)
    return blocks


def _match_legend_entries(blocks: list[list[dict]], categories) -> list[tuple]:
    """Pair legend key boxes with the categories they stand for.

    Keys are matched by their rendered label against ``str(category)``; the
    legend block with the most matches wins (a plot can also carry a border
    guide).  Returns ``[(entry, category), ...]``, empty when nothing matches —
    in which case the caller simply omits the legend hotspots.
    """
    by_label = {str(c): c for c in categories}
    best: list[dict] = []
    best_hits = 0
    for blk in blocks:
        hits = sum(1 for e in blk if e["label"] in by_label)
        if hits > best_hits:
            best, best_hits = blk, hits
    return [(e, by_label[e["label"]]) for e in best if e["label"] in by_label]


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


#: The keys of :func:`_grid_binning`'s result that are plain numbers, and so
#: survive a round trip through the analysis cache.  Everything the rendering
#: half needs -- :func:`_cell_geometry`, :func:`_build_debug_svg`,
#: :func:`_bin_labeller` -- reads only these, never ``data_full`` or the
#: per-cell arrays.
_GRID_KEYS = (
    "gs",
    "glv",
    "x_min_d",
    "x_max_d",
    "y_min_d",
    "y_max_d",
    "cell_w",
    "cell_h",
)


def _grid_geometry(b: dict) -> dict:
    """The serializable subset of a :func:`_grid_binning` result.

    ``gs`` stays an ``int`` (it is a bin count, and ``range(gs)`` draws the
    debug overlay); the bounds stay floats.
    """
    cast = {"gs": int, "glv": bool}
    return {key: cast.get(key, float)(b[key]) for key in _GRID_KEYS}


def _bin_labeller(grid: dict):
    """Rebuild :func:`_grid_binning`'s ``bin_to_label`` from a *grid* dict."""
    from .data import _LETTERS

    gs = int(grid["gs"])
    glv = grid["glv"]

    def _bin_to_label(xi: int, yi: int) -> str:
        row_from_top = gs - 1 - yi  # yi=0 is bottom → last row from top
        if glv:
            return f"{_LETTERS[row_from_top]}{xi + 1}"
        return f"{_LETTERS[xi]}{row_from_top + 1}"

    return _bin_to_label


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
    """Normalise the ``gene_url`` argument into templates plus a per-gene resolver.

    *gene_url* is a ``str`` template carrying a ``{gene}`` placeholder, a
    sequence of such templates -- a gene may have several plots, and the viewer
    shows one image per template -- or a callable, invoked here once per gene
    with the bare ``var_index`` and the alternative id, for the one case a
    template cannot express.

    Returns ``(gene_url_templates, has_gene_urls, resolve)``.  The templates are
    handed to the viewer as written and expanded there, so a gene costs one name
    in the HTML rather than one URL per template; ``resolve(g)`` is the
    callable's single URL, or ``None`` when there is no callable.
    """
    _cb = gene_url if callable(gene_url) else None
    if gene_url is None or _cb is not None:
        gene_url_templates = []
    elif isinstance(gene_url, str):
        gene_url_templates = [gene_url]
    else:
        gene_url_templates = list(gene_url)
        for template in gene_url_templates:
            if not isinstance(template, str):
                raise TypeError(
                    "gene_url: a sequence must hold '{gene}' templates - got "
                    f"{type(template).__name__}."
                )
    has_gene_urls = bool(gene_url_templates) or _cb is not None

    def _resolve(g: str):
        if _cb is not None:
            return _cb(g, data.alternative_id_for(g))  # ty: ignore
        return None

    return gene_url_templates, has_gene_urls, _resolve


# ── the two-stage cache ──────────────────────────────────────────────────────
#
# An interactive export is one cheap HTML file sitting on top of two expensive
# computations: drawing the figure, and scoring the genes.  Re-tuning what the
# viewer shows -- k, a threshold, where a gene links to -- has to touch neither.
#
# So each half writes a small set of files under a *prefix*, and the renderer
# builds the HTML from those files alone: it never opens the h5ad, never
# replays the plotter, and holds no closure over either.  The thresholds live
# in the renderer on purpose (``k`` and ``min_score`` are pure post-filters on
# the cached table), which is what makes re-tuning them free.

#: Files each cache writes, relative to its prefix.  The ppg2 wrapper declares
#: these as the outputs of its cache jobs, so they have to be exact.
FIGURE_CACHE_FILES = (".figure.png", ".figure.json")
CLUSTER_MARKERS_CACHE_FILES = (
    ".markers.parquet",
    ".genes.parquet",
    ".bins.parquet",
    ".bin_categories.parquet",
    ".grid.json",
)
MORAN_GRID_CACHE_FILES = (
    ".moran.parquet",
    ".genes.parquet",
    ".bins.parquet",
    ".grid.json",
)


def cache_paths(prefix, suffixes: Sequence[str]) -> list:
    """The files a cache with this *prefix* writes, in declaration order."""
    prefix = Path(prefix)
    return [prefix.with_name(prefix.name + suffix) for suffix in suffixes]


def _cache_file(prefix, suffix: str) -> Path:
    prefix = Path(prefix)
    return prefix.with_name(prefix.name + suffix)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


def _write_parquet(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


# -- figure cache -------------------------------------------------------------


def write_figure_cache(
    plotter, column: str, prefix, *, dpi: int = 150, legend_boxes: bool = False
) -> None:
    """Render *column* once and cache the PNG plus its CSS geometry.

    Keyed (by the caller) on the plotter's configuration, *column* and *dpi*
    alone -- nothing a viewer setting can reach -- so changing ``k`` or a
    ``gene_url`` never redraws the figure.
    """
    png, css_w, css_h, geom = _prepare_figure(
        plotter, column, dpi, legend_boxes=legend_boxes
    )
    _cache_file(prefix, ".figure.png").parent.mkdir(parents=True, exist_ok=True)
    _cache_file(prefix, ".figure.png").write_bytes(png)
    _write_json(
        _cache_file(prefix, ".figure.json"),
        {"css_w": css_w, "css_h": css_h, "geom": geom},
    )


def read_figure_cache(prefix) -> tuple[str, int, int, dict]:
    """``(img_b64, css_w, css_h, geom)`` from a :func:`write_figure_cache` prefix."""
    png = _cache_file(prefix, ".figure.png").read_bytes()
    meta = json.loads(_cache_file(prefix, ".figure.json").read_text(encoding="utf-8"))
    return (
        base64.b64encode(png).decode("ascii"),
        meta["css_w"],
        meta["css_h"],
        meta["geom"],
    )


# -- gene identity ------------------------------------------------------------


def _alt_id_values(data, genes: Sequence[str]):
    """:meth:`EmbeddingData.alternative_id_for` for many genes at once.

    The per-gene method walks every source for every call, which is fine for
    the handful of genes an export used to display but not for the whole
    ``var`` index -- and the cache has to cover the whole index, because which
    genes survive ``k``/``min_score`` is only decided when the HTML is built.
    This resolves source by source instead, filling in the genes still missing,
    which is the same first-source-that-has-it order the method uses.
    """
    import pandas as pd

    genes = list(genes)
    out = np.array([None] * len(genes), dtype=object)
    if data._alternative_id_column is None:
        return out
    col = data._alternative_id_column
    index = pd.Index(genes)
    for ad in [data.ad, *(s.ad for s in data._alternative_sources)]:
        if col not in ad.var.columns:
            continue
        todo = np.array([value is None for value in out])
        if not todo.any():
            break
        # duplicate gene symbols: the per-gene method takes the first row
        series = ad.var[col]
        series = series[~series.index.duplicated(keep="first")].dropna()
        found = series.reindex(index[todo]).to_numpy(dtype=object)
        out[todo] = [
            None if value is None or pd.isna(value) else value for value in found
        ]
    return out


def _gene_table(plotter, data, genes: Sequence[str]):
    """One row per gene: how it is displayed, and its alternative ids.

    Two alternative-id columns, because the two existing readers disagree and
    a cache is the wrong place to quietly settle that: ``alternative_id`` is
    the primary source's ``var`` value, which is what the marker TSV has always
    written (see :func:`_alt_id_lookup`), while ``url_alternative_id`` is the
    all-sources lookup that :meth:`ScatterPlotter._display_name` and a
    ``gene_url`` callable see.
    """
    import pandas as pd

    genes = list(genes)
    url_alt = _alt_id_values(data, genes)
    display = [
        gene if alt is None else f"{alt} ({gene})" for gene, alt in zip(genes, url_alt)
    ]
    if data._alternative_id_column is not None:
        primary = data.ad.var[data._alternative_id_column]
        primary = primary[~primary.index.duplicated(keep="first")]
        alternative_id = primary.reindex(pd.Index(genes)).to_numpy(dtype=object)
    else:
        alternative_id = np.array([None] * len(genes), dtype=object)
    return pd.DataFrame(
        {
            "gene": genes,
            "display_name": display,
            "alternative_id": alternative_id,
            "url_alternative_id": url_alt,
        }
    ).astype(
        {
            "display_name": "string",
            "alternative_id": "string",
            "url_alternative_id": "string",
        }
    )


class _CachedAltIds:
    """The ``alternative_id_for`` half of an EmbeddingData, read off the cache.

    :func:`_resolve_gene_url` invokes a ``gene_url`` callable with the gene's
    alternative id; the renderer has no data source to ask, so it asks this.
    """

    def __init__(self, mapping: dict):
        self._mapping = mapping

    def alternative_id_for(self, gene_name: str):
        return self._mapping.get(gene_name)


def _gene_lookups(genes_frame) -> tuple[dict, dict, dict]:
    """``(display_name, alternative_id, url_alternative_id)`` maps of a gene table."""
    import pandas as pd

    def _map(column):
        return {
            gene: (None if value is None or value is pd.NA else value)
            for gene, value in zip(genes_frame["gene"], genes_frame[column])
        }

    return _map("display_name"), _map("alternative_id"), _map("url_alternative_id")


# -- analysis caches ----------------------------------------------------------


def _bin_tables(b: dict, column: str | None):
    """The per-bin tables of a binning: cell counts, and category counts.

    Both are stored *unfiltered* -- ``min_cluster_cells`` is a post-filter on
    the counts, so leaving it to the renderer costs nothing here and makes it
    free to change there.  ``rank`` records the descending-count order
    ``value_counts`` produced, so the renderer reproduces it exactly rather
    than re-deriving an order that ties differently.
    """
    import pandas as pd

    counts = pd.DataFrame(
        {
            "xi": np.asarray(b["xi_all"], dtype=int),
            "yi": np.asarray(b["yi_all"], dtype=int),
        }
    )
    bins = (
        counts.groupby(["xi", "yi"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["xi", "yi"], kind="stable")
        .reset_index(drop=True)
    )
    if column is None:
        return bins, None

    cat_series = b["data_full"].get_column(column).series.reindex(b["all_coords"].index)
    bin_df = pd.DataFrame(
        {"xi": b["xi_all"], "yi": b["yi_all"], "cat": cat_series.values}
    )
    bin_df = bin_df[pd.notna(bin_df["cat"])]
    rows = []
    for (xi, yi), group in bin_df.groupby(["xi", "yi"], observed=True):
        # value_counts() is descending by count; keep that order as a rank so
        # the renderer does not have to re-break the ties.
        for rank, (cat, n) in enumerate(group["cat"].value_counts().items()):
            rows.append(
                {
                    "xi": int(xi),
                    "yi": int(yi),
                    "cat": str(cat),
                    "n": int(n),
                    "rank": rank,
                }
            )
    bin_categories = pd.DataFrame(
        rows, columns=["xi", "yi", "cat", "n", "rank"]
    ).astype({"xi": int, "yi": int, "cat": "string", "n": int, "rank": int})
    return bins, bin_categories


def write_cluster_markers_cache(
    plotter,
    column: str,
    prefix,
    *,
    layer: str | None = None,
    min_cells_per_group: int = 10,
) -> None:
    """Cache the pseudobulk one-vs-rest scoring behind a cluster-markers export.

    Everything here is keyed on the plotter's configuration, *column*, *layer*
    and *min_cells_per_group* -- the arguments that reach
    :func:`~mbf_singlecell_plotter.transforms.compute_cluster_markers`.  ``k``
    and ``min_score`` deliberately stay out: they filter the table this writes,
    so the renderer applies them and re-tuning them never lands here.

    Category labels are canonicalised to ``str`` so the marker table and the
    per-bin table can be joined on them after a round trip through parquet.
    """
    from .transforms import compute_cluster_markers

    data = plotter._data
    b = _grid_binning(data)
    bins, bin_categories = _bin_tables(b, column)

    marker_df = compute_cluster_markers(
        data, column, layer=layer, min_cells_per_group=min_cells_per_group
    )
    marker_df = marker_df.assign(category=marker_df["category"].astype(str))

    _write_parquet(_cache_file(prefix, ".markers.parquet"), marker_df)
    _write_parquet(
        _cache_file(prefix, ".genes.parquet"),
        _gene_table(plotter, data, list(dict.fromkeys(marker_df["gene"]))),
    )
    _write_parquet(_cache_file(prefix, ".bins.parquet"), bins)
    _write_parquet(_cache_file(prefix, ".bin_categories.parquet"), bin_categories)
    _write_json(
        _cache_file(prefix, ".grid.json"),
        {
            "column": column,
            "grid": _grid_geometry(b),
            "has_alt": data._alternative_id_column is not None,
        },
    )


def write_moran_grid_cache(
    plotter,
    column: str,
    prefix,
    *,
    min_cells: int = 3,
    var_score_column: str | None = None,
) -> None:
    """Cache the Moran's I scoring behind a moran-grid export.

    Keyed on the plotter's configuration, *column*, *min_cells* and
    *var_score_column*; ``k`` and ``min_moran`` filter the cached table and so
    belong to the renderer, exactly as for the cluster view.
    """
    from .transforms import compute_grid_moran

    data = plotter._data
    b = _grid_binning(data)
    bins, _ = _bin_tables(b, None)

    gene_df = compute_grid_moran(
        data, n_bins=b["gs"], min_cells=min_cells, var_score_column=var_score_column
    )
    # `top_bin` is an (xi, yi) tuple; parquet wants columns, and the renderer
    # puts the tuple back together before handing it to marker_genes_by_region.
    gene_df = gene_df.assign(
        top_bin_xi=[int(t[0]) for t in gene_df["top_bin"]],
        top_bin_yi=[int(t[1]) for t in gene_df["top_bin"]],
    ).drop(columns=["top_bin"])

    _write_parquet(_cache_file(prefix, ".moran.parquet"), gene_df)
    _write_parquet(
        _cache_file(prefix, ".genes.parquet"),
        _gene_table(plotter, data, list(dict.fromkeys(gene_df["gene"]))),
    )
    _write_parquet(_cache_file(prefix, ".bins.parquet"), bins)
    _write_json(
        _cache_file(prefix, ".grid.json"),
        {
            "column": column,
            "grid": _grid_geometry(b),
            "has_alt": data._alternative_id_column is not None,
        },
    )


def _read_analysis_cache(prefix, table: str) -> dict:
    """The parquet/JSON files of an analysis cache, as frames and dicts."""
    import pandas as pd

    meta = json.loads(_cache_file(prefix, ".grid.json").read_text(encoding="utf-8"))
    out = {
        "column": meta["column"],
        "grid": meta["grid"],
        "has_alt": meta["has_alt"],
        "table": pd.read_parquet(_cache_file(prefix, f".{table}.parquet")),
        "genes": pd.read_parquet(_cache_file(prefix, ".genes.parquet")),
        "bins": pd.read_parquet(_cache_file(prefix, ".bins.parquet")),
    }
    categories = _cache_file(prefix, ".bin_categories.parquet")
    if categories.exists():
        out["bin_categories"] = pd.read_parquet(categories)
    return out


def _one_shot(
    write_analysis,
    render_html,
    plotter,
    column,
    output_path,
    *,
    analysis,
    figure,
    render,
) -> None:
    """Run both cache stages into a scratch directory and render from them.

    This is what ``save_interactive_*`` is now: the same two stages the ppg2
    wrapper declares as separate jobs, minus the caching.  Sharing the code
    path is the point -- a direct call and a pipegraph run cannot drift apart,
    and the caches are not left behind for a caller who never asked for them.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="mbf-scp-") as scratch:
        # one prefix for both caches: their file suffixes do not overlap.
        prefix = Path(scratch) / "cache"
        write_figure_cache(plotter, column, prefix, **figure)
        write_analysis(plotter, column, prefix, **analysis)
        render_html(prefix, prefix, output_path, **render)


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
    gene_url: GeneUrlTemplate
    | Sequence[GeneUrlTemplate]
    | Callable[[str, str | None], str]
    | None = None,
    gene_url_inline: bool = False,
    save_tsv: bool = True,
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
        gene_url:         URL template (a ``str`` with a ``{gene}`` placeholder),
                          **or a sequence of templates** to point one gene at
                          several files -- the viewer then shows one image per
                          template, in the order given.  A callable
                          ``gene_url(gene_id, alt_gene_id=None)`` returning a URL
                          ``str`` (or ``None`` to skip a gene) still works for the
                          one case a template cannot express, and yields a single
                          URL.  When ``None`` (default) genes are plain text.
        gene_url_inline:  If ``True`` the linked resources are displayed in an
                          ``<img>`` panel below rather than opened in new
                          browser tabs (default ``False``).
        save_tsv:         If ``True`` also write a tidy ``.tsv`` of the marker
                          genes next to the HTML (same path with a ``.tsv``
                          suffix), one row per (grid cell, gene) with columns
                          ``grid_cell, gene, _display_name, moran_i, rank`` (plus
                          an ``alternative_id`` column when an alternative id
                          column is configured; default ``True``).
    """
    _one_shot(
        write_moran_grid_cache,
        render_interactive_moran_grid,
        plotter,
        column,
        output_path,
        analysis=dict(min_cells=min_cells, var_score_column=var_score_column),
        figure=dict(dpi=dpi, legend_boxes=False),
        render=dict(
            k=k,
            min_moran=min_moran,
            debug=debug,
            gene_url=gene_url,
            gene_url_inline=gene_url_inline,
            save_tsv=save_tsv,
        ),
    )


def render_interactive_moran_grid(
    figure_prefix,
    analysis_prefix,
    output_path,
    *,
    k: int = 20,
    min_moran: float = 0.2,
    debug: bool = False,
    gene_url: GeneUrlTemplate
    | Sequence[GeneUrlTemplate]
    | Callable[[str, str | None], str]
    | None = None,
    gene_url_inline: bool = False,
    save_tsv: bool = True,
) -> None:
    """Build the moran-grid HTML from a figure cache and an analysis cache.

    Touches neither the h5ad nor the plotter: every argument here is a viewer
    setting, and every one of them is cheap to change.  ``k`` and *min_moran*
    filter :func:`write_moran_grid_cache`'s table rather than the data, so
    re-tuning them costs one HTML rewrite.
    """
    from .transforms import marker_genes_by_region
    from collections import defaultdict

    img_b64, css_w, css_h, geom = read_figure_cache(figure_prefix)
    _dx, _dy = _mappers(geom)
    cache = _read_analysis_cache(analysis_prefix, "moran")
    grid = cache["grid"]
    bin_to_label = _bin_labeller(grid)
    display, alternative_id, url_alternative_id = _gene_lookups(cache["genes"])

    # ── Marker genes per bin ─────────────────────────────────────────────────
    gene_df = cache["table"]
    gene_df = gene_df.assign(
        top_bin=list(zip(gene_df["top_bin_xi"], gene_df["top_bin_yi"]))
    )
    markers = marker_genes_by_region(gene_df, k=k, min_moran=min_moran)
    gene_moran = dict(zip(gene_df["gene"], gene_df["moran_i"]))

    grid_cell_genes = defaultdict(list)  # (xi,yi) → [(gene, score)]
    for (xi, yi), genes in markers.items():
        for g in genes:
            grid_cell_genes[(int(xi), int(yi))].append(
                (g, float(gene_moran.get(g, 0.0)))
            )

    gene_url_templates, has_gene_urls, _gene_url = _resolve_gene_url(
        gene_url, _CachedAltIds(url_alternative_id)
    )

    # ── Build overlay cells for ALL occupied bins ─────────────────────────────
    cells = []
    for row in cache["bins"].itertuples(index=False):
        xi, yi, n_cells = int(row.xi), int(row.yi), int(row.n_cells)
        gene_list = grid_cell_genes.get((xi, yi), [])
        seen = set()
        deduped = []
        for gene, mi in sorted(gene_list, key=lambda t: -t[1]):
            if gene not in seen:
                seen.add(gene)
                deduped.append(
                    {
                        "name": display.get(gene, gene),
                        "gene": gene,
                        "url": _gene_url(gene),
                        "mi": round(mi, 3),
                    }
                )
        deduped = deduped[:k]

        cell = _cell_geometry(xi, yi, grid, _dx, _dy)
        cell.update(
            {
                "label": bin_to_label(xi, yi),
                "genes": deduped,
                "n_cells": n_cells,
            }
        )
        cells.append(cell)

    debug_svg = _build_debug_svg(geom, grid, _dx, _dy) if debug else ""

    html = _build_html(
        img_b64,
        css_w,
        css_h,
        cells,
        cache["column"],
        debug_svg,
        gene_url_templates=gene_url_templates,
        has_gene_urls=has_gene_urls,
        gene_url_inline=gene_url_inline,
        score_label="I",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")

    if save_tsv:
        has_alt = cache["has_alt"]
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
                    row["alternative_id"] = alternative_id.get(g["gene"])
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
    gene_url: GeneUrlTemplate
    | Sequence[GeneUrlTemplate]
    | Callable[[str, str | None], str]
    | None = None,
    gene_url_inline: bool = False,
    save_tsv: bool = True,
) -> None:
    """Save an interactive HTML view of per-cluster pseudobulk marker genes.

    The scatter is coloured by the categorical ``column``.  Each gene is scored
    per category via a pseudobulk one-vs-rest comparison (see
    :func:`~mbf_singlecell_plotter.transforms.compute_cluster_markers`).  Hovering
    a grid cell lists **every** category present in that bin as its own section
    (largest first), each with that cluster's top-*k* markers and their
    mean-difference Δ — so a small cluster sharing a bin with a large one stays
    reachable.  Clicking locks/switches the selection just like the grid view.

    Each **legend key** is a hotspot too: hovering (or clicking, to lock) one
    shows that cluster's markers over the whole embedding rather than within a
    single bin.  The key boxes are read from the rendered legend artists, so
    they line up with the PNG; if the plot has no discrete legend the hotspots
    are simply omitted.

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
                             sequence of templates to point one gene at several
                             files -- the viewer then shows one image per
                             template, in the order given.  A callable
                             ``gene_url(gene_id, alt_gene_id=None)`` still works
                             and yields a single URL.  When ``None`` (default)
                             genes are plain text.
        gene_url_inline:     If ``True`` the linked resources are shown inline in
                             an ``<img>`` panel rather than new tabs (default False).
        save_tsv:            If ``True`` also write a tidy ``.tsv`` of the marker
                             genes next to the HTML (same path with a ``.tsv``
                             suffix), one row per (cluster, gene) with columns
                             ``cluster, gene, _display_name, delta, rank`` (plus
                             an ``alternative_id`` column when an alternative id
                             column is configured; default ``True``).
    """
    _one_shot(
        write_cluster_markers_cache,
        render_interactive_cluster_markers,
        plotter,
        column,
        output_path,
        analysis=dict(layer=layer, min_cells_per_group=min_cells_per_group),
        figure=dict(dpi=dpi, legend_boxes=True),
        render=dict(
            k=k,
            min_score=min_score,
            min_cluster_cells=min_cluster_cells,
            debug=debug,
            gene_url=gene_url,
            gene_url_inline=gene_url_inline,
            save_tsv=save_tsv,
        ),
    )


def render_interactive_cluster_markers(
    figure_prefix,
    analysis_prefix,
    output_path,
    *,
    k: int = 20,
    min_score: float = 0.0,
    min_cluster_cells: int = 1,
    debug: bool = False,
    gene_url: GeneUrlTemplate
    | Sequence[GeneUrlTemplate]
    | Callable[[str, str | None], str]
    | None = None,
    gene_url_inline: bool = False,
    save_tsv: bool = True,
) -> None:
    """Build the cluster-markers HTML from a figure cache and an analysis cache.

    Touches neither the h5ad nor the plotter.  ``k``, *min_score* and
    *min_cluster_cells* are post-filters on
    :func:`write_cluster_markers_cache`'s tables, so re-tuning any of them --
    or a ``gene_url``, or the debug overlay -- costs one HTML rewrite and no
    recomputation.

    Category labels come back from parquet as ``str`` (that is how the cache
    stores them, so the marker table and the per-bin table can be joined); the
    HTML always rendered them through ``str`` anyway.
    """
    from .transforms import marker_genes_by_category

    img_b64, css_w, css_h, geom = read_figure_cache(figure_prefix)
    _dx, _dy = _mappers(geom)
    cache = _read_analysis_cache(analysis_prefix, "markers")
    grid = cache["grid"]
    column = cache["column"]
    bin_to_label = _bin_labeller(grid)
    display, alternative_id, url_alternative_id = _gene_lookups(cache["genes"])

    # ── Categories present per occupied bin (largest first) ───────────────────
    # `rank` is the order value_counts() produced when the cache was written.
    present = cache["bin_categories"]
    present = present[present["n"] >= min_cluster_cells]
    bin_categories: dict[tuple[int, int], list[tuple[Any, int]]] = {}
    for row in present.sort_values("rank", kind="stable").itertuples(index=False):
        bin_categories.setdefault((int(row.xi), int(row.yi)), []).append(
            (str(row.cat), int(row.n))
        )

    # ── Marker genes per category ─────────────────────────────────────────────
    markers = marker_genes_by_category(cache["table"], k=k, min_score=min_score)

    gene_url_templates, has_gene_urls, _gene_url = _resolve_gene_url(
        gene_url, _CachedAltIds(url_alternative_id)
    )

    # Build each category's gene chips once (identical across every bin it hits).
    cat_genes: dict[Any, list[dict]] = {}
    for cat, recs in markers.items():
        cat_genes[str(cat)] = [
            {
                "name": display.get(r["gene"], r["gene"]),
                "gene": r["gene"],
                "url": _gene_url(r["gene"]),
                "mi": round(float(r["delta"]), 3),
            }
            for r in recs[:k]
        ]

    # ── Build overlay cells for ALL occupied bins ─────────────────────────────
    cells = []
    for row in cache["bins"].itertuples(index=False):
        xi, yi, n_cells = int(row.xi), int(row.yi), int(row.n_cells)
        clusters = [
            {
                "name": f"cluster {cat}",
                "n": n,
                "genes": cat_genes.get(cat, []),
            }
            for cat, n in bin_categories.get((xi, yi), [])
        ]

        cell = _cell_geometry(xi, yi, grid, _dx, _dy)
        cell.update(
            {
                "label": bin_to_label(xi, yi),
                "clusters": clusters,
                "n_cells": n_cells,
            }
        )
        cells.append(cell)

    # ── Legend hotspots: one clickable box per legend key ─────────────────────
    # Totals are the unfiltered per-bin counts summed back up, which is the
    # whole-embedding count min_cluster_cells was never meant to touch.
    totals = cache["bin_categories"].groupby("cat", observed=True)["n"].sum()
    total_per_cat = {str(cat): int(n) for cat, n in totals.items()}
    legend_items = []
    for entry, cat in _match_legend_entries(geom["legend_blocks"], total_per_cat):
        width = min(entry["w"] + _LEGEND_PAD_RIGHT, css_w - entry["x"])
        legend_items.append(
            {
                "x": entry["x"],
                "y": entry["y"],
                "w": round(width, 1),
                "h": entry["h"],
                "label": entry["label"],
                "title": f"Cluster {cat}",
                "n_cells": total_per_cat[cat],
                "genes": cat_genes.get(cat, []),
            }
        )

    debug_svg = _build_debug_svg(geom, grid, _dx, _dy) if debug else ""

    html = _build_html(
        img_b64,
        css_w,
        css_h,
        cells,
        column,
        debug_svg,
        gene_url_templates=gene_url_templates,
        has_gene_urls=has_gene_urls,
        gene_url_inline=gene_url_inline,
        score_label="Δ",
        legend_items=legend_items,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")

    if save_tsv:
        has_alt = cache["has_alt"]
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
                    row["alternative_id"] = alternative_id.get(g["gene"])
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
    gene_url_templates: list[str] | None = None,
    has_gene_urls: bool = False,
    gene_url_inline: bool = False,
    score_label: str = "I",
    legend_items: list[Any] | None = None,
) -> str:
    legend_items = legend_items or []
    cells_json = json.dumps(cells, separators=(",", ":"))
    legend_json = json.dumps(legend_items, separators=(",", ":"))
    gene_urls_js = json.dumps(gene_url_templates or [], separators=(",", ":"))
    gene_url_inline_js = "true" if (has_gene_urls and gene_url_inline) else "false"
    score_label_js = json.dumps(score_label)

    rect_tags = []
    for i, c in enumerate(cells):
        rect_tags.append(
            f'<rect class="gc" data-i="{i}"'
            f' x="{c["x"]}" y="{c["y"]}"'
            f' width="{c["w"]}" height="{c["h"]}"/>'
        )
    # Legend keys share the overlay/index space, continuing after the grid cells.
    for j, c in enumerate(legend_items):
        rect_tags.append(
            f'<rect class="gc lg" data-i="{len(cells) + j}" rx="3"'
            f' x="{c["x"]}" y="{c["y"]}"'
            f' width="{c["w"]}" height="{c["h"]}"/>'
        )
    overlay_rects = "\n    ".join(rect_tags)

    hover_hint = (
        "Hover over a region — or a legend entry — to see its cell count and "
        "marker genes."
        if legend_items
        else "Hover over a region to see its cell count and marker genes."
    )
    placeholder = "No cells found in embedding." if not cells else hover_hint
    placeholder_js = json.dumps(placeholder)

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
.gc.lg {{ stroke: rgba(0, 0, 0, .10); stroke-width: 1; }}
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
  display: none;      /* shown as flex once a gene with URLs is clicked */
  flex-wrap: wrap;
  gap: 10px;
  align-items: flex-start;
}}
#img-wrap img {{
  max-width: 100%;
  border: 1px solid #ddd;
  border-radius: 4px;
}}
/* two or more images share the row instead of stacking full-width */
#img-wrap.multi img {{
  max-width: calc(50% - 5px);
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
<div id="img-wrap"></div>

<script>
(function () {{
  const CELLS = {cells_json};
  const LEGEND = {legend_json};
  const ITEMS = CELLS.concat(LEGEND);
  const PLACEHOLDER = {placeholder_js};
  const GENE_URLS = {gene_urls_js};
  const GENE_URL_INLINE = {gene_url_inline_js};
  const SCORE_LABEL = {score_label_js};
  const panel = document.getElementById('panel');
  const imgWrap = document.getElementById('img-wrap');
  const rects = [...document.querySelectorAll('#overlay .gc')];
  let active = null;   // index into rects / ITEMS, or null

  // Every URL of one gene: the templates with {{gene}} filled in, or the single
  // URL a gene_url callable already produced for it.
  function geneUrls(name, precomputed) {{
    if (precomputed) return [precomputed];
    return GENE_URLS.map(t => t.split('{{gene}}').join(encodeURIComponent(name)));
  }}

  function showImages(urls) {{
    imgWrap.innerHTML = '';
    urls.forEach((u) => {{
      const im = document.createElement('img');
      im.src = u;
      im.alt = 'gene image';
      imgWrap.appendChild(im);
    }});
    imgWrap.classList.toggle('multi', urls.length > 1);
    imgWrap.style.display = urls.length ? 'flex' : 'none';
  }}

  function flashBtn(btn) {{
    btn.classList.add('ok');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => {{ btn.textContent = orig; btn.classList.remove('ok'); }}, 1200);
  }}

  function chipsHtml(genes) {{
    return `<div class="chips">${{genes.map(g => {{
        const urls = geneUrls(g.gene, g.url);
        const cls = urls.length ? 'chip link' : 'chip';
        const da = urls.length ? ` data-gene="${{g.gene}}" data-url="${{g.url || ''}}"` : '';
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
    const c = ITEMS[idx];
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
          `<span class="hdr">${{c.title || ('Region ' + c.label)}} — ${{cellPart}}${{sep}}${{clPart}}</span>` +
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
          `<span class="hdr">${{c.title || ('Region ' + c.label)}} — ${{cellPart}}${{sep}}${{genesLabel(n)}}</span>` +
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
    panel.innerHTML = `<span class="ph">${{PLACEHOLDER}}</span>`;
    showImages([]);
  }}

  // Gene chip clicks (event delegation on panel)
  panel.addEventListener('click', (e) => {{
    const chip = e.target.closest('.chip.link');
    if (!chip) return;
    const urls = geneUrls(chip.dataset.gene, chip.dataset.url || null);
    if (!urls.length) return;
    if (GENE_URL_INLINE) {{
      showImages(urls);
    }} else {{
      urls.forEach((u) => window.open(u, '_blank', 'noopener,noreferrer'));
    }}
  }});

  rects.forEach((el) => {{
    const i = +el.dataset.i;
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
