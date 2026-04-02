"""Layer 4: interactive HTML export with Moran's I marker gene tooltips."""

import base64
import io
import json
from pathlib import Path

import numpy as np


def save_interactive_moran(
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
    gene_url: str | None = None,
    gene_url_inline: bool = False,
) -> None:
    """Save an interactive HTML scatter plot with Moran's I marker gene tooltips.

    The HTML file embeds the scatter plot as a PNG and adds an invisible grid
    overlay.  Hovering over a cell highlights it and shows marker genes in a
    panel below.  Clicking locks the highlight; clicking the same cell again
    returns to hover mode; clicking another cell switches the selection.

    The data panel is made square by default (5 × 5 in) unless the plotter
    already has a fixed panel size set via :meth:`~ScatterPlotter.panel_size`.

    Args:
        plotter:          Configured :class:`~mbf_singlecell_plotter.ScatterPlotter`.
        column:           Gene or obs column passed to :meth:`~ScatterPlotter.plot`.
        output_path:      Destination ``.html`` path (string or Path).
        min_cells:        Minimum cells per bin (default 3).
        k:                Marker genes stored per region (default 20).
        min_moran:        Minimum score threshold to qualify as a marker (default 0.2).
                          Applied to Moran's I or to *var_score_column* values.
        var_score_column: Column in ``adata.var`` to use as the gene score instead
                          of computing Moran's I on the fly.  Pass the column name
                          (e.g. ``"moranI"`` or ``"highly_variable_rank"``).  The
                          column must be numeric; higher values = more informative
                          genes.  When ``None`` (default), Moran's I is computed
                          from the embedding.
        dpi:              PNG resolution (default 150).  Display size is always
                          fixed at 96 dpi CSS pixels regardless of this value.
    """
    from .transforms import compute_grid_moran, marker_genes_by_region
    from .plots import _PlotWithPostDraw
    import matplotlib.pyplot as plt

    data = plotter._data

    # ── Apply square panel by default ────────────────────────────────────────
    _pl = plotter if plotter._fixed_panel_size is not None else plotter.panel_size(5, 5)

    # ── Draw ─────────────────────────────────────────────────────────────────
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
    ax_pos = ax.get_position()   # fractions of figure in [0, 1]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # CSS display size: always 96 dpi, regardless of PNG resolution
    fig_w_in, fig_h_in = fig.get_size_inches()
    css_w = round(fig_w_in * 96)
    css_h = round(fig_h_in * 96)

    # Axes bounding box in CSS pixels (SVG y=0 is at the top)
    ax_left   = ax_pos.x0 * css_w
    ax_right  = ax_pos.x1 * css_w
    ax_top    = (1.0 - ax_pos.y1) * css_h
    ax_bottom = (1.0 - ax_pos.y0) * css_h

    def _dx(x):
        return ax_left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (ax_right - ax_left)

    def _dy(y):
        # data y increases upward; CSS/SVG y increases downward
        frac = (y - ylim[0]) / (ylim[1] - ylim[0])
        return ax_bottom + frac * (ax_top - ax_bottom)

    # ── Save as PNG (base64) ──────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # ── Set up grid (bins match EmbeddingData grid cells 1:1) ────────────────
    from collections import defaultdict, Counter
    from .data import _LETTERS

    data_full = data.unfocus()
    gs  = data_full._grid_size
    glv = data_full._grid_letters_on_vertical
    x_min_d, x_max_d, y_min_d, y_max_d = data_full.full_bounds()
    cell_w = (x_max_d - x_min_d) / gs
    cell_h = (y_max_d - y_min_d) / gs

    # Use the same searchsorted binning as compute_grid_moran so counts are
    # consistent with which bin each cell actually falls into.
    all_coords = data_full.coordinates()
    x_edges = np.linspace(x_min_d, x_max_d, gs + 1)
    y_edges = np.linspace(y_min_d, y_max_d, gs + 1)
    xi_all = np.clip(np.searchsorted(x_edges[1:-1], all_coords["x"].values), 0, gs - 1)
    yi_all = np.clip(np.searchsorted(y_edges[1:-1], all_coords["y"].values), 0, gs - 1)
    bin_cell_counts: Counter = Counter(zip(xi_all.tolist(), yi_all.tolist()))

    def _bin_to_label(xi: int, yi: int) -> str:
        """Convert (xi, yi) bin indices directly to a grid label."""
        row_from_top = gs - 1 - yi   # yi=0 is bottom → last row from top
        if glv:
            return f"{_LETTERS[row_from_top]}{xi + 1}"
        return f"{_LETTERS[xi]}{row_from_top + 1}"

    # ── Compute marker genes ──────────────────────────────────────────────────
    gene_df = compute_grid_moran(
        data, n_bins=gs, min_cells=min_cells, var_score_column=var_score_column
    )
    markers = marker_genes_by_region(gene_df, k=k, min_moran=min_moran)
    gene_moran = dict(zip(gene_df["gene"], gene_df["moran_i"]))

    # ── Map bin (xi, yi) → gene list (index arithmetic, no coordinate lookup) ─
    grid_cell_genes: dict = defaultdict(list)   # (xi,yi) → [(gene, score)]
    for (xi, yi), genes in markers.items():
        for g in genes:
            grid_cell_genes[(int(xi), int(yi))].append(
                (g, float(gene_moran.get(g, 0.0)))
            )

    # ── Build overlay cells for ALL occupied bins ─────────────────────────────
    cells = []
    for (xi, yi), n_cells in sorted(bin_cell_counts.items()):
        label = _bin_to_label(xi, yi)
        row_from_top = gs - 1 - yi

        # Data-space bounds of this grid cell
        x0_d = x_min_d + xi * cell_w
        x1_d = x0_d + cell_w
        y1_d = y_max_d - row_from_top * cell_h   # top edge in data coords
        y0_d = y1_d - cell_h                      # bottom edge

        # Genes (may be empty if none pass the threshold)
        gene_list = grid_cell_genes.get((xi, yi), [])
        seen: set = set()
        deduped = []
        for gene, mi in sorted(gene_list, key=lambda t: -t[1]):
            if gene not in seen:
                seen.add(gene)
                deduped.append({"name": gene, "mi": round(mi, 3)})
        deduped = deduped[:k]

        svg_x = _dx(x0_d)
        svg_y = _dy(y1_d)
        svg_rect_w = _dx(x1_d) - svg_x
        svg_rect_h = _dy(y0_d) - svg_y

        cells.append({
            "label": label,
            "x": round(svg_x, 1), "y": round(svg_y, 1),
            "w": round(svg_rect_w, 1), "h": round(svg_rect_h, 1),
            "genes": deduped,
            "n_cells": n_cells,
        })

    # ── Debug overlay elements ────────────────────────────────────────────────
    debug_svg = ""
    if debug:
        # 1. Red dashed rect: computed axes bounding box
        debug_svg += (
            f'<!-- axes bbox -->'
            f'<rect x="{ax_left:.1f}" y="{ax_top:.1f}"'
            f' width="{ax_right - ax_left:.1f}" height="{ax_bottom - ax_top:.1f}"'
            f' fill="none" stroke="red" stroke-width="2"'
            f' stroke-dasharray="6 3" pointer-events="none"/>'
        )
        # Corner labels: data coords at the four corners of the axes
        corners = [
            (ax_left,  ax_top,    f"{xlim[0]:.2f},{ylim[1]:.2f}", "start", "hanging"),
            (ax_right, ax_top,    f"{xlim[1]:.2f},{ylim[1]:.2f}", "end",   "hanging"),
            (ax_left,  ax_bottom, f"{xlim[0]:.2f},{ylim[0]:.2f}", "start", "auto"),
            (ax_right, ax_bottom, f"{xlim[1]:.2f},{ylim[0]:.2f}", "end",   "auto"),
        ]
        for cx2, cy2, lbl, anchor, baseline in corners:
            debug_svg += (
                f'<text x="{cx2:.1f}" y="{cy2:.1f}" font-size="9"'
                f' fill="red" text-anchor="{anchor}"'
                f' dominant-baseline="{baseline}"'
                f' pointer-events="none">{lbl}</text>'
            )

        # 2. Blue outlines for ALL EmbeddingData grid cells — these should
        #    align exactly with the visible grid lines in the scatter plot.
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

    html = _build_html(img_b64, css_w, css_h, cells, column, debug_svg,
                       gene_url=gene_url, gene_url_inline=gene_url_inline)
    Path(output_path).write_text(html, encoding="utf-8")


def _build_html(
    img_b64: str,
    css_w: int,
    css_h: int,
    cells: list,
    column: str,
    debug_svg: str = "",
    *,
    gene_url: str | None = None,
    gene_url_inline: bool = False,
) -> str:
    cells_json = json.dumps(cells, separators=(",", ":"))
    gene_url_js = json.dumps(gene_url or "")
    gene_url_inline_js = "true" if (gene_url and gene_url_inline) else "false"

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
<title>Marker genes \u2014 {column}</title>
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
  const panel = document.getElementById('panel');
  const imgWrap = document.getElementById('img-wrap');
  const imgEl  = document.getElementById('img-el');
  const rects = [...document.querySelectorAll('#overlay .gc')];
  let active = null;   // index into rects / CELLS, or null

  function geneUrl(name) {{
    return GENE_URL ? GENE_URL.replace('{{gene}}', encodeURIComponent(name)) : null;
  }}

  function renderGenes(idx) {{
    const c = CELLS[idx];
    const n = c.genes.length;
    const nc = c.n_cells || 0;
    const cellPart = nc > 0
      ? `${{nc.toLocaleString()}}\u202fcell${{nc === 1 ? '' : 's'}}`
      : '';
    const genePart = n > 0
      ? `${{n}}\u202fmarker gene${{n === 1 ? '' : 's'}}`
      : 'no marker genes above threshold';
    const sep = cellPart && n > 0 ? ' \u00b7 ' : '';
    const body = n > 0
      ? `<div class="chips">${{c.genes.map(g => {{
          const url = geneUrl(g.name);
          const cls = url ? 'chip link' : 'chip';
          const da = url ? ` data-gene="${{g.name}}" data-url="${{url}}"` : '';
          return `<span class="${{cls}}"${{da}}><span>${{g.name}}</span>` +
                 `<span class="mi">I\u202f=\u202f${{g.mi.toFixed(3)}}</span></span>`;
        }}).join('')}}</div>`
      : '';
    panel.innerHTML =
      `<div class="hdr">Region ${{c.label}} \u2014 ${{cellPart}}${{sep}}${{genePart}}</div>${{body}}`;
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
