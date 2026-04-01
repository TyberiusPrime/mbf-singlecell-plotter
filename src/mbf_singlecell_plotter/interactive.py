"""Layer 4: interactive HTML export with Moran's I marker gene tooltips."""

import io
import json
import re
from pathlib import Path

import numpy as np


def _parse_svg_viewbox(svg_str: str) -> tuple:
    """Return (width, height) in SVG user units from the root element's viewBox."""
    m = re.search(r'<svg\b[^>]+\bviewBox="([^"]+)"', svg_str, re.IGNORECASE)
    if m:
        parts = m.group(1).split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])
    # Fallback: width/height attributes (strip unit suffix like 'pt', 'px')
    mw = re.search(r'<svg\b[^>]+\bwidth="([0-9.]+)', svg_str, re.IGNORECASE)
    mh = re.search(r'<svg\b[^>]+\bheight="([0-9.]+)', svg_str, re.IGNORECASE)
    if mw and mh:
        return float(mw.group(1)), float(mh.group(1))
    raise ValueError("Cannot determine SVG dimensions from the plot output.")


def save_interactive_moran(
    plotter,
    column: str,
    output_path,
    *,
    n_bins: int = 40,
    min_cells: int = 3,
    k: int = 20,
    min_moran: float = 0.2,
) -> None:
    """Save an interactive HTML scatter plot with Moran's I marker gene tooltips.

    The HTML file embeds the scatter plot as an inline SVG and adds an invisible
    grid overlay.  Hovering over a cell highlights it and shows marker genes in a
    panel below.  Clicking locks the highlight; clicking the same cell again
    returns to hover mode; clicking another cell switches the selection.

    Args:
        plotter:     Configured :class:`~mbf_singlecell_plotter.ScatterPlotter`.
        column:      Gene or obs column passed to :meth:`~ScatterPlotter.plot`.
        output_path: Destination ``.html`` path (string or Path).
        n_bins:      Moran's I grid resolution (default 40).
        min_cells:   Minimum cells per bin (default 3).
        k:           Marker genes stored per region (default 20).
        min_moran:   Minimum Moran's I to qualify as marker (default 0.2).
    """
    from .transforms import compute_grid_moran, marker_genes_by_region
    from .plots import _PlotWithPostDraw
    import matplotlib.pyplot as plt

    data = plotter._data

    # ── Draw the scatter plot ─────────────────────────────────────────────────
    p = plotter.plot(column)
    fig = p.draw()

    # Run post-draw hooks (custom colorbars / legends) that only fire on save
    if isinstance(p, _PlotWithPostDraw):
        for fn in p._post_draw_fns:
            fn(fig)

    # Scatter axes is always the first one added by plotnine
    ax = fig.axes[0]
    ax_pos = ax.get_position()   # fractions of figure, [0, 1] each
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # ── Save figure to SVG (no bbox_inches so viewBox matches figure size) ────
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    svg_str = buf.getvalue()

    svg_w, svg_h = _parse_svg_viewbox(svg_str)

    # Data → SVG coordinate helpers
    # SVG y=0 is at the top; matplotlib y=0 is at the bottom of the axes.
    ax_left   = ax_pos.x0 * svg_w
    ax_right  = ax_pos.x1 * svg_w
    ax_top    = (1.0 - ax_pos.y1) * svg_h
    ax_bottom = (1.0 - ax_pos.y0) * svg_h

    def _dx(x):
        return ax_left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (ax_right - ax_left)

    def _dy(y):
        # data y increases upward; SVG y increases downward
        frac = (y - ylim[0]) / (ylim[1] - ylim[0])
        return ax_bottom + frac * (ax_top - ax_bottom)

    # ── Compute Moran's I marker genes ────────────────────────────────────────
    coords = data.coordinates()
    x_vals = coords["x"].values
    y_vals = coords["y"].values
    x_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    y_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)

    gene_df = compute_grid_moran(data, n_bins=n_bins, min_cells=min_cells)
    markers = marker_genes_by_region(gene_df, k=k, min_moran=min_moran)
    gene_moran = dict(zip(gene_df["gene"], gene_df["moran_i"]))

    # ── Build per-cell overlay data ───────────────────────────────────────────
    cells = []
    for (xi, yi), genes in markers.items():
        xi, yi = int(xi), int(yi)
        x0_d, x1_d = x_edges[xi], x_edges[xi + 1]
        y0_d, y1_d = y_edges[yi], y_edges[yi + 1]

        # Higher data-y maps to a smaller SVG y (closer to top of image)
        svg_x = _dx(x0_d)
        svg_y = _dy(y1_d)
        svg_rect_w = _dx(x1_d) - svg_x
        svg_rect_h = _dy(y0_d) - svg_y   # positive: y0_d < y1_d ⟹ _dy(y0_d) > _dy(y1_d)

        gene_entries = [
            {"name": g, "mi": round(float(gene_moran.get(g, 0.0)), 3)}
            for g in genes
        ]
        cells.append({
            "xi": xi, "yi": yi,
            "x": round(svg_x, 2), "y": round(svg_y, 2),
            "w": round(svg_rect_w, 2), "h": round(svg_rect_h, 2),
            "genes": gene_entries,
        })

    html = _build_html(svg_str, svg_w, svg_h, cells, column)
    Path(output_path).write_text(html, encoding="utf-8")


def _build_html(
    svg_str: str,
    svg_w: float,
    svg_h: float,
    cells: list,
    column: str,
) -> str:
    cells_json = json.dumps(cells, separators=(",", ":"))

    rect_tags = []
    for i, c in enumerate(cells):
        rect_tags.append(
            f'<rect class="gc" data-i="{i}"'
            f' x="{c["x"]}" y="{c["y"]}"'
            f' width="{c["w"]}" height="{c["h"]}"/>'
        )
    overlay_rects = "\n    ".join(rect_tags)

    placeholder = (
        "No regions with Moran&#8217;s I above threshold &mdash; "
        "try lowering <code>min_moran</code> or <code>min_cells</code>."
        if not cells
        else "Hover over a highlighted region to see marker genes."
    )

    # Panel width in pt (same units as the SVG viewBox)
    panel_w_pt = f"{svg_w:.0f}pt"

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
/* ── scatter wrapper ── */
#wrap {{
  display: inline-block;
  position: relative;
  line-height: 0;
  box-shadow: 0 1px 6px rgba(0,0,0,.12);
  border-radius: 3px;
  overflow: hidden;
}}
#wrap > svg:first-child {{ display: block; }}

/* ── interactive overlay ── */
#overlay {{
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
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
  width: {panel_w_pt};
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
</style>
</head>
<body>
<h1>Marker genes &mdash; {column}</h1>
<div id="wrap">
{svg_str}
  <svg id="overlay" xmlns="http://www.w3.org/2000/svg"
       viewBox="0 0 {svg_w:.2f} {svg_h:.2f}">
    {overlay_rects}
  </svg>
</div>
<div id="panel"><span class="ph">{placeholder}</span></div>

<script>
(function () {{
  const CELLS = {cells_json};
  const panel = document.getElementById('panel');
  const rects = [...document.querySelectorAll('#overlay .gc')];
  let active = null;   // index into rects / CELLS, or null

  function renderGenes(idx) {{
    const c = CELLS[idx];
    const n = c.genes.length;
    const chips = c.genes.map(g =>
      `<span class="chip"><span>${{g.name}}</span><span class="mi">I\u202f=\u202f${{g.mi.toFixed(3)}}</span></span>`
    ).join('');
    panel.innerHTML =
      `<div class="hdr">Region (${{c.xi}},\u2009${{c.yi}}) \u2014 ` +
      `${{n}}\u202fmarker gene${{n === 1 ? '' : 's'}}</div>` +
      `<div class="chips">${{chips}}</div>`;
  }}

  function clearPanel() {{
    panel.innerHTML = '<span class="ph">Hover over a highlighted region to see marker genes.</span>';
  }}

  rects.forEach((el, i) => {{
    el.addEventListener('mouseenter', () => {{
      if (active === null) {{ el.classList.add('hov'); renderGenes(i); }}
    }});

    el.addEventListener('mouseleave', () => {{
      if (active === null) {{ el.classList.remove('hov'); clearPanel(); }}
    }});

    el.addEventListener('click', () => {{
      if (active === i) {{
        // Clicking the active cell → deactivate, revert to hover
        el.classList.remove('act');
        el.classList.add('hov');
        active = null;
        renderGenes(i);   // mouse is still over, so keep showing
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
