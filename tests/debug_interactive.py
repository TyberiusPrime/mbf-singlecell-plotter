"""Debug script for save_interactive_moran overlay alignment.

Run with:
    PYTHONPATH=src python tests/debug_interactive.py

Opens (or writes) two HTML files:
  debug_interactive_debug.html  -- with debug overlays
  debug_interactive_plain.html  -- without debug overlays

What to check in the browser
-----------------------------
debug version:
  - The RED dashed rectangle should sit exactly on the scatter plot axes border.
    If it is offset or wrong-sized, ax.get_position() or the CSS-pixel mapping
    is wrong.
  - The BLUE outlined rectangles should tile the region where the dots are.
    If they are shifted relative to the dots, the data→SVG coordinate mapping
    is wrong (xlim/ylim mismatch).
  - Corner labels (red, small) show the data-space coordinates at each corner
    of the computed axes box — verify they match the axis tick values visible
    in the PNG.

plain version:
  - Hover / click the four coloured quadrants.  Each quadrant should show
    exactly its own gene:
        top-left  (blue)  → gene_TL
        top-right (red)   → gene_TR
        bottom-left (green) → gene_BL
        bottom-right (orange) → gene_BR
  - The highlight square should sit on top of the corresponding coloured cloud,
    not on a neighbouring one.

Data layout
-----------
400 cells placed in tight 2 × 2 clusters:

    TL (-1.5,+1.5)   TR (+1.5,+1.5)
    BL (-1.5,-1.5)   BR (+1.5,-1.5)

Each quadrant has exactly one gene with expression = 5.0; all others = 0.
With n_bins=8 the four clusters each fall cleanly inside one or two bins.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from mbf_singlecell_plotter import ScatterPlotter

# ── Build synthetic data ──────────────────────────────────────────────────────

rng = np.random.default_rng(0)
n_per_quad = 100
spread = 0.25   # tight cluster radius

centers = {
    "TL": (-1.5,  1.5),
    "TR": ( 1.5,  1.5),
    "BL": (-1.5, -1.5),
    "BR": ( 1.5, -1.5),
}
gene_names = ["gene_TL", "gene_TR", "gene_BL", "gene_BR"]
quad_colors = {"TL": "#4477CC", "TR": "#CC3333", "BL": "#33AA55", "BR": "#FF8800"}

coords_list, quad_labels, cell_ids = [], [], []
for quad, (cx, cy) in centers.items():
    xy = rng.normal([cx, cy], spread, size=(n_per_quad, 2))
    coords_list.append(xy)
    quad_labels.extend([quad] * n_per_quad)

coords = np.vstack(coords_list)
n_cells = len(coords)
n_genes = len(gene_names)

# Expression: each gene is 5.0 only in its quadrant, 0 elsewhere
X = np.zeros((n_cells, n_genes), dtype=np.float32)
for gi, quad in enumerate(centers.keys()):
    start = gi * n_per_quad
    end   = start + n_per_quad
    X[start:end, gi] = 5.0

adata = ad.AnnData(
    X=sp.csr_matrix(X),
    obs=pd.DataFrame(
        {"quadrant": pd.Categorical(quad_labels)},
        index=[f"c{i}" for i in range(n_cells)],
    ),
    var=pd.DataFrame(index=gene_names),
    obsm={"X_umap": coords},
)

# ── Build plotter ─────────────────────────────────────────────────────────────

sp_plotter = (
    ScatterPlotter(adata, "umap")
    .colormap_discrete({q: quad_colors[q] for q in centers})
)

# ── Generate HTML files ───────────────────────────────────────────────────────

common_kwargs = dict(
    n_bins=8,
    min_cells=3,
    k=4,
    min_moran=0.05,
    dpi=120,
)

debug_path = "debug_interactive_debug.html"
plain_path = "debug_interactive_plain.html"

print("Generating debug HTML …")
sp_plotter.save_interactive_moran(
    "quadrant", debug_path, debug=True, **common_kwargs
)
print(f"  → {debug_path}")

print("Generating plain HTML …")
sp_plotter.save_interactive_moran(
    "quadrant", plain_path, debug=False, **common_kwargs
)
print(f"  → {plain_path}")

# ── Sanity checks ─────────────────────────────────────────────────────────────

import re, json

for path, label in [(debug_path, "debug"), (plain_path, "plain")]:
    html = open(path).read()
    m = re.search(r'const CELLS = (\[.+?\]);', html)
    cells = json.loads(m.group(1))
    labels = [c["label"] for c in cells]
    genes_per_cell = {c["label"]: [g["name"] for g in c["genes"]] for c in cells}
    print(f"\n[{label}] {len(cells)} interactive regions: {sorted(labels)}")
    for lbl, genes in sorted(genes_per_cell.items()):
        print(f"  {lbl}: {genes}")

print("\nOpen in browser:")
print(f"  file://{debug_path}")
print(f"  file://{plain_path}")
