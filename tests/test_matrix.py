"""
Combinatorial image regression test matrix.

Axes
----
data_type  : numerical (S100A8) | categorical (leiden)
borders    : off | on  — on requires scikit-image
grid_mode  : off | plain | labels | coords
focus      : off | on

2 × 2 × 4 × 2 = 32 combinations, one reference image each.
"""

import itertools

import matplotlib
matplotlib.use("Agg")
import pytest

from image_comparison import assert_plotnine_matches
from conftest import CELL_TYPE_COLUMN

DATA_TYPES = ["numerical", "categorical"]
BORDERS    = [False, True]
GRID_MODES = ["off", "plain", "labels", "coords"]
FOCUS      = [False, True]

GENE_FOR = {"numerical": "S100A8", "categorical": CELL_TYPE_COLUMN}
DOT_SIZE = 5


def _build_plotter(data, borders, grid_mode, focus):
    from mbf_singlecell_plotter import ScatterPlotter

    sp = ScatterPlotter().set_source(data).style(dot_size=DOT_SIZE)

    if borders:
        sp = sp.with_borders(cell_type_column=CELL_TYPE_COLUMN)

    if grid_mode == "plain":
        sp = sp.with_grid()
    elif grid_mode == "labels":
        sp = sp.with_grid(labels=True)
    elif grid_mode == "coords":
        sp = sp.with_grid(coords=True)

    if focus:
        coords = data.coordinates()
        x_mid = coords["x"].median()
        y_mid = coords["y"].median()
        sp = sp.focus_on(x=(coords["x"].min(), x_mid), y=(coords["y"].min(), y_mid))

    return sp


@pytest.mark.parametrize(
    "data_type,borders,grid_mode,focus",
    list(itertools.product(DATA_TYPES, BORDERS, GRID_MODES, FOCUS)),
)
def test_plot_matrix(data, data_type, borders, grid_mode, focus):
    if borders:
        pytest.importorskip("skimage", reason="scikit-image required for boundary computation")

    sp = _build_plotter(data, borders, grid_mode, focus)
    gene = GENE_FOR[data_type]
    name = f"matrix_{data_type}_borders{int(borders)}_grid_{grid_mode}_focus{int(focus)}"
    p = sp.plot(gene)
    assert_plotnine_matches(p, name)
