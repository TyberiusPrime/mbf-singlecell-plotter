"""Shared pytest fixtures for mbf-singlecell-plotter tests."""

import pytest
import anndata
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before importing pyplot

import sys
from pathlib import Path

EXAMPLE_DATA = Path(__file__).parent.parent / "example_data" / "scanpy-pbmc3k_stripped.h5ad"

SAMPLE_GENES = [
    "S100A8", "HIST1H2AH", "LST1", "KRBOX4", "FBXL14",
    "CST3", "TYROBP", "CD79A", "ZNF256", "NEFH",
]

# The categorical cluster column in the example data (leiden clusters)
CELL_TYPE_COLUMN = "leiden"


@pytest.fixture(scope="session")
def ad():
    """Load the example AnnData once per test session."""
    return anndata.read_h5ad(EXAMPLE_DATA)


@pytest.fixture(scope="session")
def plotter(ad):
    """A ScanpyPlotter with UMAP embedding and leiden-cluster boundaries.

    Requires scikit-image for boundary computation.  Tests using this fixture
    are automatically skipped when scikit-image is not installed.
    """
    pytest.importorskip("skimage", reason="scikit-image required for boundary computation")
    from mbf_singlecell_plotter import ScanpyPlotter
    return ScanpyPlotter(ad, "umap", cell_type_column=CELL_TYPE_COLUMN)


@pytest.fixture(scope="session")
def plotter_no_boundary(ad):
    """A ScanpyPlotter without boundary computation (faster for unit tests)."""
    from mbf_singlecell_plotter import ScanpyPlotter
    return ScanpyPlotter(ad, "umap", cell_type_column=None)
