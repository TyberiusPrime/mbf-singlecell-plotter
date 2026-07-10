"""Shared pytest fixtures for mbf-singlecell-plotter tests."""

import pytest
import anndata
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend; must be set before importing pyplot

import shutil
import numpy as np
from pathlib import Path

FAILURES_DIR = Path(__file__).parent / "failures"


def pytest_sessionstart(session):
    """Clear stale failure images before the run begins."""
    if FAILURES_DIR.exists():
        shutil.rmtree(FAILURES_DIR)


EXAMPLE_DATA = (
    Path(__file__).parent.parent / "example_data" / "scanpy-pbmc3k_stripped.h5ad"
)

SAMPLE_GENES = [
    "S100A8",
    "HIST1H2AH",
    "LST1",
    "KRBOX4",
    "FBXL14",
    "CST3",
    "TYROBP",
    "CD79A",
    "ZNF256",
    "NEFH",
]

# The categorical cluster column in the example data (leiden clusters)
CELL_TYPE_COLUMN = "leiden"

# A coarse (3-category) grouping derived from leiden, for 2-D facet tests.
COARSE_COLUMN = "coarse"

# A leiden → descriptive-name mapping, for testing multi-character facet/row labels.
CELL_TYPE_LABEL_COLUMN = "cell_type_label"
_CELL_TYPE_LABELS = {
    "0": "CD4 Naive T cells",
    "1": "CD14+ Monocytes",
    "2": "B cells",
    "3": "CD8 T cells",
    "4": "NK cells",
    "5": "FCGR3A+ Monocytes",
    "6": "Dendritic cells",
    "7": "Megakaryocytes",
    "8": "Platelets",
}


@pytest.fixture(scope="session")
def ad():
    """Load the example AnnData once per test session."""
    res = anndata.read_h5ad(EXAMPLE_DATA)
    half_and_half = pd.Series(True, index=res.obs.index)
    half_and_half.iloc[: len(half_and_half) // 2] = False
    res.obs["bool"] = half_and_half
    # Coarse cluster grouping: low / mid / high leiden index → 3 categories.
    coarse = pd.Series("M", index=res.obs.index, dtype=object)
    leiden_int = res.obs["leiden"].astype(int)
    coarse[leiden_int < 3] = "L"
    coarse[leiden_int >= 6] = "H"
    res.obs[COARSE_COLUMN] = coarse.astype("category")
    res.obs[CELL_TYPE_LABEL_COLUMN] = (
        res.obs["leiden"].astype(str).map(_CELL_TYPE_LABELS).astype("category")
    )
    return res


@pytest.fixture(scope="session")
def data(ad):
    """An EmbeddingData with UMAP embedding."""
    from mbf_singlecell_plotter import EmbeddingData

    return EmbeddingData(ad, "umap")


@pytest.fixture(scope="session")
def plotter_no_boundary(data):
    """A ScatterPlotter without border overlay (faster for unit tests)."""
    from mbf_singlecell_plotter import ScatterPlotter

    return ScatterPlotter().set_source(data)


@pytest.fixture(scope="session")
def plotter(data):
    """A ScatterPlotter with (leiden-)cluster boundary overlay.

    Requires scikit-image for boundary computation.  Tests using this fixture
    are automatically skipped when scikit-image is not installed.
    """
    pytest.importorskip(
        "skimage", reason="scikit-image required for boundary computation"
    )
    from mbf_singlecell_plotter import ScatterPlotter

    return (
        ScatterPlotter()
        .set_source(data)
        .with_borders(cell_type_column=CELL_TYPE_COLUMN)
    )


@pytest.fixture
def assert_image(request):
    """Assert a plot or array matches a reference image.

    The reference image name is derived automatically from the test class and
    function: ``ClassName__test_function_name``.  No manual name string needed.
    """

    def _assert(p_or_array, **kwargs):
        cls = request.node.cls.__name__ if request.node.cls else None
        fn = request.node.originalname or request.node.name
        name = f"{cls}__{fn}" if cls else fn

        if isinstance(p_or_array, np.ndarray):
            from image_comparison import assert_array_matches

            assert_array_matches(p_or_array, name, **kwargs)
        else:
            from image_comparison import assert_plotnine_matches

            assert_plotnine_matches(p_or_array, name, **kwargs)

    return _assert
