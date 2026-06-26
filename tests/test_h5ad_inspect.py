"""Tests for the h5ad-inspect backed data source.

These tests exercise H5adFacade + EmbeddingData (via set_source) against the
same example file used by the rest of the test suite.  All tests are
automatically skipped when h5ad-inspect is not on PATH.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from mbf_singlecell_plotter import (
    EmbeddingData,
    ScatterPlotter,
    is_h5ad_inspect_available,
)
from mbf_singlecell_plotter import H5adFacade

EXAMPLE_H5AD = (
    Path(__file__).parent.parent / "example_data" / "scanpy-pbmc3k_stripped.h5ad"
)

pytestmark = pytest.mark.skipif(
    not is_h5ad_inspect_available(),
    reason="h5ad-inspect not on PATH",
)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def facade():
    return H5adFacade(EXAMPLE_H5AD)


@pytest.fixture(scope="module")
def ed(facade):
    return EmbeddingData(facade, "umap")


# ── is_h5ad_inspect_available ────────────────────────────────────────────────


def test_availability_returns_bool():
    assert isinstance(is_h5ad_inspect_available(), bool)


def test_availability_true_when_binary_present():
    # If we reach this point the marker hasn't skipped us, so it must be True.
    assert is_h5ad_inspect_available() is True


# ── facade basics ────────────────────────────────────────────────────────────


class TestFacadeIndices:
    def test_obs_names_length(self, facade):
        assert len(facade.obs_names) == 2638

    def test_obs_names_type(self, facade):
        assert isinstance(facade.obs_names, pd.Index)

    def test_obs_names_first_cell(self, facade):
        assert facade.obs_names[0] == "AAACATACAACCAC-1"

    def test_var_names_present(self, facade):
        assert len(facade.var_names) > 0
        assert isinstance(facade.var_names, pd.Index)

    def test_obsm_keys_include_umap(self, facade):
        assert "X_umap" in facade.obsm

    def test_obsm_keys_include_pca(self, facade):
        assert "X_pca" in facade.obsm


# ── obs column types ──────────────────────────────────────────────────────────


class TestObsColumns:
    """Covers categorical, numeric, and (via ScatterPlotter) the full pathway."""

    def test_categorical_column_dtype(self, facade):
        s = facade.obs["leiden"]
        assert isinstance(s.dtype, pd.CategoricalDtype), (
            f"Expected CategoricalDtype, got {s.dtype}"
        )

    def test_categorical_column_length(self, facade):
        s = facade.obs["leiden"]
        assert len(s) == len(facade.obs_names)

    def test_categorical_column_values_are_strings(self, facade):
        s = facade.obs["leiden"]
        # Each category label should be a string (e.g. '0', '1', ..., '8')
        for cat in s.cat.categories:
            assert isinstance(cat, str)

    def test_numeric_column_dtype(self, facade):
        s = facade.obs["n_genes"]
        assert pd.api.types.is_numeric_dtype(s)

    def test_numeric_column_positive(self, facade):
        s = facade.obs["n_genes"]
        assert (s > 0).all()

    def test_numeric_column_total_counts(self, facade):
        s = facade.obs["total_counts"]
        assert pd.api.types.is_numeric_dtype(s)
        assert (s > 0).all()

    def test_obs_contains(self, facade):
        assert "leiden" in facade.obs
        assert "n_genes" in facade.obs
        assert "__nonexistent__" not in facade.obs


# ── obsm (embedding) ─────────────────────────────────────────────────────────


class TestObsm:
    def test_umap_shape(self, facade):
        arr = facade.obsm["X_umap"]
        assert arr.shape == (2638, 2)

    def test_umap_dtype_float(self, facade):
        arr = facade.obsm["X_umap"]
        assert np.issubdtype(arr.dtype, np.floating)

    def test_pca_has_many_components(self, facade):
        arr = facade.obsm["X_pca"]
        assert arr.shape[0] == 2638
        assert arr.shape[1] >= 2

    def test_missing_key_raises(self, facade):
        with pytest.raises(KeyError):
            facade.obsm["X_nonexistent"]


# ── X matrix (binary gene-expression) ────────────────────────────────────────


class TestXBinary:
    def test_column_length(self, facade):
        # First gene in var_index
        col = facade.X[:, 0]
        assert len(col) == len(facade.obs_names)

    def test_column_dtype_float64(self, facade):
        col = facade.X[:, 0]
        assert col.dtype == np.float64

    def test_column_writable(self, facade):
        col = facade.X[:, 0]
        # np.frombuffer produces read-only; the facade must return a copy
        col[0] = col[0]  # should not raise ValueError

    def test_single_column_is_1d(self, facade):
        assert facade.X[:, 0].ndim == 1

    def test_column_by_name_matches_positional(self, facade):
        gene = str(facade.var_names[2])
        np.testing.assert_array_equal(facade.X[:, gene], facade.X[:, 2])

    def test_rejects_bulk_indexing(self, facade):
        # Single-column access only on .X; bulk must go through get_X_csr()
        import pytest as _pytest

        with _pytest.raises(NotImplementedError):
            facade.X[np.array([0, 1])]
        with _pytest.raises(NotImplementedError):
            facade.X[:, [0, 1]]


# ── bulk X matrix (get_X_csr) ────────────────────────────────────────────────


class TestGetXCsr:
    def test_returns_csr(self, facade):
        from scipy import sparse as sp

        X = facade.get_X_csr()
        assert sp.issparse(X)
        assert X.format == "csr"

    def test_shape(self, facade):
        X = facade.get_X_csr()
        assert X.shape == (len(facade.obs_names), len(facade.var_names))

    def test_columns_align_with_var_names(self, facade):
        # column j of the bulk matrix must match X[:, j] (single-column fetch)
        X = facade.get_X_csr().toarray()
        for j in range(len(facade.var_names)):
            np.testing.assert_array_equal(X[:, j], facade.X[:, j])

    def test_row_slicing_matches_single_columns(self, facade):
        # the access pattern used by compute_grid_moran
        rows = np.array([3, 7, 100, 500])
        block = facade.get_X_csr()[rows].toarray()
        for j in range(len(facade.var_names)):
            np.testing.assert_array_equal(block[:, j], facade.X[:, j][rows])

    def test_row_mean_matches_column_mean(self, facade):
        rows = np.array([0, 4, 9, 200, 1000])
        block = facade.get_X_csr()[rows]
        row_mean = np.asarray(block.mean(axis=0)).ravel()
        col_means = np.array(
            [facade.X[:, j][rows].mean() for j in range(len(facade.var_names))]
        )
        np.testing.assert_allclose(row_mean, col_means)

    def test_cached(self, facade):
        assert facade.get_X_csr() is facade.get_X_csr()


# ── var_names ordering (physical / AnnData order) ────────────────────────────


class TestVarNamesOrder:
    def test_matches_var_index_column(self, facade):
        # var_names must be in the same order as var columns / X columns
        # (the file's native order), not alphabetically sorted.
        n_cells = facade.var["n_cells"]
        # n_cells is fetched in native order, so it must align positionally
        # with var_names by gene.
        assert list(facade.var_names) == list(n_cells.index)

    def test_var_names_column_aligned_with_get_X_csr(self, facade):
        # The gene at var_names[0] must correspond to column 0 of the bulk X.
        first_gene = str(facade.var_names[0])
        np.testing.assert_array_equal(
            facade.get_X_csr().toarray()[:, 0], facade.X[:, first_gene]
        )


# ── EmbeddingData integration ─────────────────────────────────────────────────


class TestEmbeddingDataFromFacade:
    def test_coordinates_shape(self, ed):
        coords = ed.coordinates()
        assert coords.shape == (2638, 2)
        assert set(coords.columns) == {"x", "y"}

    def test_get_categorical_column(self, ed):
        series, name = ed.get_column("leiden")
        assert name == "leiden"
        assert isinstance(series.dtype, pd.CategoricalDtype)
        assert len(series) == 2638

    def test_get_numeric_column(self, ed):
        series, name = ed.get_column("n_genes")
        assert name == "n_genes"
        assert pd.api.types.is_numeric_dtype(series)

    def test_get_gene_expression(self, ed):
        # CST3 is in the stripped example file
        series, name = ed.get_column("CST3")
        assert name == "CST3"
        assert series.dtype == np.float64
        assert len(series) == 2638

    def test_missing_column_raises(self, ed):
        with pytest.raises(KeyError):
            ed.get_column("__definitely_not_there__")


# ── ScatterPlotter.set_source with path ───────────────────────────────────────


class TestSetSourcePath:
    def test_string_path(self):
        plotter = ScatterPlotter().set_source(str(EXAMPLE_H5AD), embedding="umap")
        series, name = plotter.get_column("leiden")
        assert isinstance(series.dtype, pd.CategoricalDtype)

    def test_pathlib_path(self):
        plotter = ScatterPlotter().set_source(EXAMPLE_H5AD, embedding="umap")
        series, name = plotter.get_column("n_genes")
        assert pd.api.types.is_numeric_dtype(series)

    def test_plot_categorical_runs(self):
        plotter = (
            ScatterPlotter().set_source(EXAMPLE_H5AD, embedding="umap").without_grid()
        )
        p = plotter.plot("leiden")
        assert p is not None

    def test_plot_numeric_runs(self):
        plotter = (
            ScatterPlotter().set_source(EXAMPLE_H5AD, embedding="umap").without_grid()
        )
        p = plotter.plot("n_genes")
        assert p is not None


# ── alternative_id_column label via facade ───────────────────────────────────


class TestAlternativeIdLabel:
    """The example file stores Ensembl ids in ``var['gene_ids']``; using it as
    the ``alternative_id_column`` should produce ``"alt_id (symbol)"`` labels."""

    @pytest.fixture(scope="class")
    def ed_alt(self, facade):
        return EmbeddingData(facade, "umap", alternative_id_column="gene_ids")

    def test_helper_resolves_symbol(self, ed_alt):
        assert ed_alt.alternative_id_for("S100A8") == "ENSG00000143546"

    def test_helper_returns_none_for_obs_column(self, ed_alt):
        # obs columns are not in var.index -> no alternative id
        assert ed_alt.alternative_id_for("n_genes") is None

    def test_helper_returns_none_without_column(self, facade):
        ed = EmbeddingData(facade, "umap")
        assert ed.alternative_id_for("S100A8") is None

    def test_plot_title_shows_alt_id_and_symbol(self, ed_alt):
        sp = ScatterPlotter().set_source(ed_alt).without_grid()
        p = sp.plot("S100A8")
        assert p.labels.get("title", None) == "ENSG00000143546 (S100A8)"

    def test_plot_by_alt_id_labelled_with_symbol(self, ed_alt):
        # addressing a gene by its alternative id still resolves to the symbol
        sp = ScatterPlotter().set_source(ed_alt).without_grid()
        p = sp.plot("ENSG00000143546")
        assert p.labels.get("title", None) == "ENSG00000143546 (S100A8)"

    def test_plot_obs_column_unaffected(self, ed_alt):
        sp = ScatterPlotter().set_source(ed_alt).without_grid()
        p = sp.plot("leiden")
        assert p.labels.get("title", None) == "leiden"

    @staticmethod
    def _colorbar_name(p):
        for s in p.scales:
            n = getattr(s, "name", None)
            if n is not None:
                return n
        return None

    def test_colorbar_keeps_log2_suffix_with_alt_id(self, ed_alt):
        # the alt-id relabel must only affect the title, not the colourbar —
        # the ": log2 expression" stance (and is_gene detection) must survive.
        sp = ScatterPlotter().set_source(ed_alt).without_grid()
        p = sp.plot("S100A8")
        assert self._colorbar_name(p) == "S100A8: log2 expression"

    def test_user_colorbar_title_not_overwritten_by_alt_id(self, ed_alt):
        sp = (
            ScatterPlotter().set_source(ed_alt).without_grid().colormap(title="my cbar")
        )
        p = sp.plot("S100A8")
        assert p.labels.get("title", None) == "ENSG00000143546 (S100A8)"
        assert self._colorbar_name(p) == "my cbar"
