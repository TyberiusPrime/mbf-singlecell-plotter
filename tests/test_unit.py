"""
Unit tests for mbf-singlecell-plotter — no image comparison, fast.

Example data has:
  obs columns: n_genes, total_counts, leiden  (categorical, 9 clusters)
  obsm keys:   X_pca, X_umap
  var index:   gene names like "S100A8", no "NAME ID" spaces
"""

import numpy as np
import pandas as pd
import pytest

from mbf_singlecell_plotter import (
    ScanpyPlotter,
    map_to_integers,
    unmap,
)

# The categorical obs column in the example data
CAT_COL = "leiden"
NUMERIC_COL = "n_genes"


# ---------------------------------------------------------------------------
# map_to_integers / unmap
# ---------------------------------------------------------------------------

class TestMapToIntegers:
    def test_full_range(self):
        s = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
        result = map_to_integers(s, upper=100)
        assert result.iloc[0] == 0
        assert result.iloc[-1] == 99

    def test_explicit_min_max(self):
        s = pd.Series([2.0, 3.0, 4.0])
        result = map_to_integers(s, upper=10, min=0.0, max=10.0)
        # 2 → 0.2*9=1
        assert result.iloc[0] == 1

    def test_dtype_is_int(self):
        s = pd.Series(np.linspace(0, 1, 50))
        result = map_to_integers(s, upper=256)
        assert np.issubdtype(result.dtype, np.integer)

    def test_upper_bound_not_exceeded(self):
        s = pd.Series(np.linspace(0, 1, 200))
        result = map_to_integers(s, upper=100)
        assert result.max() <= 99


class TestUnmap:
    def test_roundtrip(self):
        s = pd.Series([0.0, 25.0, 50.0, 75.0, 100.0])
        res = 1000
        mapped = map_to_integers(s, upper=res)
        recovered = unmap(mapped.astype(float), s, res)
        # approximate due to integer rounding
        np.testing.assert_allclose(recovered, s, atol=0.2)

    def test_monotone(self):
        s = pd.Series(np.linspace(0, 10, 20))
        res = 100
        mapped = map_to_integers(s, upper=res)
        recovered = unmap(mapped.astype(float), s, res)
        assert (recovered.diff().dropna() >= 0).all()


# ---------------------------------------------------------------------------
# ScanpyPlotter: data accessors
# ---------------------------------------------------------------------------

class TestGetColumn:
    def test_categorical_obs_column(self, plotter_no_boundary):
        series, name = plotter_no_boundary.get_column(CAT_COL)
        assert len(series) > 0
        assert name == CAT_COL

    def test_numeric_obs_column(self, plotter_no_boundary):
        series, name = plotter_no_boundary.get_column(NUMERIC_COL)
        assert len(series) > 0
        assert name == NUMERIC_COL

    def test_gene_by_name(self, plotter_no_boundary):
        series, name = plotter_no_boundary.get_column("S100A8")
        assert len(series) > 0
        assert series.dtype in (np.float32, np.float64)

    def test_missing_column_raises(self, plotter_no_boundary):
        with pytest.raises(KeyError):
            plotter_no_boundary.get_column("__nonexistent_column__")


class TestGetCoordinateDataframe:
    def test_columns_present(self, plotter_no_boundary):
        df = plotter_no_boundary.get_coordinate_dataframe()
        assert set(df.columns) == {"x", "y"}

    def test_length_matches_cells(self, plotter_no_boundary, ad):
        df = plotter_no_boundary.get_coordinate_dataframe()
        assert len(df) == ad.n_obs

    def test_finite_values(self, plotter_no_boundary):
        df = plotter_no_boundary.get_coordinate_dataframe()
        assert np.isfinite(df["x"].values).all()
        assert np.isfinite(df["y"].values).all()

    def test_index_matches_obs_index(self, plotter_no_boundary, ad):
        df = plotter_no_boundary.get_coordinate_dataframe()
        assert df.index.equals(ad.obs.index)


class TestGetCellTypeCategories:
    def test_returns_categories(self, plotter_no_boundary):
        # get_cell_type_categories() requires cell_type_column to be set,
        # but plotter_no_boundary has cell_type_column=None — call with explicit pdf
        df = plotter_no_boundary.get_coordinate_dataframe()
        df = df.assign(cell_type=plotter_no_boundary.ad.obs[CAT_COL].values)
        cats = plotter_no_boundary.get_cell_type_categories(df)
        assert len(cats) > 0

    def test_natural_sort_order(self, plotter_no_boundary):
        df = plotter_no_boundary.get_coordinate_dataframe()
        df = df.assign(cell_type=plotter_no_boundary.ad.obs[CAT_COL].values)
        cats = list(plotter_no_boundary.get_cell_type_categories(df))
        from natsort import natsorted
        assert cats == natsorted(cats)

    def test_via_plotter_with_column(self, plotter):
        """plotter has cell_type_column=leiden — requires scikit-image (skipped if absent)."""
        cats = plotter.get_cell_type_categories()
        assert len(cats) > 0


class TestGetClusterCenters:
    def test_leiden_centers(self, plotter_no_boundary):
        centers = plotter_no_boundary.get_cluster_centers(CAT_COL)
        assert "x" in centers.columns
        assert "y" in centers.columns
        assert "grid" in centers.columns

    def test_number_of_centers(self, plotter_no_boundary):
        centers = plotter_no_boundary.get_cluster_centers(CAT_COL)
        assert len(centers) == plotter_no_boundary.ad.obs[CAT_COL].nunique()

    def test_numeric_raises(self, plotter_no_boundary):
        with pytest.raises(ValueError, match="numeric"):
            plotter_no_boundary.get_cluster_centers("S100A8")

    def test_grid_label_format(self, plotter_no_boundary):
        centers = plotter_no_boundary.get_cluster_centers(CAT_COL)
        for label in centers["grid"]:
            assert isinstance(label, str) and len(label) >= 2
            assert label[0].isalpha(), f"Grid label '{label}' should start with a letter"

    def test_coordinates_within_embedding_range(self, plotter_no_boundary):
        centers = plotter_no_boundary.get_cluster_centers(CAT_COL)
        df = plotter_no_boundary.get_coordinate_dataframe()
        assert centers["x"].between(df["x"].min(), df["x"].max()).all()
        assert centers["y"].between(df["y"].min(), df["y"].max()).all()


class TestPointToGrid:
    def test_returns_tuple_of_two(self, plotter_no_boundary):
        r = plotter_no_boundary.point_to_grid(0, 12, 0, 12, 6, 6)
        assert isinstance(r, tuple) and len(r) == 2

    def test_center_point(self, plotter_no_boundary):
        n = plotter_no_boundary.grid_size  # 12 by default
        # Center of the space should land in a middle cell
        r = plotter_no_boundary.point_to_grid(0, n, 0, n, n / 2, n / 2)
        assert r is not None

    def test_near_origin(self, plotter_no_boundary):
        # Small x, small y — should be first column, last row (bottom-left)
        r = plotter_no_boundary.point_to_grid(0, 12, 0, 12, 0.01, 0.01)
        letter, number = r
        assert letter == "A"
        assert number == 12  # bottom row is highest number when letters_on_vertical=False

    def test_out_of_x_range_raises(self, plotter_no_boundary):
        with pytest.raises(AssertionError):
            plotter_no_boundary.point_to_grid(0, 10, 0, 10, 11, 5)

    def test_out_of_y_range_raises(self, plotter_no_boundary):
        with pytest.raises(AssertionError):
            plotter_no_boundary.point_to_grid(0, 10, 0, 10, 5, 11)


class TestGetGridCoordinate:
    def test_returns_string(self, plotter_no_boundary):
        df = plotter_no_boundary.get_coordinate_dataframe()
        # Use the median point to avoid edge-of-range issues
        x, y = df["x"].median(), df["y"].median()
        label = plotter_no_boundary.get_grid_coordinate(x, y)
        assert isinstance(label, str)

    def test_label_starts_with_letter(self, plotter_no_boundary):
        df = plotter_no_boundary.get_coordinate_dataframe()
        x, y = df["x"].median(), df["y"].median()
        label = plotter_no_boundary.get_grid_coordinate(x, y)
        assert label[0].isalpha()

    def test_label_contains_digit(self, plotter_no_boundary):
        df = plotter_no_boundary.get_coordinate_dataframe()
        x, y = df["x"].median(), df["y"].median()
        label = plotter_no_boundary.get_grid_coordinate(x, y)
        assert any(c.isdigit() for c in label)


class TestGetGridCoordinates:
    # NOTE: get_grid_coordinates() ignores x_limits/y_limits (params accepted but unused),
    # and point_to_grid() has an IndexError when a cell coordinate rounds to index == grid_size.
    # These tests document the current buggy behaviour with xfail markers.

    @pytest.mark.xfail(reason="point_to_grid IndexError on cells at embedding boundary", strict=False)
    def test_length(self, plotter_no_boundary, ad):
        coords = plotter_no_boundary.get_grid_coordinates()
        assert len(coords) == ad.n_obs

    @pytest.mark.xfail(reason="point_to_grid IndexError on cells at embedding boundary", strict=False)
    def test_all_strings(self, plotter_no_boundary):
        coords = plotter_no_boundary.get_grid_coordinates()
        assert all(isinstance(c, str) for c in coords)


class TestGridLocalHistogram:
    def test_basic(self, plotter_no_boundary):
        df, bbox = plotter_no_boundary.grid_local_histogram(CAT_COL, min_cells=5)
        assert "category" in df.columns
        assert "frequency" in df.columns
        assert "total" in df.columns
        assert len(bbox) == 4

    def test_frequencies_sum_to_one_per_bin(self, plotter_no_boundary):
        df, _ = plotter_no_boundary.grid_local_histogram(CAT_COL, min_cells=5)
        grouped = df.groupby(["x", "y"])["frequency"].sum()
        np.testing.assert_allclose(grouped.values, 1.0, atol=1e-6)

    def test_min_cells_filter(self, plotter_no_boundary):
        df_strict, _ = plotter_no_boundary.grid_local_histogram(CAT_COL, min_cells=200)
        df_lax, _ = plotter_no_boundary.grid_local_histogram(CAT_COL, min_cells=1)
        strict_bins = set(zip(df_strict["x"], df_strict["y"]))
        lax_bins = set(zip(df_lax["x"], df_lax["y"]))
        assert strict_bins.issubset(lax_bins)

    def test_bbox_covers_embedding(self, plotter_no_boundary):
        _, bbox = plotter_no_boundary.grid_local_histogram(CAT_COL)
        x_min, x_max, y_min, y_max = bbox
        assert x_min < x_max
        assert y_min < y_max


# ---------------------------------------------------------------------------
# Constructor edge cases
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_invalid_embedding_raises(self, ad):
        with pytest.raises(KeyError):
            ScanpyPlotter(ad, "nonexistent_embedding")

    def test_grid_size_too_large_raises(self, ad):
        with pytest.raises(ValueError):
            ScanpyPlotter(ad, "umap", cell_type_column=None, grid_size=27)

    def test_no_cell_type_column(self, ad):
        sp = ScanpyPlotter(ad, "umap", cell_type_column=None)
        assert not hasattr(sp, "boundary_dataframe")

    def test_umap_embedding_resolved(self, ad):
        sp = ScanpyPlotter(ad, "umap", cell_type_column=None)
        # Should have resolved "umap" to "X_umap"
        assert "umap" in sp.embedding

    @pytest.mark.xfail(reason="Constructor does 'X_' + embedding which crashes for tuple", strict=True)
    def test_pca_tuple_embedding(self, ad):
        # The constructor tries `"X_" + embedding` before checking isinstance(tuple),
        # so passing a PCA tuple currently raises TypeError.
        sp = ScanpyPlotter(ad, ("pca", 0, 1), cell_type_column=None)
        df = sp.get_coordinate_dataframe()
        assert set(df.columns) == {"x", "y"}
        assert len(df) == ad.n_obs

    def test_custom_colors_list(self, ad):
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        sp = ScanpyPlotter(ad, "umap", cell_type_column=None, colors=colors)
        import matplotlib.colors as mcolors
        assert isinstance(sp.cell_type_color_map, mcolors.ListedColormap)
