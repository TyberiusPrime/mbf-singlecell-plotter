"""
Unit tests for mbf-singlecell-plotter — no image comparison, fast.

Example data has:
  obs columns: n_genes, total_counts, leiden  (categorical, 9 clusters)
  obsm keys:   X_pca, X_umap
  var index:   gene names like "S100A8", no "NAME ID" spaces
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import shutil

from mbf_singlecell_plotter import (
    EmbeddingData,
    ScatterPlotter,
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
# EmbeddingData: data accessors
# ---------------------------------------------------------------------------

class TestGetColumn:
    def test_categorical_obs_column(self, data):
        series, name = data.get_column(CAT_COL)
        assert len(series) > 0
        assert name == CAT_COL

    def test_numeric_obs_column(self, data):
        series, name = data.get_column(NUMERIC_COL)
        assert len(series) > 0
        assert name == NUMERIC_COL

    def test_gene_by_name(self, data):
        series, name = data.get_column("S100A8")
        assert len(series) > 0
        assert series.dtype in (np.float32, np.float64)

    def test_missing_column_raises(self, data):
        with pytest.raises(KeyError):
            data.get_column("__nonexistent_column__")


class TestGetCoordinateDataframe:
    def test_columns_present(self, data):
        df = data.coordinates()
        assert set(df.columns) == {"x", "y"}

    def test_length_matches_cells(self, data, ad):
        df = data.coordinates()
        assert len(df) == ad.n_obs

    def test_finite_values(self, data):
        df = data.coordinates()
        assert np.isfinite(df["x"].values).all()
        assert np.isfinite(df["y"].values).all()

    def test_index_matches_obs_index(self, data, ad):
        df = data.coordinates()
        assert df.index.equals(ad.obs.index)


class TestGetClusterCenters:
    def test_leiden_centers(self, data):
        centers = data.cluster_centers(CAT_COL)
        assert "x" in centers.columns
        assert "y" in centers.columns
        assert "grid" in centers.columns

    def test_number_of_centers(self, data, ad):
        centers = data.cluster_centers(CAT_COL)
        assert len(centers) == ad.obs[CAT_COL].nunique()

    def test_numeric_raises(self, data):
        with pytest.raises(ValueError, match="numeric"):
            data.cluster_centers("S100A8")

    def test_grid_label_format(self, data):
        centers = data.cluster_centers(CAT_COL)
        for label in centers["grid"]:
            assert isinstance(label, str) and len(label) >= 2
            assert label[0].isalpha(), f"Grid label '{label}' should start with a letter"

    def test_coordinates_within_embedding_range(self, data):
        centers = data.cluster_centers(CAT_COL)
        df = data.coordinates()
        assert centers["x"].between(df["x"].min(), df["x"].max()).all()
        assert centers["y"].between(df["y"].min(), df["y"].max()).all()


class TestPointToGrid:
    def test_returns_tuple_of_two(self, data):
        r = data.point_to_grid(0, 12, 0, 12, 6, 6)
        assert isinstance(r, tuple) and len(r) == 2

    def test_center_point(self, data):
        n = data.grid_size  # 12 by default
        r = data.point_to_grid(0, n, 0, n, n / 2, n / 2)
        assert r is not None

    def test_near_origin(self, data):
        # Small x, small y — should be first column, last row (bottom-left)
        r = data.point_to_grid(0, 12, 0, 12, 0.01, 0.01)
        letter, number = r
        assert letter == "A"
        assert number == 12  # bottom row is highest number when letters_on_vertical=False

    def test_out_of_x_range_raises(self, data):
        with pytest.raises(AssertionError):
            data.point_to_grid(0, 10, 0, 10, 11, 5)

    def test_out_of_y_range_raises(self, data):
        with pytest.raises(AssertionError):
            data.point_to_grid(0, 10, 0, 10, 5, 11)


class TestGetGridCoordinate:
    def test_returns_string(self, data):
        df = data.coordinates()
        x, y = df["x"].median(), df["y"].median()
        label = data.grid_coordinate(x, y)
        assert isinstance(label, str)

    def test_label_starts_with_letter(self, data):
        df = data.coordinates()
        x, y = df["x"].median(), df["y"].median()
        label = data.grid_coordinate(x, y)
        assert label[0].isalpha()

    def test_label_contains_digit(self, data):
        df = data.coordinates()
        x, y = df["x"].median(), df["y"].median()
        label = data.grid_coordinate(x, y)
        assert any(c.isdigit() for c in label)


class TestGetGridCoordinates:
    def test_length(self, data, ad):
        coords = data.grid_coordinates()
        assert len(coords) == ad.n_obs

    def test_all_strings(self, data):
        coords = data.grid_coordinates()
        assert all(isinstance(c, str) for c in coords)


class TestGridLocalHistogram:
    def test_basic(self, data):
        df = data.grid_local_histogram(CAT_COL, min_cells=5)
        assert "category" in df.columns
        assert "frequency" in df.columns
        assert "total" in df.columns

    def test_frequencies_sum_to_one_per_bin(self, data):
        df = data.grid_local_histogram(CAT_COL, min_cells=5)
        grouped = df.groupby(["x", "y"])["frequency"].sum()
        np.testing.assert_allclose(grouped.values, 1.0, atol=1e-6)

    def test_min_cells_filter(self, data):
        df_strict = data.grid_local_histogram(CAT_COL, min_cells=200)
        df_lax = data.grid_local_histogram(CAT_COL, min_cells=1)
        strict_bins = set(zip(df_strict["x"], df_strict["y"]))
        lax_bins = set(zip(df_lax["x"], df_lax["y"]))
        assert strict_bins.issubset(lax_bins)

    def test_bounds_covers_embedding(self, data):
        x_min, x_max, y_min, y_max = data.bounds()
        assert x_min < x_max
        assert y_min < y_max


# ---------------------------------------------------------------------------
# Constructor edge cases
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_invalid_embedding_raises(self, ad):
        with pytest.raises(KeyError):
            EmbeddingData(ad, "nonexistent_embedding")

    def test_grid_size_too_large_raises(self, ad):
        with pytest.raises(ValueError):
            EmbeddingData(ad, "umap", grid_size=27)

    def test_no_borders_by_default(self, ad):
        sp = ScatterPlotter().set_source(ad, "umap")
        assert sp._border_config is None

    def test_umap_embedding_resolved(self, ad):
        data = EmbeddingData(ad, "umap")
        # Should have resolved "umap" to "X_umap"
        assert "umap" in data.embedding

    def test_pca_tuple_embedding(self, ad):
        # The PCA-tuple bug is now fixed — isinstance checked before string concat
        data = EmbeddingData(ad, ("pca", 0, 1))
        df = data.coordinates()
        assert set(df.columns) == {"x", "y"}
        assert len(df) == ad.n_obs

    def test_custom_colors_list(self):
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        sp = ScatterPlotter().colormap_discrete(colors)
        assert sp._cat_colors == colors

    def test_custom_colors_dict(self):
        colors = {"T-cell": "#FF0000", "B-cell": "#00FF00", "NK": "#0000FF"}
        sp = ScatterPlotter().colormap_discrete(colors)
        assert sp._cat_colors == colors

    def _make_bool_ad(self):
        """Minimal AnnData with a boolean obs column and UMAP coords."""
        n = 20
        ad = anndata.AnnData(X=np.zeros((n, 2)))
        ad.obs["flag"] = [i % 2 == 0 for i in range(n)]
        ad.obsm["X_umap"] = np.column_stack([
            np.linspace(0, 1, n), np.linspace(0, 1, n)
        ])
        return ad

    def test_bool_column_expression_is_categorical_str(self):
        """After _build_categorical the expression column must be a string Categorical."""
        ad = self._make_bool_ad()
        sp = ScatterPlotter(ad)
        # Access the built dataframe indirectly via the ggplot data
        p = sp.plot("flag")
        expr = p.data["expression"]
        assert hasattr(expr, "cat"), "expression should be Categorical"
        assert all(isinstance(c, str) for c in expr.cat.categories)


# ---------------------------------------------------------------------------
# panel_size
# ---------------------------------------------------------------------------


class TestPanelSize:
    def test_attributes_set(self, plotter_no_boundary):
        """panel_size() registers a fixed-panel post-draw hook on the ggplot."""
        from mbf_singlecell_plotter.plots import _PlotWithPostDraw
        p = plotter_no_boundary.panel_size(2.0, 3.0).plot("S100A8")
        assert isinstance(p, _PlotWithPostDraw)
        assert len(p._post_draw_fns) == 1

    def test_larger_panel_produces_larger_image(self, plotter_no_boundary):
        """A bigger panel_size yields a bigger rendered figure."""
        import io
        from PIL import Image

        p_small = plotter_no_boundary.panel_size(1, 1).plot("S100A8")
        p_large = plotter_no_boundary.panel_size(4, 4).plot("S100A8")

        buf_small, buf_large = io.BytesIO(), io.BytesIO()
        p_small.save(buf_small, format="png", dpi=100, verbose=False)
        p_large.save(buf_large, format="png", dpi=100, verbose=False)
        buf_small.seek(0)
        buf_large.seek(0)

        img_small = Image.open(buf_small)
        img_large = Image.open(buf_large)

        assert img_large.size[0] > img_small.size[0], "Wider panel → wider image"
        assert img_large.size[1] > img_small.size[1], "Taller panel → taller image"

    def test_same_panel_different_legends_different_figure_size(
        self, plotter_no_boundary
    ):
        """Same panel_size, different legend content → different total figure size."""
        import io
        from PIL import Image

        # numerical → continuous colorbar (narrow)
        p_num = plotter_no_boundary.panel_size(3, 3).plot("S100A8")
        # categorical → discrete legend (wider right margin)
        p_cat = plotter_no_boundary.panel_size(3, 3).plot("leiden")

        buf_num, buf_cat = io.BytesIO(), io.BytesIO()
        p_num.save(buf_num, format="png", dpi=100, verbose=False)
        p_cat.save(buf_cat, format="png", dpi=100, verbose=False)
        buf_num.seek(0)
        buf_cat.seek(0)

        img_num = Image.open(buf_num)
        img_cat = Image.open(buf_cat)

        assert img_num.size != img_cat.size, (
            "Different legend sizes should give different total figure sizes"
        )


# ---------------------------------------------------------------------------
# focus_on (grid-label and float-range forms)
# ---------------------------------------------------------------------------

# Synthetic 2×2 grid: four corner cells at data coords (0,0)..(10,10).
# grid_size=2 → cell_w = cell_h = 5.0
#   Labels (default):          Labels (vertical-letters):
#     A1 (top-left)   B1          1A (top-left)   2A
#     A2 (bottom-left) B2         1B (bottom-left) 2B

@pytest.fixture(scope="module")
def grid2_ad():
    coords = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0]])
    ad = anndata.AnnData(np.zeros((4, 1), dtype=np.float32))
    ad.obs_names = ["c0", "c1", "c2", "c3"]
    ad.var_names = ["g0"]
    ad.obsm["X_test"] = coords
    return ad


@pytest.fixture(scope="module")
def grid2_data(grid2_ad):
    return EmbeddingData(grid2_ad, "test", grid_size=2)


@pytest.fixture(scope="module")
def grid2_data_vl(grid2_ad):
    """Vertical-letters orientation: column=number, row=letter."""
    return EmbeddingData(grid2_ad, "test", grid_size=2, grid_letters_on_vertical=True)


@pytest.fixture(scope="module")
def grid2_plotter(grid2_ad):
    return ScatterPlotter().set_source(grid2_ad, "test").with_grid(grid_size=2)


class TestFocusOn:
    # ── happy-path: default orientation (grid-label form) ────────────────────

    def test_top_left_cell(self, grid2_data):
        b = grid2_data.focus_on("A1", "A1").bounds()
        assert b == pytest.approx((0.0, 5.0, 5.0, 10.0))

    def test_bottom_right_cell(self, grid2_data):
        b = grid2_data.focus_on("B2", "B2").bounds()
        assert b == pytest.approx((5.0, 10.0, 0.0, 5.0))

    def test_full_grid_matches_full_bounds(self, grid2_data):
        b = grid2_data.focus_on("A1", "B2").bounds()
        assert b == pytest.approx(grid2_data.full_bounds())

    def test_top_row_both_columns(self, grid2_data):
        b = grid2_data.focus_on("A1", "B1").bounds()
        assert b == pytest.approx((0.0, 10.0, 5.0, 10.0))

    def test_left_column_both_rows(self, grid2_data):
        b = grid2_data.focus_on("A1", "A2").bounds()
        assert b == pytest.approx((0.0, 5.0, 0.0, 10.0))

    def test_lowercase_accepted(self, grid2_data):
        b = grid2_data.focus_on("a1", "b2").bounds()
        assert b == pytest.approx(grid2_data.full_bounds())

    # ── happy-path: vertical-letters orientation ─────────────────────────────

    def test_vl_top_left_cell(self, grid2_data_vl):
        b = grid2_data_vl.focus_on("1A", "1A").bounds()
        assert b == pytest.approx((0.0, 5.0, 5.0, 10.0))

    def test_vl_bottom_right_cell(self, grid2_data_vl):
        b = grid2_data_vl.focus_on("2B", "2B").bounds()
        assert b == pytest.approx((5.0, 10.0, 0.0, 5.0))

    def test_vl_full_grid(self, grid2_data_vl):
        b = grid2_data_vl.focus_on("1A", "2B").bounds()
        assert b == pytest.approx(grid2_data_vl.full_bounds())

    # ── ScatterPlotter: focus_on_grid() without_grid() raises ────────────────

    def test_without_grid_raises(self, grid2_plotter):
        with pytest.raises(ValueError, match="without_grid"):
            grid2_plotter.without_grid().focus_on_grid("A1", "B2")

    def test_without_grid_error_mentions_focus_on_grid(self, grid2_plotter):
        with pytest.raises(ValueError, match="focus_on_grid"):
            grid2_plotter.without_grid().focus_on_grid("A1", "B2")

    # ── error: bad label format ───────────────────────────────────────────────

    def test_missing_number(self, grid2_data):
        with pytest.raises(ValueError, match="letter\\+number"):
            grid2_data.focus_on("A", "B2")

    def test_missing_letter(self, grid2_data):
        with pytest.raises(ValueError, match="letter\\+number"):
            grid2_data.focus_on("1", "B2")

    def test_vl_wrong_format(self, grid2_data_vl):
        with pytest.raises(ValueError, match="number\\+letter"):
            grid2_data_vl.focus_on("A1", "2B")

    # ── error: out-of-range column/row ───────────────────────────────────────

    def test_column_letter_out_of_range(self, grid2_data):
        with pytest.raises(ValueError, match="A..B"):
            grid2_data.focus_on("C1", "C1")

    def test_column_letter_mentions_grid_size(self, grid2_data):
        with pytest.raises(ValueError, match="grid_size=2"):
            grid2_data.focus_on("Z1", "Z1")

    def test_row_zero_invalid(self, grid2_data):
        with pytest.raises(ValueError, match="1..2"):
            grid2_data.focus_on("A0", "A1")

    def test_row_too_large(self, grid2_data):
        with pytest.raises(ValueError, match="1..2"):
            grid2_data.focus_on("A1", "A3")

    # ── swapped args are silently corrected ──────────────────────────────────

    def test_column_swapped(self, grid2_data):
        assert grid2_data.focus_on("B1", "A2").bounds() == pytest.approx(
            grid2_data.focus_on("A1", "B2").bounds()
        )

    def test_row_swapped(self, grid2_data):
        assert grid2_data.focus_on("A2", "B1").bounds() == pytest.approx(
            grid2_data.focus_on("A1", "B2").bounds()
        )

    def test_both_swapped(self, grid2_data):
        assert grid2_data.focus_on("B2", "A1").bounds() == pytest.approx(
            grid2_data.focus_on("A1", "B2").bounds()
        )


# ---------------------------------------------------------------------------
# ScatterPlotter.background()
# ---------------------------------------------------------------------------


class TestBackground:
    def test_default_off(self):
        p = ScatterPlotter()
        assert p._background_enabled is False

    def test_enable(self):
        p = ScatterPlotter().background()
        assert p._background_enabled is True

    def test_color_and_size(self):
        p = ScatterPlotter().background(color="#AABBCC", dot_size=0.3)
        assert p._background_color == "#AABBCC"
        assert p._background_dot_size == pytest.approx(0.3)

    def test_immutable(self):
        original = ScatterPlotter()
        updated = original.background(color="#FF0000")
        assert original._background_enabled is False
        assert updated._background_color == "#FF0000"

    def test_disable(self):
        p = ScatterPlotter().background().background(enabled=False)
        assert p._background_enabled is False


# ---------------------------------------------------------------------------
# Alternative / fallback sources
# ---------------------------------------------------------------------------


def _make_alt_ad(primary, extra_gene="EXTRA_GENE", extra_obs="extra_annot",
                 shuffle=False, drop=0, superset_extra=0):
    """Build a secondary AnnData sharing primary's obs_names.

    shuffle: reverse the obs_names order so the alternative stores values in a
             different order than the primary (to exercise reindex alignment).
    drop:    drop this many trailing cells from the alternative (subset).
    superset_extra: prepend this many brand-new cells (superset).
    """
    obs_names = list(primary.obs_names)
    n = len(obs_names)

    new_names = obs_names
    if superset_extra:
        new_names = [f"extra_{i}" for i in range(superset_extra)] + obs_names
        n = len(new_names)
    if drop:
        new_names = new_names[: len(new_names) - drop]
        n = len(new_names)

    g = np.arange(n, dtype=np.float32) * 1.0  # distinct, order-sensitive values
    X = g.reshape(-1, 1)
    alt = anndata.AnnData(X=X)
    alt.obs_names = new_names
    alt.var_names = [extra_gene]
    # an obs column only present in the alternative
    alt.obs[extra_obs] = pd.Categorical(["a", "b", "c"][i % 3] for i in range(n))

    if shuffle:
        order = list(range(n))[::-1]
        alt = alt[order].copy()
    return alt


class TestAlternativeSourcesEmbeddingData:
    def test_primary_hit_skips_alternative(self, ad):
        data = EmbeddingData(ad, "umap")
        alt = _make_alt_ad(ad, extra_gene="S100A8")  # S100A8 also in primary
        data2 = data.add_alternative_source(alt)
        series, name = data2.get_column("S100A8")
        # primary's S100A8 values are real expression, not 0..n integers
        assert name == "S100A8"
        assert not np.allclose(series.values, np.arange(len(series)))

    def test_missing_gene_from_alternative(self, ad):
        data = EmbeddingData(ad, "umap")
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data2 = data.add_alternative_source(alt)
        series, name = data2.get_column("EXTRA_GENE")
        assert name == "EXTRA_GENE"
        assert len(series) == ad.n_obs
        # values 0..n-1 from the alternative, in primary's obs order
        assert np.allclose(series.values, np.arange(ad.n_obs))
        # index must be the primary's obs_names
        assert series.index.equals(ad.obs_names)

    def test_missing_obs_column_from_alternative(self, ad):
        data = EmbeddingData(ad, "umap")
        alt = _make_alt_ad(ad, extra_obs="extra_annot")
        data2 = data.add_alternative_source(alt)
        series, name = data2.get_column("extra_annot")
        assert name == "extra_annot"
        assert len(series) == ad.n_obs
        assert series.index.equals(ad.obs_names)

    def test_alternative_reindex_alignment_when_shuffled(self, ad):
        """Alternative stores values in reversed obs order — reindex must fix it."""
        data = EmbeddingData(ad, "umap")
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE", shuffle=True)
        data2 = data.add_alternative_source(alt)
        series, _ = data2.get_column("EXTRA_GENE")
        # After reindex onto primary obs_names, value i must sit at primary position i
        expected = pd.Series(
            np.arange(ad.n_obs, dtype=np.float32), index=ad.obs_names
        )
        np.testing.assert_allclose(series.values, expected.values)
        assert series.index.equals(ad.obs_names)

    def test_alternative_superset_is_subsetted(self, ad):
        """Alternative has extra cells not in primary — they must be dropped."""
        data = EmbeddingData(ad, "umap")
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE", superset_extra=7)
        assert len(alt.obs_names) == ad.n_obs + 7
        data2 = data.add_alternative_source(alt)
        series, _ = data2.get_column("EXTRA_GENE")
        assert len(series) == ad.n_obs
        assert series.index.equals(ad.obs_names)

    def test_alternative_missing_cells_become_nan(self, ad):
        """Primary cells absent from the alternative become NaN."""
        data = EmbeddingData(ad, "umap")
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE", drop=5)
        data2 = data.add_alternative_source(alt)
        series, _ = data2.get_column("EXTRA_GENE")
        assert len(series) == ad.n_obs
        # the dropped cells are the last 5 of the primary
        missing = set(ad.obs_names[-5:]) - set(alt.obs_names)
        assert missing, "precondition: some primary cells missing from alt"
        assert series.loc[list(missing)].isna().all()

    def test_not_found_anywhere_raises(self, ad):
        data = EmbeddingData(ad, "umap")
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data2 = data.add_alternative_source(alt)
        with pytest.raises(KeyError):
            data2.get_column("__does_not_exist__")

    def test_first_alternative_wins(self, ad):
        data = EmbeddingData(ad, "umap")
        alt_a = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        alt_b = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        # give alt_b distinguishable values
        alt_b.X = alt_b.X * 100
        data2 = data.add_alternative_source(alt_a).add_alternative_source(alt_b)
        series, _ = data2.get_column("EXTRA_GENE")
        # alt_a registered first → its values (0..n-1) win
        assert np.allclose(series.values, np.arange(ad.n_obs))

    def test_second_alternative_used_when_first_lacks_it(self, ad):
        data = EmbeddingData(ad, "umap")
        alt_empty = _make_alt_ad(ad, extra_gene="OTHER_GENE")
        alt_real = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data2 = data.add_alternative_source(alt_empty).add_alternative_source(alt_real)
        series, name = data2.get_column("EXTRA_GENE")
        assert name == "EXTRA_GENE"
        assert np.allclose(series.values, np.arange(ad.n_obs))

    def test_immutability_add_alternative_source(self, ad):
        data = EmbeddingData(ad, "umap")
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data2 = data.add_alternative_source(alt)
        assert data.alternative_sources == []
        assert len(data2.alternative_sources) == 1
        # adding another must not mutate data2's list
        data3 = data2.add_alternative_source(alt)
        assert len(data2.alternative_sources) == 1
        assert len(data3.alternative_sources) == 2

    def test_constructor_accepts_alternative_sources(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap", alternative_sources=[alt])
        assert len(data.alternative_sources) == 1
        series, _ = data.get_column("EXTRA_GENE")
        assert np.allclose(series.values, np.arange(ad.n_obs))

    def test_coerce_unwraps_embedding_data(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        # alt has no embedding; add one so it can be wrapped as EmbeddingData
        alt.obsm["X_umap"] = np.column_stack([
            np.linspace(0, 1, alt.n_obs), np.linspace(0, 1, alt.n_obs)
        ])
        alt_data = EmbeddingData(alt, "umap")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt_data)
        series, _ = data.get_column("EXTRA_GENE")
        assert np.allclose(series.values, np.arange(ad.n_obs))


class TestAlternativeSourcesPlotter:
    def test_plotter_add_alternative_source_resolves_gene(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        sp = ScatterPlotter().set_source(ad, "umap").add_alternative_source(alt)
        series, name = sp.get_column("EXTRA_GENE")
        assert name == "EXTRA_GENE"
        assert len(series) == ad.n_obs

    def test_plotter_add_alternative_source_immutable(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        base = ScatterPlotter().set_source(ad, "umap")
        extended = base.add_alternative_source(alt)
        assert len(base._data.alternative_sources) == 0
        assert len(extended._data.alternative_sources) == 1

    def test_plotter_requires_source_first(self, ad):
        with pytest.raises(RuntimeError):
            ScatterPlotter().add_alternative_source(ad)

    def test_plotter_alternatives_survive_with_grid(self, ad):
        """with_grid rebuilds EmbeddingData — alternatives must be preserved."""
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        sp = (
            ScatterPlotter()
            .set_source(ad, "umap")
            .add_alternative_source(alt)
            .with_grid(grid_size=8)
        )
        assert len(sp._data.alternative_sources) == 1
        series, _ = sp.get_column("EXTRA_GENE")
        assert len(series) == ad.n_obs
        assert sp._data._grid_size == 8


class TestAlternativeSourceH5adFacade:
    """H5adFacade implements the same interface as AnnData — verify it is accepted
    as an alternative source and resolves columns correctly."""

    def test_h5ad_facade_accepted_as_alternative(self, ad, tmp_path):
        pytest.importorskip("h5py")
        from mbf_singlecell_plotter import H5adFacade

        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        path = tmp_path / "alt.h5ad"
        alt.write_h5ad(path)

        data = EmbeddingData(ad, "umap").add_alternative_source(H5adFacade(path))
        series, name = data.get_column("EXTRA_GENE")
        assert name == "EXTRA_GENE"
        assert series.index.equals(ad.obs_names)
        assert np.allclose(series.values, np.arange(ad.n_obs))

    def test_path_coerced_to_h5ad_facade(self, ad, tmp_path):
        # only meaningful when h5ad-inspect binary is available
        if not shutil.which("h5ad-inspect"):
            pytest.skip("h5ad-inspect not on PATH")
        from mbf_singlecell_plotter import H5adFacade

        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        path = tmp_path / "alt.h5ad"
        alt.write_h5ad(path)

        data = EmbeddingData(ad, "umap").add_alternative_source(str(path))
        assert isinstance(data.alternative_sources[0].ad, H5adFacade)
        series, _ = data.get_column("EXTRA_GENE")
        assert np.allclose(series.values, np.arange(ad.n_obs))


# ---------------------------------------------------------------------------
# Alternative sources: naming + tuple routing
# ---------------------------------------------------------------------------


class TestAlternativeSourceNames:
    def test_register_with_name(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="imp")
        assert data.alternative_sources[0].name == "imp"

    def test_tuple_routes_to_named_source(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="imp")
        series, name = data.get_column(("imp", "EXTRA_GENE"))
        assert name == "EXTRA_GENE"
        assert series.index.equals(ad.obs_names)
        assert np.allclose(series.values, np.arange(ad.n_obs))

    def test_tuple_reindexes_shuffled_alternative(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE", shuffle=True)
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="imp")
        series, _ = data.get_column(("imp", "EXTRA_GENE"))
        expected = pd.Series(
            np.arange(ad.n_obs, dtype=np.float32), index=ad.obs_names
        )
        np.testing.assert_allclose(series.values, expected.values)

    def test_tuple_routes_to_named_not_primary(self, ad):
        """Tuple addresses the named source even when primary also has the column."""
        alt = _make_alt_ad(ad, extra_gene="S100A8")  # S100A8 is also in primary
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="v2")
        series, name = data.get_column(("v2", "S100A8"))
        assert name == "S100A8"
        # alt's values are 0..n-1 (from _make_alt_ad), distinct from primary expr
        assert np.allclose(series.values, np.arange(ad.n_obs))

    def test_tuple_unknown_name_raises(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="imp")
        with pytest.raises(KeyError):
            data.get_column(("nope", "EXTRA_GENE"))

    def test_tuple_missing_column_in_named_source_raises(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="imp")
        with pytest.raises(KeyError):
            data.get_column(("imp", "NOT_THERE"))

    def test_tuple_bad_shape_raises(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="imp")
        with pytest.raises(KeyError):
            data.get_column(("imp", "EXTRA_GENE", "extra"))

    def test_plain_string_fallback_unaffected_by_names(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="imp")
        # plain string still falls back to the alternative
        series, _ = data.get_column("EXTRA_GENE")
        assert np.allclose(series.values, np.arange(ad.n_obs))
        # and primary still wins for shared columns
        series2, _ = data.get_column("S100A8")
        assert not np.allclose(series2.values, np.arange(ad.n_obs))

    def test_unnamed_source_skipped_by_tuple_routing(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt)  # no name
        with pytest.raises(KeyError):
            data.get_column(("anything", "EXTRA_GENE"))
        # but the fallback search still finds it
        series, _ = data.get_column("EXTRA_GENE")
        assert np.allclose(series.values, np.arange(ad.n_obs))

    def test_duplicate_name_raises(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="x")
        with pytest.raises(ValueError):
            data.add_alternative_source(_make_alt_ad(ad), name="x")

    def test_constructor_accepts_named_tuples(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap", alternative_sources=[("imp", alt)])
        assert data.alternative_sources[0].name == "imp"
        series, _ = data.get_column(("imp", "EXTRA_GENE"))
        assert np.allclose(series.values, np.arange(ad.n_obs))


class TestAlternativeSourceNamesPlotter:
    def test_plotter_named_and_tuple_routing(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        sp = (
            ScatterPlotter()
            .set_source(ad, "umap")
            .add_alternative_source(alt, name="imp")
        )
        series, name = sp.get_column(("imp", "EXTRA_GENE"))
        assert name == "EXTRA_GENE"
        assert np.allclose(series.values, np.arange(ad.n_obs))

    def test_plot_accepts_tuple_column(self, ad):
        import plotnine as p9

        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        sp = (
            ScatterPlotter()
            .set_source(ad, "umap")
            .add_alternative_source(alt, name="imp")
        )
        p = sp.plot(("imp", "EXTRA_GENE"))
        assert isinstance(p, p9.ggplot)
        # pulls from the named alternative (values 0..n-1), reindexed to primary
        assert "expression" in p.data.columns
        assert set(p.data["expression"]).issubset(set(np.arange(ad.n_obs)))



