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

    def test_bool_column_works(self, data):
        """Bool columns are discrete categories — must not raise."""
        df = data.grid_local_histogram("bool", min_cells=5)
        assert "category" in df.columns
        assert set(df["category"].unique()).issubset({"True", "False"})

    def test_bool_frequencies_sum_to_one_per_bin(self, data):
        df = data.grid_local_histogram("bool", min_cells=5)
        grouped = df.groupby(["x", "y"])["frequency"].sum()
        np.testing.assert_allclose(grouped.values, 1.0, atol=1e-6)

    def test_numeric_still_raises(self, data):
        with pytest.raises(ValueError, match="category types only"):
            data.grid_local_histogram(NUMERIC_COL)


# ---------------------------------------------------------------------------
# prepare_density_df — 2D density transform (incl. faceting)
# ---------------------------------------------------------------------------


class TestPrepareDensityDf:
    def test_basic_columns(self, data):
        from mbf_singlecell_plotter.transforms import prepare_density_df

        df = prepare_density_df(data, bins=80)
        assert set(["x", "y", "density", "x_width", "y_width"]).issubset(df.columns)
        assert "facet" not in df.columns
        assert (df["density"] > 0).all()

    def test_facet_adds_facet_column(self, data):
        from mbf_singlecell_plotter.transforms import prepare_density_df

        facet_vals, _ = data.get_column(CAT_COL)
        df = prepare_density_df(data, bins=80, facet=facet_vals)
        assert "facet" in df.columns
        assert isinstance(df["facet"].dtype, pd.CategoricalDtype)
        # categorical order preserved from the source column
        assert list(df["facet"].cat.categories) == list(facet_vals.cat.categories)

    def test_facet_shared_global_edges(self, data):
        """Every facet panel must share the same coordinate frame (global edges)."""
        from mbf_singlecell_plotter.transforms import prepare_density_df

        facet_vals, _ = data.get_column(CAT_COL)
        df = prepare_density_df(data, bins=80, facet=facet_vals)
        single = prepare_density_df(data, bins=80)
        # bin width (and thus the shared edge spacing) matches the un-faceted run
        assert df["x_width"].iloc[0] == pytest.approx(single["x_width"].iloc[0])
        assert df["y_width"].iloc[0] == pytest.approx(single["y_width"].iloc[0])
        # every facet's occupied bin centres lie on the shared global grid
        global_x = np.isclose(
            df["x"].values[:, None], single["x"].values[None, :], atol=1e-9
        ).any(axis=1)
        global_y = np.isclose(
            df["y"].values[:, None], single["y"].values[None, :], atol=1e-9
        ).any(axis=1)
        assert global_x.all() and global_y.all()

    def test_facet_cell_count_partitioned(self, data, ad):
        """Per-facet density sums equal the number of cells in that facet."""
        from mbf_singlecell_plotter.transforms import prepare_density_df

        facet_vals, _ = data.get_column(CAT_COL)
        df = prepare_density_df(data, bins=120, facet=facet_vals)
        coords = data.coordinates()
        true_counts = coords.groupby(
            facet_vals.reindex(coords.index), observed=True
        ).size()
        for cat, sub in df.groupby("facet", observed=True):
            assert sub["density"].sum() == pytest.approx(true_counts[cat])

    def test_facet_preserves_total(self, data):
        """Concatenated facet densities sum to the same total as the un-faceted run."""
        from mbf_singlecell_plotter.transforms import prepare_density_df

        facet_vals, _ = data.get_column(CAT_COL)
        facet_total = prepare_density_df(data, bins=80, facet=facet_vals)["density"].sum()
        plain_total = prepare_density_df(data, bins=80)["density"].sum()
        assert facet_total == pytest.approx(plain_total)


class TestPrepareDensityDf2D:
    """prepare_density_df 2-D (facet_grid) faceting."""

    def test_adds_row_and_col_columns(self, data):
        from mbf_singlecell_plotter.transforms import prepare_density_df

        row_vals, _ = data.get_column("bool")
        col_vals, _ = data.get_column("coarse")
        df = prepare_density_df(
            data, bins=80, facet_row=row_vals, facet_col=col_vals
        )
        assert "facet_row" in df.columns
        assert "facet_col" in df.columns
        assert isinstance(df["facet_row"].dtype, pd.CategoricalDtype)
        assert isinstance(df["facet_col"].dtype, pd.CategoricalDtype)

    def test_grid_partitions_cells(self, data):
        """Each (row, col) cell combination's density sums to its cell count."""
        from mbf_singlecell_plotter.transforms import prepare_density_df

        row_vals, _ = data.get_column("bool")
        col_vals, _ = data.get_column("coarse")
        df = prepare_density_df(
            data, bins=80, facet_row=row_vals, facet_col=col_vals
        )
        coords = data.coordinates()
        r = row_vals.reindex(coords.index)
        c = col_vals.reindex(coords.index)
        true = coords.groupby([r, c], observed=True).size()
        for (rv, cv), sub in df.groupby(
            ["facet_row", "facet_col"], observed=True
        ):
            assert sub["density"].sum() == pytest.approx(true[(rv, cv)])

    def test_2d_preserves_total(self, data):
        """Concatenated 2-D grid densities sum to the un-faceted total."""
        from mbf_singlecell_plotter.transforms import prepare_density_df

        row_vals, _ = data.get_column("bool")
        col_vals, _ = data.get_column("coarse")
        grid_total = prepare_density_df(
            data, bins=80, facet_row=row_vals, facet_col=col_vals
        )["density"].sum()
        plain_total = prepare_density_df(data, bins=80)["density"].sum()
        assert grid_total == pytest.approx(plain_total)

    def test_rows_only_grid(self, data):
        """facet_row without facet_col still partitions by the single variable."""
        from mbf_singlecell_plotter.transforms import prepare_density_df

        row_vals, _ = data.get_column("coarse")
        df = prepare_density_df(data, bins=80, facet_row=row_vals)
        assert "facet_row" in df.columns
        assert "facet_col" not in df.columns
        assert set(df["facet_row"].cat.categories) == {"L", "M", "H"}


class TestPlotDensityParity:
    """plot_density honours focus_on / focus_on_grid and title like plot()."""

    def test_default_title_is_embedding_name(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.plot_density()
        assert isinstance(p, p9.ggplot)
        # "X_umap" → "umap", matching the embedding-label convention
        assert p.labels.get("title", None) == "umap"

    def test_title_override(self, plotter_no_boundary):
        p = plotter_no_boundary.title("my density").plot_density()
        assert p.labels.get("title", None) == "my density"

    def test_focus_on_grid_applies_coord_cartesian(self, plotter_no_boundary, data):
        import plotnine as p9

        x_min, x_max, y_min, y_max = plotter_no_boundary.focus_on_grid(
            "K12", "G9"
        )._data.bounds()
        p = plotter_no_boundary.focus_on_grid("K12", "G9").plot_density()
        assert isinstance(p.coordinates, p9.coords.coord_cartesian)
        assert p.coordinates.limits.x == (x_min, x_max)
        assert p.coordinates.limits.y == (y_min, y_max)

    def test_no_focus_leaves_limits_unset(self, plotter_no_boundary):
        # plotnine defaults every ggplot to coord_cartesian(), so the signal
        # that focus is *not* active is that limits are unset (None).
        p = plotter_no_boundary.plot_density()
        assert p.coordinates.limits.x is None
        assert p.coordinates.limits.y is None

    def test_focus_float_range(self, plotter_no_boundary, data):
        import plotnine as p9

        coords = data.coordinates()
        xlo, xhi = float(coords["x"].min()), float(coords["x"].median())
        ylo, yhi = float(coords["y"].min()), float(coords["y"].median())
        p = (
            plotter_no_boundary
            .focus_on(x=(xlo, xhi), y=(ylo, yhi))
            .plot_density()
        )
        assert isinstance(p.coordinates, p9.coords.coord_cartesian)
        assert p.coordinates.limits.x == (xlo, xhi)
        assert p.coordinates.limits.y == (ylo, yhi)


# ---------------------------------------------------------------------------
# plot_histogram — global value_counts bar plot
# ---------------------------------------------------------------------------


class TestPlotHistogram:
    def test_returns_ggplot_and_default_title(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.plot_histogram(CAT_COL)
        assert isinstance(p, p9.ggplot)
        assert p.labels.get("title", None) == CAT_COL

    def test_title_override(self, plotter_no_boundary):
        p = plotter_no_boundary.title("counts!").plot_histogram(CAT_COL)
        assert p.labels.get("title", None) == "counts!"

    def test_counts_match_value_counts(self, plotter_no_boundary, ad):
        p = plotter_no_boundary.plot_histogram(CAT_COL)
        vc = ad.obs[CAT_COL].astype(str).value_counts(sort=False)
        got = p.data.set_index("category")["count"]
        assert got.sort_index().equals(vc.sort_index())
        # total must equal the number of cells
        assert p.data["count"].sum() == ad.n_obs

    def test_numeric_column_raises(self, plotter_no_boundary):
        with pytest.raises(ValueError, match="numeric"):
            plotter_no_boundary.plot_histogram(NUMERIC_COL)

    def test_uses_configured_colors(self, plotter_no_boundary):
        """Rendered bars must use the colormap_discrete palette."""
        import matplotlib.colors as mc

        colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff",
                  "#00ffff", "#000000", "#ffffff", "#888888"]
        p = plotter_no_boundary.colormap_discrete(colors).plot_histogram(CAT_COL)
        ax = p.draw().axes[0]
        # geom_col bars are a PolyCollection in current plotnine
        from matplotlib.collections import PolyCollection

        fcs = set()
        for pc in ax.collections:
            if isinstance(pc, PolyCollection):
                fcs.update(mc.to_hex(rgba) for rgba in pc.get_facecolors())
        assert fcs, "no bars drawn"
        assert fcs.issubset(set(colors))

    def test_fill_legend_hidden(self, plotter_no_boundary):
        p = plotter_no_boundary.plot_histogram(CAT_COL)
        scale = next(s for s in p.scales if "fill" in s.aesthetics)
        assert scale.guide is None

    def test_facet_counts_partition_cells(self, plotter_no_boundary, ad):
        p = plotter_no_boundary.facet("coarse", n_col=3).plot_histogram(CAT_COL)
        assert "facet" in p.data.columns
        # totals equal the number of cells
        assert p.data["count"].sum() == ad.n_obs
        # per-facet counts match the ground truth
        truth = (
            ad.obs.assign(_c=ad.obs[CAT_COL].astype(str))
            .groupby("coarse", observed=True)["_c"]
            .value_counts()
            .reset_index(name="count")
        )
        truth.columns = ["facet", "category", "count"]
        merged = p.data.merge(truth, on=["facet", "category"])
        assert merged["count_x"].equals(merged["count_y"])

    def test_facet_2d_counts_partition_cells(self, plotter_no_boundary, ad):
        p = plotter_no_boundary.facet_2d("bool", "coarse").plot_histogram(CAT_COL)
        assert {"facet_row", "facet_col"}.issubset(p.data.columns)
        assert p.data["count"].sum() == ad.n_obs

    def test_facet_uses_facet_wrap(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.facet("coarse", n_col=3).plot_histogram(CAT_COL)
        assert type(p.facet) is p9.facet_wrap

    def test_normalize_divides_by_factor(self, plotter_no_boundary, ad):
        """Counts divided by n_obs should sum to 1.0 and the max category hits exactly its fraction."""
        total = ad.n_obs
        p = plotter_no_boundary.plot_histogram(CAT_COL, normalize=total)
        # each count is now a fraction of total
        assert abs(p.data["count"].sum() - 1.0) < 1e-9

    def test_normalize_single_value_hits_one(self, plotter_no_boundary, ad):
        """Normalizing by a specific bin's count makes that bin exactly 1.0."""
        vc = ad.obs[CAT_COL].astype(str).value_counts(sort=False)
        cat_name, target_val = str(vc.index[0]), int(vc.iloc[0])
        p = plotter_no_boundary.plot_histogram(CAT_COL, normalize=target_val)
        assert abs(p.data.loc[p.data["category"] == cat_name, "count"].iloc[0] - 1.0) < 1e-9

    def test_normalize_none_keeps_raw_counts(self, plotter_no_boundary, ad):
        """Explicit normalize=None behaves identically to omitting it."""
        p = plotter_no_boundary.plot_histogram(CAT_COL, normalize=None)
        assert p.data["count"].sum() == ad.n_obs


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

    def test_panel_size_respected_in_composition(self, plotter_no_boundary):
        """panel_size() must be honoured when plots are joined with `/` or `|`.

        plotnine's Compose.save/draw bypass ggplot.save_helper, so the
        post-draw hooks (incl. _apply_fixed_panel) used to be skipped for
        stacked / side-by-side plots.  After the fix each member plot's panel
        is resized to the requested size.
        """
        from mbf_singlecell_plotter.plots import _first_panel_axes

        def _panel_inches(fig):
            ax = _first_panel_axes(fig)
            pos = ax.get_position()
            fw, fh = fig.get_size_inches()
            return pos.width * fw, pos.height * fh

        # ── vertical stack (p1 / p2) ──────────────────────────────────────
        p1 = plotter_no_boundary.panel_size(3, 3).plot("S100A8")
        p2 = plotter_no_boundary.panel_size(3, 3).plot("S100A8")
        fig = (p1 / p2).draw()
        w, h = _panel_inches(fig)
        assert abs(w - 3.0) < 0.05, f"stacked panel width {w:.2f} != 3.0"
        assert abs(h - 3.0) < 0.05, f"stacked panel height {h:.2f} != 3.0"

        # ── side-by-side (p1 | p2) ────────────────────────────────────────
        p3 = plotter_no_boundary.panel_size(2, 4).plot("S100A8")
        p4 = plotter_no_boundary.panel_size(2, 4).plot("S100A8")
        fig2 = (p3 | p4).draw()
        w2, h2 = _panel_inches(fig2)
        assert abs(w2 - 2.0) < 0.05, f"beside panel width {w2:.2f} != 2.0"
        assert abs(h2 - 4.0) < 0.05, f"beside panel height {h2:.2f} != 4.0"

    def test_composition_panel_size_via_save(self, plotter_no_boundary):
        """panel_size also works through the normal .save() path of a stack."""
        import io
        from PIL import Image

        p1 = plotter_no_boundary.panel_size(3, 3).plot("S100A8")
        p2 = plotter_no_boundary.panel_size(3, 3).plot("S100A8")

        buf = io.BytesIO()
        (p1 / p2).save(buf, format="png", dpi=100, verbose=False)
        buf.seek(0)
        img = Image.open(buf)
        # Two 3x3in panels stacked (plus per-plot margins) → clearly taller
        # than wide, and tall enough that each panel is ~3in at 100 dpi.
        assert img.size[1] > img.size[0], "stacked image should be taller than wide"
        # 2 panels × 3in × 100dpi = 600px of panel alone.
        assert img.size[1] >= 600, (
            f"stacked image height {img.size[1]}px too small for 2×3in panels"
        )


class TestFacet2D:
    """facet_2d() API and its mutual exclusivity with facet()."""

    def test_facet_2d_sets_row_and_col(self, plotter_no_boundary):
        pt = plotter_no_boundary.facet_2d("bool", "coarse")
        assert pt._facet_row_variable == "bool"
        assert pt._facet_col_variable == "coarse"
        assert pt._facet_variable is None

    def test_facet_after_2d_clears_2d(self, plotter_no_boundary):
        pt = plotter_no_boundary.facet_2d("bool", "coarse").facet("leiden", n_col=3)
        assert pt._facet_variable == "leiden"
        assert pt._facet_row_variable is None
        assert pt._facet_col_variable is None

    def test_2d_after_facet_clears_1d(self, plotter_no_boundary):
        pt = plotter_no_boundary.facet("leiden", n_col=3).facet_2d("bool", "coarse")
        assert pt._facet_row_variable == "bool"
        assert pt._facet_col_variable == "coarse"
        assert pt._facet_variable is None

    def test_unfacet_clears_2d(self, plotter_no_boundary):
        pt = plotter_no_boundary.facet_2d("bool", "coarse").unfacet()
        assert pt._facet_variable is None
        assert pt._facet_row_variable is None
        assert pt._facet_col_variable is None

    def test_2d_uses_facet_grid(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.facet_2d("bool", "coarse").plot("S100A8")
        assert type(p.facet) is p9.facet_grid

    def test_1d_uses_facet_wrap(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.facet("leiden", n_col=3).plot("S100A8")
        assert type(p.facet) is p9.facet_wrap

    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason=(
            "plotnine facet_grid(drop=True) mishandles NaN in a facetting "
            "variable: the NaN is added as an extra level when building the "
            "cartesian product of panels (len(layout) > nrow*ncol), but "
            "ninteraction(drop=True) does not assign it a ROW/COL index, so "
            "only nrow*ncol axes are created. Strips.setup then indexes "
            "self.axs out of range. facet_2d() uses facet_grid with the "
            "default drop=True, so any NaN in the row or column variable "
            "crashes the draw."
        ),
    )
    def test_2d_facet_variable_with_nan_draws(self):
        """2-D facetting where the column variable has NaN must still draw.

        Reproduces the IndexError at ``plotnine/facets/strips.py`` line 184
        (``ax = self.axs[layout_info.panel_index]``) reported when the 2nd
        axis has more than 2 values. The actual trigger is a NaN in a
        facetting variable combined with the default ``drop=True``.
        """
        np.random.seed(0)
        n = 400
        ad = anndata.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        ad.obs_names = [f"c{i}" for i in range(n)]
        ad.var_names = ["gene0"]
        ad.obsm["X_umap"] = np.column_stack([
            np.random.rand(n), np.random.rand(n)
        ])
        # Row variable: a clean categorical.
        ad.obs["row_var"] = pd.Categorical(
            np.random.choice(["a", "b", "c"], n)
        )
        # Column variable: 3 categories with some unannotated (NaN) cells.
        col = np.random.choice(
            ["Missense", "Synonymous", "Frameshift"], n
        ).astype(object)
        col[:25] = None
        ad.obs["col_var"] = pd.Categorical(col)

        p = (
            ScatterPlotter()
            .set_source(ad, "umap")
            .facet_2d("row_var", "col_var")
            .plot("gene0")
        )
        p.draw()


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

    def test_rejects_dict_source(self, ad):
        data = EmbeddingData(ad, "umap")
        with pytest.raises(TypeError):
            data.add_alternative_source({"x": lambda d: d.get_column("n_genes").series})

    def test_constructor_rejects_dict_source(self, ad):
        with pytest.raises(TypeError):
            EmbeddingData(
                ad,
                "umap",
                alternative_sources=[{"x": lambda d: d.get_column("n_genes").series}],
            )


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


# ---------------------------------------------------------------------------
# Derived sources: on-demand computed columns
# ---------------------------------------------------------------------------


class TestDerivedSources:
    def test_plain_string_computed_on_demand(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {"double_genes": lambda d: d.get_column("n_genes").series * 2}
        )
        series, name = data.get_column("double_genes")
        assert name == "double_genes"
        assert series.index.equals(ad.obs_names)
        expected = ad.obs["n_genes"].astype(float) * 2
        np.testing.assert_allclose(series.values, expected.values)

    def test_callable_receives_embedding_data_and_combines(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {
                "ratio": lambda d: (
                    d.get_column("n_genes").series / d.get_column("total_counts").series
                )
            }
        )
        series, _ = data.get_column("ratio")
        assert series.index.equals(ad.obs_names)
        expected = ad.obs["n_genes"].astype(float) / ad.obs["total_counts"].astype(float)
        np.testing.assert_allclose(series.values, expected.values)

    def test_named_tuple_routing(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {"x": lambda d: d.get_column("n_genes").series + 1},
            name="calc",
        )
        series, name = data.get_column(("calc", "x"))
        assert name == "x"
        assert series.index.equals(ad.obs_names)
        expected = ad.obs["n_genes"].astype(float) + 1
        np.testing.assert_allclose(series.values, expected.values)

    def test_tuple_unknown_name_raises(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {"x": lambda d: d.get_column("n_genes").series}
        )
        with pytest.raises(KeyError):
            data.get_column(("nope", "x"))

    def test_tuple_missing_column_in_named_derived_raises(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {"x": lambda d: d.get_column("n_genes").series}, name="calc"
        )
        with pytest.raises(KeyError):
            data.get_column(("calc", "not_there"))

    def test_unnamed_derived_plain_string_fallback(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {"derived_only": lambda d: d.get_column("n_genes").series * 10}
        )
        # cannot be tuple-addressed without a name
        with pytest.raises(KeyError):
            data.get_column(("anything", "derived_only"))
        # but plain string finds it
        series, _ = data.get_column("derived_only")
        assert series.index.equals(ad.obs_names)

    def test_primary_wins_over_derived(self, ad):
        """Primary source is consulted first — derived callable not invoked."""
        calls = []

        def fn(d):
            calls.append(1)
            return pd.Series(99.0, index=ad.obs_names)

        data = EmbeddingData(ad, "umap").add_derived_source({"n_genes": fn})
        series, _ = data.get_column("n_genes")
        # primary's real n_genes values, not the derived 99.0
        assert not np.allclose(series.values, 99.0)
        assert calls == []  # never invoked because primary hit

    def test_derived_checked_before_alternatives(self, ad):
        alt = _make_alt_ad(ad, extra_gene="DUP")  # alt gives values 0..n-1
        data = (
            EmbeddingData(ad, "umap")
            .add_alternative_source(alt)
            .add_derived_source(
                {"DUP": lambda d: pd.Series(42.0, index=ad.obs_names)}
            )
        )
        series, _ = data.get_column("DUP")
        # derived (42.0) wins over the alternative (0..n-1)
        assert np.allclose(series.values, 42.0)

    def test_recomputed_each_call_no_cache(self, ad):
        calls = []

        def fn(d):
            calls.append(1)
            return d.get_column("n_genes").series

        data = EmbeddingData(ad, "umap").add_derived_source({"tracked": fn})
        data.get_column("tracked")
        data.get_column("tracked")
        assert len(calls) == 2

    def test_non_callable_value_raises_at_registration(self, ad):
        data = EmbeddingData(ad, "umap")
        with pytest.raises(TypeError):
            data.add_derived_source({"bad": 5})

    def test_non_series_return_raises(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {"bad": lambda d: [1, 2, 3]}
        )
        with pytest.raises(TypeError):
            data.get_column("bad")

    def test_non_dict_arg_raises(self, ad):
        data = EmbeddingData(ad, "umap")
        with pytest.raises(TypeError):
            data.add_derived_source("not a dict")

    def test_duplicate_name_vs_alternative_raises(self, ad):
        alt = _make_alt_ad(ad, extra_gene="EXTRA_GENE")
        data = EmbeddingData(ad, "umap").add_alternative_source(alt, name="dup")
        with pytest.raises(ValueError):
            data.add_derived_source({"x": lambda d: d.get_column("n_genes").series}, name="dup")

    def test_duplicate_name_among_derived_raises(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {"x": lambda d: d.get_column("n_genes").series}, name="d"
        )
        with pytest.raises(ValueError):
            data.add_derived_source({"y": lambda d: d.get_column("n_genes").series}, name="d")

    def test_immutability(self, ad):
        data = EmbeddingData(ad, "umap")
        data2 = data.add_derived_source({"x": lambda d: d.get_column("n_genes").series})
        assert data.derived_sources == []
        assert len(data2.derived_sources) == 1
        # alternative list shared-but-immutable: original untouched
        data2.get_column("x")  # works on the copy

    def test_property_returns_records(self, ad):
        data = EmbeddingData(ad, "umap").add_derived_source(
            {"x": lambda d: d.get_column("n_genes").series}, name="calc"
        )
        rec = data.derived_sources[0]
        assert rec.name == "calc"
        assert set(rec.columns) == {"x"}

    def test_constructor_accepts_derived(self, ad):
        data = EmbeddingData(
            ad,
            "umap",
            derived_sources=[("calc", {"x": lambda d: d.get_column("n_genes").series * 3})],
        )
        assert data.derived_sources[0].name == "calc"
        series, _ = data.get_column(("calc", "x"))
        expected = ad.obs["n_genes"].astype(float) * 3
        np.testing.assert_allclose(series.values, expected.values)

    def test_constructor_bare_dict_derived(self, ad):
        data = EmbeddingData(
            ad,
            "umap",
            derived_sources=[{"x": lambda d: d.get_column("n_genes").series}],
        )
        assert data.derived_sources[0].name is None
        series, name = data.get_column("x")
        assert name == "x"

    def test_derived_source_record_accepted(self, ad):
        from mbf_singlecell_plotter import DerivedSource

        rec = DerivedSource("calc", {"x": lambda d: d.get_column("n_genes").series})
        data = EmbeddingData(ad, "umap").add_derived_source(rec)
        assert data.derived_sources[0].name == "calc"
        series, _ = data.get_column(("calc", "x"))
        assert series.index.equals(ad.obs_names)


class TestDerivedSourcesPlotter:
    def test_plotter_add_derived_and_tuple_routing(self, ad):
        sp = (
            ScatterPlotter()
            .set_source(ad, "umap")
            .add_derived_source(
                {"x": lambda d: d.get_column("n_genes").series * 2},
                name="calc",
            )
        )
        series, name = sp.get_column(("calc", "x"))
        assert name == "x"
        expected = ad.obs["n_genes"].astype(float) * 2
        np.testing.assert_allclose(series.values, expected.values)

    def test_plotter_requires_set_source(self, ad):
        with pytest.raises(RuntimeError):
            ScatterPlotter().add_derived_source(
                {"x": lambda d: d.get_column("n_genes").series}
            )

    def test_plot_accepts_derived_tuple(self, ad):
        import plotnine as p9

        sp = (
            ScatterPlotter()
            .set_source(ad, "umap")
            .add_derived_source(
                {"score": lambda d: d.get_column("n_genes").series * 2},
                name="calc",
            )
        )
        p = sp.plot(("calc", "score"))
        assert isinstance(p, p9.ggplot)
        assert "expression" in p.data.columns


# ---------------------------------------------------------------------------
# Cell filtering (set_filter / unfilter)
# ---------------------------------------------------------------------------


class TestFilterEmbeddingData:
    def test_has_filter_default_false(self, data):
        assert data.has_filter is False

    def test_set_filter_returns_copy_and_sets_flag(self, data):
        filtered = data.set_filter(lambda d: d.get_column("leiden").series == "0")
        assert data.has_filter is False
        assert filtered.has_filter is True

    def test_coordinates_subsetted(self, data, ad):
        full = data.coordinates()
        keep = ad.obs["leiden"] == "0"
        filtered = data.set_filter(lambda d, m=keep.values: m)
        sub = filtered.coordinates()
        assert len(sub) == int(keep.sum())
        assert len(sub) < ad.n_obs
        # kept cells are exactly the leiden-0 cells
        assert set(sub.index) == set(ad.obs_names[keep])

    def test_get_column_subsetted(self, data, ad):
        keep = ad.obs["leiden"] == "0"
        filtered = data.set_filter(lambda d, m=keep.values: m)
        series, _ = filtered.get_column("S100A8")
        assert len(series) == int(keep.sum())

    def test_get_column_preserves_index_order(self, data, ad):
        keep = ad.obs["leiden"] == "0"
        filtered = data.set_filter(lambda d, m=keep.values: m)
        series, _ = filtered.get_column("leiden")
        coords = filtered.coordinates()
        assert list(series.index) == list(coords.index)

    def test_filter_sees_full_dataset(self, data, ad):
        """The filter callable must operate on the complete (unfiltered) data."""
        seen_n = []
        n_total = ad.n_obs

        def fn(d):
            seen_n.append(len(d.coordinates()))
            return d.get_column("leiden").series == "0"

        filtered = data.set_filter(fn)
        filtered.coordinates()
        assert seen_n == [n_total]

    def test_bounds_ignore_filter(self, data, ad):
        keep = ad.obs["leiden"] == "0"
        filtered = data.set_filter(lambda d, m=keep.values: m)
        full_coords = data.coordinates()
        fb = filtered.bounds()
        fb_full = filtered.full_bounds()
        assert fb == pytest.approx(
            (
                float(full_coords["x"].min()),
                float(full_coords["x"].max()),
                float(full_coords["y"].min()),
                float(full_coords["y"].max()),
            )
        )
        assert fb_full == pytest.approx(fb)

    def test_filter_evaluated_once_and_cached(self, data):
        calls = []

        def fn(d):
            calls.append(1)
            return d.get_column("leiden").series == "0"

        filtered = data.set_filter(fn)
        filtered.coordinates()
        filtered.get_column("S100A8")
        filtered.coordinates()
        assert len(calls) == 1

    def test_set_filter_recomputes_mask(self, data):
        calls = []

        def fn(d):
            calls.append(1)
            return d.get_column("leiden").series == "0"

        filtered = data.set_filter(fn)
        filtered.coordinates()  # first eval
        assert len(calls) == 1
        refiltered = filtered.set_filter(fn)
        refiltered.coordinates()  # cache reset → re-eval
        assert len(calls) == 2

    def test_unfilter_removes_mask(self, data, ad):
        filtered = data.set_filter(lambda d: d.get_column("leiden").series == "0")
        assert filtered.has_filter is True
        unfiltered = filtered.unfilter()
        assert unfiltered.has_filter is False
        assert len(unfiltered.coordinates()) == ad.n_obs

    def test_set_filter_none_removes_mask(self, data, ad):
        filtered = data.set_filter(lambda d: d.get_column("leiden").series == "0")
        cleared = filtered.set_filter(None)
        assert cleared.has_filter is False
        assert len(cleared.coordinates()) == ad.n_obs

    def test_wrong_length_mask_raises(self, data):
        filtered = data.set_filter(lambda d: np.array([True, False]))
        with pytest.raises(ValueError, match="n_obs"):
            filtered.coordinates()

    def test_series_mask_supported(self, data, ad):
        keep = ad.obs["leiden"] == "0"

        def fn(d):
            return pd.Series(keep.values, index=ad.obs_names)

        filtered = data.set_filter(fn)
        sub = filtered.coordinates()
        assert len(sub) == int(keep.sum())

    def test_int_mask_coerced_to_bool(self, data, ad):
        keep = (ad.obs["leiden"] == "0").astype(int).values

        def fn(d, m=keep):
            return m

        filtered = data.set_filter(fn)
        sub = filtered.coordinates()
        assert len(sub) == int((ad.obs["leiden"] == "0").sum())

    def test_immutability(self, data, ad):
        filtered = data.set_filter(lambda d: d.get_column("leiden").series == "0")
        assert data.has_filter is False
        assert len(data.coordinates()) == ad.n_obs

    def test_derived_source_called_with_filter_disabled(self, data, ad):
        """A derived source inside the filter callable sees full data; the
        resulting column from get_column still respects the active filter."""
        calls = []

        def derived_fn(d):
            calls.append(len(d.coordinates()))
            return d.get_column("n_genes").series * 2

        enriched = data.add_derived_source({"dbl": derived_fn})
        filtered = enriched.set_filter(
            lambda d: d.get_column("dbl").series > d.get_column("n_genes").series
        )
        # every cell satisfies dbl > n_genes, so nothing is dropped
        sub, _ = filtered.get_column("dbl")
        assert len(sub) == ad.n_obs
        # the derived callable is invoked once for the filter mask and once for
        # the public get_column — both times it sees the full dataset
        assert calls == [ad.n_obs, ad.n_obs]


class TestFilterPlotter:
    def test_plotter_requires_source_first(self):
        with pytest.raises(RuntimeError):
            ScatterPlotter().set_filter(lambda d: d.get_column("x"))

    def test_plot_subset_uses_filtered_cells(self, plotter_no_boundary, ad):
        keep = ad.obs["leiden"] == "0"
        filtered_pt = plotter_no_boundary.set_filter(
            lambda d, m=keep.values: m
        )
        # categorical plot: p.data holds every kept cell (no zero/clip split)
        p = filtered_pt.plot("leiden")
        assert len(p.data) == int(keep.sum())

    def test_plot_bounds_match_full_dataset(self, plotter_no_boundary, data):
        keep_mask = data.get_column("leiden").series == "0"
        filtered_pt = plotter_no_boundary.set_filter(lambda d, m=keep_mask.values: m)
        x_min, x_max, y_min, y_max = filtered_pt._data.bounds()
        full = plotter_no_boundary._data.bounds()
        assert (x_min, x_max, y_min, y_max) == pytest.approx(full)

    def test_unfilter_restores_all_cells(self, plotter_no_boundary, ad):
        keep = ad.obs["leiden"] == "0"
        filtered = plotter_no_boundary.set_filter(lambda d, m=keep.values: m)
        restored = filtered.unfilter()
        p = restored.plot("leiden")
        assert len(p.data) == ad.n_obs

    def test_filter_survives_with_grid(self, ad):
        keep = ad.obs["leiden"] == "0"
        sp = (
            ScatterPlotter()
            .set_source(ad, "umap")
            .set_filter(lambda d, m=keep.values: m)
            .with_grid(grid_size=8)
        )
        assert sp._data.has_filter is True
        p = sp.plot("leiden")
        assert len(p.data) == int(keep.sum())
        assert sp._data._grid_size == 8

    def test_filter_immutability(self, plotter_no_boundary, ad):
        keep = ad.obs["leiden"] == "0"
        filtered = plotter_no_boundary.set_filter(lambda d, m=keep.values: m)
        assert plotter_no_boundary._data.has_filter is False
        assert filtered._data.has_filter is True
        # original still plots all cells
        p = plotter_no_boundary.plot("leiden")
        assert len(p.data) == ad.n_obs

