"""
Image regression tests for mbf-singlecell-plotter.

On first run: reference images are saved to tests/reference_images/.
On subsequent runs: images are compared pixel-by-pixel; failures are written to tests/failures/.

Run with REGENERATE_REFS=1 to regenerate all reference images.

Fixture choice:
  plotter_no_boundary  — no scikit-image needed; no border overlay for these tests
  plotter              — requires scikit-image; tests boundary rendering specifically

Image names are derived automatically from the test class + function name via the
``assert_image`` fixture — no hardcoded name strings needed.
"""

import pytest
import matplotlib

matplotlib.use("Agg")

from conftest import CELL_TYPE_COLUMN as CAT_COL

# Dot size used across all scatter tests for legible reference images
DOT_SIZE = 2


# ---------------------------------------------------------------------------
# plot — numerical (gene expression), no boundary overlay
# ---------------------------------------------------------------------------


class TestPlotScatterNumerical:
    def test_s100a8_basic(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.zeros(zero_value=-0.50)
            .style(dot_size=DOT_SIZE)
            .plot("S100A8")
        )
        assert_image(p)

    def test_lst1_no_zeros(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .layers(zeros=False)
            .plot("LST1")
        )
        assert_image(p)

    def test_cst3_clip_quantile(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .colormap(max_quantile=0.99)
            .plot("CST3")
        )
        assert_image(p)

    def test_cst3_clip_quantile_05(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .colormap(max_quantile=0.50)
            .plot("CST3")
        )
        assert_image(p)

    def test_cst3_clip_quantile_10(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .colormap(max_quantile=1.0)
            .plot("CST3")
        )
        assert_image(p)

    def test_cd79a_no_spines(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.style(dot_size=DOT_SIZE, panel_border=False).plot(
            "CD79A"
        )
        assert_image(p)

    def test_zeros_vs_no_zeros(self, plotter_no_boundary, assert_image):
        """Side-by-side: left=zeros shown, right=zeros hidden."""
        import numpy as np
        from image_comparison import _plotnine_to_array

        p_with = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .zeros(color="green", dot_size=DOT_SIZE, zero_value=0.0)
            .plot("S100A8")
        )
        p_without = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .zeros(color="green", dot_size=DOT_SIZE)
            .layers(zeros=False)
            .plot("S100A8")
        )

        a = _plotnine_to_array(p_with)
        b = _plotnine_to_array(p_without)

        h = max(a.shape[0], b.shape[0])

        def pad_h(arr):
            if arr.shape[0] < h:
                pad = np.full((h - arr.shape[0], arr.shape[1], 3), 255, dtype=np.uint8)
                return np.concatenate([arr, pad], axis=0)
            return arr

        assert_image(np.concatenate([pad_h(a), pad_h(b)], axis=1))

    def test_returns_ggplot(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot("S100A8")
        assert isinstance(p, p9.ggplot)


# ---------------------------------------------------------------------------
# plot — categorical, no boundary overlay
# ---------------------------------------------------------------------------


class TestPlotScatterCategorical:
    def test_leiden_clusters(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot(CAT_COL)
        assert_image(p)

    def test_leiden_clusters_no_outliers(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .layers(outliers=False)
            .plot(CAT_COL)
        )
        assert_image(p)

    def test_leiden_clusters_outlier_shape(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .outlier(shape="x")
            .plot(CAT_COL)
        )
        assert_image(p)

    def test_leiden_clusters_outlier_shape_only_outlier(
        self, plotter_no_boundary, assert_image
    ):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .layers(borders=False, outliers=True, zeros=False, data=False)
            .outlier(shape="x")
            .plot(CAT_COL)
        )
        assert_image(p)

    def test_leiden_flip_order(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .flip_draw_order(True)
            .plot(CAT_COL)
        )
        assert_image(p)


# ---------------------------------------------------------------------------
# plot_density
# ---------------------------------------------------------------------------


class TestPlotCellDensity:
    def test_basic(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.plot_density()
        assert_image(p)

    def test_no_spines(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.style(panel_border=False).plot_density()
        assert_image(p)

    def test_custom_bins(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.plot_density(bins=50)
        assert_image(p)

    def test_returns_ggplot(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.plot_density()
        assert isinstance(p, p9.ggplot)


# ---------------------------------------------------------------------------
# Grid coordinate overlays
# ---------------------------------------------------------------------------


class TestGridOverlays:
    """Visual tests for the various grid rendering modes."""

    def test_no_grid(self, plotter_no_boundary, assert_image):
        """Grid lines only — no axis-tick replacement, no cell labels."""
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).without_grid().plot("S100A8")
        assert_image(p)

    def test_draw_grid_no_coords(self, plotter_no_boundary, assert_image):
        """Grid lines only — no axis-tick replacement, no cell labels."""
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(coords=False)
            .plot("S100A8")
        )
        assert_image(p)

    def test_draw_grid(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).with_grid().plot("S100A8")
        assert_image(p)

    def test_draw_grid_with_labels(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(labels=True)
            .plot("S100A8")
        )
        assert_image(p)

    def test_grid_axes(self, plotter_no_boundary, assert_image):
        """grid coords replaces axis ticks with grid-cell labels."""
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(coords=True)
            .plot("S100A8")
        )
        assert_image(p)

    def test_grid_custom_color(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(labels=False, color="#CC0000")
            .plot("S100A8")
        )
        assert_image(p)

    def test_grid_custom_color_labels(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(labels=True, color="#CC0000", label_color="#CC00CC")
            .plot("S100A8")
        )
        assert_image(p)

    def test_categorical_with_grid(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(
                labels=True,
            )
            .plot(CAT_COL)
        )
        assert_image(p)


# ---------------------------------------------------------------------------
# Boundary (border) rendering — requires scikit-image (skips if absent)
# ---------------------------------------------------------------------------


class TestBoundaryRendering:
    """Tests that specifically exercise the cell-type boundary overlay."""

    def test_numerical_with_borders(self, plotter, assert_image):
        # todo rename zero_value to min_value or such
        p = plotter.zeros(zero_value=-0.5).style(dot_size=DOT_SIZE).plot("S100A8")
        assert_image(p)

    def test_categorical_with_borders(self, plotter, assert_image):
        plotter._data.ad.obs["even"] = (
            plotter._data.ad.obs["leiden"].astype(int) % 2 == 0
        ).replace({True: "yes", False: "no"})
        p = plotter.style(dot_size=1).plot("even")
        assert_image(p)

    def test_cell_density_with_borders(self, plotter, assert_image):
        p = plotter.plot_density()
        assert_image(p)

    def test_border_size_small(self, plotter, assert_image):
        p = plotter.with_borders(size=5).style(dot_size=DOT_SIZE).plot("S100A8")
        assert_image(p)

    def test_border_size_large(self, plotter, assert_image):
        p = (
            plotter.with_borders(size=30)
            .zeros(zero_value=-0.5)
            .style(dot_size=DOT_SIZE)
            .plot("S100A8")
        )
        assert_image(p)

    def test_borders_with_grid(self, plotter, assert_image):
        p = plotter.style(dot_size=DOT_SIZE).with_grid(labels=True).plot("S100A8")
        assert_image(p)


# ---------------------------------------------------------------------------
# plot_scatter — cell density transform (now plot_density)
# ---------------------------------------------------------------------------


class TestPlotScatterCellDensity:
    def test_cell_density(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.plot_density()
        assert_image(p)


# ---------------------------------------------------------------------------
# plot — faceting
# ---------------------------------------------------------------------------


class TestPlotScatterFacet:
    def test_facet_by_leiden(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.facet(CAT_COL, n_col=3)
            .style(dot_size=DOT_SIZE)
            .plot("S100A8")
        )
        assert_image(p)


# ---------------------------------------------------------------------------
# plot — axis limits (focus_on)
# ---------------------------------------------------------------------------


class TestPlotScatterLimits:
    def test_custom_limits(self, plotter_no_boundary, data, assert_image):
        coords = data.coordinates()
        x_mid = coords["x"].median()
        y_mid = coords["y"].median()
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .focus_on(x=(coords["x"].min(), x_mid), y=(coords["y"].min(), y_mid))
            .plot("S100A8")
        )
        assert_image(p)


# ---------------------------------------------------------------------------
# plot_grid_local_histogram  (plotnine output)
# ---------------------------------------------------------------------------


class TestPlotGridLocalHistogram:
    def test_basic(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.plot_grid_histogram(CAT_COL, min_cell_count=10)
        assert_image(p)

    def test_high_min_cells(self, plotter_no_boundary, assert_image):
        """Fewer grid cells shown when min_cell_count is large."""
        p = plotter_no_boundary.plot_grid_histogram(CAT_COL, min_cell_count=100)
        assert_image(p)
