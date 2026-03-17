"""
Image regression tests for mbf-singlecell-plotter.

On first run: reference images are saved to tests/reference_images/.
On subsequent runs: images are compared pixel-by-pixel; failures are written to tests/failures/.

Run with REGENERATE_REFS=1 to regenerate all reference images.

Fixture choice:
  plotter_no_boundary  — no scikit-image needed; no border overlay for these tests
  plotter              — requires scikit-image; tests boundary rendering specifically
"""

import pytest
import matplotlib

matplotlib.use("Agg")

from image_comparison import assert_plotnine_matches
from conftest import CELL_TYPE_COLUMN as CAT_COL
from mbf_singlecell_plotter import DEFAULT_COLORS

# Dot size used across all scatter tests for legible reference images
DOT_SIZE = 2


# ---------------------------------------------------------------------------
# plot — numerical (gene expression), no boundary overlay
# ---------------------------------------------------------------------------


class TestPlotScatterNumerical:
    def test_s100a8_basic(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.zeros(zero_value=-0.50)
            .style(dot_size=DOT_SIZE)
            .plot("S100A8")
        )
        assert_plotnine_matches(p, "scatter_S100A8_basic")

    def test_lst1_no_zeros(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .layers(zeros=False)
            .plot("LST1")
        )
        assert_plotnine_matches(p, "scatter_LST1_no_zeros")

    def test_cst3_clip_quantile(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .colormap(max_quantile=0.99)
            .plot("CST3")
        )
        assert_plotnine_matches(p, "scatter_CST3_clip99")

    def test_cst3_clip_quantile_05(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .colormap(max_quantile=0.50)
            .plot("CST3")
        )
        assert_plotnine_matches(p, "scatter_CST3_clip50")

    def test_cst3_clip_quantile_10(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .colormap(max_quantile=1.0)
            .plot("CST3")
        )
        assert_plotnine_matches(p, "scatter_CST3_clip100")

    def test_cd79a_no_spines(self, plotter_no_boundary):
        p = plotter_no_boundary.style(dot_size=DOT_SIZE, spines=False).plot("CD79A")
        assert_plotnine_matches(p, "scatter_CD79A_no_spines")

    def test_zeros_vs_no_zeros(self, plotter_no_boundary):
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

        from image_comparison import assert_array_matches

        assert_array_matches(
            np.concatenate([pad_h(a), pad_h(b)], axis=1),
            "scatter_S100A8_zeros_comparison",
        )

    def test_returns_ggplot(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot("S100A8")
        assert isinstance(p, p9.ggplot)


# ---------------------------------------------------------------------------
# plot — categorical, no boundary overlay
# ---------------------------------------------------------------------------


class TestPlotScatterCategorical:
    def test_leiden_clusters(self, plotter_no_boundary):
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot(CAT_COL)
        assert_plotnine_matches(p, "scatter_leiden")

    def test_leiden_flip_order(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .flip_draw_order(True)
            .plot(CAT_COL)
        )
        assert_plotnine_matches(p, "scatter_leiden_flip_order")


# ---------------------------------------------------------------------------
# plot_density
# ---------------------------------------------------------------------------


class TestPlotCellDensity:
    def test_basic(self, plotter_no_boundary):
        p = plotter_no_boundary.plot_density()
        assert_plotnine_matches(p, "cell_density_basic")

    def test_no_spines(self, plotter_no_boundary):
        p = plotter_no_boundary.style(spines=False).plot_density()
        assert_plotnine_matches(p, "cell_density_no_spines")

    def test_custom_bins(self, plotter_no_boundary):
        p = plotter_no_boundary.plot_density(bins=50)
        assert_plotnine_matches(p, "cell_density_bins_50")

    def test_returns_ggplot(self, plotter_no_boundary):
        import plotnine as p9

        p = plotter_no_boundary.plot_density()
        assert isinstance(p, p9.ggplot)


# ---------------------------------------------------------------------------
# Grid coordinate overlays
# ---------------------------------------------------------------------------


class TestGridOverlays:
    """Visual tests for the various grid rendering modes."""

    def test_draw_grid(self, plotter_no_boundary):
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).with_grid().plot("S100A8")
        assert_plotnine_matches(p, "grid_draw")

    def test_draw_grid_with_labels(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(labels=True)
            .plot("S100A8")
        )
        assert_plotnine_matches(p, "grid_draw_labeled")

    def test_grid_axes(self, plotter_no_boundary):
        """grid coords replaces axis ticks with grid-cell labels."""
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(coords=True)
            .plot("S100A8")
        )
        assert_plotnine_matches(p, "grid_axes")

    def test_grid_custom_color(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(labels=False, color="#CC0000")
            .plot("S100A8")
        )
        assert_plotnine_matches(p, "grid_draw_red")

    def test_grid_custom_color_labels(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(labels=True, color="#CC0000", label_color="#CC00CC")
            .plot("S100A8")
        )
        assert_plotnine_matches(p, "grid_draw_red_labels_purple")

    def test_categorical_with_grid(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(
                labels=True,
            )
            .plot(CAT_COL)
        )
        assert_plotnine_matches(p, "grid_leiden_labeled")


# ---------------------------------------------------------------------------
# Boundary (border) rendering — requires scikit-image (skips if absent)
# ---------------------------------------------------------------------------


class TestBoundaryRendering:
    """Tests that specifically exercise the cell-type boundary overlay."""

    def test_numerical_with_borders(self, plotter):
        p = plotter.style(dot_size=DOT_SIZE).plot("S100A8")
        assert_plotnine_matches(p, "border_scatter_S100A8")

    def test_categorical_with_borders(self, plotter):
        p = plotter.style(dot_size=DOT_SIZE).plot(CAT_COL)
        assert_plotnine_matches(p, "border_scatter_leiden")

    def test_cell_density_with_borders(self, plotter):
        p = plotter.plot_density()
        assert_plotnine_matches(p, "border_cell_density")

    def test_border_size_small(self, plotter):
        p = plotter.with_borders(size=5).style(dot_size=DOT_SIZE).plot("S100A8")
        assert_plotnine_matches(p, "border_scatter_S100A8_border5")

    def test_border_size_large(self, plotter):
        p = (
            plotter.with_borders(size=30)
            .zeros(zero_value=-0.5)
            .style(dot_size=DOT_SIZE)
            .plot("S100A8")
        )
        assert_plotnine_matches(p, "border_scatter_S100A8_border30")

    def test_borders_with_grid(self, plotter):
        p = plotter.style(dot_size=DOT_SIZE).with_grid(labels=True).plot("S100A8")
        assert_plotnine_matches(p, "border_scatter_S100A8_grid")


# ---------------------------------------------------------------------------
# plot_scatter — cell density transform (now plot_density)
# ---------------------------------------------------------------------------


class TestPlotScatterCellDensity:
    def test_cell_density(self, plotter_no_boundary):
        p = plotter_no_boundary.plot_density()
        assert_plotnine_matches(p, "scatter_cell_density_transform")


# ---------------------------------------------------------------------------
# plot — faceting
# ---------------------------------------------------------------------------


class TestPlotScatterFacet:
    def test_facet_by_leiden(self, plotter_no_boundary):
        p = (
            plotter_no_boundary.facet(CAT_COL, n_col=3)
            .style(dot_size=DOT_SIZE)
            .plot("S100A8")
        )
        assert_plotnine_matches(p, "scatter_S100A8_facet_leiden")


# ---------------------------------------------------------------------------
# plot — axis limits (focus_on)
# ---------------------------------------------------------------------------


class TestPlotScatterLimits:
    def test_custom_limits(self, plotter_no_boundary, data):
        coords = data.coordinates()
        x_mid = coords["x"].median()
        y_mid = coords["y"].median()
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .focus_on(x=(coords["x"].min(), x_mid), y=(coords["y"].min(), y_mid))
            .plot("S100A8")
        )
        assert_plotnine_matches(p, "scatter_S100A8_custom_limits")


# ---------------------------------------------------------------------------
# plot_grid_local_histogram  (plotnine output)
# ---------------------------------------------------------------------------


class TestPlotGridLocalHistogram:
    def test_basic(self, plotter_no_boundary):
        p = plotter_no_boundary.plot_grid_histogram(CAT_COL, min_cell_count=10)
        assert_plotnine_matches(p, "grid_local_histogram_leiden")

    def test_high_min_cells(self, plotter_no_boundary):
        """Fewer grid cells shown when min_cell_count is large."""
        p = plotter_no_boundary.plot_grid_histogram(CAT_COL, min_cell_count=100)
        assert_plotnine_matches(p, "grid_local_histogram_leiden_min100")
