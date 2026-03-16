"""
Image regression tests for mbf-singlecell-plotter.

On first run: reference images are saved to tests/reference_images/.
On subsequent runs: images are compared pixel-by-pixel; failures are written to tests/failures/.

Run with REGENERATE_REFS=1 to regenerate all reference images.

Fixture choice:
  plotter_no_boundary  — no scikit-image needed; border_cell_types=False for these tests
  plotter              — requires scikit-image; tests boundary rendering specifically
"""

import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from image_comparison import assert_image_matches, assert_array_matches, assert_plotnine_matches
from conftest import CELL_TYPE_COLUMN as CAT_COL
from mbf_singlecell_plotter import default_colors

# Dot size used across all scatter tests for legible reference images
DOT_SIZE = 5


# ---------------------------------------------------------------------------
# plot_scatter — numerical (gene expression), no boundary overlay
# ---------------------------------------------------------------------------

class TestPlotScatterNumerical:
    def test_s100a8_basic(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "S100A8", border_cell_types=False, dot_size=DOT_SIZE
        )
        assert_image_matches(parts.fig, "scatter_S100A8_basic")

    def test_lst1_no_zeros(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "LST1", plot_zeros=False, border_cell_types=False, dot_size=DOT_SIZE
        )
        assert_image_matches(parts.fig, "scatter_LST1_no_zeros")

    def test_cst3_clip_quantile(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "CST3", clip_quantile=0.99, border_cell_types=False, dot_size=DOT_SIZE
        )
        assert_image_matches(parts.fig, "scatter_CST3_clip99")

    def test_cd79a_no_spines(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "CD79A", show_spines=False, border_cell_types=False, dot_size=DOT_SIZE
        )
        assert_image_matches(parts.fig, "scatter_CD79A_no_spines")

    def test_tyrobp_no_zeros(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "TYROBP", plot_zeros=False, border_cell_types=False, dot_size=DOT_SIZE
        )
        assert_image_matches(parts.fig, "scatter_TYROBP_no_zeros")

    def test_zeros_vs_no_zeros(self, plotter_no_boundary):
        """Side-by-side: left=zeros shown, right=zeros hidden.

        plot_scatter always creates its own figure so we render both
        separately and stitch the arrays together horizontally.
        """
        import numpy as np
        from image_comparison import _fig_to_array

        parts_with = plotter_no_boundary.plot_scatter(
            "S100A8", border_cell_types=False, dot_size=DOT_SIZE, plot_zeros=True,
            zero_dot_size=DOT_SIZE,
            zero_color='green',
        )
        parts_without = plotter_no_boundary.plot_scatter(
            "S100A8", border_cell_types=False, dot_size=DOT_SIZE, plot_zeros=False,
            zero_dot_size=DOT_SIZE,
            zero_color='green',
        )
        parts_with.fig.suptitle("plot_zeros=True")
        parts_without.fig.suptitle("plot_zeros=False")

        a = _fig_to_array(parts_with.fig)
        b = _fig_to_array(parts_without.fig)

        h = max(a.shape[0], b.shape[0])
        def pad_h(arr):
            if arr.shape[0] < h:
                pad = np.full((h - arr.shape[0], arr.shape[1], 3), 255, dtype=np.uint8)
                return np.concatenate([arr, pad], axis=0)
            return arr

        assert_array_matches(
            np.concatenate([pad_h(a), pad_h(b)], axis=1),
            "scatter_S100A8_zeros_comparison",
        )

    def test_returns_scatter_parts(self, plotter_no_boundary):
        from mbf_singlecell_plotter import ScatterParts
        parts = plotter_no_boundary.plot_scatter(
            "S100A8", border_cell_types=False, dot_size=DOT_SIZE
        )
        assert isinstance(parts, ScatterParts)
        plt.close(parts.fig)


# ---------------------------------------------------------------------------
# plot_scatter — categorical, no boundary overlay
# ---------------------------------------------------------------------------

class TestPlotScatterCategorical:
    def test_leiden_clusters(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            CAT_COL, border_cell_types=False, dot_size=DOT_SIZE
        )
        assert_image_matches(parts.fig, "scatter_leiden")

    def test_leiden_flip_order(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            CAT_COL, flip_order=True, border_cell_types=False, dot_size=DOT_SIZE
        )
        assert_image_matches(parts.fig, "scatter_leiden_flip_order")


# ---------------------------------------------------------------------------
# plot_scatter — cell density transform
# ---------------------------------------------------------------------------

class TestPlotScatterCellDensity:
    def test_cell_density(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "transform:cell_density", border_cell_types=False
        )
        assert_image_matches(parts.fig, "scatter_cell_density_transform")


# ---------------------------------------------------------------------------
# plot_scatter — faceting
# ---------------------------------------------------------------------------

class TestPlotScatterFacet:
    def test_facet_by_leiden(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "S100A8", facet_variable=CAT_COL, n_col=3,
            border_cell_types=False, dot_size=DOT_SIZE,
        )
        assert_image_matches(parts.fig, "scatter_S100A8_facet_leiden")


# ---------------------------------------------------------------------------
# plot_scatter — axis limits
# ---------------------------------------------------------------------------

class TestPlotScatterLimits:
    def test_custom_limits(self, plotter_no_boundary):
        df = plotter_no_boundary.get_coordinate_dataframe()
        x_mid = df["x"].median()
        y_mid = df["y"].median()
        parts = plotter_no_boundary.plot_scatter(
            "S100A8",
            border_cell_types=False,
            dot_size=DOT_SIZE,
            x_limits=(df["x"].min(), x_mid),
            y_limits=(df["y"].min(), y_mid),
        )
        assert_image_matches(parts.fig, "scatter_S100A8_custom_limits")


# ---------------------------------------------------------------------------
# plot_cell_density — no boundary overlay
# ---------------------------------------------------------------------------

class TestPlotCellDensity:
    def test_basic(self, plotter_no_boundary):
        fig, ax, h = plotter_no_boundary.plot_cell_density(border_cell_types=False)
        assert_image_matches(fig, "cell_density_basic")

    def test_no_spines(self, plotter_no_boundary):
        fig, ax, h = plotter_no_boundary.plot_cell_density(
            border_cell_types=False, show_spines=False
        )
        assert_image_matches(fig, "cell_density_no_spines")

    def test_custom_bins(self, plotter_no_boundary):
        fig, ax, h = plotter_no_boundary.plot_cell_density(border_cell_types=False, bins=50)
        assert_image_matches(fig, "cell_density_bins_50")

    def test_returns_three_tuple(self, plotter_no_boundary):
        result = plotter_no_boundary.plot_cell_density(border_cell_types=False)
        assert len(result) == 3
        fig, ax, h = result
        plt.close(fig)


# ---------------------------------------------------------------------------
# Grid coordinate overlays — draw_grid / edge_grid / label_grid / grid_axes
# ---------------------------------------------------------------------------

class TestGridOverlays:
    """Visual tests for the various grid rendering modes."""

    def test_draw_grid(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "S100A8", border_cell_types=False, dot_size=DOT_SIZE,
            draw_grid=True,
        )
        assert_image_matches(parts.fig, "grid_draw")

    def test_draw_grid_with_labels(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "S100A8", border_cell_types=False, dot_size=DOT_SIZE,
            draw_grid=True, label_grid=True,
        )
        assert_image_matches(parts.fig, "grid_draw_labeled")

    def test_grid_axes(self, plotter_no_boundary):
        """grid_axes replaces axis ticks with grid-cell labels."""
        parts = plotter_no_boundary.plot_scatter(
            "S100A8", border_cell_types=False, dot_size=DOT_SIZE,
            grid_axes=True,
        )
        assert_image_matches(parts.fig, "grid_axes")

    def test_grid_custom_color(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "S100A8", border_cell_types=False, dot_size=DOT_SIZE,
            draw_grid=True, label_grid=True, grid_color="#CC0000",
        )
        assert_image_matches(parts.fig, "grid_draw_red")

    def test_categorical_with_grid(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            CAT_COL, border_cell_types=False, dot_size=DOT_SIZE,
            draw_grid=True, label_grid=True,
        )
        assert_image_matches(parts.fig, "grid_leiden_labeled")


# ---------------------------------------------------------------------------
# Boundary (border) rendering — requires scikit-image (skips if absent)
# ---------------------------------------------------------------------------

class TestBoundaryRendering:
    """Tests that specifically exercise the cell-type boundary overlay."""

    def test_numerical_with_borders(self, plotter):
        parts = plotter.plot_scatter("S100A8", dot_size=DOT_SIZE)
        assert_image_matches(parts.fig, "border_scatter_S100A8")

    def test_categorical_with_borders(self, plotter):
        parts = plotter.plot_scatter(CAT_COL, dot_size=DOT_SIZE)
        assert_image_matches(parts.fig, "border_scatter_leiden")

    def test_cell_density_with_borders(self, plotter):
        fig, ax, h = plotter.plot_cell_density()
        assert_image_matches(fig, "border_cell_density")

    def test_border_size_small(self, plotter):
        parts = plotter.plot_scatter("S100A8", dot_size=DOT_SIZE, border_size=5)
        assert_image_matches(parts.fig, "border_scatter_S100A8_border5")

    def test_border_size_large(self, plotter):
        parts = plotter.plot_scatter("S100A8", dot_size=DOT_SIZE, border_size=30)
        assert_image_matches(parts.fig, "border_scatter_S100A8_border30")

    def test_borders_with_grid(self, plotter):
        parts = plotter.plot_scatter(
            "S100A8", dot_size=DOT_SIZE,
            draw_grid=True, label_grid=True,
        )
        assert_image_matches(parts.fig, "border_scatter_S100A8_grid")

    def test_plot_data_false(self, plotter):
        """Border-only: no scatter data, just the cell-type boundary overlay."""
        parts = plotter.plot_scatter("S100A8", dot_size=DOT_SIZE, plot_data=False)
        assert_image_matches(parts.fig, "border_only_S100A8")


# ---------------------------------------------------------------------------
# plot_grid_local_histogram  (plotnine output)
# ---------------------------------------------------------------------------

class TestPlotGridLocalHistogram:
    def test_basic(self, plotter_no_boundary):
        p = plotter_no_boundary.plot_grid_local_histogram(
            CAT_COL, default_colors, min_cell_count=10
        )
        assert_plotnine_matches(p, "grid_local_histogram_leiden")

    def test_high_min_cells(self, plotter_no_boundary):
        """Fewer grid cells shown when min_cell_count is large."""
        p = plotter_no_boundary.plot_grid_local_histogram(
            CAT_COL, default_colors, min_cell_count=100
        )
        assert_plotnine_matches(p, "grid_local_histogram_leiden_min100")
