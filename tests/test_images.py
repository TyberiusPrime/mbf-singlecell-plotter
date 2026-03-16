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

from image_comparison import assert_image_matches
from conftest import CELL_TYPE_COLUMN as CAT_COL


# ---------------------------------------------------------------------------
# plot_scatter — numerical (gene expression), no boundary overlay
# ---------------------------------------------------------------------------

class TestPlotScatterNumerical:
    def test_s100a8_basic(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter("S100A8", border_cell_types=False)
        assert_image_matches(parts.fig, "scatter_S100A8_basic")

    def test_lst1_no_zeros(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter("LST1", plot_zeros=False, border_cell_types=False)
        assert_image_matches(parts.fig, "scatter_LST1_no_zeros")

    def test_cst3_clip_quantile(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter("CST3", clip_quantile=0.99, border_cell_types=False)
        assert_image_matches(parts.fig, "scatter_CST3_clip99")

    def test_cd79a_no_spines(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter("CD79A", show_spines=False, border_cell_types=False)
        assert_image_matches(parts.fig, "scatter_CD79A_no_spines")

    @pytest.mark.xfail(reason="point_to_grid IndexError on cells at embedding boundary", strict=False)
    def test_s100a8_with_grid(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(
            "S100A8", draw_grid=True, label_grid=True, border_cell_types=False
        )
        assert_image_matches(parts.fig, "scatter_S100A8_grid")

    def test_tyrobp_no_zeros(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter("TYROBP", plot_zeros=False, border_cell_types=False)
        assert_image_matches(parts.fig, "scatter_TYROBP_no_zeros")

    def test_returns_scatter_parts(self, plotter_no_boundary):
        from mbf_singlecell_plotter import ScatterParts
        parts = plotter_no_boundary.plot_scatter("S100A8", border_cell_types=False)
        assert isinstance(parts, ScatterParts)
        plt.close(parts.fig)


# ---------------------------------------------------------------------------
# plot_scatter — categorical, no boundary overlay
# ---------------------------------------------------------------------------

class TestPlotScatterCategorical:
    def test_leiden_clusters(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(CAT_COL, border_cell_types=False)
        assert_image_matches(parts.fig, "scatter_leiden")

    def test_leiden_flip_order(self, plotter_no_boundary):
        parts = plotter_no_boundary.plot_scatter(CAT_COL, flip_order=True, border_cell_types=False)
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
            "S100A8", facet_variable=CAT_COL, n_col=3, border_cell_types=False
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
# Boundary rendering — requires scikit-image (skips if absent)
# ---------------------------------------------------------------------------

class TestBoundaryRendering:
    """Tests that specifically exercise the cell-type boundary overlay."""

    def test_scatter_with_borders(self, plotter):
        parts = plotter.plot_scatter("S100A8")
        assert_image_matches(parts.fig, "scatter_S100A8_with_borders")

    def test_categorical_with_borders(self, plotter):
        parts = plotter.plot_scatter(CAT_COL)
        assert_image_matches(parts.fig, "scatter_leiden_with_borders")

    def test_cell_density_with_borders(self, plotter):
        fig, ax, h = plotter.plot_cell_density()
        assert_image_matches(fig, "cell_density_with_borders")


# ---------------------------------------------------------------------------
# plot_grid_local_histogram  (plotnine output)
# ---------------------------------------------------------------------------

class TestPlotGridLocalHistogram:
    @pytest.mark.xfail(reason="point_to_grid IndexError on cells at embedding boundary", strict=False)
    def test_basic(self, plotter_no_boundary):
        import io
        import numpy as np
        from PIL import Image

        p = plotter_no_boundary.plot_grid_local_histogram(
            CAT_COL, plotter_no_boundary.cell_type_color_map, min_cell_count=10
        )
        buf = io.BytesIO()
        p.save(buf, format="png", dpi=100, verbose=False)
        buf.seek(0)
        img_array = np.array(Image.open(buf).convert("RGB"))

        from image_comparison import (
            REFERENCE_DIR, FAILURES_DIR, REGENERATE,
            _save_png, _load_png, _build_diff_image,
            DEFAULT_TOLERANCE, DEFAULT_PIXEL_THRESHOLD,
        )
        import shutil
        name = "grid_local_histogram_leiden"
        ref_path = REFERENCE_DIR / f"{name}.png"
        FAILURES_DIR.mkdir(parents=True, exist_ok=True)

        if REGENERATE or not ref_path.exists():
            _save_png(img_array, ref_path)
            return

        ref = _load_png(ref_path)
        if ref.shape != img_array.shape:
            _save_png(img_array, FAILURES_DIR / f"{name}_actual.png")
            diff = _build_diff_image(ref, img_array)
            _save_png(diff, FAILURES_DIR / f"{name}_diff.png")
            raise AssertionError(
                f"Shape mismatch for '{name}': {ref.shape} vs {img_array.shape}"
            )

        diff_mask = np.abs(ref.astype(int) - img_array.astype(int)).max(axis=2) > DEFAULT_PIXEL_THRESHOLD
        bad_fraction = diff_mask.mean()
        if bad_fraction > DEFAULT_TOLERANCE:
            _save_png(img_array, FAILURES_DIR / f"{name}_actual.png")
            shutil.copy(ref_path, FAILURES_DIR / f"{name}_reference.png")
            diff = _build_diff_image(ref, img_array)
            _save_png(diff, FAILURES_DIR / f"{name}_diff.png")
            raise AssertionError(
                f"Image mismatch for '{name}': {bad_fraction:.2%} pixels differ\n"
                f"  reference: {FAILURES_DIR / (name + '_reference.png')}\n"
                f"  actual:    {FAILURES_DIR / (name + '_actual.png')}\n"
                f"  diff:      {FAILURES_DIR / (name + '_diff.png')}"
            )
