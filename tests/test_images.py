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
import numpy as np
import pandas as pd
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
            plotter_no_boundary.zeros(max_zero_value=-0.50)
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
            .zeros(color="green", dot_size=DOT_SIZE, max_zero_value=0.0)
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

    def test_background_numerical(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=.1)
            .background(color="#00FF00", dot_size=DOT_SIZE)
            .layers(zeros=False)
            .plot("S100A8")
        )
        assert_image(p)

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

    def test_background_categorical(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=.1)
            .background(color="#CCCCCC", dot_size=DOT_SIZE)
            #.layerz(zeros=False)
            .plot(CAT_COL)
        )
        assert_image(p)

    def test_background_categorical_facet(self, plotter_no_boundary, assert_image):
        p = (
            plotter_no_boundary.style(dot_size=.1)
            .background(color="#CCCCCC", dot_size=DOT_SIZE)
            .facet('leiden')
            #.layerz(zeros=False)
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

    def test_no_grid(self, plotter_no_boundary, assert_image):
        """Density with the grid disabled — plain axis ticks, no grid lines."""
        p = plotter_no_boundary.without_grid().plot_density()
        assert_image(p)

    def test_grid_labels(self, plotter_no_boundary, assert_image):
        """Density honouring the grid: lines + cell-interior labels + axis ticks."""
        p = plotter_no_boundary.with_grid(labels=True).plot_density()
        assert_image(p)

    def test_grid_no_coords(self, plotter_no_boundary, assert_image):
        """Grid lines drawn but axis ticks left as plain coordinates."""
        p = plotter_no_boundary.with_grid(coords=False).plot_density()
        assert_image(p)

    def test_facet_by_leiden(self, plotter_no_boundary, assert_image):
        """Density faceted by cluster — one heatmap panel per leiden group."""
        p = plotter_no_boundary.facet(CAT_COL, n_col=3).plot_density()
        assert_image(p)

    def test_facet_with_grid(self, plotter_no_boundary, assert_image):
        """Faceted density combined with the grid overlay."""
        p = (
            plotter_no_boundary
            .facet(CAT_COL, n_col=3)
            .with_grid(labels=True)
            .plot_density()
        )
        assert_image(p)

    def test_focus_on_grid(self, plotter_no_boundary, assert_image):
        """Density zoomed to a grid window via focus_on_grid."""
        p = plotter_no_boundary.focus_on_grid("K12", "G9").plot_density()
        assert_image(p)

    def test_title_override(self, plotter_no_boundary, assert_image):
        """Custom title applied via .title()."""
        p = plotter_no_boundary.title("cell density").plot_density()
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

    def test_grid_coord_labels(self, plotter_no_boundary, assert_image):
        """Grid with embedding-coordinate (x, y) labels instead of letter strings."""
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(labels="coords")
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

    def test_vertical_letters_scatter(self, plotter_no_boundary, assert_image):
        """Grid with letters on the vertical axis instead of horizontal."""
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_grid(labels=True, vertical_letters=True)
            .plot("S100A8")
        )
        assert_image(p)


# ---------------------------------------------------------------------------
# Boundary (border) rendering — requires scikit-image (skips if absent)
# ---------------------------------------------------------------------------


class TestBoundaryRendering:
    """Tests that specifically exercise the cell-type boundary overlay."""

    def test_numerical_with_borders(self, plotter, assert_image):
        p = plotter.zeros(max_zero_value=-0.5).style(dot_size=DOT_SIZE).plot("S100A8")
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
            .zeros(max_zero_value=-0.5)
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
# plot / plot_density — 2-D faceting (facet_2d → facet_grid)
# ---------------------------------------------------------------------------


class TestFacet2D:
    def test_scatter_2d_grid(self, plotter_no_boundary, assert_image):
        """Scatter faceted into a 2-D grid (rows=bool, cols=coarse)."""
        p = (
            plotter_no_boundary
            .facet_2d("bool", "coarse")
            .style(dot_size=DOT_SIZE)
            .plot("S100A8")
        )
        assert_image(p)

    def test_density_2d_grid(self, plotter_no_boundary, assert_image):
        """Density heatmap faceted into a 2-D grid (rows=bool, cols=coarse)."""
        p = plotter_no_boundary.facet_2d("bool", "coarse").plot_density()
        assert_image(p)

    def test_2d_fixed_panel(self, plotter_no_boundary, assert_image):
        """2-D facet grid with a fixed per-panel size."""
        p = (
            plotter_no_boundary
            .facet_2d("bool", "coarse")
            .style(dot_size=DOT_SIZE)
            .panel_size(2, 2)
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

    def test_grid_limits(self, plotter_no_boundary, data, assert_image):
        coords = data.coordinates()
        x_mid = coords["x"].median()
        y_mid = coords["y"].median()
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .focus_on_grid("K12", "G9")
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

    def test_vertical_letters(self, plotter_no_boundary, assert_image):
        """Grid histogram with letters on the vertical axis."""
        p = plotter_no_boundary.with_grid(vertical_letters=True).plot_grid_histogram(
            CAT_COL, min_cell_count=10
        )
        assert_image(p)

    def test_vertical_stacking(self, plotter_no_boundary, assert_image):
        """Grid histogram with bars stacked vertically instead of horizontally."""
        p = plotter_no_boundary.plot_grid_histogram(
            CAT_COL, min_cell_count=10, vertical=True
        )
        assert_image(p)

    def test_scale_by_count(self, plotter_no_boundary, assert_image):
        """Grid histogram where tile size reflects cell observation count."""
        p = plotter_no_boundary.plot_grid_histogram(
            CAT_COL, min_cell_count=10, scale_by_count=True
        )
        assert_image(p)


# ---------------------------------------------------------------------------
# plot_histogram — global value_counts bar plot
# ---------------------------------------------------------------------------


class TestPlotHistogram:
    def test_basic(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.plot_histogram(CAT_COL)
        assert_image(p)

    def test_custom_colors(self, plotter_no_boundary, assert_image):
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
            "#00FFFF", "#000000", "#FFFFFF", "#888888",
        ]
        p = plotter_no_boundary.colormap_discrete(colors).plot_histogram(CAT_COL)
        assert_image(p)

    def test_facet(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.facet("coarse", n_col=3).plot_histogram(CAT_COL)
        assert_image(p)

    def test_facet_2d(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.facet_2d("bool", "coarse").plot_histogram(CAT_COL)
        assert_image(p)

    def test_fixed_panel(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.panel_size(3, 3).plot_histogram(CAT_COL)
        assert_image(p)


# ---------------------------------------------------------------------------
# panel_size — fixed scatter-panel dimensions
# ---------------------------------------------------------------------------


class TestPanelSize:
    def test_numerical_fixed_panel(self, plotter_no_boundary, assert_image):
        """Numerical plot with a fixed 3×3-inch panel."""
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).panel_size(3, 3).plot("S100A8")
        assert_image(p)

    def test_categorical_fixed_panel(self, plotter_no_boundary, assert_image):
        """Categorical plot with the same 3×3-inch panel — legend is wider."""
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).panel_size(3, 3).plot(CAT_COL)
        assert_image(p)

    def test_numerical_fixed_panel_with_borders(self, plotter, assert_image):
        """Fixed panel with cell-type border overlay — extra right-side decoration."""
        p = (
            plotter.zeros(max_zero_value=-0.5)
            .style(dot_size=DOT_SIZE)
            .panel_size(3, 3)
            .plot("S100A8")
        )
        assert_image(p)

    def test_grid_histogram_fixed_panel(self, plotter_no_boundary, assert_image):
        """panel_size applied to a grid histogram."""
        p = plotter_no_boundary.panel_size(3, 3).plot_grid_histogram(
            CAT_COL, min_cell_count=10
        )
        assert_image(p)

    def test_density_fixed_panel(self, plotter_no_boundary, assert_image):
        """panel_size applied to a density heatmap."""
        p = plotter_no_boundary.panel_size(3, 3).plot_density()
        assert_image(p)

    def test_density_facet_fixed_panel(self, plotter_no_boundary, assert_image):
        """panel_size applied to a faceted density heatmap (multi-row, multi-col).

        Exercises the fixed-panel resize across a facet grid; density plots take
        the frozen-layout path (the colourbar freezes the engine) which is
        facet-correct, so every panel should be 3x3 in.
        """
        p = (
            plotter_no_boundary
            .facet(CAT_COL, n_col=3)
            .panel_size(3, 3)
            .plot_density()
        )
        assert_image(p)

    def test_numerical_facet_fixed_panel(self, plotter_no_boundary, assert_image):
        """panel_size applied to a faceted numerical plot (multi-row, multi-col).

        Exercises the fixed-panel resize across a facet grid: every panel must
        come out 3x3 in.  _apply_fixed_panel's live-layout branch scales the
        figure-size correction by the facet's ncol/nrow.
        """
        p = (
            plotter_no_boundary
            .style(dot_size=DOT_SIZE)
            .facet(CAT_COL, n_col=3)
            .panel_size(3, 3)
            .plot("S100A8")
        )
        assert_image(p)

    def test_panel_size_in_composition(self, plotter_no_boundary, assert_image):
        """panel_size must be honoured when two plots are stacked with ``/``.

        plotnine's ``Compose`` bypasses ``ggplot.save_helper``, so without the
        composition post-draw patch the ``_apply_fixed_panel`` hook would never
        run and both panels would fall back to the default figure size.
        """
        p1 = plotter_no_boundary.style(dot_size=DOT_SIZE).panel_size(3, 3).plot("S100A8")
        p2 = plotter_no_boundary.style(dot_size=DOT_SIZE).panel_size(3, 3).plot(CAT_COL)
        assert_image(p1 / p2)


class TestColormaps:
    def test_numeric_manual_colors(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.colormap(cmap=["red", "white", "blue"]).plot("S100A8")
        assert_image(p)

    def test_numeric_map(self, plotter_no_boundary, assert_image):
        import matplotlib.cm

        p = plotter_no_boundary.colormap(
            cmap=matplotlib.cm.Reds, upper_clip_color="green"
        ).plot("S100A8")
        assert_image(p)

    def test_categorical_color_list(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.colormap_discrete(
            [
                "red",
                "grey",
                "blue",
                "purple",
                "green",
                "lime",
                "pink",
                "yellow",
                "darkgreen",
            ]
        ).plot("leiden")
        assert_image(p)

    def test_categorical_color_list_too_few(self, plotter_no_boundary):
        with pytest.raises(ValueError, match="not enough colors"):
            plotter_no_boundary.colormap_discrete(
                [
                    "red",
                    "grey",
                ]
            ).plot("leiden")

    def test_categorical_color_dict(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.colormap_discrete(
            {
                "9": "black",  # that's ignored.
                "8": "red",
                "7": "grey",
                "6": "blue",
                "5": "purple",
                "4": "green",
                "3": "lime",
                "2": "pink",
                "1": "yellow",
                "0": "darkgreen",
            }
        ).plot("leiden")
        assert_image(p)

    def test_categorical_color_map_missing(self, plotter_no_boundary):
        with pytest.raises(
            ValueError,
            match="not enough colors: dict is missing entries for: \\['7', '8'\\]",
        ):
            plotter_no_boundary.colormap_discrete(
                {
                    "6": "blue",
                    "5": "purple",
                    "4": "green",
                    "3": "lime",
                    "2": "pink",
                    "1": "yellow",
                    "0": "darkgreen",
                }
            ).plot("leiden")

    def test_bool(self, plotter_no_boundary, assert_image):
        p = plotter_no_boundary.colormap_discrete({True: "blue", False: "red"}).plot(
            "bool"
        )
        assert_image(p)


# ---------------------------------------------------------------------------
# plot_embedding_color — 2D positional color gradient
# ---------------------------------------------------------------------------


class TestPlotEmbeddingColor:
    def test_pca_in_umap(self, plotter_no_boundary, assert_image):
        """Color cells by PCA position, plotted in UMAP space.

        Each cell's color encodes where it sits in the first two PCA dimensions:
        red=top-left, blue=top-right, yellow=bottom-left, green=bottom-right.
        """
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "pca", show_legend=True
        )
        assert_image(p)

    def test_umap_in_umap(self, plotter_no_boundary, assert_image):
        """Self-referential: color cells by their own UMAP position.

        Should produce a smooth rainbow gradient perfectly matching the layout.
        """
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "umap", show_legend=True
        )
        assert_image(p)

    def test_custom_corners(self, plotter_no_boundary, assert_image):
        """Custom corner colors: cyan / magenta / white / black."""
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "pca",
            corner_colors=("#00CCCC", "#CC00CC", "#FFFFFF", "#111111"),
            show_legend=True,
        )
        assert_image(p)

    def test_no_legend(self, plotter_no_boundary, assert_image):
        """Verify show_legend=False suppresses the inset."""
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "pca", show_legend=False
        )
        assert_image(p)

    def test_region_grid(self, plotter_no_boundary, assert_image):
        """Grid-label region: only cells in A1–I12 of PCA space get the gradient."""
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "pca", region=("A1", "I12")
        )
        assert_image(p)

    def test_region_float(self, plotter_no_boundary, assert_image):
        """2-tuple bounding-box shorthand: gradient restricted to a centred box in PCA space."""
        ad = plotter_no_boundary._data.ad
        pca = ad.obsm["X_pca"][:, :2]
        cx = float(pca[:, 0].mean())
        cy = float(pca[:, 1].mean())
        hw_x = float((pca[:, 0].max() - pca[:, 0].min()) * 0.25)
        hw_y = float((pca[:, 1].max() - pca[:, 1].min()) * 0.25)
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "pca",
            region=((cx - hw_x, cy + hw_y), (cx + hw_x, cy - hw_y)),
        )
        assert_image(p)

    def test_region_rect(self, plotter_no_boundary, assert_image):
        """Rectangular 4-corner region in PCA space (axis-aligned box)."""
        ad = plotter_no_boundary._data.ad
        pca = ad.obsm["X_pca"][:, :2]
        cx = float(pca[:, 0].mean())
        cy = float(pca[:, 1].mean())
        hw_x = float((pca[:, 0].max() - pca[:, 0].min()) * 0.25)
        hw_y = float((pca[:, 1].max() - pca[:, 1].min()) * 0.25)
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "pca",
            region=(
                (cx - hw_x, cy + hw_y),  # top_left
                (cx + hw_x, cy + hw_y),  # top_right
                (cx - hw_x, cy - hw_y),  # bottom_left
                (cx + hw_x, cy - hw_y),  # bottom_right
            ),
        )
        assert_image(p)

    def test_region_quad(self, plotter_no_boundary, assert_image):
        """Non-rectangular quad region in PCA space — tilted parallelogram."""
        ad = plotter_no_boundary._data.ad
        pca = ad.obsm["X_pca"][:, :2]
        cx = float(pca[:, 0].mean())
        cy = float(pca[:, 1].mean())
        rng_x = float((pca[:, 0].max() - pca[:, 0].min()) * 0.3)
        rng_y = float((pca[:, 1].max() - pca[:, 1].min()) * 0.3)
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "pca",
            region=(
                (cx - rng_x * 0.3, cy + rng_y),  # top_left
                (cx + rng_x, cy + rng_y * 0.4),  # top_right
                (cx - rng_x, cy - rng_y * 0.4),  # bottom_left
                (cx + rng_x * 0.3, cy - rng_y),  # bottom_right
            ),
        )
        assert_image(p)

    def test_region_quad_show_region(self, plotter_no_boundary, assert_image):
        """show_region=True draws the region corners and connecting lines."""
        ad = plotter_no_boundary._data.ad
        pca = ad.obsm["X_pca"][:, :2]
        cx = float(pca[:, 0].mean())
        cy = float(pca[:, 1].mean())
        rng_x = float((pca[:, 0].max() - pca[:, 0].min()) * 0.3)
        rng_y = float((pca[:, 1].max() - pca[:, 1].min()) * 0.3)
        p = plotter_no_boundary.style(dot_size=DOT_SIZE).plot_embedding_color(
            "pca",
            region=(
                (cx - rng_x * 0.3, cy + rng_y),
                (cx + rng_x, cy + rng_y * 0.4),
                (cx - rng_x, cy - rng_y * 0.4),
                (cx + rng_x * 0.3, cy - rng_y),
            ),
            show_region=True,
        )
        assert_image(p)

    def test_region_rect_same_embedding(self, plotter_no_boundary, assert_image):
        """Rectangular region in PCA space, plotted in UMAP embedding."""
        ad = plotter_no_boundary._data.ad
        pca = ad.obsm["X_pca"][:, :2]
        cx = float(pca[:, 0].mean())
        cy = float(pca[:, 1].mean())
        hw_x = float((pca[:, 0].max() - pca[:, 0].min()) * 0.25)
        hw_y = float((pca[:, 1].max() - pca[:, 1].min()) * 0.25)
        p = (
            plotter_no_boundary.style(dot_size=DOT_SIZE)
            .with_embedding_label("umap")
            .plot_embedding_color(
                "umap",
                region=(
                    (cx - hw_x, cy + hw_y),
                    (cx + hw_x, cy + hw_y),
                    (cx - hw_x, cy - hw_y),
                    (cx + hw_x, cy - hw_y),
                ),
            )
        )
        assert_image(p)


# ---------------------------------------------------------------------------
# Reproducer: categorical legend overflow with many categories
# ---------------------------------------------------------------------------


class TestCategoricalLegendFit:
    """Reproducer: categorical legend should fit within the canvas when there
    are ~13 categories plotted at a small panel size and saved at a fixed
    width×height."""

    def test_13_categories_panel_3x3(self, ad, assert_image):
        """13-category legend with panel_size(3,3) saved at width=5, height=4."""
        from mbf_singlecell_plotter import EmbeddingData, ScatterPlotter

        # 13 distinct colors to go with 13 categories
        colors_13 = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
            "#469990", "#dcbeff", "#9A6324",
        ]

        n = len(ad.obs)
        ad.obs["cat13"] = pd.Categorical(
            [f"CellType_{i:02d}" for i in np.arange(n) % 13]
        )
        data = EmbeddingData(ad, "umap")
        p = (
            ScatterPlotter()
            .set_source(data)
            .with_grid(labels=False, coords=True, grid_size=12)
            .without_borders()
            .style(dot_size=0.1)
            .zeros(dot_size=0.1)
            .panel_size(3, 3)
            .colormap_discrete(colors_13)
            .plot("cat13")
        )
        assert_image(p, width=5, height=4, dpi=300)

    def test_14_categories_panel_3x3(self, ad, assert_image):
        """14-category legend with panel_size(3,3) saved at width=5, height=4 — triggers overflow."""
        from mbf_singlecell_plotter import EmbeddingData, ScatterPlotter

        colors_14 = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
            "#469990", "#dcbeff", "#9A6324", "#800000",
        ]

        n = len(ad.obs)
        ad.obs["cat14"] = pd.Categorical(
            [f"CellType_{i:02d}" for i in np.arange(n) % 14]
        )
        data = EmbeddingData(ad, "umap")
        p = (
            ScatterPlotter()
            .set_source(data)
            .with_grid(labels=False, coords=True, grid_size=12)
            .without_borders()
            .style(dot_size=0.1)
            .zeros(dot_size=0.1)
            .panel_size(3, 3)
            .colormap_discrete(colors_14)
            .plot("cat14")
        )
        assert_image(p, width=5, height=4, dpi=300)
