"""Layer 3: plotnine plot builders."""

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotnine as p9
from natsort import natsorted

from .data import EmbeddingData, ColumnData, _LETTERS
from .theme import DEFAULT_COLORS, embedding_theme


# ── Custom matplotlib colorbar legend ────────────────────────────────────────


class _PlotWithCustomLegend(p9.ggplot):
    """ggplot subclass that replaces the auto color guide with a custom matplotlib colorbar."""

    def save_helper(self, **kwargs):
        sv = super().save_helper(**kwargs)
        _draw_numerical_legend(sv.figure, **self._legend_config)
        return sv


class _PlotWithVerticalLegendTitle(p9.ggplot):
    """ggplot subclass that adds a vertical title to the right-side categorical legend."""

    def save_helper(self, **kwargs):
        sv = super().save_helper(**kwargs)
        _draw_vertical_cat_legend_title(sv.figure, self._cat_legend_title, self._cat_legend_fontsize)
        return sv


def _draw_vertical_cat_legend_title(fig, title: str, fontsize: float) -> None:
    """Add a rotated title to the left of the categorical legend offsetbox."""
    le = fig.get_layout_engine()
    if le is not None:
        le.execute(fig)
        fig.set_layout_engine(None)

    # Find the FlexibleAnchoredOffsetbox that plotnine uses for the legend
    legend_box = None
    for child in fig.get_children():
        if type(child).__name__ == "FlexibleAnchoredOffsetbox":
            legend_box = child
            break

    if legend_box is None:
        return

    bb = legend_box.get_window_extent()
    fig_bb = bb.transformed(fig.transFigure.inverted())

    cx = fig_bb.x0 - 0.025
    cy = (fig_bb.y0 + fig_bb.y1) / 2
    fig.text(cx, cy, title, rotation=90, va="center", ha="center", fontsize=fontsize)


def _draw_numerical_legend(
    fig,
    *,
    expr_min: float,
    clip_val: float,
    cmap_colors: list,
    has_zeros: bool,
    zero_color: str,
    has_clips: bool,
    upper_clip_color: str,
    cbar_title: str,
    breaks: list,
    labels: list,
    base_size: float = 12,
    title_position: str = "side",
) -> None:
    """Add a custom colorbar with rectangular extension boxes to a plotnine figure."""
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    # Finalise axes positions via the plotnine layout engine, then freeze it
    le = fig.get_layout_engine()
    if le is not None:
        le.execute(fig)
        fig.set_layout_engine(None)

    # Build colormap with over/under colors for extensions
    cmap = LinearSegmentedColormap.from_list("_cbar", cmap_colors)
    if has_zeros:
        cmap.set_under(zero_color)
    if has_clips:
        cmap.set_over(upper_clip_color)

    if has_zeros and has_clips:
        extend = "both"
    elif has_zeros:
        extend = "min"
    elif has_clips:
        extend = "max"
    else:
        extend = "neither"

    norm = mpl.colors.Normalize(vmin=expr_min, vmax=clip_val)

    extendfrac = 0.05  # extension boxes = 5% of bar height each

    # ── Compute combined bounding box of all data axes ────────────────────────
    # For faceted plots there are multiple panels; we want the colorbar anchored
    # to the right edge of the rightmost panel and spanning the full grid height.
    all_axes = fig.axes
    grid_x1     = max(ax.get_position().x1 for ax in all_axes)
    grid_y0     = min(ax.get_position().y0 for ax in all_axes)
    grid_height = max(ax.get_position().y1 for ax in all_axes) - grid_y0

    # For single-panel figures keep a reference to the one axes for shrinking.
    main_ax = all_axes[0]

    # ── Layout: title | gap | bar (tick labels auto to the right) ────────────
    gap = 0.008
    bar_frac = 0.035  # colorbar bar width as fraction of figure

    if title_position == "top":
        # No side title column; bar starts right after the grid
        bar_left = grid_x1 + gap
        legend_width = gap + bar_frac + 0.10
    else:
        # "side": vertical title text to the left of the bar
        title_half = 0.025  # half-width of title text area
        bar_left = grid_x1 + 2 * title_half + gap
        legend_width = 2 * title_half + gap + bar_frac + 0.10

    needed_right = grid_x1 + legend_width

    if needed_right > 0.99:
        # Scale all data axes proportionally so the legend fits within the figure.
        target_x1 = 0.99 - legend_width
        scale = target_x1 / grid_x1
        for ax in all_axes:
            p = ax.get_position()
            ax.set_position([p.x0 * scale, p.y0, p.width * scale, p.height])
        grid_x1 = max(ax.get_position().x1 for ax in all_axes)
        if title_position == "top":
            bar_left = grid_x1 + gap
        else:
            bar_left = grid_x1 + 2 * title_half + gap

    cbar_ax = fig.add_axes([bar_left, grid_y0, bar_frac, grid_height])

    # Drop the boundary ticks that would overlap with extension box labels.
    tick_breaks = list(breaks)
    tick_labels = list(labels)
    if has_clips and tick_breaks:
        tick_breaks = tick_breaks[:-1]
        tick_labels = tick_labels[:-1]
    if has_zeros and tick_breaks:
        tick_breaks = tick_breaks[1:]
        tick_labels = tick_labels[1:]

    cb = mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap=cmap,
        norm=norm,
        extend=extend,
        extendrect=True,
        extendfrac=extendfrac,
        orientation="vertical",
        ticks=tick_breaks,
    )
    cb.set_ticklabels(tick_labels)
    legend_fontsize = base_size * 0.9
    cb.ax.tick_params(labelsize=legend_fontsize, length=3)
    cb.ax.yaxis.set_tick_params(which="both", labelleft=False, labelright=True)

    # ── Extension box labels ──────────────────────────────────────────────────
    # The colorbar gradient fills transAxes [0, 1]; the extension boxes are
    # rendered *outside* that range: bottom extension at [-extendfrac, 0] and
    # top extension at [1, 1+extendfrac].  Their centres are therefore at
    # -extendfrac/2 and 1+extendfrac/2 in transAxes coordinates.
    if has_zeros:
        cb.ax.text(
            1.08,
            -extendfrac / 2,
            "0",
            transform=cb.ax.transAxes,
            va="center",
            ha="left",
            fontsize=legend_fontsize,
            clip_on=False,
        )
    if has_clips:
        cb.ax.text(
            1.08,
            1.0 + extendfrac / 2,
            f">{labels[-1]}",
            transform=cb.ax.transAxes,
            va="center",
            ha="left",
            fontsize=legend_fontsize,
            clip_on=False,
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    if title_position == "top":
        # Horizontal title above the bar
        title_cx = bar_left + bar_frac / 2
        title_cy = grid_y0 + grid_height + 0.02
        fig.text(
            title_cx,
            title_cy,
            cbar_title,
            rotation=0,
            va="bottom",
            ha="center",
            fontsize=legend_fontsize,
        )
    else:
        # Vertical title to the LEFT of the bar
        title_cx = grid_x1 + title_half
        title_cy = grid_y0 + grid_height * 0.5
        fig.text(
            title_cx,
            title_cy,
            cbar_title,
            rotation=90,
            va="center",
            ha="center",
            fontsize=legend_fontsize,
        )


@dataclass(frozen=True)
class BorderConfig:
    size: float = 15
    resolution: int = 200
    blur: float = 1.1
    threshold: float = 0.95


@dataclass(frozen=True)
class GridConfig:
    labels: bool = False
    coords: bool = False
    vertical_letters: bool = False
    grid_size: int = 12
    color: str = "#777777"
    label_color: str = "#777777"


class ScatterPlotter:
    """Reusable builder for embedding scatter plots.

    Configure once, plot many genes::

        plotter = (
            ScatterPlotter()
            .set_source(ad, embedding="umap")
            .dot_size(5)
            .with_borders(cell_type_column="leiden")
            .with_grid(labels=True)
        )
        plotter.plot("S100A8")
        plotter.plot("leiden")
    """

    def __init__(
        self, ad_or_data=None, embedding: str = "umap", base_size=12, fig_size=None
    ):
        self._data: Optional[EmbeddingData] = None
        self._cell_type_column: Optional[str] = None

        # basic plot options
        self.base_size = base_size
        self.fig_size = fig_size

        # dot appearance
        self._dot_size: float = 1
        self._panel_border: bool = True
        self._spine_color: str = "#555555"
        self._tick_color: str = "#555555"
        self._bg_color: str = "#FFFFFF"
        self._legend_title_position: Optional[str] = None  # None → auto ("top" for cat, "side" for num)
        self._anti_overplot: bool = True
        self._flip_order: bool = False
        self._outlier_quantile: float = 0.95
        self._outlier_shape: Optional[str] = None  # None → same shape as main dots

        # colormap config (numerical)
        self._cmap = None  # None → default blue-magenta, or list of color strings
        self._max_quantile: float = 0.95
        self._upper_clip_color: str = "#FF0000"
        self._cbar_title: Optional[str] = None  # None → auto from gene name

        # zero handling
        self._zero_color: str = "#D0D0D0"
        self._zero_dot_size: float = 3
        self._zero_value: Optional[float] = None

        # categorical colormap
        self._cat_colors: Optional[list] = None  # None → DEFAULT_COLORS

        # layer visibility
        self._layer_borders: bool = True
        self._layer_zeros: bool = True
        self._layer_data: bool = True
        self._layer_outliers: bool = True

        # optional layers
        self._border_config: Optional[BorderConfig] = None
        self._boundary_cache: dict = {"df": None}
        self._grid_config: Optional[GridConfig] = GridConfig(coords=True)
        self._facet_variable: Optional[str] = None
        self._n_col: int = 2
        self._title_override = _UNSET

        if ad_or_data is not None:
            if isinstance(ad_or_data, EmbeddingData):
                self._data = ad_or_data
            else:
                self._data = EmbeddingData(ad_or_data, embedding)

    # ── source ──────────────────────────────────────────────────────────────

    def set_source(
        self,
        ad_or_data,
        embedding: str = "umap",
    ) -> "ScatterPlotter":
        """Attach data source. Accepts AnnData or an existing EmbeddingData."""
        new = copy.copy(self)
        if isinstance(ad_or_data, EmbeddingData):
            new._data = ad_or_data
        else:
            # Preserve grid config from existing _data if present
            grid_size = self._data._grid_size if self._data is not None else 12
            glv = (
                self._data._grid_letters_on_vertical
                if self._data is not None
                else False
            )
            new._data = EmbeddingData(
                ad_or_data,
                embedding,
                grid_size=grid_size,
                grid_letters_on_vertical=glv,
            )
        new._boundary_cache = {"df": None}
        return new

    # ── dot appearance ───────────────────────────────────────────────────────

    def style(
        self,
        *,
        dot_size: Optional[float] = None,
        panel_border: Optional[bool] = None,
        spine_color: Optional[str] = None,
        tick_color: Optional[str] = None,
        bg_color: Optional[str] = None,
        legend_title_position: Optional[str] = None,
    ) -> "ScatterPlotter":
        """Configure visual appearance. Only supplied arguments are changed.

        Args:
            dot_size:    Point size for the main scatter layer.
            panel_border: Show/hide the panel border (True = show).
            spine_color: Hex color for the panel border (default ``"#555555"``).
            tick_color:  Hex color for axis ticks and tick labels (default ``"#555555"``).
            bg_color:    Background hex color (e.g. ``"#FFFFFF"``).
            legend_title_position: ``"top"`` (title above legend, horizontal) or
                ``"side"`` (title left of legend, vertical).  ``None`` → auto
                (``"top"`` for categorical, ``"side"`` for numerical).
        """
        new = copy.copy(self)
        if dot_size is not None:
            new._dot_size = dot_size
        if panel_border is not None:
            new._panel_border = panel_border
        if spine_color is not None:
            new._spine_color = spine_color
        if tick_color is not None:
            new._tick_color = tick_color
        if bg_color is not None:
            new._bg_color = bg_color
        if legend_title_position is not None:
            new._legend_title_position = legend_title_position
        return new

    def flip_draw_order(self, value: bool = True) -> "ScatterPlotter":
        """Reverse categorical draw order (last category drawn on top when False)."""
        new = copy.copy(self)
        new._flip_order = value
        return new

    def outlier(
        self,
        *,
        shape: Optional[str] = None,
        quantile: Optional[float] = None,
    ) -> "ScatterPlotter":
        """Configure the categorical outlier replot pass.

        Args:
            shape:    Marker shape for outlier points (e.g. ``"^"``, ``"D"``).
                      None keeps the same shape as the main scatter dots.
            quantile: Distance quantile above which a point is an outlier
                      (default 0.95).
        """
        new = copy.copy(self)
        if shape is not None:
            new._outlier_shape = shape
        if quantile is not None:
            new._outlier_quantile = quantile
        return new

    # ── colormap (numerical) ─────────────────────────────────────────────────

    def colormap(
        self,
        cmap=None,
        *,
        max_quantile: float = 0.95,
        upper_clip_color: str = "#FF0000",
        title: Optional[str] = None,
    ) -> "ScatterPlotter":
        """Configure the continuous colormap.

        cmap may be a list of color strings, a matplotlib colormap, or None
        (default: black→blue→magenta).
        """
        new = copy.copy(self)
        new._cmap = cmap
        new._max_quantile = max_quantile
        new._upper_clip_color = upper_clip_color
        new._cbar_title = title
        return new

    # ── categorical colors ───────────────────────────────────────────────────

    def colormap_discrete(self, cmap_or_list_or_dict) -> "ScatterPlotter":
        """Set the discrete color palette for categorical data.

        Accepts:
        - A list of hex color strings (positional, cycling).
        - A dict mapping category name → hex color string.
        - A matplotlib ``ListedColormap`` or similar (uses ``.colors``).
        """
        new = copy.copy(self)
        if isinstance(cmap_or_list_or_dict, dict):
            new._cat_colors = cmap_or_list_or_dict
        elif isinstance(cmap_or_list_or_dict, list):
            new._cat_colors = cmap_or_list_or_dict
        else:
            # Assume matplotlib ListedColormap or similar
            new._cat_colors = list(cmap_or_list_or_dict.colors)
        return new

    # ── zero handling ────────────────────────────────────────────────────────

    def zeros(
        self,
        *,
        color: Optional[str] = None,
        dot_size: Optional[float] = None,
        zero_value: Optional[float] = None,
    ) -> "ScatterPlotter":
        """Configure zero-value rendering (appearance only; use layers(zeros=) to toggle visibility)."""
        new = copy.copy(self)
        if color is not None:
            new._zero_color = color
        if dot_size is not None:
            new._zero_dot_size = dot_size
        if zero_value is not None:
            new._zero_value = zero_value
        return new

    # ── borders ──────────────────────────────────────────────────────────────

    def with_borders(
        self,
        *,
        cell_type_column: Optional[str] = None,
        size: float = 15,
        resolution: int = 200,
        blur: float = 1.1,
        threshold: float = 0.95,
    ) -> "ScatterPlotter":
        new = copy.copy(self)
        new._border_config = BorderConfig(
            size=size, resolution=resolution, blur=blur, threshold=threshold
        )
        if cell_type_column is not None:
            new._cell_type_column = cell_type_column
        # Reset cache only if image-processing params changed
        old = self._border_config
        if (
            old is None
            or old.resolution != resolution
            or old.blur != blur
            or old.threshold != threshold
            or cell_type_column != self._cell_type_column
        ):
            new._boundary_cache = {"df": None}
        # else: share the same cache dict (shallow copy) — only size changed
        return new

    def without_borders(self) -> "ScatterPlotter":
        new = copy.copy(self)
        new._border_config = None
        return new

    # ── layer visibility ─────────────────────────────────────────────────────

    def layers(
        self,
        *,
        borders: Optional[bool] = None,
        zeros: Optional[bool] = None,
        data: Optional[bool] = None,
        outliers: Optional[bool] = None,
    ) -> "ScatterPlotter":
        """Toggle individual rendering layers.

        Args:
            borders:  Show/hide the cell-type border overlay.
            zeros:    Show/hide the zero-expression underlay.  When False,
                      zero-valued points are folded into the data layer and
                      coloured by the gradient instead of a flat zero colour.
            data:     Show/hide the main scatter layer.
            outliers: Show/hide the categorical outlier replot pass.
        """
        new = copy.copy(self)
        if borders is not None:
            new._layer_borders = borders
        if zeros is not None:
            new._layer_zeros = zeros
        if data is not None:
            new._layer_data = data
        if outliers is not None:
            new._layer_outliers = outliers
        return new

    # ── grid overlay ─────────────────────────────────────────────────────────

    def with_grid(
        self,
        *,
        labels: Optional[bool] = None,
        coords: Optional[bool] = None,
        vertical_letters: Optional[bool] = None,
        grid_size: Optional[int] = None,
        color: Optional[str] = None,
        label_color: Optional[str] = None,
    ) -> "ScatterPlotter":
        """
        labels: show "A1", "B2" text labels inside each grid cell
        coords: use grid coords as axis tick labels

        Only supplied arguments are changed; unspecified ones inherit from the
        current grid config (or GridConfig defaults if no grid is set yet).
        """
        cur = self._grid_config if self._grid_config is not None else GridConfig()
        resolved_grid_size = grid_size if grid_size is not None else cur.grid_size
        resolved_vl = vertical_letters if vertical_letters is not None else cur.vertical_letters
        if resolved_grid_size > 26:
            raise ValueError("grid_size max is 26")
        new = copy.copy(self)
        new._grid_config = GridConfig(
            labels=labels if labels is not None else cur.labels,
            coords=coords if coords is not None else cur.coords,
            vertical_letters=resolved_vl,
            grid_size=resolved_grid_size,
            color=color if color is not None else cur.color,
            label_color=label_color if label_color is not None else cur.label_color,
        )
        # Sync EmbeddingData grid settings if needed
        if new._data is not None and (
            new._data._grid_size != resolved_grid_size
            or new._data._grid_letters_on_vertical != resolved_vl
        ):
            new._data = EmbeddingData(
                new._data.ad,
                new._data._embedding
                if new._data._embedding_cols is None
                else (
                    new._data._embedding,
                    new._data._embedding_cols[0],
                    new._data._embedding_cols[1],
                ),
                alternative_id_column=new._data._alternative_id_column,
                grid_size=resolved_grid_size,
                grid_letters_on_vertical=resolved_vl,
            )
        return new

    def without_grid(self) -> "ScatterPlotter":
        new = copy.copy(self)
        new._grid_config = None
        return new

    # ── viewport ─────────────────────────────────────────────────────────────

    def focus_on(self, *, x: tuple, y: tuple) -> "ScatterPlotter":
        """Restrict viewport to a coordinate window.

        Args:
            x: (x_min, x_max)
            y: (y_min, y_max)
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .focus_on()")
        new = copy.copy(self)
        new._data = self._data.focus_on(x=x, y=y)
        return new

    def unfocus(self) -> "ScatterPlotter":
        if self._data is None:
            raise RuntimeError("call .set_source() before .unfocus()")
        new = copy.copy(self)
        new._data = self._data.unfocus()
        return new

    # ── faceting ─────────────────────────────────────────────────────────────

    def facet(self, variable: str, n_col: int = 2) -> "ScatterPlotter":
        new = copy.copy(self)
        new._facet_variable = variable
        new._n_col = n_col
        return new

    def unfacet(self) -> "ScatterPlotter":
        new = copy.copy(self)
        new._facet_variable = None
        return new

    # ── title ────────────────────────────────────────────────────────────────

    def title(self, t: str) -> "ScatterPlotter":
        new = copy.copy(self)
        new._title_override = t
        return new

    # ── terminal ─────────────────────────────────────────────────────────────

    def plot(self, column: str) -> p9.ggplot:
        """Build and return a plotnine ggplot for the given obs column or gene."""
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot()")

        data = self._data
        coords = data.coordinates()
        x_min, x_max, y_min, y_max = data.bounds()

        # Load expression data
        expr, expr_name = data.get_column(column)
        is_numerical = (
            (expr.dtype != "object")
            and (expr.dtype != "category")
            and (expr.dtype != "bool")
        )
        if not is_numerical and expr.dtype != "category":
            expr = expr.astype("category")

        df = coords.copy()
        df["expression"] = expr

        # Facet column
        if self._facet_variable is not None:
            facet_vals, _ = data.get_column(self._facet_variable)
            df["facet"] = facet_vals

        # Figure size
        if self.fig_size is None:
            if self._facet_variable is not None:
                n_facets = df["facet"].nunique()
                n_row = (n_facets + self._n_col - 1) // self._n_col
                fig_size = (6 * self._n_col, 5 * n_row)
            else:
                fig_size = (6, 5)

        # Build plot
        legend_config = None
        if is_numerical:
            p, legend_config = self._build_numerical(df, expr_name)
        else:
            p = self._build_categorical(df, expr_name)

        # Focus viewport
        if data.has_focus:
            p = p + p9.coord_cartesian(xlim=(x_min, x_max), ylim=(y_min, y_max))

        # Facet
        if self._facet_variable is not None:
            p = p + p9.facet_wrap("~facet", ncol=self._n_col)

        # Title
        if self._title_override is not _UNSET:
            p = p + p9.labs(title=self._title_override)
        elif is_numerical:
            p = p + p9.labs(title=expr_name)

        # Theme (must come before grid axis ticks so theme_void doesn't override them)
        p = p + embedding_theme(
            base_size=self.base_size,
            show_spines=self._panel_border,
            bg_color=self._bg_color,
            spine_color=self._spine_color,
        )
        p = p + p9.theme(figure_size=fig_size)

        # Grid axis tick labels (applied after theme so they survive theme_void)
        if self._grid_config is not None and self._grid_config.coords:
            p = self._add_grid_axis_ticks(p)

        # Wrap numerical plots so the custom legend is injected at save time
        if legend_config is not None:
            p.__class__ = _PlotWithCustomLegend
            p._legend_config = legend_config

        return p

    def plot_density(self, bins: int = 200, quantile: float = 0.99) -> p9.ggplot:
        """Build a 2D cell-density heatmap.

        Args:
            bins:     Number of bins per axis for the 2D histogram.
            quantile: Upper quantile at which density is clipped (default 0.99).
                      Set to 1.0 for no clipping (uses the built-in plotnine legend).
                      Any value < 1.0 clips the colour scale and draws the same
                      custom matplotlib colourbar as numerical scatter plots.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_density()")

        from .transforms import prepare_density_df

        df = prepare_density_df(self._data, bins=bins)

        has_clips = quantile < 1.0
        if has_clips:
            clip_val = float(df["density"].quantile(quantile))
        else:
            clip_val = float(df["density"].max())

        nonzero = df["density"][df["density"] > 0]
        density_min = float(nonzero.min()) if len(nonzero) > 0 else 0.0

        cmap_colors = ["#BFBFFF", "#0000FF"]
        breaks = list(np.linspace(density_min, clip_val, 5))
        labels = [f"{b:.2f}" for b in breaks]

        p = (
            p9.ggplot(df, p9.aes("x", "y", fill="density"))
            + p9.geom_tile(p9.aes(width="x_width", height="y_width"))
            + p9.scale_fill_gradientn(
                colors=cmap_colors,
                limits=(density_min, clip_val),
                breaks=breaks,
                labels=labels,
                na_value="#FFFFFF",
                name="density",
            )
        )

        if has_clips:
            p = p + p9.guides(fill="none")

        # Boundary overlay
        if self._layer_borders and self._border_config is not None and self._cell_type_column is not None:
            bdf = self._get_boundary_df()
            border_pt = self._border_config.size / 10
            for color in bdf["color"].unique():
                sub = bdf[bdf["color"] == color]
                p = p + p9.geom_point(
                    data=sub,
                    mapping=p9.aes("x", "y"),
                    color=color,
                    size=border_pt,
                    inherit_aes=False,
                )

        p = p + embedding_theme(
            base_size=self.base_size,
            show_spines=self._panel_border,
            bg_color=self._bg_color,
            spine_color=self._spine_color,
        )
        p = p + p9.theme(figure_size=(6, 5))

        if has_clips:
            legend_config = dict(
                expr_min=density_min,
                clip_val=clip_val,
                cmap_colors=cmap_colors,
                has_zeros=False,
                zero_color="#FFFFFF",
                has_clips=True,
                upper_clip_color=cmap_colors[-1],
                cbar_title="density",
                breaks=breaks,
                labels=labels,
                base_size=self.base_size,
                title_position=self._legend_title_position or "side",
            )
            p.__class__ = _PlotWithCustomLegend
            p._legend_config = legend_config

        return p

    def plot_grid_histogram(self, column: str, min_cell_count: int = 10) -> p9.ggplot:
        """Build a grid-local category frequency heatmap (plotnine)."""
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_grid_histogram()")

        hdf = self._data.grid_local_histogram(column, min_cell_count)
        hdf["category"] = pd.Categorical(
            hdf["category"], sorted(hdf["category"].unique())
        )
        hdf = hdf.sort_values(["x", "y", "category"])
        cats = list(hdf["category"].cat.categories)
        colors = self._colors_as_list(cats)

        factor = 0.8
        hdf["frequency"] = hdf["frequency"] * factor

        x_offset = []
        for _ignored, group in hdf.groupby(["x", "y"]):
            x_offset.extend(group["frequency"].cumsum().shift(fill_value=0))
        hdf["x_offset"] = x_offset

        hdf["x_plot"] = (
            hdf["x"] - hdf["frequency"] / 2 - hdf["x_offset"] - (1 - factor) / 2
        )
        hdf["y_plot"] = hdf["y"] + 0.5

        grid_size = self._data._grid_size
        _x_ticks, _y_ticks, x_labels, y_labels = self._data.grid_labels()
        x_ticks = list(range(grid_size))
        y_ticks = list(range(grid_size))

        hdf = hdf[::-1]

        p = (
            p9.ggplot(
                hdf,
                p9.aes(
                    x="x",
                    y="y",
                    width="frequency",
                    height=factor,
                    fill="category",
                ),
            )
            + p9.theme_bw()
            + p9.geom_hline(
                p9.aes(yintercept="xx"),
                data=pd.DataFrame({"xx": list(range(grid_size + 1))}),
                color="#D0D0D0",
            )
            + p9.geom_vline(
                p9.aes(xintercept="xx"),
                data=pd.DataFrame({"xx": [x - 1 for x in range(grid_size + 1)]}),
                color="#D0D0D0",
            )
            + p9.geom_tile(p9.aes(x="x_plot", y="y_plot"))
            + p9.coord_fixed()
            + p9.scale_x_continuous(
                expand=(0, 0.5, 0, 0.5),
                breaks=x_ticks,
                labels=x_labels,
            )
            + p9.scale_y_continuous(
                expand=(0, 0.5, 0, 0.5),
                breaks=y_ticks,
                labels=y_labels,
            )
            + p9.scale_fill_manual(colors)
            + p9.theme(
                axis_title_x=p9.element_blank(),
                axis_title_y=p9.element_blank(),
                panel_grid=p9.element_blank(),
                axis_ticks_length=3,
                axis_title=p9.element_blank(),
            )
        )
        return p

    def render(self, column: str, path: str, **kwargs):
        """Plot and save to *path*. ``**kwargs`` are forwarded to :meth:`plotnine.ggplot.save`."""
        self.plot(column).save(path, **kwargs)

    # ── internals ────────────────────────────────────────────────────────────

    def _colors_as_list(self, cats: list) -> list:
        """Return an ordered color list for *cats*.

        Handles both list (positional cycling) and dict (name → color) forms
        of ``_cat_colors``.
        """
        if isinstance(self._cat_colors, dict):
            return [
                self._cat_colors.get(str(c), DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                for i, c in enumerate(cats)
            ]
        return self._cat_colors or DEFAULT_COLORS

    def _get_cmap_colors(self) -> list:
        if self._cmap is None:
            return ["#000000", "#0000FF", "#FF00FF"]
        if isinstance(self._cmap, list):
            return self._cmap
        # Assume matplotlib colormap object — sample 10 colors
        import matplotlib.colors as mcolors

        return [mcolors.to_hex(self._cmap(i / 9)) for i in range(10)]

    def _get_boundary_df(self) -> pd.DataFrame:
        if self._boundary_cache["df"] is None:
            from .transforms import compute_boundaries

            # Resolve cats so dict-based palettes can be ordered correctly
            cell_types, _ = self._data.get_column(self._cell_type_column)
            if hasattr(cell_types, "cat"):
                cats = list(cell_types.cat.categories)
            else:
                cats = natsorted(cell_types.unique())
            colors = self._colors_as_list(cats)
            bc = self._border_config
            self._boundary_cache["df"] = compute_boundaries(
                data=self._data,
                cell_type_column=self._cell_type_column,
                colors=colors,
                resolution=bc.resolution,
                blur=bc.blur,
                threshold=bc.threshold,
            )
        return self._boundary_cache["df"]

    def _add_border_layers(self, p: p9.ggplot) -> p9.ggplot:
        bdf = self._get_boundary_df()
        border_pt = self._border_config.size / 10
        for color in bdf["color"].unique():
            sub = bdf[bdf["color"] == color]
            p = p + p9.geom_point(
                data=sub,
                mapping=p9.aes("x", "y"),
                color=color,
                size=border_pt,
                inherit_aes=False,
            )
        return p

    def _build_numerical(
        self,
        df: pd.DataFrame,
        expr_name: str,
    ) -> p9.ggplot:
        zero_val = self._zero_value
        if zero_val is None:
            zero_val = df["expression"].min()

        # When the zeros layer is hidden, fold zero-valued points into the
        # gradient so they get a colour rather than being dropped.
        if self._layer_zeros:
            df_zeros = df[df["expression"] <= zero_val]
            df_nonzero = df[~(df["expression"] <= zero_val)].copy()
        else:
            df_zeros = df.iloc[:0]  # empty — nothing to render as flat colour
            df_nonzero = df.copy()  # all points go through the gradient

        if len(df_nonzero) == 0:
            clip_val = 1.0
            expr_min = 0.0
        else:
            clip_val = float(df_nonzero["expression"].quantile(self._max_quantile))
            expr_min = float(df_nonzero["expression"].min())

        # Split into gradient range and clipped-above values
        df_normal = df_nonzero[df_nonzero["expression"] <= clip_val].copy()
        df_above = df_nonzero[df_nonzero["expression"] > clip_val].copy()

        if self._anti_overplot:
            df_normal = df_normal.sort_values("expression")

        df_normal["expression_plot"] = df_normal["expression"]

        p = p9.ggplot(df_normal, p9.aes("x", "y", color="expression_plot"))

        # Grid lines first so they render behind all other layers
        if self._grid_config is not None:
            p = self._add_grid_layers(p)

        # Boundary layer (behind scatter)
        if self._layer_borders and self._border_config is not None and self._cell_type_column is not None:
            p = self._add_border_layers(p)

        # Zero underlay
        if self._layer_zeros and len(df_zeros) > 0:
            p = p + p9.geom_point(
                data=df_zeros,
                mapping=p9.aes("x", "y"),
                color=self._zero_color,
                size=self._zero_dot_size,
                inherit_aes=False,
            )

        # Main scatter (gradient range)
        if self._layer_data:
            p = p + p9.geom_point(size=self._dot_size)

        # Clipped values drawn on top in clip color
        if self._layer_data and len(df_above) > 0:
            p = p + p9.geom_point(
                data=df_above,
                mapping=p9.aes("x", "y"),
                color=self._upper_clip_color,
                size=self._dot_size,
                inherit_aes=False,
            )

        # Color scale
        cmap_colors = self._get_cmap_colors()
        breaks = list(np.linspace(expr_min, clip_val, 5))
        labels = [f"{b:.2f}" for b in breaks]

        cbar_name = (
            self._cbar_title
            if self._cbar_title is not None
            else (expr_name + ": log2 expression")
        )
        p = p + p9.scale_color_gradientn(
            colors=cmap_colors,
            limits=(expr_min, clip_val),
            breaks=breaks,
            labels=labels,
            name=cbar_name,
        )
        # Suppress plotnine's auto-guide; we'll draw a custom matplotlib one
        p = p + p9.guides(color="none")

        legend_config = dict(
            expr_min=expr_min,
            clip_val=clip_val,
            cmap_colors=cmap_colors,
            has_zeros=len(df_zeros) > 0,
            zero_color=self._zero_color,
            has_clips=len(df_above) > 0,
            upper_clip_color=self._upper_clip_color,
            cbar_title=cbar_name,
            breaks=breaks,
            labels=labels,
            base_size=self.base_size,
            title_position=self._legend_title_position or "side",
        )
        return p, legend_config

    def _build_categorical(
        self,
        df: pd.DataFrame,
        expr_name: str,
    ) -> p9.ggplot:
        if df["expression"].dtype == "category":
            cats = list(df["expression"].cat.categories)
        else:
            cats = natsorted(df["expression"].unique())

        colors = self._colors_as_list(cats)
        color_values = {str(c): colors[i % len(colors)] for i, c in enumerate(cats)}

        # Sort for draw order (flip_order controls which appears on top)
        cat_order = {c: i for i, c in enumerate(cats)}
        df = df.copy()
        df["_sort_key"] = df["expression"].map(cat_order)
        df = df.sort_values("_sort_key", ascending=not self._flip_order).drop(
            columns=["_sort_key"]
        )

        p = p9.ggplot(df, p9.aes("x", "y", color="expression"))

        # Grid lines first so they render behind all other layers
        if self._grid_config is not None:
            p = self._add_grid_layers(p)

        # Boundary layer (behind scatter)
        if self._layer_borders and self._border_config is not None and self._cell_type_column is not None:
            p = self._add_border_layers(p)

        # Scatter
        if self._layer_data:
            p = p + p9.geom_point(size=self._dot_size)

        # Outlier replot
        if self._layer_outliers:
            outlier_dfs = []
            for cat in cats:
                sdf = df[df["expression"] == cat]
                if len(sdf) == 0:
                    continue
                center_x, center_y = sdf["x"].mean(), sdf["y"].mean()
                dist = np.sqrt((sdf["x"] - center_x) ** 2 + (sdf["y"] - center_y) ** 2)
                thresh = dist.quantile(self._outlier_quantile)
                outlier_dfs.append(sdf[dist > thresh])
            if outlier_dfs:
                df_outliers = pd.concat(outlier_dfs)
                extra = {} if self._outlier_shape is None else {"shape": self._outlier_shape}
                p = p + p9.geom_point(
                    data=df_outliers,
                    size=self._dot_size,
                    inherit_aes=True,
                    **extra,
                )

        # Legend title position: auto defaults to "top" for categorical
        title_pos = self._legend_title_position or "top"
        if title_pos == "side":
            # Suppress plotnine's title; we'll draw a rotated one via hook
            p = p + p9.scale_color_manual(values=color_values, name=expr_name)
            p = p + p9.theme(legend_title=p9.element_blank())
            p.__class__ = _PlotWithVerticalLegendTitle
            p._cat_legend_title = expr_name
            p._cat_legend_fontsize = self.base_size * 0.9
        else:
            # "top" is plotnine's default
            p = p + p9.scale_color_manual(values=color_values, name=expr_name)

        return p

    def _add_grid_layers(self, p: p9.ggplot) -> p9.ggplot:
        gc = self._grid_config
        x_min, x_max, y_min, y_max = self._data.bounds()
        x_grid = np.linspace(x_min, x_max, gc.grid_size + 1)
        y_grid = np.linspace(y_min, y_max, gc.grid_size + 1)

        p = p + p9.geom_vline(
            data=pd.DataFrame({"xintercept": x_grid}),
            mapping=p9.aes(xintercept="xintercept"),
            color=gc.color,
            linetype="solid",
            alpha=0.5,
            size=0.3,
        )
        p = p + p9.geom_hline(
            data=pd.DataFrame({"yintercept": y_grid}),
            mapping=p9.aes(yintercept="yintercept"),
            color=gc.color,
            linetype="solid",
            alpha=0.5,
            size=0.3,
        )

        if gc.labels:
            cell_width = (x_max - x_min) / gc.grid_size
            cell_height = (y_max - y_min) / gc.grid_size
            rows = []
            for i in range(gc.grid_size):
                for j in range(gc.grid_size):
                    cell_x = x_min + (i + 0.5) * cell_width
                    cell_y = y_min + (j + 0.5) * cell_height
                    label = self._point_to_grid_label(
                        gc, x_min, x_max, y_min, y_max, cell_x, cell_y
                    )
                    rows.append({"x": cell_x, "y": cell_y, "label": label})
            labels_df = pd.DataFrame(rows)
            p = p + p9.geom_text(
                data=labels_df,
                mapping=p9.aes("x", "y", label="label"),
                color=gc.label_color,
                size=8,
                alpha=0.7,
                inherit_aes=False,
            )
        return p

    def _add_grid_axis_ticks(self, p: p9.ggplot) -> p9.ggplot:
        gc = self._grid_config
        x_positions, y_positions, x_labels, y_labels = self._data.grid_labels()
        p = (
            p
            + p9.scale_x_continuous(breaks=list(x_positions), labels=list(x_labels))
            + p9.scale_y_continuous(breaks=list(y_positions), labels=list(y_labels))
            + p9.theme(
                axis_text_x=p9.element_text(size=12 / self.base_size * 12, color=self._tick_color),
                axis_text_y=p9.element_text(size=12 / self.base_size * 12, color=self._tick_color),
                axis_ticks_major_x=p9.element_line(color=self._tick_color, size=0.5),
                axis_ticks_major_y=p9.element_line(color=self._tick_color, size=0.5),
                axis_ticks_length_major=5,
            )
        )
        return p

    @staticmethod
    def _point_to_grid_label(
        gc: GridConfig,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        x: float,
        y: float,
    ) -> str:
        x_step = (x_max - x_min) / gc.grid_size
        y_step = (y_max - y_min) / gc.grid_size
        x_index = min(int(round((x - x_min) / x_step)), gc.grid_size - 1)
        y_index = min(int(round((y - y_min) / y_step)), gc.grid_size - 1)
        letters = _LETTERS[: gc.grid_size]
        non_letters = list(range(1, gc.grid_size + 1))
        if gc.vertical_letters:
            letters_rev = letters[::-1]
            letter = letters_rev[y_index]
            number = non_letters[x_index]
            return f"{number}{letter}"
        else:
            letter = letters[x_index]
            number = non_letters[gc.grid_size - 1 - y_index]
            return f"{letter}{number}"


# sentinel for "not set"
class _UnsetType:
    pass


_UNSET = _UnsetType()
