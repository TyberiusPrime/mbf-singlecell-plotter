"""Layer 3: plotnine plot builders."""

import copy
from dataclasses import dataclass, field
from typing import Optional


class _DoNotUpdateType:
    """Sentinel type — distinguishes 'not supplied' from explicit None."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "DoNotUpdate"


#: Pass this as a default argument value to mean "leave the existing setting unchanged".
DoNotUpdate = _DoNotUpdateType()

import numpy as np
import pandas as pd
import plotnine as p9
from natsort import natsorted

from .data import EmbeddingData, ColumnData, _LETTERS
from .theme import DEFAULT_COLORS_BORDERS, DEFAULT_COLORS_CATEGORIES, embedding_theme
from .colorbar import sc_guide_colorbar


# ── Custom matplotlib colorbar legend ────────────────────────────────────────


class _PlotWithPostDraw(p9.ggplot):
    """ggplot subclass that runs accumulated post-draw hooks on the figure.

    Use ``_ensure_post_draw(p)`` to promote any ggplot to this class, then
    append callables ``fn(fig)`` to ``p._post_draw_fns``.
    """

    def save_helper(self, **kwargs):
        sv = super().save_helper(**kwargs)
        for fn in self._post_draw_fns:
            fn(sv.figure)
        return sv


def _ensure_post_draw(p: p9.ggplot) -> "_PlotWithPostDraw":
    """Promote *p* to _PlotWithPostDraw (idempotent); initialise _post_draw_fns."""
    if not isinstance(p, _PlotWithPostDraw):
        p.__class__ = _PlotWithPostDraw
        p._post_draw_fns = []
    return p


def _apply_fixed_panel(fig, panel_w: float, panel_h: float) -> None:
    """Resize *fig* so the scatter panel is exactly panel_w × panel_h inches."""
    from plotnine._mpl.layout_manager._spaces import LayoutSpaces

    fw, fh = fig.get_size_inches()
    le = fig.get_layout_engine()
    spaces = LayoutSpaces(le.plot)
    l_in = spaces.l.total * fw
    r_in = spaces.r.total * fw
    b_in = spaces.b.total * fh
    t_in = spaces.t.total * fh
    fig.set_size_inches(l_in + panel_w + r_in, b_in + panel_h + t_in)
    le.execute(fig)
    fig.canvas.draw()
    for _ in range(3):
        ax = fig.get_axes()[0]
        pos = ax.get_position()
        cur_fw, cur_fh = fig.get_size_inches()
        actual_w = pos.width * cur_fw
        actual_h = pos.height * cur_fh
        fig.set_size_inches(
            cur_fw + (panel_w - actual_w),
            cur_fh + (panel_h - actual_h),
        )
        le.execute(fig)
        fig.canvas.draw()


# ── 2-D embedding colour legend ──────────────────────────────────────────────

_EMBEDDING_COLOR_DEFAULTS = ("#FF4444", "#4444FF", "#FFCC00", "#44BB44")


def _make_2d_color_image(corner_colors, size: int = 64) -> np.ndarray:
    """Return an (H, W, 3) float32 gradient image for the 2D legend.

    Row 0 = top (s=1=high y), col 0 = left (t=0=low x).
    corner_colors: (top_left, top_right, bottom_left, bottom_right).
    """
    import matplotlib.colors as mcolors

    tl = np.array(mcolors.to_rgb(corner_colors[0]))
    tr = np.array(mcolors.to_rgb(corner_colors[1]))
    bl = np.array(mcolors.to_rgb(corner_colors[2]))
    br = np.array(mcolors.to_rgb(corner_colors[3]))

    xs = np.linspace(0, 1, size)  # t: left → right
    ys = np.linspace(1, 0, size)  # s: top row → bottom row (row 0 = s=1 = top)
    T, S = np.meshgrid(xs, ys)

    img = (
        (1 - T[..., None]) * S[..., None] * tl
        + T[..., None] * S[..., None] * tr
        + (1 - T[..., None]) * (1 - S[..., None]) * bl
        + T[..., None] * (1 - S[..., None]) * br
    )
    return np.clip(img, 0, 1).astype(np.float32)


def _draw_embedding_color_legend(
    fig, *, corner_colors, ref_name: str, base_size: float = 12, size: int = 64
):
    """Add a 2D colour-gradient square to the right of the main scatter axes."""
    # Execute layout and freeze — same pattern as _draw_numerical_legend
    le = fig.get_layout_engine()
    if le is not None:
        le.execute(fig)
        fig.set_layout_engine(None)

    legend_fontsize = base_size * 0.9

    fig_w, fig_h = fig.get_size_inches()
    all_axes = fig.axes
    grid_x1 = max(ax.get_position().x1 for ax in all_axes)
    grid_y0 = min(ax.get_position().y0 for ax in all_axes)
    grid_height = max(ax.get_position().y1 for ax in all_axes) - grid_y0

    # Square sized at 35 % of the grid height; convert to equal inches for width
    side_h = grid_height * 0.35
    side_w = side_h * fig_h / fig_w  # same size in inches → different fig fraction

    gap = 0.015
    legend_total_w = gap + side_w + 0.06  # gap + square + right-side tick labels

    # Shrink data axes proportionally if the legend would overflow the figure
    needed_right = grid_x1 + legend_total_w
    if needed_right > 0.99:
        target_x1 = 0.99 - legend_total_w
        scale = target_x1 / grid_x1
        for ax in all_axes:
            p = ax.get_position()
            ax.set_position([p.x0 * scale, p.y0, p.width * scale, p.height])
        grid_x1 = max(ax.get_position().x1 for ax in all_axes)

    # Centre the square vertically on the grid
    lx = grid_x1 + gap
    ly = grid_y0 + (grid_height - side_h) / 2

    ax_leg = fig.add_axes([lx, ly, side_w, side_h])
    img = _make_2d_color_image(corner_colors, size=size)
    ax_leg.imshow(img, aspect="auto", origin="upper")

    ax_leg.set_title(ref_name, fontsize=legend_fontsize, pad=3, color="#333333")

    ax_leg.set_xticks([0, size - 1])
    ax_leg.set_xticklabels(["←", "→"], fontsize=legend_fontsize, color="#333333")
    ax_leg.xaxis.tick_top()

    ax_leg.set_yticks([0, size - 1])
    ax_leg.set_yticklabels(["↑", "↓"], fontsize=legend_fontsize, color="#333333")
    ax_leg.yaxis.tick_right()

    ax_leg.tick_params(length=2, pad=2)
    for sp in ax_leg.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#777777")


def _draw_embedding_label(fig, *, label: str, fontsize: float, color: str = "#777777") -> None:
    """Place a small label outside the panel in the lower-left corner.

    x aligns with the left edge of the y-axis tick labels (via tight bbox);
    the top of the text aligns with the bottom of the panel frame.
    """
    if not fig.axes:
        return
    ax = fig.axes[0]

    # x: left edge of tight bbox (includes y-axis tick labels)
    # y: bottom of tight bbox (includes x-axis tick labels)
    fig_w, fig_h = fig.get_size_inches()
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = ax.get_tightbbox(renderer)
        if tight is None:
            raise ValueError("no tight bbox")
        x_frac = tight.x0 / (fig_w * fig.dpi)
        y_frac = tight.y0 / (fig_h * fig.dpi)
        if not (0 <= x_frac <= 1) or not (0 <= y_frac <= 1):
            raise ValueError(f"fracs out of range: x={x_frac}, y={y_frac}")
    except Exception:
        x_frac = ax.get_position().x0
        y_frac = ax.get_position().y0

    fig.text(x_frac, y_frac, label, ha="left", va="bottom", fontsize=fontsize, color=color)


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
    border_cats: dict = None,
    border_legend_dot_size: float = 4,
) -> None:
    """Add a custom colorbar with rectangular extension boxes to a plotnine figure."""
    import matplotlib as mpl
    import matplotlib.lines
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
    grid_x1 = max(ax.get_position().x1 for ax in all_axes)
    grid_y0 = min(ax.get_position().y0 for ax in all_axes)
    grid_height = max(ax.get_position().y1 for ax in all_axes) - grid_y0

    # For single-panel figures keep a reference to the one axes for shrinking.
    main_ax = all_axes[0]

    # ── Layout: title | gap | bar (tick labels auto to the right) ────────────
    gap = 0.008
    bar_frac = 0.035  # colorbar bar width as fraction of figure

    if title_position == "top":
        bar_left = grid_x1 + gap
        legend_width = gap + bar_frac + 0.10
    else:
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

    # ── Border category legend (matplotlib patches, left of the colorbar) ────
    if border_cats:
        handles = [
            mpl.lines.Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markersize=border_legend_dot_size,
                label=str(cat),
            )
            for cat, color in border_cats.items()
        ]
        legend_fontsize = max(6, base_size * 0.7)
        border_leg = fig.legend(
            handles=handles,
            title="borders",
            loc="upper left",
            bbox_to_anchor=(grid_x1 + gap, grid_y0 + grid_height),
            bbox_transform=fig.transFigure,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=False,
            handlelength=0.8,
            handletextpad=0.4,
            borderpad=0.3,
        )
        # Shift the colorbar right to clear the border legend
        fig.canvas.draw()
        try:
            _r = fig.canvas.renderer
        except AttributeError:
            _r = None
        bb = border_leg.get_window_extent(_r)
        bb_fig = bb.transformed(fig.transFigure.inverted())
        bar_left = max(bar_left, bb_fig.x1 + gap)
        cbar_ax.set_position([bar_left, grid_y0, bar_frac, grid_height])
        if title_position != "top":
            # Re-centre the vertical title
            for txt in fig.texts:
                if txt.get_text() == cbar_title:
                    txt.set_position((bar_left - title_half, grid_y0 + grid_height * 0.5))
                    break

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
    colors: tuple = field(default_factory=lambda: tuple(DEFAULT_COLORS_BORDERS))
    legend: bool = True
    legend_dot_size: float = 4
    legend_title: Optional[str] = None  # None → use the cell_type column name


@dataclass(frozen=True)
class GridConfig:
    labels: object = False  # False/None = off, True/"letters" = A1 labels, "coords" = (x,y) labels
    coords: bool = False
    vertical_letters: bool = False
    grid_size: int = 12
    color: str = "#777777"
    label_color: str = "#777777"
    label_size: Optional[float] = None  # None → 5 for "coords", 8 for letters


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
        self._fixed_panel_size: Optional[tuple] = None

        # dot appearance
        self._dot_size: float = 1
        self._legend_dot_size: float = 4
        self._panel_border: bool = True
        self._spine_color: str = "#555555"
        self._tick_color: str = "#555555"
        self._bg_color: str = "#FFFFFF"
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

        # background layer (all cells, fixed colour, behind data)
        self._background_enabled: bool = False
        self._background_color: str = "#D0D0D0"
        self._background_dot_size: float = 1

        # categorical colormap
        self._cat_colors: Optional[list] = None  # None → DEFAULT_COLORS_CATEGORIES
        self._cat_colors_title: Optional[str] = None  # None → auto from column name

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

        # embedding label
        self._embedding_label: bool = False
        self._embedding_label_size: Optional[float] = None

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

    def get_column(self, name: str):
        """Return ``(series, column_name)`` for an obs column or gene.

        Delegates to :meth:`EmbeddingData.get_column` — useful for downstream
        callers who want to inspect expression or metadata without reaching
        into the internal data layer.
        """
        if self._data is None:
            raise RuntimeError("No data source set — call set_source() first.")
        return self._data.get_column(name)

    # ── dot appearance ───────────────────────────────────────────────────────

    def style(
        self,
        *,
        dot_size: Optional[float] = None,
        legend_dot_size: Optional[float] = None,  # None = keep current (default 4)
        panel_border: Optional[bool] = None,
        spine_color: Optional[str] = None,
        tick_color: Optional[str] = None,
        bg_color: Optional[str] = None,
    ) -> "ScatterPlotter":
        """Configure visual appearance. Only supplied arguments are changed.

        Args:
            dot_size:        Point size for the main scatter layer.
            legend_dot_size: Override the dot size shown in the categorical legend
                             (default: same as dot_size).
            panel_border:    Show/hide the panel border (True = show).
            spine_color:     Hex color for the panel border (default ``"#555555"``).
            tick_color:      Hex color for axis ticks and tick labels (default ``"#555555"``).
            bg_color:        Background hex color (e.g. ``"#FFFFFF"``).
        """
        new = copy.copy(self)
        if dot_size is not None:
            new._dot_size = dot_size
        if legend_dot_size is not None:
            new._legend_dot_size = legend_dot_size
        if panel_border is not None:
            new._panel_border = panel_border
        if spine_color is not None:
            new._spine_color = spine_color
        if tick_color is not None:
            new._tick_color = tick_color
        if bg_color is not None:
            new._bg_color = bg_color
        return new

    def flip_draw_order(self, value: bool = True) -> "ScatterPlotter":
        """Reverse categorical draw order (last category drawn on top when False)."""
        new = copy.copy(self)
        new._flip_order = value
        return new

    def outlier(
        self,
        *,
        shape=DoNotUpdate,
        quantile=DoNotUpdate,
    ) -> "ScatterPlotter":
        """Configure the categorical outlier replot pass.

        Args:
            shape:    Marker shape for outlier points (e.g. ``"^"``, ``"D"``).
                      ``None`` resets to the same shape as main scatter dots.
            quantile: Distance quantile above which a point is an outlier
                      (default 0.95).
        """
        new = copy.copy(self)
        if shape is not DoNotUpdate:
            new._outlier_shape = shape
        if quantile is not DoNotUpdate:
            new._outlier_quantile = quantile
        return new

    # ── colormap (numerical) ─────────────────────────────────────────────────

    def colormap(
        self,
        cmap=DoNotUpdate,
        *,
        max_quantile=DoNotUpdate,
        upper_clip_color=DoNotUpdate,
        title=DoNotUpdate,
    ) -> "ScatterPlotter":
        """Configure the continuous colormap.

        cmap may be a list of color strings, a matplotlib colormap, or None
        (default: black→blue→magenta).  Pass ``None`` explicitly to reset to
        the default palette.  Unspecified arguments are left unchanged.
        """
        new = copy.copy(self)
        if cmap is not DoNotUpdate:
            new._cmap = cmap
        if max_quantile is not DoNotUpdate:
            new._max_quantile = max_quantile
        if upper_clip_color is not DoNotUpdate:
            new._upper_clip_color = upper_clip_color
        if title is not DoNotUpdate:
            new._cbar_title = title
        return new

    # ── categorical colors ───────────────────────────────────────────────────

    def colormap_discrete(
        self,
        cmap_or_list_or_dict=DoNotUpdate,
        *,
        title=DoNotUpdate,
    ) -> "ScatterPlotter":
        """Set the discrete color palette and/or legend title for categorical data.

        cmap_or_list_or_dict accepts:
        - A list of hex color strings (positional, cycling).
        - A dict mapping category name → hex color string.
        - A matplotlib ``ListedColormap`` or similar (uses ``.colors``).
        - ``DoNotUpdate`` (default) — leave the palette unchanged.

        title: Legend title for the color scale.  ``None`` resets to the
        auto-derived column name; ``DoNotUpdate`` leaves the current title.
        """
        new = copy.copy(self)
        if cmap_or_list_or_dict is not DoNotUpdate:
            if isinstance(cmap_or_list_or_dict, (dict, list)):
                new._cat_colors = cmap_or_list_or_dict
            else:
                # Assume matplotlib ListedColormap or similar
                new._cat_colors = list(cmap_or_list_or_dict.colors)
        if title is not DoNotUpdate:
            new._cat_colors_title = title
        return new

    # ── zero handling ────────────────────────────────────────────────────────

    def zeros(
        self,
        *,
        color=DoNotUpdate,
        dot_size=DoNotUpdate,
        max_zero_value=DoNotUpdate,
    ) -> "ScatterPlotter":
        """Configure zero-value rendering (appearance only; use layers(zeros=) to toggle visibility)."""
        new = copy.copy(self)
        if color is not DoNotUpdate:
            new._zero_color = color
        if dot_size is not DoNotUpdate:
            new._zero_dot_size = dot_size
        if max_zero_value is not DoNotUpdate:
            new._zero_value = max_zero_value
        return new

    def background(
        self,
        *,
        enabled: bool = True,
        color=DoNotUpdate,
        dot_size=DoNotUpdate,
    ) -> "ScatterPlotter":
        """Add a background layer plotting all cells in a fixed colour behind the data.

        Args:
            enabled:   Turn the background layer on (True) or off (False).
            color:     Dot colour for all background cells (default ``"#D0D0D0"``).
            dot_size:  Dot size for background cells (default 1).
        """
        new = copy.copy(self)
        new._background_enabled = enabled
        if color is not DoNotUpdate:
            new._background_color = color
        if dot_size is not DoNotUpdate:
            new._background_dot_size = dot_size
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
        colors: Optional[list] = None,
        legend: bool = True,
        legend_dot_size: float = 4,
        legend_title: Optional[str] = None,
    ) -> "ScatterPlotter":
        new = copy.copy(self)
        resolved_colors = tuple(colors) if colors is not None else tuple(DEFAULT_COLORS_BORDERS)
        new._border_config = BorderConfig(
            size=size, resolution=resolution, blur=blur, threshold=threshold,
            colors=resolved_colors, legend=legend, legend_dot_size=legend_dot_size,
            legend_title=legend_title,
        )
        if cell_type_column is not None:
            new._cell_type_column = cell_type_column
        # Reset cache if anything that affects the boundary image changes
        old = self._border_config
        if (
            old is None
            or old.resolution != resolution
            or old.blur != blur
            or old.threshold != threshold
            or old.colors != resolved_colors
            or cell_type_column != self._cell_type_column
        ):
            new._boundary_cache = {"df": None}
        # else: share the same cache dict (shallow copy) — only size/legend changed
        return new

    def without_borders(self) -> "ScatterPlotter":
        new = copy.copy(self)
        new._border_config = None
        return new

    # ── layer visibility ─────────────────────────────────────────────────────

    def layers(
        self,
        *,
        borders=DoNotUpdate,
        zeros=DoNotUpdate,
        data=DoNotUpdate,
        outliers=DoNotUpdate,
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
        if borders is not DoNotUpdate:
            new._layer_borders = borders
        if zeros is not DoNotUpdate:
            new._layer_zeros = zeros
        if data is not DoNotUpdate:
            new._layer_data = data
        if outliers is not DoNotUpdate:
            new._layer_outliers = outliers
        return new

    # ── grid overlay ─────────────────────────────────────────────────────────

    def with_grid(
        self,
        *,
        labels=None,
        coords: Optional[bool] = None,
        vertical_letters: Optional[bool] = None,
        grid_size: Optional[int] = None,
        color: Optional[str] = None,
        label_color: Optional[str] = None,
        label_size: Optional[float] = None,
    ) -> "ScatterPlotter":
        """Configure the grid overlay.

        Args:
            labels:     Cell-interior labels.  ``False``/``None`` = off;
                        ``True``/``"letters"`` = grid-label strings (e.g. ``"A1"``);\
                        ``"coords"`` = ``(x, y)`` embedding-coordinate strings.
            coords:     Replace axis tick labels with grid-cell identifiers.
            label_size: Font size for cell-interior labels.  Defaults to 5 for
                        ``"coords"`` and 8 for letter labels.
            vertical_letters: Put letters on the vertical axis (default: horizontal).
            grid_size, color, label_color: passed through unchanged if ``None``.

        Only supplied arguments are changed; unspecified ones inherit from the
        current grid config (or GridConfig defaults if no grid is set yet).
        """
        cur = self._grid_config if self._grid_config is not None else GridConfig()
        resolved_grid_size = grid_size if grid_size is not None else cur.grid_size
        resolved_vl = (
            vertical_letters if vertical_letters is not None else cur.vertical_letters
        )
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
            label_size=label_size if label_size is not None else cur.label_size,
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

    def focus_on(self, *args, x: tuple = None, y: tuple = None) -> "ScatterPlotter":
        """Restrict viewport to a coordinate window.

        Accepts either two grid label strings::

            plotter.focus_on("A1", "C5")

        or explicit coordinate ranges (keyword-only)::

            plotter.focus_on(x=(x_min, x_max), y=(y_min, y_max))
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .focus_on()")
        new = copy.copy(self)
        if args:
            new._data = self._data.focus_on(*args)
        else:
            new._data = self._data.focus_on(x=x, y=y)
        return new

    def focus_on_grid(self, cell_min: str, cell_max: str) -> "ScatterPlotter":
        """Restrict viewport to the rectangle from cell_min (top-left) to cell_max (bottom-right).

        Raises RuntimeError if no source has been set.
        Raises ValueError if the grid has been disabled with without_grid().
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .focus_on_grid()")
        if self._grid_config is None:
            raise ValueError(
                "focus_on_grid() requires a grid; call .with_grid() to re-enable "
                "or remove .without_grid()"
            )
        new = copy.copy(self)
        new._data = self._data._focus_on_grid(cell_min, cell_max)
        return new

    def unfocus(self) -> "ScatterPlotter":
        if self._data is None:
            raise RuntimeError("call .set_source() before .unfocus()")
        new = copy.copy(self)
        new._data = self._data.unfocus()
        return new

    def panel_size(self, width: float, height: float) -> "ScatterPlotter":
        """Fix the scatter-panel (data area) to *width* × *height* inches.

        The figure size grows to accommodate the panel plus whatever space the
        legends, title, and axis labels require — so plots with different
        legends remain comparable.
        """
        new = copy.copy(self)
        new._fixed_panel_size = (width, height)
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

    # ── embedding label ───────────────────────────────────────────────────────

    def with_embedding_label(
        self, show: bool = True, size=DoNotUpdate
    ) -> "ScatterPlotter":
        """Show the embedding name in the lower-left corner of each plot.

        Args:
            show: Whether to show the label (default True).
            size: Font size in points. Defaults to half the base font size.
                  ``DoNotUpdate`` leaves the current size unchanged.
        """
        new = copy.copy(self)
        new._embedding_label = show
        if size is not DoNotUpdate:
            new._embedding_label_size = size
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
                has_border_legend = (
                    self._layer_borders
                    and self._border_config is not None
                    and self._border_config.legend
                )
                # Two legends on the right need more width
                fig_size = (8 if has_border_legend else 6, 5)

        # Build plot
        if is_numerical:
            p = self._build_numerical(df, expr_name)
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
        else:
            p = p + p9.labs(title=expr_name)

        # Theme (must come before grid axis ticks so theme_void doesn't override them)
        p = p + embedding_theme(
            base_size=self.base_size,
            show_spines=self._panel_border,
            bg_color=self._bg_color,
            spine_color=self._spine_color,
        )
        p = p + p9.theme(figure_size=fig_size, legend_box="horizontal")

        # Grid axis tick labels (applied after theme so they survive theme_void)
        if self._grid_config is not None and self._grid_config.coords:
            p = self._add_grid_axis_ticks(p)
        elif self._grid_config is None:
            p = self._add_plain_axis_ticks(p)

        # Fixed panel size
        if self._fixed_panel_size is not None:
            w, h = self._fixed_panel_size
            p = _ensure_post_draw(p)
            p._post_draw_fns.append(lambda fig, _w=w, _h=h: _apply_fixed_panel(fig, _w, _h))

        # Embedding label (after fixed-panel so tight-bbox reflects final size)
        if self._embedding_label:
            p = self._add_embedding_label(p, data)

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
        if (
            self._layer_borders
            and self._border_config is not None
            and self._cell_type_column is not None
        ):
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

        if self._grid_config is None:
            p = self._add_plain_axis_ticks(p)

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
            )
            p = _ensure_post_draw(p)
            p._post_draw_fns.append(lambda fig, _c=legend_config: _draw_numerical_legend(fig, **_c))

        if self._embedding_label:
            p = self._add_embedding_label(p, self._data)

        return p

    def plot_moran_markers(
        self,
        n_bins: int = 40,
        min_cells: int = 3,
        k: int = 20,
        min_moran: float = 0.2,
        genes_shown: int = 3,
        density_bins: int = 100,
        label_size: float = 7.0,
    ) -> p9.ggplot:
        """Build a density heatmap annotated with Moran's I marker genes per region.

        Bins cells into an ``n_bins × n_bins`` grid, computes Moran's I for every
        gene, and overlays the top-*genes_shown* spatially coherent markers as text
        labels at each region's bin centre.

        Args:
            n_bins:        Grid resolution for Moran's I binning (default 40).
            min_cells:     Minimum cells per bin (default 3).
            k:             Marker genes computed per region (default 20).
            min_moran:     Minimum Moran's I to qualify as a marker (default 0.2).
            genes_shown:   How many gene names to show per label (default 3).
            density_bins:  Resolution of the density background (default 100).
            label_size:    Font size for the gene labels (default 7).
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_moran_markers()")

        from .transforms import compute_grid_moran, marker_genes_by_region, prepare_density_df

        data = self._data
        gene_df = compute_grid_moran(data, n_bins=n_bins, min_cells=min_cells)
        markers = marker_genes_by_region(gene_df, k=k, min_moran=min_moran)

        # Build per-region label DataFrame (one row per occupied region with markers)
        if markers:
            # Use the first gene in each bin to look up the bin centre coordinates
            first_gene_per_bin = (
                gene_df[gene_df["top_bin"].isin(markers.keys())]
                .groupby("top_bin", group_keys=False)
                .apply(lambda g: g.nlargest(1, "moran_i"))
            )
            label_rows = []
            for _, row in first_gene_per_bin.iterrows():
                genes = markers[row["top_bin"]][:genes_shown]
                label_rows.append({
                    "x":     row["top_bin_x"],
                    "y":     row["top_bin_y"],
                    "label": "\n".join(genes),
                })
            label_df = pd.DataFrame(label_rows)
        else:
            label_df = pd.DataFrame({"x": [], "y": [], "label": []})

        density_df = prepare_density_df(data, bins=density_bins)

        p = (
            p9.ggplot(density_df, p9.aes("x", "y", fill="density"))
            + p9.geom_tile(p9.aes(width="x_width", height="y_width"))
            + p9.scale_fill_gradientn(
                colors=["#FFFFFF", "#BFBFFF", "#0000FF"],
                na_value="#FFFFFF",
                name="density",
            )
            + p9.guides(fill="none")
        )

        if len(label_df) > 0:
            p = p + p9.geom_label(
                data=label_df,
                mapping=p9.aes("x", "y", label="label"),
                size=label_size,
                color="#111111",
                fill="#FFFFFFCC",
                label_size=0.2,
                inherit_aes=False,
            )

        p = p + embedding_theme(
            base_size=self.base_size,
            show_spines=self._panel_border,
            bg_color=self._bg_color,
            spine_color=self._spine_color,
        )
        p = p + p9.theme(figure_size=(7, 6))

        if self._grid_config is not None:
            p = self._add_grid_layers(p)
            p = self._add_grid_axis_ticks(p)
        else:
            p = self._add_plain_axis_ticks(p)

        if self._embedding_label:
            p = self._add_embedding_label(p, data)

        return p

    def save_interactive_moran(
        self,
        column: str,
        output_path,
        min_cells: int = 3,
        k: int = 20,
        min_moran: float = 0.2,
        var_score_column: str | None = None,
        dpi: int = 150,
        debug: bool = False,
        gene_url: str | None = None,
        gene_url_inline: bool = False,
    ) -> None:
        """Save an interactive HTML scatter plot with marker gene tooltips.

        Renders the scatter for *column* as a PNG, then overlays an invisible
        grid.  Hovering over a cell highlights it (yellow tint) and shows its
        top-k marker genes in a panel below.  Clicking locks the selection;
        clicking the same cell deactivates it; clicking another cell switches.

        The spatial binning resolution is taken from the plotter's grid size so
        that bins align exactly with the visible grid cells.

        The data panel defaults to 5 × 5 in (square) unless
        :meth:`panel_size` has already been called on this plotter.

        Args:
            column:           Gene or obs column to plot.
            output_path:      Destination ``.html`` file path.
            min_cells:        Minimum cells per bin (default 3).
            k:                Marker genes stored per region (default 20).
            min_moran:        Minimum score to qualify as a marker (default 0.2).
                              Applied to Moran's I or to *var_score_column* values.
            var_score_column: Column in ``adata.var`` to use as the gene score
                              instead of computing Moran's I on the fly (e.g.
                              ``"moranI"`` or ``"highly_variable_rank"``).
                              Must be numeric; higher = more informative.
                              When ``None`` (default), Moran's I is computed.
            dpi:              PNG resolution for the scatter image (default 150).
            gene_url:         URL template for gene links.  ``{gene}`` is
                              replaced with the gene name.  When ``None``
                              (default) genes are plain text.
            gene_url_inline:  If ``True`` the linked resource is displayed in
                              an ``<img>`` panel below rather than opened in a
                              new browser tab (default ``False``).
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .save_interactive_moran()")
        from .interactive import save_interactive_moran as _impl
        _impl(
            self, column, output_path,
            min_cells=min_cells, k=k, min_moran=min_moran,
            var_score_column=var_score_column,
            dpi=dpi, debug=debug,
            gene_url=gene_url, gene_url_inline=gene_url_inline,
        )

    def plot_grid_histogram(
        self,
        column: str,
        min_cell_count: int = 10,
        vertical: bool = False,
        scale_by_count: bool = False,
        fill_fraction: float | None = None,
    ) -> p9.ggplot:
        """Build a grid-local category frequency heatmap (plotnine).

        Parameters
        ----------
        vertical:
            If True, bars stack vertically within each cell instead of
            horizontally (the default).
        scale_by_count:
            If True, scale each cell's tile area proportionally to the number
            of observations in that cell (sqrt scaling so area ∝ count).
            Cells with fewer observations appear as smaller tiles.
        fill_fraction:
            Fraction of each grid square covered by the tiles at full size
            (0 < fill_fraction ≤ 1).  Defaults to 1.0 when ``scale_by_count``
            is True, 0.8 otherwise.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_grid_histogram()")

        hdf = self._data.grid_local_histogram(column, min_cell_count)
        hdf["category"] = pd.Categorical(
            hdf["category"], sorted(hdf["category"].unique())
        )
        hdf = hdf.sort_values(["x", "y", "category"])
        cats = list(hdf["category"].cat.categories)
        colors = self._colors_as_list(cats)

        if fill_fraction is None:
            fill_fraction = 1.0 if scale_by_count else 0.8
        factor = fill_fraction
        if scale_by_count:
            # sqrt so that linear dimension ∝ sqrt(count) and area ∝ count
            hdf["cell_factor"] = factor * np.sqrt(
                hdf["total"] / hdf["total"].max()
            )
        else:
            hdf["cell_factor"] = factor

        hdf["frequency"] = hdf["frequency"] * hdf["cell_factor"]

        offset = []
        for _ignored, group in hdf.groupby(["x", "y"]):
            offset.extend(group["frequency"].cumsum().shift(fill_value=0))

        if vertical:
            hdf["y_offset"] = offset
            hdf["x_plot"] = hdf["x"] - 0.5
            # vertical: bars stack bottom→top; x is fixed at cell centre
            hdf["y_plot"] = (
                hdf["y"]
                + (1 - hdf["cell_factor"]) / 2
                + hdf["y_offset"]
                + hdf["frequency"] / 2
            )
            hdf["xmin"] = hdf["x"] - 0.5 - hdf["cell_factor"] / 2
            hdf["xmax"] = hdf["x"] - 0.5 + hdf["cell_factor"] / 2
            hdf["ymin"] = hdf["y_plot"] - hdf["frequency"] / 2
            hdf["ymax"] = hdf["y_plot"] + hdf["frequency"] / 2
        else:
            # horizontal: bars stack left→right; y is fixed at cell centre
            hdf["x_offset"] = offset
            hdf["x_plot"] = (
                hdf["x"]
                - hdf["frequency"] / 2
                - hdf["x_offset"]
                - (1 - hdf["cell_factor"]) / 2
            )
            hdf["xmin"] = hdf["x_plot"] - hdf["frequency"] / 2
            hdf["xmax"] = hdf["x_plot"] + hdf["frequency"] / 2
            hdf["ymin"] = hdf["y"] + (1 - hdf["cell_factor"]) / 2
            hdf["ymax"] = hdf["y"] + 1 - (1 - hdf["cell_factor"]) / 2

        bar_geom = p9.geom_rect(
            p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="category")
        )
        global_aes = p9.aes()

        grid_size = self._data._grid_size
        _x_ticks, _y_ticks, x_labels, y_labels = self._data.grid_labels()
        # Ticks at cell centres: bars are centred at (x - 0.5) and (y + 0.5),
        # matching the scatter plot where axis labels also sit at cell centres.
        x_ticks = [-0.5 + i for i in range(grid_size)]
        y_ticks = [0.5 + i for i in range(grid_size)]

        hdf = hdf[::-1]

        p = (
            p9.ggplot(hdf, global_aes)
            + embedding_theme(base_size=self.base_size, show_spines=True)
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
            + bar_geom
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

        if self._fixed_panel_size is not None:
            w, h = self._fixed_panel_size
            p = _ensure_post_draw(p)
            p._post_draw_fns.append(lambda fig, _w=w, _h=h: _apply_fixed_panel(fig, _w, _h))

        return p

    def plot_embedding_color(
        self,
        reference_embedding,
        *,
        corner_colors=_EMBEDDING_COLOR_DEFAULTS,
        region=None,
        outside_color: str = "#C0C0C0",
        show_legend: bool = False,
        show_region: bool = False,
        dot_size: Optional[float] = None,
    ) -> p9.ggplot:
        """Plot cells in the current embedding colored by 2D position in another embedding.

        Each cell receives a color from a bilinear gradient defined by four corner
        colors at its normalized (x, y) position in the reference embedding.  This
        lets you see how the layout of one embedding corresponds to another.

        Args:
            reference_embedding: Embedding name (str) or EmbeddingData for color assignment.
            corner_colors:        4-tuple ``(top_left, top_right, bottom_left, bottom_right)``.
                                  Default: red / blue / yellow / green.
            region:               Optional ``(corner1, corner2)`` restricting which cells
                                  receive the gradient.  Each corner is a grid label string
                                  (e.g. ``"A1"``) or an ``(x, y)`` float tuple in reference-
                                  embedding coordinates.  ``corner1`` is the top-left (>=)
                                  and ``corner2`` the bottom-right (<=), matching the
                                  ``focus_on_grid`` convention.  Cells outside the box get
                                  *outside_color*.
            outside_color:        Color for cells outside *region* (default ``"#C0C0C0"``).
            show_legend:          Add a small 2D color legend inset (default False).
            dot_size:             Point size; defaults to the plotter's dot_size.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_embedding_color()")

        from .transforms import prepare_embedding_color_df

        data = self._data
        x_min, x_max, y_min, y_max = data.bounds()

        if isinstance(reference_embedding, str):
            ref_name = reference_embedding
            ref_data = EmbeddingData(data.ad, reference_embedding)
        elif isinstance(reference_embedding, EmbeddingData):
            ref_name = reference_embedding.embedding
            ref_data = reference_embedding
        else:
            raise ValueError("reference_embedding must be a str or EmbeddingData")

        df = prepare_embedding_color_df(
            data,
            ref_data,
            corner_colors=corner_colors,
            region=region,
            outside_color=outside_color,
        )

        dot = dot_size if dot_size is not None else self._dot_size

        p = p9.ggplot(df, p9.aes("x", "y", color="color"))

        # Grid lines first so they render behind all other layers
        if self._grid_config is not None:
            p = self._add_grid_layers(p)

        # Boundary layer (behind scatter)
        if (
            self._layer_borders
            and self._border_config is not None
            and self._cell_type_column is not None
        ):
            p = self._add_border_layers(p)

        # Main scatter — identity scale reads hex color strings directly
        if self._layer_data:
            p = p + p9.geom_point(size=dot)

        p = p + p9.scale_color_identity(guide=None)

        # Region outline overlay (in reference-embedding coordinates)
        if show_region and region is not None:
            from .transforms import _corner_to_bounds
            import numpy as _np
            if len(region) == 2:
                if isinstance(region[0], str) or isinstance(region[1], str):
                    c1 = _corner_to_bounds(region[0], ref_data)
                    c2 = _corner_to_bounds(region[1], ref_data)
                    xlo = min(c1[0], c1[1], c2[0], c2[1])
                    xhi = max(c1[0], c1[1], c2[0], c2[1])
                    ylo = min(c1[2], c1[3], c2[2], c2[3])
                    yhi = max(c1[2], c1[3], c2[2], c2[3])
                else:
                    (x0, y0), (x1, y1) = region
                    xlo, xhi = min(x0, x1), max(x0, x1)
                    ylo, yhi = min(y0, y1), max(y0, y1)
                # polygon order: tl → tr → br → bl
                corners = [(xlo, yhi), (xhi, yhi), (xhi, ylo), (xlo, ylo)]
            else:
                pts4 = sorted([_np.array(c, dtype=float) for c in region], key=lambda p: -p[1])
                top_two = sorted(pts4[:2], key=lambda pt: pt[0])
                bot_two = sorted(pts4[2:], key=lambda pt: pt[0])
                tl_, tr_ = top_two[0], top_two[1]
                bl_, br_ = bot_two[0], bot_two[1]
                corners = [tl_, tr_, br_, bl_]
            # closed polygon + corner markers
            cx = [c[0] for c in corners] + [corners[0][0]]
            cy = [c[1] for c in corners] + [corners[0][1]]
            region_path_df = pd.DataFrame({"x": cx, "y": cy})
            region_pts_df = pd.DataFrame({"x": [c[0] for c in corners], "y": [c[1] for c in corners]})
            p = p + p9.geom_path(
                data=region_path_df,
                mapping=p9.aes(x="x", y="y"),
                color="#000000",
                size=0.6,
                inherit_aes=False,
            )
            p = p + p9.geom_point(
                data=region_pts_df,
                mapping=p9.aes(x="x", y="y"),
                color="#000000",
                size=1.5,
                inherit_aes=False,
            )

        # Focus viewport
        if data.has_focus:
            p = p + p9.coord_cartesian(xlim=(x_min, x_max), ylim=(y_min, y_max))

        # Title
        if self._title_override is not _UNSET:
            p = p + p9.labs(title=self._title_override)
        else:
            p = p + p9.labs(title=ref_name)

        # Theme
        p = p + embedding_theme(
            base_size=self.base_size,
            show_spines=self._panel_border,
            bg_color=self._bg_color,
            spine_color=self._spine_color,
        )
        p = p + p9.theme(figure_size=(6, 5))

        # Grid axis ticks
        if self._grid_config is not None and self._grid_config.coords:
            p = self._add_grid_axis_ticks(p)
        elif self._grid_config is None:
            p = self._add_plain_axis_ticks(p)

        # Fixed panel size
        if self._fixed_panel_size is not None:
            w, h = self._fixed_panel_size
            p = _ensure_post_draw(p)
            p._post_draw_fns.append(lambda fig, _w=w, _h=h: _apply_fixed_panel(fig, _w, _h))

        # 2D colour legend (to the right of the figure)
        if show_legend:
            _cfg = {"corner_colors": corner_colors, "ref_name": ref_name, "base_size": self.base_size}
            p = _ensure_post_draw(p)
            p._post_draw_fns.append(lambda fig, _c=_cfg: _draw_embedding_color_legend(fig, **_c))

        # Embedding label — runs after legend so tight-bbox is stable
        if self._embedding_label:
            p = self._add_embedding_label(p, data)

        return p

    # ── internals ────────────────────────────────────────────────────────────

    def _add_embedding_label(self, p: p9.ggplot, data: "EmbeddingData") -> p9.ggplot:
        label = data.embedding
        if label.startswith("X_"):
            label = label[2:]
        fontsize = (
            self._embedding_label_size
            if self._embedding_label_size is not None
            else self.base_size  
        )
        p = _ensure_post_draw(p)
        p._post_draw_fns.append(
            lambda fig, _l=label, _fs=fontsize: _draw_embedding_label(fig, label=_l, fontsize=_fs)
        )
        return p

    def _colors_as_list(self, cats: list) -> list:
        """Return an ordered color list for *cats*.

        Handles both list (positional cycling) and dict (name → color) forms
        of ``_cat_colors``.
        """
        if isinstance(self._cat_colors, dict):
            # Normalize keys to str so {True: 'red'} and {'True': 'red'} both work
            normalized = {str(k): v for k, v in self._cat_colors.items()}
            missing = sorted([str(c) for c in cats if str(c) not in normalized])
            if missing:
                raise ValueError(
                    f"not enough colors: dict is missing entries for: {missing}"
                )
            return [normalized[str(c)] for c in cats]
        colors = self._cat_colors or DEFAULT_COLORS_CATEGORIES
        if len(colors) < len(cats):
            raise ValueError(
                f"not enough colors: {len(colors)} provided for {len(cats)} categories"
            )
        return colors

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

            bc = self._border_config
            self._boundary_cache["df"] = compute_boundaries(
                data=self._data,
                cell_type_column=self._cell_type_column,
                colors=list(bc.colors),
                resolution=bc.resolution,
                blur=bc.blur,
                threshold=bc.threshold,
            )
        return self._boundary_cache["df"]

    def _border_cat_to_color(self) -> dict:
        """Return ordered {category: hex_color} mapping for the border palette."""
        cell_types, _ = self._data.get_column(self._cell_type_column)
        cats = (
            list(cell_types.cat.categories)
            if hasattr(cell_types, "cat")
            else natsorted(cell_types.unique())
        )
        colors = list(self._border_config.colors)
        return {cat: colors[i % len(colors)] for i, cat in enumerate(cats)}

    def _add_border_layers(self, p: p9.ggplot) -> p9.ggplot:
        bdf = self._get_boundary_df()
        bc = self._border_config
        border_pt = bc.size / 10

        # Render border dots (fixed color per group, avoids plotnine scale conflicts)
        for color in bdf["color"].unique():
            sub = bdf[bdf["color"] == color]
            p = p + p9.geom_point(
                data=sub,
                mapping=p9.aes("x", "y"),
                color=color,
                size=border_pt,
                inherit_aes=False,
            )

        if bc.legend:
            cat_to_color = self._border_cat_to_color()

            # One invisible phantom point per category — carries the fill aesthetic
            # for the legend. Uses fill (not color) so it doesn't clash with the
            # scatter's color scale. Placed at (x_min, y_min) which is already
            # within the data bounds so the axis range is unaffected.
            x_min, _, y_min, _ = self._data.bounds()
            cats = list(cat_to_color.keys())
            legend_df = pd.DataFrame({"cell_type": cats, "x": x_min, "y": y_min})
            p = (
                p
                + p9.geom_point(
                    data=legend_df,
                    mapping=p9.aes("x", "y", fill="cell_type"),
                    shape="o",
                    color="none",
                    size=border_pt * 1.5,
                    alpha=0,
                    inherit_aes=False,
                )
                + p9.scale_fill_manual(
                    values=cat_to_color,
                    name=bc.legend_title if bc.legend_title is not None else self._cell_type_column,
                    guide=p9.guide_legend(override_aes={"alpha": 1, "size": bc.legend_dot_size}),
                )
            )

        return p

    def _build_numerical(
        self,
        df: pd.DataFrame,
        expr_name: str,
    ) -> p9.ggplot:
        zero_val = self._zero_value if self._zero_value is not None else 0.0

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
        else:
            clip_val = float(df_nonzero["expression"].quantile(self._max_quantile))

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
        if (
            self._layer_borders
            and self._border_config is not None
            and self._cell_type_column is not None
        ):
            p = self._add_border_layers(p)

        # Background layer (all cells, fixed colour, behind data)
        if self._background_enabled:
            p = self._add_background_layer(p, df)

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
        cbar_name = (
            self._cbar_title
            if self._cbar_title is not None
            else (expr_name + ": log2 expression")
        )
        has_zeros = self._layer_zeros and len(df_zeros) > 0
        has_clips = len(df_above) > 0
        zero_val_str = "0" if abs(zero_val) < 1e-9 else f"{zero_val:.3g}"
        data_min = float(df["expression"].min())
        zero_label = (
            f"≤{zero_val_str}" if has_zeros and data_min < zero_val - 1e-9
            else zero_val_str
        )
        # Use MaxNLocator to get ≥7 "nice" break values so that after removing
        # the boundary ticks that duplicate the extension-box labels (1 at each
        # end), at least 5 ticks remain.
        import matplotlib.ticker as _ticker
        cbar_breaks = list(
            _ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]).tick_values(
                zero_val, clip_val
            )
        )
        p = p + p9.scale_color_gradientn(
            colors=cmap_colors,
            limits=(zero_val, clip_val),
            breaks=cbar_breaks,
            name=cbar_name,
            guide=sc_guide_colorbar(
                zero_color=self._zero_color if has_zeros else None,
                zero_label=zero_label,
                upper_clip_color=self._upper_clip_color if has_clips else None,
                clip_label=f">{clip_val:.3g}",
                key_height_pt=(
                    round(self._fixed_panel_size[1] * 72 * 0.70)
                    if self._fixed_panel_size is not None
                    else None
                ),
            ),
        )
        return p

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

        # Convert to str-Categorical so plotnine matches the string-keyed color_values
        # dict (needed for non-string dtypes such as bool).  Categorical avoids
        # allocating a full object array of strings for every row.
        cats = [str(c) for c in cats]
        df["expression"] = pd.Categorical(df["expression"].astype(str), categories=cats)

        p = p9.ggplot(df, p9.aes("x", "y", color="expression"))

        # Grid lines first so they render behind all other layers
        if self._grid_config is not None:
            p = self._add_grid_layers(p)

        # Boundary layer (behind scatter)
        if (
            self._layer_borders
            and self._border_config is not None
            and self._cell_type_column is not None
        ):
            p = self._add_border_layers(p)

        # Background layer (all cells, fixed colour, behind data)
        if self._background_enabled:
            p = self._add_background_layer(p, df)

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
                extra = (
                    {}
                    if self._outlier_shape is None
                    else {"shape": self._outlier_shape}
                )
                p = p + p9.geom_point(
                    data=df_outliers,
                    size=self._dot_size,
                    inherit_aes=True,
                    **extra,
                )

        # When panel height is fixed, switch to multi-column legend if the
        # categories would overflow the available vertical space.
        ncol = 1
        if self._fixed_panel_size is not None:
            panel_h = self._fixed_panel_size[1]
            n_cats = len(cats)
            title_pt = self.base_size * 2.5
            available_pt = panel_h * 72 - title_pt
            default_key_h = self.base_size * 1.2  # plotnine default ≈ 14.4 pt
            max_per_col = max(1, int(available_pt / default_key_h))
            if n_cats > max_per_col:
                ncol = -(-n_cats // max_per_col)  # ceiling division

        p = p + p9.scale_color_manual(
            values=color_values,
            name=self._cat_colors_title if self._cat_colors_title is not None else expr_name,
            guide=p9.guide_legend(
                override_aes={"size": self._legend_dot_size, "shape": "o"},
                ncol=ncol,
            ),
        )

        return p

    def _add_background_layer(self, p: p9.ggplot, df: pd.DataFrame) -> p9.ggplot:
        """Add a fixed-colour layer of all cells behind the data layers."""
        bg_df = df[["x", "y"]].copy()
        return p + p9.geom_point(
            data=bg_df,
            mapping=p9.aes("x", "y"),
            color=self._background_color,
            size=self._background_dot_size,
            inherit_aes=False,
            show_legend=False,
        )

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
                    if gc.labels == "coords":
                        label = f"{cell_x:.2f}\n{cell_y:.2f}"
                    else:
                        label = self._point_to_grid_label(
                            gc, x_min, x_max, y_min, y_max, cell_x, cell_y
                        )
                    rows.append({"x": cell_x, "y": cell_y, "label": label})
            labels_df = pd.DataFrame(rows)
            default_size = 5 if gc.labels == "coords" else 8
            lsize = gc.label_size if gc.label_size is not None else default_size
            p = p + p9.geom_text(
                data=labels_df,
                mapping=p9.aes("x", "y", label="label"),
                color=gc.label_color,
                size=lsize,
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
                axis_text_x=p9.element_text(
                    size=12 / self.base_size * 12, color=self._tick_color
                ),
                axis_text_y=p9.element_text(
                    size=12 / self.base_size * 12, color=self._tick_color
                ),
                axis_ticks_major_x=p9.element_line(color=self._tick_color, size=0.5),
                axis_ticks_major_y=p9.element_line(color=self._tick_color, size=0.5),
                axis_ticks_length_major=5,
            )
        )
        return p

    def _add_plain_axis_ticks(self, p: p9.ggplot) -> p9.ggplot:
        """Restore regular axis ticks (major + minor) when no grid is active."""

        return (
            p
            + p9.scale_x_continuous(minor_breaks=5)
            + p9.scale_y_continuous(minor_breaks=5)
            + p9.theme(
                panel_grid_major=p9.element_line(color="#d0d0d0"),
                panel_grid_minor=p9.element_line(color="#e0e0e0"),
                axis_text_x=p9.element_text(color=self._tick_color),
                axis_text_y=p9.element_text(color=self._tick_color),
                axis_ticks_major_x=p9.element_line(color=self._tick_color, size=0.7),
                axis_ticks_major_y=p9.element_line(color=self._tick_color, size=0.7),
                axis_ticks_minor_x=p9.element_line(color=self._tick_color, size=0.4),
                axis_ticks_minor_y=p9.element_line(color=self._tick_color, size=0.4),
                axis_ticks_length_major=6,
                axis_ticks_length_minor=3,
            )
        )

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
            return f"{letter}{number}"
        else:
            letter = letters[x_index]
            number = non_letters[gc.grid_size - 1 - y_index]
            return f"{letter}{number}"


# sentinel for "not set"
class _UnsetType:
    pass


_UNSET = _UnsetType()
