"""Layer 3: plotnine plot builders."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Dict, Callable, override, Any

import numpy as np
import pandas as pd
import plotnine as p9
from natsort import natsorted

from .data import EmbeddingData, _LETTERS
from .theme import DEFAULT_COLORS_BORDERS, DEFAULT_COLORS_CATEGORIES, embedding_theme
from .colorbar import sc_guide_colorbar


class _DoNotUpdateType:
    """Sentinel type — distinguishes 'not supplied' from explicit None."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @override
    def __repr__(self):
        return "DoNotUpdate"


#: Pass this as a default argument value to mean "leave the existing setting unchanged".
DoNotUpdate = _DoNotUpdateType()


# ── Custom matplotlib colorbar legend ────────────────────────────────────────


class _PlotWithPostDraw(p9.ggplot):
    """ggplot subclass that runs accumulated post-draw hooks on the figure.

    Use ``_ensure_post_draw(p)`` to promote any ggplot to this class, then
    append callables ``fn(fig)`` to ``p._post_draw_fns``.

    Hooks are run after the figure is drawn in three situations:

    * ``save_helper`` — the regular ``ggplot.save`` path for a single plot.
    * ``interactive`` — the module runs the hooks explicitly after ``draw``.
    * plotnine *compositions* (``p1 / p2``, ``p1 | p2``) — see
      :func:`_patch_composition_post_draw`, which makes ``Compose.draw`` run
      every member plot's hooks once the shared figure has been laid out.
    """

    _post_draw_fns: list[Any]

    @override
    def save_helper(self, *args, **kwargs):
        sv = super().save_helper(*args, **kwargs)
        for fn in self._post_draw_fns:
            fn(sv.figure)
        return sv


def _ensure_post_draw(p: p9.ggplot) -> "_PlotWithPostDraw":
    """Promote *p* to _PlotWithPostDraw (idempotent); initialise _post_draw_fns."""
    if not isinstance(p, _PlotWithPostDraw):
        p.__class__ = _PlotWithPostDraw
        p._post_draw_fns = []  # ty: ignore
    return p  # ty: ignore


def _first_panel_axes(fig):
    """Return a representative scatter-panel axes of *fig*.

    For plain / faceted single plots the first axes is the panel.  In a
    plotnine composition some helper axes (legends, tag titles) may precede
    the panels, so prefer the first visible axes that actually holds data.
    """
    axes = fig.get_axes()
    if not axes:
        raise RuntimeError("figure has no axes; cannot measure panel size")
    for ax in axes:
        if not ax.get_visible():
            continue
        pos = ax.get_position()
        if pos.width <= 0 or pos.height <= 0:
            continue
        if ax.collections or ax.images or ax.lines:
            return ax
    return axes[0]


def _apply_fixed_panel(fig, panel_w: float, panel_h: float) -> None:
    """Resize *fig* so the scatter panel is exactly panel_w × panel_h inches.

    Works for single plots, faceted plots and plotnine compositions
    (``p1 / p2`` & friends).  The composition case is handled by iterating
    against the live ``PlotnineCompositionLayoutEngine``.
    """
    le = fig.get_layout_engine()

    if le is None:
        # Layout already frozen (e.g. by a custom post-draw colourbar): axes
        # positions are fixed figure fractions, so size the figure directly from
        # the current panel position.  No iteration is needed because the frozen
        # positions do not change when the figure is resized.
        ax = _first_panel_axes(fig)
        pos = ax.get_position()
        fig.set_size_inches(panel_w / pos.width, panel_h / pos.height)
        fig.canvas.draw()
        return

    from plotnine._mpl.layout_manager._engine import (
        PlotnineCompositionLayoutEngine,
    )

    if isinstance(le, PlotnineCompositionLayoutEngine):
        _apply_fixed_panel_composition(fig, le, panel_w, panel_h)
        return

    fw, fh = fig.get_size_inches()
    from plotnine._mpl.layout_manager._spaces import LayoutSpaces

    # Facet grid dimensions (1x1 for un-faceted plots).
    facet = le.plot.facet
    n_col = max(1, int(getattr(facet, "ncol", 1) or 1))
    n_row = max(1, int(getattr(facet, "nrow", 1) or 1))

    spaces = LayoutSpaces(le.plot)
    l_in = spaces.l.total * fw
    r_in = spaces.r.total * fw
    b_in = spaces.b.total * fh
    t_in = spaces.t.total * fh
    # Initial estimate: reserve width/height for *every* panel in the grid.
    # Inter-panel gaps are ignored here and corrected by the iteration below.
    fig.set_size_inches(
        l_in + n_col * panel_w + r_in,
        b_in + n_row * panel_h + t_in,
    )
    le.execute(fig)
    fig.canvas.draw()
    for _ in range(5):
        ax = _first_panel_axes(fig)
        pos = ax.get_position()
        cur_fw, cur_fh = fig.get_size_inches()
        actual_w = pos.width * cur_fw
        actual_h = pos.height * cur_fh
        # Each panel only claims 1/n_col of any figure-width change (and 1/n_row
        # of a height change), so scale the correction by the grid dimensions.
        fig.set_size_inches(
            cur_fw + n_col * (panel_w - actual_w),
            cur_fh + n_row * (panel_h - actual_h),
        )
        le.execute(fig)
        fig.canvas.draw()


def _apply_fixed_panel_composition(fig, le, panel_w: float, panel_h: float) -> None:
    """``_apply_fixed_panel`` variant for plotnine composition figures.

    The composition shares one figure between several plots.  After
    ``harmonise`` every panel has equal width and height, so resizing the
    figure so that *one* panel reaches the target size sizes them all.  We
    iterate because the surrounding margins (titles, axis labels, legends)
    are absolute and only settle once the layout engine re-executes.
    """
    cmp = le.composition
    n_col = max(1, int(getattr(cmp, "ncol", 1) or 1))
    n_row = max(1, int(getattr(cmp, "nrow", 1) or 1))

    for _ in range(10):
        le.execute(fig)
        fig.canvas.draw()
        ax = _first_panel_axes(fig)
        pos = ax.get_position()
        cur_fw, cur_fh = fig.get_size_inches()
        actual_w = pos.width * cur_fw
        actual_h = pos.height * cur_fh
        if abs(actual_w - panel_w) < 0.005 and abs(actual_h - panel_h) < 0.005:
            break
        # As with facets: a figure-size change is split across the grid, so a
        # single panel only grows by 1/n_col (width) or 1/n_row (height) of it.
        fig.set_size_inches(
            cur_fw + n_col * (panel_w - actual_w),
            cur_fh + n_row * (panel_h - actual_h),
        )


def _patch_composition_post_draw() -> None:
    """Make plotnine compositions run each member plot's post-draw hooks.

    ``plotnine``'s ``Compose.draw`` / ``Compose.save`` build a shared figure
    and apply their own layout engine, bypassing ``ggplot.save_helper`` where
    our hooks normally run.  Without this patch, features registered through
    :func:`_ensure_post_draw` — most notably :func:`panel_size`
    (``_apply_fixed_panel``) — are silently ignored for stacked / side-by-side
    plots.  We wrap ``Compose.draw`` to settle the layout and then run every
    member plot's hooks once (deduplicated per figure).
    """
    import plotnine.composition._compose as _cmp

    if getattr(_cmp.Compose.draw, "_msp_patched", False):
        return
    _orig_draw = _cmp.Compose.draw

    def _collect_plots(cmp, out):
        for item in cmp:
            if isinstance(item, _cmp.Compose):
                _collect_plots(item, out)
            else:
                out.append(item)

    def _draw(self, *, show: bool = False):
        figure = _orig_draw(self, show=show)
        if getattr(figure, "_msp_post_draw_done", False):
            return figure
        plots = []
        _collect_plots(self, plots)
        hook_plots = [
            p
            for p in plots
            if isinstance(p, _PlotWithPostDraw) and getattr(p, "_post_draw_fns", None)
        ]
        if hook_plots:
            # Settle the composition layout before hooks read panel geometry.
            le = figure.get_layout_engine()
            if le is not None:
                le.execute(figure)
                figure.canvas.draw()
            for p in hook_plots:
                for fn in p._post_draw_fns:
                    fn(figure)
        figure._msp_post_draw_done = True
        return figure

    _draw._msp_patched = True
    _cmp.Compose.draw = _draw


_patch_composition_post_draw()


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


def _draw_embedding_label(
    fig, *, label: str, fontsize: float, color: str = "#777777"
) -> None:
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

    fig.text(
        x_frac, y_frac, label, ha="left", va="bottom", fontsize=fontsize, color=color
    )


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
    border_cats: Optional[dict] = None,
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
                [0],
                [0],
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
                    txt.set_position(
                        (bar_left - title_half, grid_y0 + grid_height * 0.5)
                    )
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


def _normalize_discrete_palette(value):
    """Coerce one discrete-palette spec into ``None`` | list | dict.

    Accepts a list/tuple of colors, a ``{category: color}`` dict, ``None``
    (= use the defaults) or a matplotlib ``ListedColormap``-alike.
    """
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    # Assume matplotlib ListedColormap or similar
    return list(value.colors)


def _palette_key(column) -> str | tuple:
    """Palette lookup key for a column spec.

    Source-routed ``(source, column)`` specs keep their source, so a routed
    column is addressed exactly as it is plotted and can carry a palette of its
    own, separate from a same-named column of the primary source.
    """
    if isinstance(column, tuple) and len(column) == 2:
        return (str(column[0]), str(column[1]))
    return str(column)


def _is_per_column_palette(mapping: dict) -> bool:
    """True if *mapping* is column → palette rather than category → color.

    The two dict forms are told apart by their values: a color is a string,
    a palette is anything else (list/tuple/dict/ListedColormap/None).
    Mixing both in one dict is an error.
    """
    if not mapping:
        return False
    is_palette = [not isinstance(v, str) for v in mapping.values()]
    if all(is_palette):
        return True
    if any(is_palette):
        cat_like = sorted(str(k) for k, v in mapping.items() if isinstance(v, str))
        raise ValueError(
            "colormap_discrete() got a dict mixing category → color entries "
            f"({cat_like}) with column → palette entries. Pass either a flat "
            "{category: color} palette, or a {column: palette} mapping whose "
            "values are lists/dicts of colors."
        )
    return False


@dataclass(frozen=True)
class BorderConfig:
    size: float = 15
    resolution: int = 200
    blur: float = 1.1
    threshold: float = 0.95
    # None → resolved at draw time: the per-column discrete palette for the
    # cell type column if one is set, else DEFAULT_COLORS_BORDERS.
    colors: Optional[tuple] = None
    legend: bool = True
    legend_dot_size: float = 4
    legend_dot_alpha: float = 1
    legend_title: Optional[str] = None  # None → use the cell_type column name
    respect_filter: bool = False  # False → borders computed from unfiltered cells


@dataclass(frozen=True)
class GridConfig:
    labels: object = (
        False  # False/None = off, True/"letters" = A1 labels, "coords" = (x,y) labels
    )
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
        self._dot_alpha: float = 1
        self._legend_dot_size: float = 4
        self._legend_dot_alpha: float = 1
        self._panel_border: bool = True
        self._spine_color: str = "#555555"
        self._tick_color: str = "#555555"
        self._bg_color: str = "#FFFFFF"
        self._anti_overplot: bool = True
        self._anti_overplot_ascending = True
        self._anti_overplot_seed: Optional[int] = None
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
        self._cat_colors: Optional[List[str] | Dict[str, str]] = (
            None  # None → DEFAULT_COLORS_CATEGORIES
        )
        # {column: palette} mapping (key "" = fallback for any other column);
        # keys are _palette_key()s, mutually exclusive with flat _cat_colors.
        self._cat_colors_by_column: Optional[Dict[str | tuple, object]] = None
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
        # 2D faceting (facet_grid rows ~ cols); mutually exclusive with facet().
        self._facet_row_variable: Optional[str] = None
        self._facet_col_variable: Optional[str] = None
        self._facet_args: dict = {}
        self._title_override: str | Callable[[str], str] | _UnsetType = _UNSET

        # embedding label
        self._embedding_label: bool = False
        self._embedding_label_size: Optional[float] = None

        # theme overwrites - passed to p9.theme()
        self._theme_overwrites = {}

        if ad_or_data is not None:
            if isinstance(ad_or_data, EmbeddingData):
                self._data = ad_or_data
            elif isinstance(ad_or_data, (str, Path)):
                from .h5ad_source import _require_h5ad_inspect, H5adFacade

                _require_h5ad_inspect()
                self._data = EmbeddingData(H5adFacade(Path(ad_or_data)), embedding)
            else:
                self._data = EmbeddingData(ad_or_data, embedding)

    # ── source ──────────────────────────────────────────────────────────────

    def set_source(
        self,
        ad_or_data,
        embedding: Union[str, tuple, None] = "umap",
        alternative_id_column=None,
        layer: str = "X",
        transform: Optional[Callable[["np.ndarray"], "np.ndarray"]] = None,
    ) -> "ScatterPlotter":
        """Attach data source. Accepts AnnData, a path to an .h5ad file, or a full EmbeddingData.

        Constructs an EmbeddingData from the anndata/h5ad file,
        passing in alternative_id_column, layer, and transform if provided.

        *embedding* is an ``obsm`` key of this source (``"umap"`` finds
        ``"X_umap"``), a ``(key, col1, col2)`` tuple to pick two columns of one
        array, a ``(source_name, key)`` tuple to read it from a named
        alternative source instead, or ``None`` to choose it later.  When the
        coordinates live in a *different file* than the expression matrix, keep
        that file the primary source and follow up with
        :meth:`set_embedding_source` — then columns still resolve under their
        plain names::

            plotter = (
                ScatterPlotter()
                .set_source("expression.h5ad", embedding=None)
                .set_embedding_source("coordinates.h5ad", "umap")
            )
        """
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
            if isinstance(ad_or_data, (str, Path)):
                from .h5ad_source import _require_h5ad_inspect, H5adFacade

                _require_h5ad_inspect()
                ad_or_data = H5adFacade(Path(ad_or_data))
            new._data = EmbeddingData(
                ad_or_data,
                embedding,
                grid_size=grid_size,
                grid_letters_on_vertical=glv,
                alternative_id_column=alternative_id_column,
                transform=transform,
                layer=layer,
            )
        new._boundary_cache = {"df": None}
        return new

    def set_embedding(self, embedding) -> "ScatterPlotter":
        """Plot on a different embedding, leaving the sources untouched.

        *embedding* takes the same forms as :meth:`set_source`'s argument.
        The plotter is immutable — a new copy is returned.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .set_embedding()")
        new = copy.copy(self)
        new._data = self._data.set_embedding(embedding)
        new._boundary_cache = {"df": None}
        return new

    def set_embedding_source(
        self,
        source,
        embedding="umap",
        name=None,
        layer="X",
        transform=None,
    ) -> "ScatterPlotter":
        """Take the embedding from *source* while the primary source stays put.

        For the common split where one file holds the expression matrix and
        another the coordinates (plus the ``obs`` annotation that goes with
        them).  Keep the expression file as the primary source, name the other
        one here, and every column keeps its plain name::

            plotter = (
                ScatterPlotter()
                .set_source("expression.h5ad", embedding=None)
                .set_embedding_source("coordinates.h5ad", "umap")
            )
            plotter.plot("S100A8")          # not ("coordinates", "S100A8")

        *source* may be an ``AnnData``, an :class:`H5adFacade`, a path to an
        ``.h5ad`` file, or another ``EmbeddingData``; it is registered as an
        alternative source, so its ``obs`` columns are available to
        :meth:`plot` / :meth:`get_column` under their plain names too.
        Coordinates are reindexed onto the primary ``obs_names`` (extra cells
        dropped, primary cells missing from *source* → ``NaN``).

        *embedding* is an ``obsm`` key of *source* (or a ``(key, col1, col2)``
        tuple).  *name* is the name the source is registered under (default:
        a reserved one, which a second call replaces — so switching embedding
        file stays a single edit).  *layer* and *transform* apply to feature
        columns read from *source*.

        The plotter is immutable — a new copy is returned.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .set_embedding_source()")
        new = copy.copy(self)
        new._data = self._data.set_embedding_source(
            source, embedding, name=name, layer=layer, transform=transform
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

    def add_alternative_source(
        self, source, name=None, layer="X", transform=None
    ) -> "ScatterPlotter":
        """Register a fallback source for column / gene lookup.

        When a name passed to :meth:`plot` or :meth:`get_column` is not found
        in the primary source, each registered alternative source is tried in
        registration order.  The first one that resolves the name wins, and its
        values are reindexed onto the primary source's ``obs_names`` so they
        align with the embedding (extra cells dropped, missing cells → NaN).

        If *name* is given, the source can additionally be addressed
        explicitly via ``plot((name, column))`` or
        ``get_column((name, column))`` — which resolves *column* from that
        specific source only.  Names must be unique among registered
        alternatives.

        *layer* selects which expression matrix feature columns are read from
        (``'X'`` → ``.X``, else ``.layers[layer]``).  *transform*, when given,
        is a callable applied to each feature column read from this source.

        ``source`` may be an ``AnnData``, an :class:`H5adFacade`, a path to an
        ``.h5ad`` file (requires ``h5ad-inspect``), or another
        ``EmbeddingData``.  The plotter is immutable — a new copy is returned.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .add_alternative_source()")
        new = copy.copy(self)
        new._data = self._data.add_alternative_source(
            source, name=name, layer=layer, transform=transform
        )
        return new

    def add_derived_source(self, derived, name=None) -> "ScatterPlotter":
        """Register a computed (derived) source for column / gene lookup.

        *derived* is a ``{column_name: callable}`` mapping where each callable
        receives the underlying :class:`EmbeddingData` and returns a
        :class:`pandas.Series` indexed by the primary ``obs_names`` — so it can
        pull from the primary source or any registered alternative via
        :meth:`get_column` and combine the results.  Columns are computed on
        demand (once per lookup, no caching).

        Derived columns are found by :meth:`plot` and :meth:`get_column` both
        as plain strings (checked after the primary source but before
        alternative sources) and, when *name* is given, explicitly via
        ``plot((name, column_name))`` / ``get_column((name, column_name))``.
        *name* must be unique among all alternative and derived sources.

        The plotter is immutable — a new copy is returned.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .add_derived_source()")
        new = copy.copy(self)
        new._data = self._data.add_derived_source(derived, name=name)
        return new

    # ── dot appearance ───────────────────────────────────────────────────────

    def style(
        self,
        *,
        dot_size: Optional[float] = None,
        dot_alpha: Optional[float] = None,
        legend_dot_size: Optional[float] = None,  # None = keep current (default 4)
        legend_dot_alpha: Optional[float] = None,  # None = keep current (default 1)
        panel_border: Optional[bool] = None,
        spine_color: Optional[str] = None,
        tick_color: Optional[str] = None,
        bg_color: Optional[str] = None,
    ) -> "ScatterPlotter":
        """Configure visual appearance. Only supplied arguments are changed.

        Args:
            dot_size:        Point size for the main scatter layer.
            dot_alpha:       Point transparency for the main scatter layer
                             (0–1, default 1).
            legend_dot_size: Override the dot size shown in the categorical legend
                             (default: same as dot_size).
            legend_dot_alpha: Override the dot transparency shown in the categorical
                              legend (0–1, default 1).
            panel_border:    Show/hide the panel border (True = show).
            spine_color:     Hex color for the panel border (default ``"#555555"``).
            tick_color:      Hex color for axis ticks and tick labels (default ``"#555555"``).
            bg_color:        Background hex color (e.g. ``"#FFFFFF"``).
        """
        new = copy.copy(self)
        if dot_size is not None:
            new._dot_size = dot_size
        if dot_alpha is not None:
            new._dot_alpha = dot_alpha
        if legend_dot_size is not None:
            new._legend_dot_size = legend_dot_size
        if legend_dot_alpha is not None:
            new._legend_dot_alpha = legend_dot_alpha
        if panel_border is not None:
            new._panel_border = panel_border
        if spine_color is not None:
            new._spine_color = spine_color
        if tick_color is not None:
            new._tick_color = tick_color
        if bg_color is not None:
            new._bg_color = bg_color
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
        cmap_or_list_or_dict: None | List[str] | Dict[str, str] = DoNotUpdate,
        *,
        title: str | _DoNotUpdateType = DoNotUpdate,
    ) -> "ScatterPlotter":
        """Set the discrete color palette and/or legend title for categorical data.

        cmap_or_list_or_dict accepts:
        - A list of hex color strings (positional, cycling).
        - A dict mapping category name → hex color string.
        - A matplotlib ``ListedColormap`` or similar (uses ``.colors``).
        - A dict mapping *column name* → any of the above, so one plotter
          colors every column consistently (see below).
        - ``None`` — reset to the built-in default palette.
        - ``DoNotUpdate`` (default) — leave the palette unchanged.

        Per-column palettes::

            plotter.colormap_discrete({
                "leiden": ["#98414f", "#776431", ...],   # positional
                "genotype": {"wt": "#333333", "ko": "#cc0000"},  # by category
                ("imputed", "leiden"): [...],            # source-routed column
                "": ["#111111", "#eeeeee"],              # fallback, any other column
            })

        Whenever a categorical column is plotted, its palette is looked up by
        the column spec as passed to the plot method — including
        ``(source, column)`` tuples, which are keyed with their source, so a
        routed column can differ from the primary column of the same name.  The
        lookup then falls back to the name the spec resolved to, the ``""``
        entry, and finally the built-in defaults.  That way a single plotter (or
        a single configured base plotter that others are derived from)
        guarantees column X is always drawn in the same colors, without having
        to remember to re-apply the right palette per builder.

        The two dict forms are told apart by their values: string values mean
        ``{category: color}``, anything else (list/dict/ListedColormap/None)
        means ``{column: palette}``.  Mixing both raises ``ValueError``.
        Setting a palette replaces the previous one — the mapping is not merged
        with an earlier call.

        title: Legend title for the color scale.  ``None`` resets to the
        auto-derived column name; ``DoNotUpdate`` leaves the current title.
        """
        new = copy.copy(self)
        if cmap_or_list_or_dict is not DoNotUpdate:
            if isinstance(cmap_or_list_or_dict, dict) and _is_per_column_palette(
                cmap_or_list_or_dict
            ):
                new._cat_colors = None
                new._cat_colors_by_column = {
                    _palette_key(k): _normalize_discrete_palette(v)
                    for k, v in cmap_or_list_or_dict.items()
                }
            else:
                new._cat_colors = _normalize_discrete_palette(cmap_or_list_or_dict)
                new._cat_colors_by_column = None
            # Borders may draw their colors from the per-column mapping, and the
            # boundary df caches the resolved colors — drop it if it no longer
            # matches (cheap no-op when nothing was cached).
            if (
                self._border_config is not None
                and self._border_config.colors is None
                and self._boundary_cache["df"] is not None
            ):
                new._boundary_cache = {"df": None}
        if title is not DoNotUpdate:
            assert not isinstance(title, _DoNotUpdateType)
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

    # overplotting

    def anti_overplot(
        self,
        enabled: bool = True,
        ascending: bool = True,
        *,
        seed: Optional[int] = None,
    ) -> "ScatterPlotter":
        """Control point draw order to mitigate overplotting bias.

        This does not jitter point positions — it only changes the order in
        which (already-fixed) point positions are painted, since later-drawn
        points cover earlier ones.

        For numerical data (default ``enabled=True``): points are sorted by
        expression value before drawing, so the highest values end up on top.
        Set ``ascending=False`` to put the lowest values on top instead — this
        applies to the *whole* point stack, including the zero-value underlay
        and the clipped (``>max_quantile``) points, so with ``ascending=False``
        the clipped points sit at the *bottom* rather than on top. The
        colorbar is flipped to match, so the colorbar end corresponding to the
        values drawn on top is always at the top of the bar.

        For categorical data (default ``enabled=True``): points are grouped
        and drawn by category, so the last category (in its natural /
        categorical order) ends up on top. Set ``ascending=False`` to reverse
        which category ends up on top.

        In both cases, ``enabled=False`` draws points in their original
        (dataset) row order instead of sorting/grouping them.

        Passing ``seed`` overrides ``enabled``/``ascending`` and instead
        draws all points in a fully randomized, reproducible order. This is
        especially useful for categorical data, where grouping by category
        always biases which category visually dominates in overlapping
        regions — a random seed gives an unbiased, still-reproducible draw
        order.


        Contrast to plotting the outlier layer (outlier()), which enabled
        by default for categorical data, draws outliers on top *after*
        this anti overplotting measure.
        """
        new = copy.copy(self)
        new._anti_overplot = enabled
        new._anti_overplot_ascending = ascending
        new._anti_overplot_seed = seed
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
        legend_dot_alpha: float = 1,
        legend_title: Optional[str] = None,
        respect_filter: bool = False,
    ) -> "ScatterPlotter":
        """Overlay cell-type region borders.

        Args:
            colors: Border palette (positional, cycling over the cell type
                    categories).  ``None`` (default) auto-resolves: the
                    per-column palette from :meth:`colormap_discrete` if one is
                    configured for *cell_type_column* (or via its ``""``
                    fallback), else ``DEFAULT_COLORS_BORDERS``.
        """
        new = copy.copy(self)
        # None = resolve lazily at draw time (see _resolve_border_colors)
        resolved_colors = tuple(colors) if colors is not None else None
        new._border_config = BorderConfig(
            size=size,
            resolution=resolution,
            blur=blur,
            threshold=threshold,
            colors=resolved_colors,
            legend=legend,
            legend_dot_size=legend_dot_size,
            legend_dot_alpha=legend_dot_alpha,
            legend_title=legend_title,
            respect_filter=respect_filter,
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
            or old.respect_filter != respect_filter
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
            new._data = new._data._replace(
                grid_size=resolved_grid_size,
                grid_letters_on_vertical=resolved_vl,
            )
        return new

    def without_grid(self) -> "ScatterPlotter":
        new = copy.copy(self)
        new._grid_config = None
        return new

    # ── viewport ─────────────────────────────────────────────────────────────

    def focus_on(self, region) -> "ScatterPlotter":
        """Restrict viewport to *region* (a soft zoom; every cell is still drawn).

        *region* is a 2-corner ``(corner1, corner2)`` box; each corner is either a
        grid-label string or an ``(x, y)`` coordinate pair::

            plotter.focus_on(("A1", "C5"))                     # grid labels
            plotter.focus_on(((x_min, y_min), (x_max, y_max)))  # raw coordinates

        Grid-label corners require a grid; raises ValueError if the grid was
        disabled with ``without_grid()``.  Use :meth:`hard_filter` to instead
        restrict the data and re-span the grid over just the region.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .focus_on()")
        uses_grid_labels = isinstance(region, (tuple, list)) and any(
            isinstance(c, str) for c in region
        )
        if uses_grid_labels and self._grid_config is None:
            raise ValueError(
                "focus_on() with grid labels requires a grid; call .with_grid() "
                "to re-enable or remove .without_grid() (or pass (x, y) corners)"
            )
        new = copy.copy(self)
        new._data = self._data.focus_on(region)
        return new

    def unfocus(self) -> "ScatterPlotter":
        if self._data is None:
            raise RuntimeError("call .set_source() before .unfocus()")
        new = copy.copy(self)
        new._data = self._data.unfocus()
        return new

    # ── cell filtering ──────────────────────────────────────────────────────

    def set_filter(self, filter_fn) -> "ScatterPlotter":
        """Keep only the cells selected by *filter_fn* in subsequent plots.

        *filter_fn* is a callable that receives the underlying
        :class:`EmbeddingData` (with the filter disabled, so it sees the full
        dataset) and returns a boolean vector (array or ``Series``) of length
        ``n_obs`` marking the cells to keep.  The filter is evaluated lazily
        (on first use) and cached until the next :meth:`set_filter` call.

        The filter restricts the cells shown by :meth:`plot` and friends, but
        **not** the coordinate bounds — the embedding frame always reflects the
        full dataset, matching :meth:`focus_on`.

        Pass ``None`` to remove an existing filter.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .set_filter()")
        new = copy.copy(self)
        new._data = self._data.set_filter(filter_fn)
        # Boundaries are derived from coordinates()/get_column(), which respect
        # the filter — drop any cached boundary image so it is recomputed.
        new._boundary_cache = {"df": None}
        return new

    def hard_filter(self, spec) -> "ScatterPlotter":
        """Restrict to *spec* with the bounds/grid following the kept subset.

        Accepts the same *spec* grammar as :meth:`set_filter` (a callable, or a
        region / list of regions).  Unlike the soft :meth:`set_filter`, this
        re-spans the grid and coordinate frame over just the restricted cells, so
        downstream grid analyses (Moran's I, interactive per-cell views) recompute
        over the smaller cells.  Pass ``None`` to remove the filter.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .hard_filter()")
        new = copy.copy(self)
        new._data = self._data.hard_filter(spec)
        new._boundary_cache = {"df": None}
        return new

    def unfilter(self) -> "ScatterPlotter":
        """Remove any cell filter set via :meth:`set_filter` / :meth:`hard_filter`."""
        if self._data is None:
            raise RuntimeError("call .set_source() before .unfilter()")
        new = copy.copy(self)
        new._data = self._data.unfilter()
        new._boundary_cache = {"df": None}
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

    def facet(
        self,
        variable: str,
        n_col: int = 2,
        n_row: Optional[int] = DoNotUpdate,
        scales: str = DoNotUpdate,
        shrink: bool = DoNotUpdate,
        labeller: str = DoNotUpdate,
        as_table: bool = DoNotUpdate,
        drop: bool = DoNotUpdate,
        dir: str = DoNotUpdate,
    ) -> "ScatterPlotter":
        new = copy.copy(self)
        new._facet_variable = variable
        if dir is not DoNotUpdate and dir not in ("h", "v"):
            raise ValueError("dir must be 'h' or 'v'")
        new._n_col = n_col
        # facet() and facet_2d() are mutually exclusive.
        new._facet_row_variable = None
        new._facet_col_variable = None
        raw: dict = {
            "ncol": n_col,
            "nrow": n_row,
            "scales": scales,
            "shrink": shrink,
            "labeller": labeller,
            "as_table": as_table,
            "drop": drop,
            "dir": dir,
        }
        new._facet_args = {k: v for k, v in raw.items() if v is not DoNotUpdate}
        return new

    def facet_2d(
        self,
        row_variable: str,
        col_variable: str,
        margins: bool = DoNotUpdate,
        scales: str = DoNotUpdate,
        space: str = DoNotUpdate,
        shrink: bool = DoNotUpdate,
        labeller: str = DoNotUpdate,
        as_table: bool = DoNotUpdate,
        drop: bool = DoNotUpdate,
    ) -> "ScatterPlotter":
        """Facet into a 2-D grid (``facet_grid(row ~ col)``).

        *row_variable* labels the grid rows, *col_variable* labels the columns,
        matching ggplot's ``facet_grid`` convention.  Unsets any regular
        :meth:`facet` configuration, and vice versa.
        """
        new = copy.copy(self)
        new._facet_row_variable = row_variable
        new._facet_col_variable = col_variable
        # facet() and facet_2d() are mutually exclusive.
        new._facet_variable = None
        raw: dict = {
            "margins": margins,
            "scales": scales,
            "space": space,
            "shrink": shrink,
            "labeller": labeller,
            "as_table": as_table,
            "drop": drop,
        }
        new._facet_args = {k: v for k, v in raw.items() if v is not DoNotUpdate}
        return new

    def unfacet(self) -> "ScatterPlotter":
        new = copy.copy(self)
        new._facet_variable = None
        new._facet_row_variable = None
        new._facet_col_variable = None
        new._facet_args = {}
        return new

    # ── faceting (internals) ────────────────────────────────────────────────

    def _is_faceted(self) -> bool:
        return (
            self._facet_variable is not None
            or self._facet_row_variable is not None
            or self._facet_col_variable is not None
        )

    def _facet_grid_dims(self, df: pd.DataFrame) -> "tuple[int, int]":
        """Return (n_col, n_row) panel counts for figure sizing."""
        if self._facet_variable is not None:
            n_facets = df["facet"].nunique()
            n_row = -(-n_facets // self._n_col)  # ceil division
            return self._n_col, n_row
        n_col = df["facet_col"].nunique() if self._facet_col_variable is not None else 1
        n_row = df["facet_row"].nunique() if self._facet_row_variable is not None else 1
        return n_col, n_row

    def _apply_facet_layer(self, p: p9.ggplot) -> p9.ggplot:
        """Attach the configured facet_wrap / facet_grid to *p*."""
        if self._facet_variable is not None:
            return p + p9.facet_wrap("~facet", **self._facet_args)
        if self._facet_row_variable is not None or (
            self._facet_col_variable is not None
        ):
            kwargs: dict = {}
            if self._facet_row_variable is not None:
                kwargs["rows"] = "facet_row"
            if self._facet_col_variable is not None:
                kwargs["cols"] = "facet_col"
            kwargs.update(self._facet_args)
            return p + p9.facet_grid(**kwargs)
        return p

    def _facet_theme_overwrites(self) -> dict:
        """Theme kwargs implied by the current faceting mode.

        A 2-D ``facet_grid`` (with row strips) places the row labels on the
        right, between the panels and the colour bar.  Horizontal labels eat
        horizontal space there, so they are rotated to match plotnine's usual
        ``strip_text_y`` convention and free up the gap to the legend.
        """
        if self._facet_row_variable is not None:
            return {"strip_text_y": p9.element_text(angle=-90)}
        return {}

    def _theme_kwargs(self, **defaults) -> dict:
        """Merge a plot's own theme defaults with faceting and user overwrites.

        Splatting ``**self._theme_overwrites`` alongside explicit keywords in
        the same ``p9.theme(...)`` call raises ``TypeError: got multiple values
        for keyword argument`` as soon as the user themes something the plot
        already sets itself (``figure_size``, ``axis_text``, ...).  Merging
        first makes that an override instead of an error: later wins, so the
        user's ``theme()`` beats the faceting implication, which beats the
        plot's default.

        Faceting theming sits in the middle because it is derived state: it
        reflects the facet layout the plot just built, but the user asked for
        their overwrites by name.
        """
        return {**defaults, **self._facet_theme_overwrites(), **self._theme_overwrites}

    # ── title ────────────────────────────────────────────────────────────────

    def title(self, t: str | Callable[[str], str]) -> "ScatterPlotter":
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

    # changing apperance via plotnine theming

    def theme(self, **theme_args) -> "ScatterPlotter":
        """Change plotnine theming args.

        See https://plotnine.org/reference/#themeables
        for a list of valid arguments.

        example:
        p = p.theme(strip_text=p9.element_text(size=24))
        """
        new = copy.copy(self)
        new._theme_overwrites = {**self._theme_overwrites, **theme_args}
        return new

    # ── terminal ─────────────────────────────────────────────────────────────

    def plot(self, column: Union[str, tuple]) -> p9.ggplot:
        """Build and return a plotnine ggplot for the given obs column or gene.

        *column* may be a plain name (resolved against the primary source, then
        any registered alternative sources) or a ``(source_name, column)``
        tuple to pull the column from a specific named alternative source.
        """
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
        # Leave non-numerical dtypes as-is (object/bool/category) — forcing an
        # early astype("category") here would bake in pandas' default
        # (lexicographic) category order and make _build_categorical's own
        # dtype check always take the "already category" branch, silently
        # skipping its natsorted(...) fallback for non-category columns.
        df = coords.copy()
        df["expression"] = expr

        # Facet columns
        if self._facet_variable is not None:
            facet_vals, _ = data.get_column(self._facet_variable)
            df["facet"] = facet_vals
        if self._facet_row_variable is not None:
            row_vals, _ = data.get_column(self._facet_row_variable)
            df["facet_row"] = row_vals
        if self._facet_col_variable is not None:
            col_vals, _ = data.get_column(self._facet_col_variable)
            df["facet_col"] = col_vals

        # Figure size
        if self.fig_size is None:
            if self._is_faceted():
                n_col, n_row = self._facet_grid_dims(df)
                fig_size = (6 * n_col, 5 * n_row)
            else:
                has_border_legend = (
                    self._layer_borders
                    and self._border_config is not None
                    and self._border_config.legend
                    and self._cell_type_column is not None
                )
                # Two legends on the right need more width
                fig_size = (8 if has_border_legend else 6, 5)
        else:
            fig_size = self.fig_size

        # Build plot
        is_gene = data.is_gene(column)
        if is_numerical:
            p = self._build_numerical(df, expr_name, is_gene=is_gene)
        else:
            p = self._build_categorical(df, expr_name, column=column)

        # Focus viewport
        if data.has_focus:
            p = p + p9.coord_cartesian(xlim=(x_min, x_max), ylim=(y_min, y_max))

        # Facet
        p = self._apply_facet_layer(p)

        # Title
        if callable(self._title_override):
            title = self._title_override(self._display_name(data, expr_name))
        elif self._title_override is not _UNSET:
            title = self._title_override
        else:
            # expr_name is in var.index space (the gene symbol); _display_name
            # expands it to "alt_id (symbol)" when an alternative id column is
            # set.  The colourbar name and is_gene detection keep using the bare
            # symbol.
            title = self._display_name(data, expr_name)
        p = p + p9.labs(title=title)

        # Theme (must come before grid axis ticks so theme_void doesn't override them)
        p = self._apply_embedding_theme(p)
        p = p + p9.theme(
            **self._theme_kwargs(
                figure_size=fig_size,
                legend_box="horizontal",
            )
        )

        # Grid axis tick labels (applied after theme so they survive theme_void)
        p = self._apply_axis_ticks(p, grid_layers=False)

        # Fixed panel size
        p = self._register_fixed_panel(p)

        # Embedding label (after fixed-panel so tight-bbox reflects final size)
        p = self._maybe_add_embedding_label(p, data)

        return p

    def plot_density(
        self,
        bins: int = 200,
        quantile: float = 0.99,
        cmap_colors=None,
        include_counts=False,
        count_text_size=5,
    ) -> p9.ggplot:
        """Build a 2D cell-density heatmap.

        Args:
            bins:     Number of bins per axis for the 2D histogram.
            quantile: Upper quantile at which density is clipped (default 0.99).
                      Set to 1.0 for no clipping (uses the built-in plotnine legend).
                      Any value < 1.0 clips the colour scale and draws the same
                      custom matplotlib colourbar as numerical scatter plots.

        Honours the plotter's grid overlay (``with_grid``), faceting
        (``facet``), viewport (``focus_on``) and title
        (``title``) configuration, mirroring :meth:`plot`.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_density()")

        from .transforms import prepare_density_df

        data = self._data
        x_min, x_max, y_min, y_max = data.bounds()

        facet_series = None
        if self._facet_variable is not None:
            facet_series, _ = data.get_column(self._facet_variable)

        facet_row_series = None
        facet_col_series = None
        if self._facet_row_variable is not None:
            facet_row_series, _ = data.get_column(self._facet_row_variable)
        if self._facet_col_variable is not None:
            facet_col_series, _ = data.get_column(self._facet_col_variable)

        df = prepare_density_df(
            data,
            bins=bins,
            facet=facet_series,
            facet_row=facet_row_series,
            facet_col=facet_col_series,
        )

        has_clips = quantile < 1.0
        if has_clips:
            clip_val = float(df["density"].quantile(quantile))
        else:
            clip_val = float(df["density"].max())

        nonzero = df["density"][df["density"] > 0]
        density_min = float(nonzero.min()) if len(nonzero) > 0 else 0.0

        if cmap_colors is None:
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
        if self._layer_borders and self._border_config is not None:
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

        if include_counts:
            count_df = df[df["density"] > 0]
            count_df = count_df.assign(
                text=count_df["density"].round(0).astype(int).astype(str)
            )
            p = p + p9.geom_text(
                p9.aes("x", "y", label="text"), size=count_text_size, data=count_df
            )

        # Focus viewport (mirrors .plot())
        if data.has_focus:
            p = p + p9.coord_cartesian(xlim=(x_min, x_max), ylim=(y_min, y_max))

        # Facet (mirrors .plot())
        p = self._apply_facet_layer(p)

        # Title (mirrors .plot() / .plot_embedding_color())
        if self._title_override is not _UNSET:
            p = p + p9.labs(title=self._title_override)
        else:
            emb_name = data.embedding
            if emb_name.startswith("X_"):
                emb_name = emb_name[2:]
            p = p + p9.labs(title=emb_name)

        # Figure size — grows with facets, like .plot()
        if self.fig_size is None:
            if self._is_faceted():
                n_col, n_row = self._facet_grid_dims(df)
                fig_size = (6 * n_col, 5 * n_row)
            else:
                fig_size = (6, 5)
        else:
            fig_size = self.fig_size

        p = self._apply_embedding_theme(p)
        p = p + p9.theme(**self._theme_kwargs(figure_size=fig_size))

        # Grid overlay + axis ticks — parity with .plot() / .plot_moran_markers()
        p = self._apply_axis_ticks(p, grid_layers=True)

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
            p._post_draw_fns.append(
                lambda fig, _c=legend_config: _draw_numerical_legend(fig, **_c)
            )

        # Fixed panel size — registered after the colorbar so the resize wins
        # (the custom legend scales the panel horizontally; must run first).
        p = self._register_fixed_panel(p)

        p = self._maybe_add_embedding_label(p, data)

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

        from .transforms import (
            compute_grid_moran,
            marker_genes_by_region,
            prepare_density_df,
        )

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
                label_rows.append(
                    {
                        "x": row["top_bin_x"],
                        "y": row["top_bin_y"],
                        "label": "\n".join(genes),
                    }
                )
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

        p = self._apply_embedding_theme(p)
        p = p + p9.theme(**self._theme_kwargs(figure_size=(7, 6)))

        p = self._apply_axis_ticks(p, grid_layers=True)

        p = self._maybe_add_embedding_label(p, data)

        return p

    def get_morans_i_markers(
        self,
        k: int = 20,
        min_moran: float = 0.2,
        min_cells: int = 3,
        var_score_column: str | None = None,
    ) -> pd.DataFrame:
        """Return marker genes per current grid cell as a tidy DataFrame.

        Bins the embedding at the plotter's grid resolution, computes Moran's I
        for every gene, and returns the top-*k* markers (above *min_moran*) for
        each occupied grid cell — one row per (cell, gene) pair, ready to plot
        individually.  The bins align 1:1 with the visible grid cells shown by
        :meth:`with_grid`.

        Args:
            k:                Maximum marker genes kept per grid cell (default 20).
            min_moran:        Minimum Moran's I for a gene to qualify (default 0.2).
            min_cells:        Minimum cells per bin (passed to compute_grid_moran).
            var_score_column: If given, use ``adata.var[var_score_column]`` as the
                              gene score instead of computing Moran's I on the fly.

        Returns:
            DataFrame with columns:

            * ``cell``    — grid-cell label (e.g. ``"A1"``) of the gene's top bin
            * ``gene``    — gene name
            * ``moran_i`` — Moran's I score (or *var_score_column* value)
            * ``rank``    — 1-based rank within the cell (1 = highest score)

            Rows are sorted by ``cell`` then ``rank``. Grid cells with no
            qualifying genes are omitted.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .get_morans_i_markers()")

        from .transforms import compute_grid_moran, marker_genes_by_region

        gene_df = compute_grid_moran(
            self._data,
            n_bins=self._data._grid_size,
            min_cells=min_cells,
            var_score_column=var_score_column,
        )
        markers = marker_genes_by_region(gene_df, k=k, min_moran=min_moran)

        out = {
            "grid_cell": [],
            "grid_cell_numeric": [],
            "gene": [],
            "morans_i": [],
            "rank_in_grid_cell": [],
        }
        if self._data._alternative_id_column is not None:
            out["alternative_id"] = []
            alternative_ids = self._data.ad.var[
                self._data._alternative_id_column
            ].to_dict()
        else:
            alternative_ids = {}  # so we get a key error
        _, _, grid_labels_x, grid_labels_ys = self._data.grid_labels()
        for (xi, yi), genes in markers.items():
            for rank, gene in enumerate(genes, start=1):
                x_label = grid_labels_x[xi]
                y_label = grid_labels_ys[yi]
                grid_cell_label = f"{x_label}{y_label}"
                out["grid_cell"].append(grid_cell_label)
                out["grid_cell_numeric"].append((xi, yi))
                out["gene"].append(gene)
                out["morans_i"].append(
                    gene_df.loc[
                        (gene_df["top_bin"] == (xi, yi)) & (gene_df["gene"] == gene),
                        "moran_i",
                    ].values[0]
                )
                out["rank_in_grid_cell"].append(rank)
                if self._data._alternative_id_column is not None:
                    out["alternative_id"].append(alternative_ids[gene])
        return pd.DataFrame(out)

    def get_cluster_markers(
        self,
        column: str,
        k: int = 20,
        min_score: float = 0.0,
        min_cells_per_group: int = 10,
        layer: str | None = None,
    ) -> pd.DataFrame:
        """Return marker genes per category of *column* as a tidy DataFrame.

        Scores every gene per category with a pseudobulk one-vs-rest comparison
        (mean difference gated by expression; see
        :func:`~mbf_singlecell_plotter.transforms.compute_cluster_markers`) and
        returns the top-*k* markers (with ``score`` above *min_score*) for each
        category — one row per (category, gene) pair, ready to plot individually.

        This is the per-*cluster* analogue of :meth:`get_morans_i_markers`
        (which scores per *spatial grid cell* via Moran's I).

        Args:
            column:              Categorical (or bool) obs column with cluster
                                 labels.
            k:                   Maximum marker genes kept per category
                                 (default 20).
            min_score:           Minimum combined score for a gene to qualify
                                 (default 0.0 → keep genes up-regulated versus
                                 the rest).
            min_cells_per_group: Categories with fewer cells are skipped
                                 (default 10).
            layer:               Expression layer for marker computation
                                 (``None`` = the source's configured layer; pass
                                 a raw / log-normalized layer key to score on
                                 that).

        Returns:
            DataFrame with columns:

            * ``category``         — category (cluster) label
            * ``gene``             — gene name
            * ``delta``            — ``mean_in_cluster - mean_in_rest``
                                     (log fold-change on log-normalized data;
                                     z-score difference on scaled data)
            * ``mean_expr``        — the category's pseudobulk mean expression
            * ``score``            — combined score used for ranking
            * ``rank_in_category`` — 1-based rank within the category
                                     (1 = highest score)

            When an alternative id column is configured, an ``alternative_id``
            column is added.  Rows are sorted by ``category`` then
            ``rank_in_category``. Categories with no qualifying genes are omitted.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .get_cluster_markers()")

        from .transforms import compute_cluster_markers

        marker_df = compute_cluster_markers(
            self._data,
            column,
            layer=layer,
            min_cells_per_group=min_cells_per_group,
        )

        filtered = marker_df[marker_df["score"] > min_score]
        parts = []
        for _cat, grp in filtered.groupby("category", observed=True, sort=True):
            top = grp.nlargest(k, "score").reset_index(drop=True)
            top = top.assign(rank_in_category=range(1, len(top) + 1))
            parts.append(top)

        cols = ["category", "gene", "delta", "mean_expr", "score", "rank_in_category"]
        if not parts:
            out = pd.DataFrame({c: [] for c in cols})
        else:
            out = pd.concat(parts, ignore_index=True)[cols]

        if self._data._alternative_id_column is not None:
            alternative_ids = self._data.ad.var[
                self._data._alternative_id_column
            ].to_dict()
            out["alternative_id"] = out["gene"].map(alternative_ids)

        return out

    def save_interactive_moran_grid(
        self,
        column: str,
        output_path,
        min_cells: int = 3,
        k: int = 20,
        min_moran: float = 0.2,
        var_score_column: str | None = None,
        dpi: int = 150,
        debug: bool = False,
        gene_url: str | Callable[[str, str | None], str] | None = None,
        gene_url_inline: bool = False,
        save_tsv: bool = False,
    ) -> None:
        """Save an interactive HTML scatter plot with per-bin marker gene tooltips.

        Renders the scatter for *column* as a PNG, then overlays an invisible
        grid.  Hovering over a cell highlights it (yellow tint) and shows its
        top-k marker genes — ranked by Moran's I spatial autocorrelation — in a
        panel below.  Clicking locks the selection; clicking the same cell
        deactivates it; clicking another cell switches.

        For per-*cluster* markers (differential expression against the rest of a
        categorical column) use :meth:`save_interactive_cluster_markers` instead.

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
            gene_url:         URL template for gene links, or a callable.
                              A ``str`` is treated as a template with a
                              ``{gene}`` placeholder (replaced with the gene
                              name).  Alternatively pass a callable
                              ``gene_url(gene_id, alt_gene_id=None)`` returning a
                              URL ``str`` (or ``None`` to skip a gene);
                              ``gene_id`` is the bare ``var_index`` symbol and
                              ``alt_gene_id`` the alternative id value when an
                              alternative id column is configured.  When ``None``
                              (default) genes are plain text.
            gene_url_inline:  If ``True`` the linked resource is displayed in
                              an ``<img>`` panel below rather than opened in a
                              new browser tab (default ``False``).
            save_tsv:         If ``True`` also write a tidy ``.tsv`` of the marker
                              genes next to the HTML (same path with a ``.tsv``
                              suffix), one row per (grid cell, gene) with columns
                              ``grid_cell, gene, _display_name, moran_i, rank``
                              (plus an ``alternative_id`` column when an
                              alternative id column is configured; default
                              ``False``).
        """
        if self._data is None:
            raise RuntimeError(
                "call .set_source() before .save_interactive_moran_grid()"
            )
        from .interactive import save_interactive_moran_grid as _impl

        _impl(
            self,
            column,
            output_path,
            min_cells=min_cells,
            k=k,
            min_moran=min_moran,
            var_score_column=var_score_column,
            dpi=dpi,
            debug=debug,
            gene_url=gene_url,
            gene_url_inline=gene_url_inline,
            save_tsv=save_tsv,
        )

    def save_interactive_cluster_markers(
        self,
        column: str,
        output_path,
        k: int = 20,
        min_score: float = 0.0,
        min_cells_per_group: int = 10,
        min_cluster_cells: int = 1,
        layer: str | None = None,
        dpi: int = 150,
        debug: bool = False,
        gene_url: str | Callable[[str, str | None], str] | None = None,
        gene_url_inline: bool = False,
        save_tsv: bool = False,
    ) -> None:
        """Save an interactive HTML view of per-cluster pseudobulk marker genes.

        The scatter is coloured by the categorical *column*.  Each gene is scored
        per category with a pseudobulk one-vs-rest comparison (mean difference
        gated by expression; see
        :func:`~mbf_singlecell_plotter.transforms.compute_cluster_markers`).
        Hovering a grid cell lists **every** category present in that bin as its
        own section (largest first), each with that cluster's top-*k* markers and
        their mean-difference Δ — so a small cluster sharing a bin with a large
        one stays reachable.  Clicking locks/switches the selection.

        Each **legend key** is clickable too: it shows that cluster's markers
        across the whole embedding instead of within one bin.  The hotspots are
        derived from the rendered legend artists, and are omitted when the plot
        has no discrete legend.

        Unlike :meth:`save_interactive_moran_grid` (markers per spatial bin via
        Moran's I), markers here are computed per category of *column*.

        Args:
            column:              Categorical obs column (cluster labels); also the
                                 column the scatter is coloured by.
            output_path:         Destination ``.html`` file path.
            k:                   Marker genes shown per cluster (default 20).
            min_score:           Minimum combined score to qualify (default 0.0 →
                                 keep genes up-regulated versus the rest).
            min_cells_per_group: Categories with fewer cells are skipped
                                 (default 10).
            min_cluster_cells:   A category must have at least this many cells
                                 *within a bin* to be listed for it (default 1 →
                                 list every category present; raise to hide stray
                                 cells that bleed across bin borders).
            layer:               Expression layer for marker computation (``None``
                                 = the source's configured layer; pass a raw /
                                 log-normalized layer key to score on that).
            dpi:                 PNG resolution for the scatter image (default 150).
            gene_url:            URL template with a ``{gene}`` placeholder, or a
                                 callable ``gene_url(gene_id, alt_gene_id=None)``.
                                 When ``None`` (default) genes are plain text.
            gene_url_inline:     If ``True`` the linked resource is displayed
                                 inline in an ``<img>`` panel rather than a new
                                 browser tab (default ``False``).
            save_tsv:            If ``True`` also write a tidy ``.tsv`` of the
                                 marker genes next to the HTML (same path with a
                                 ``.tsv`` suffix), one row per (cluster, gene)
                                 with columns
                                 ``cluster, gene, _display_name, delta, rank``
                                 (plus an ``alternative_id`` column when an
                                 alternative id column is configured; default
                                 ``False``).
        """
        if self._data is None:
            raise RuntimeError(
                "call .set_source() before .save_interactive_cluster_markers()"
            )
        from .interactive import save_interactive_cluster_markers as _impl

        _impl(
            self,
            column,
            output_path,
            k=k,
            min_score=min_score,
            min_cells_per_group=min_cells_per_group,
            min_cluster_cells=min_cluster_cells,
            layer=layer,
            dpi=dpi,
            debug=debug,
            gene_url=gene_url,
            gene_url_inline=gene_url_inline,
            save_tsv=save_tsv,
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

        Honours the plotter's faceting (:meth:`facet` / :meth:`facet_2d`),
        computing a separate grid-local histogram per facet group.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_grid_histogram()")

        data = self._data

        facet_series = None
        if self._facet_variable is not None:
            facet_series, _ = data.get_column(self._facet_variable)

        facet_row_series = None
        facet_col_series = None
        if self._facet_row_variable is not None:
            facet_row_series, _ = data.get_column(self._facet_row_variable)
        if self._facet_col_variable is not None:
            facet_col_series, _ = data.get_column(self._facet_col_variable)

        hdf = data.grid_local_histogram(
            column,
            min_cell_count,
            facet=facet_series,
            facet_row=facet_row_series,
            facet_col=facet_col_series,
        )
        hdf["category"] = pd.Categorical(
            hdf["category"], sorted(hdf["category"].unique())
        )
        facet_cols = [
            c for c in ("facet", "facet_row", "facet_col") if c in hdf.columns
        ]
        hdf = hdf.sort_values(facet_cols + ["x", "y", "category"])
        cats = list(hdf["category"].cat.categories)
        colors = self._colors_as_list(cats, column=column)

        if fill_fraction is None:
            fill_fraction = 1.0 if scale_by_count else 0.8
        factor = fill_fraction
        if scale_by_count:
            # sqrt so that linear dimension ∝ sqrt(count) and area ∝ count
            hdf["cell_factor"] = factor * np.sqrt(hdf["total"] / hdf["total"].max())
        else:
            hdf["cell_factor"] = factor

        hdf["frequency"] = hdf["frequency"] * hdf["cell_factor"]

        offset = []
        for _ignored, group in hdf.groupby(facet_cols + ["x", "y"], observed=True):
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
                **self._theme_kwargs(
                    axis_title_x=p9.element_blank(),
                    axis_title_y=p9.element_blank(),
                    panel_grid=p9.element_blank(),
                    axis_ticks_length=3,
                    axis_title=p9.element_blank(),
                )
            )
        )

        p = self._apply_facet_layer(p)

        p = self._register_fixed_panel(p)

        return p

    def plot_histogram(
        self,
        column: str,
        normalize_to: Optional[str] = None,
        stat_bin_args: Optional[dict] = None,
    ) -> p9.ggplot:
        """Build a histogram of *column*.

        Categorical columns get a bar plot of ``value_counts`` (bars coloured
        per category via :meth:`colormap_discrete`, fill legend hidden since
        the x-axis already labels the categories). Numeric columns get a
        binned ``geom_histogram``. Dispatches to
        :meth:`_plot_histogram_categorical` / :meth:`_plot_histogram_numeric`
        based on *column*'s dtype.

        Honours the plotter's faceting (:meth:`facet` / :meth:`facet_2d`,
        computing counts/bins per facet group), :meth:`title`,
        :meth:`panel_size` and :meth:`theme` configuration, mirroring the
        other ``plot_*`` methods.  Returns the plotnine ``ggplot`` object.

        Raises:
            RuntimeError: if no data source has been set.

        **Parameters**

        - ``normalize_to``: categorical columns only, ignored for numeric
          ones. Optional column value. When provided, every bin count is
          divided by this value's count so that it becomes ``1.0`` on the
          y-axis.
        - ``stat_bin_args``: numeric columns only, ignored for categorical
          ones. Optional dict of ``stat_bin`` keyword arguments (e.g.
          ``bins``, ``binwidth``, ``boundary``), forwarded to
          ``geom_histogram``.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_histogram()")

        data = self._data
        expr, expr_name = data.get_column(column)

        is_numerical = (
            expr.dtype != "object" and expr.dtype != "category" and expr.dtype != "bool"
        )
        if is_numerical:
            return self._plot_histogram_numeric(
                data, expr, expr_name, stat_bin_args or {}, column=column
            )
        else:
            return self._plot_histogram_categorical(
                data, expr, expr_name, normalize_to, column=column
            )

    def _plot_histogram_categorical(
        self,
        data,
        expr: pd.Series,
        expr_name: str,
        normalize_to: Optional[str],
        column=None,
    ) -> p9.ggplot:
        if expr.dtype == "category":
            cats = list(expr.cat.categories)
        else:
            cats = natsorted([c for c in expr.unique() if not pd.isna(c)])
        cats_str = [str(c) for c in cats]

        expr_clean = expr.dropna()
        df = pd.DataFrame({"category": expr_clean.astype(str)})
        df["category"] = pd.Categorical(df["category"], categories=cats_str)

        facet_cols = []
        if self._facet_variable is not None:
            fv, _ = data.get_column(self._facet_variable)
            df["facet"] = pd.Categorical(fv.reindex(expr_clean.index).astype(str))
            facet_cols.append("facet")
        if self._facet_row_variable is not None:
            rv, _ = data.get_column(self._facet_row_variable)
            df["facet_row"] = pd.Categorical(rv.reindex(expr_clean.index).astype(str))
            facet_cols.append("facet_row")
        if self._facet_col_variable is not None:
            cv, _ = data.get_column(self._facet_col_variable)
            df["facet_col"] = pd.Categorical(cv.reindex(expr_clean.index).astype(str))
            facet_cols.append("facet_col")

        if facet_cols:
            vc = df.groupby(facet_cols, observed=True, sort=False)[
                "category"
            ].value_counts(sort=False)
            counts = vc.reset_index(name="count")
        else:
            vc = df["category"].value_counts(sort=False)
            counts = pd.DataFrame(
                {"category": vc.index.to_numpy(), "count": vc.to_numpy()}
            )

        if normalize_to is not None:
            if facet_cols:
                norm_rows = counts[
                    (counts["category"] == normalize_to) & (counts["count"] > 0)
                ]
                missing_facets = (
                    counts[facet_cols]
                    .drop_duplicates()
                    .merge(
                        norm_rows[facet_cols],
                        on=facet_cols,
                        how="left",
                        indicator=True,
                    )
                    .query('_merge == "left_only"')
                    .drop(columns=["_merge"])
                )
                if len(missing_facets) > 0:
                    bad = missing_facets.to_dict("records")
                    raise ValueError(
                        f"normalize_to={normalize_to!r} not found in facet group(s): {bad}"
                    )
                counts = _normalize_counts_per_facet(counts, norm_rows, facet_cols)
            else:
                norm_rows = counts[
                    (counts["category"] == normalize_to) & (counts["count"] > 0)
                ]
                if len(norm_rows) == 0:
                    raise ValueError(
                        f"normalize_to={normalize_to!r} not found in column. Type issue maybe?"
                    )
                counts = _normalize_counts(counts, norm_rows["count"].sum())

        colors = self._colors_as_list(cats_str, column=column, resolved_name=expr_name)
        color_values = {c: colors[i % len(colors)] for i, c in enumerate(cats_str)}
        legend_title = (
            self._cat_colors_title if self._cat_colors_title is not None else expr_name
        )

        p = (
            p9.ggplot(counts, p9.aes(x="category", y="count", fill="category"))
            + p9.geom_col()
            + p9.scale_fill_manual(
                values=color_values,
                name=legend_title,
                guide=None,
            )
        )
        if normalize_to is not None:
            p = (
                p
                + p9.geom_hline(yintercept=1.0, color="black", linetype="dashed")
                + p9.labs(y=f'Relative to "{normalize_to}"')
            )

        p = self._apply_facet_layer(p)

        if self._title_override is not _UNSET:
            p = p + p9.labs(title=self._title_override)
        else:
            p = p + p9.labs(title=expr_name)

        if self.fig_size is None:
            if self._is_faceted():
                n_col, n_row = self._facet_grid_dims(counts)
                fig_size = (6 * n_col, 5 * n_row)
            else:
                fig_size = (6, 5)
        else:
            fig_size = self.fig_size

        p = p + p9.theme_minimal(base_size=self.base_size)
        p = p + p9.theme(
            **self._theme_kwargs(
                figure_size=fig_size,
                panel_background=p9.element_rect(fill=self._bg_color, color=None),
                panel_border=p9.element_rect(
                    color=self._spine_color, size=0.5, fill=None
                ),
                panel_grid_major=p9.element_line(color="#E0E0E0", size=0.3),
                panel_grid_minor=p9.element_blank(),
                axis_text=p9.element_text(color=self._tick_color),
                axis_ticks_major_x=p9.element_line(color=self._tick_color, size=0.5),
                axis_ticks_major_y=p9.element_line(color=self._tick_color, size=0.5),
            )
        )

        p = self._register_fixed_panel(p)

        return p

    def _plot_histogram_numeric(
        self,
        data,
        expr: pd.Series,
        expr_name: str,
        stat_bin_args: dict,
        column=None,
    ) -> p9.ggplot:
        df = pd.DataFrame({"value": expr})

        facet_cols = []
        if self._facet_variable is not None:
            fv, _ = data.get_column(self._facet_variable)
            df["facet"] = pd.Categorical(fv.reindex(expr.index).astype(str))
            facet_cols.append("facet")
        if self._facet_row_variable is not None:
            rv, _ = data.get_column(self._facet_row_variable)
            df["facet_row"] = pd.Categorical(rv.reindex(expr.index).astype(str))
            facet_cols.append("facet_row")
        if self._facet_col_variable is not None:
            cv, _ = data.get_column(self._facet_col_variable)
            df["facet_col"] = pd.Categorical(cv.reindex(expr.index).astype(str))
            facet_cols.append("facet_col")

        color = self._colors_as_list(
            [expr_name], column=column, resolved_name=expr_name
        )[0]

        p = (
            p9.ggplot(df, p9.aes(x="value"))
            + p9.geom_histogram(fill=color, **stat_bin_args)
            + p9.labs(x=self._display_name(data, expr_name), y="count")
        )

        p = self._apply_facet_layer(p)

        if self._title_override is not _UNSET:
            p = p + p9.labs(title=self._title_override)
        else:
            p = p + p9.labs(title=self._display_name(data, expr_name))

        if self.fig_size is None:
            if self._is_faceted():
                n_col, n_row = self._facet_grid_dims(df)
                fig_size = (6 * n_col, 5 * n_row)
            else:
                fig_size = (6, 5)
        else:
            fig_size = self.fig_size

        p = p + p9.theme_minimal(base_size=self.base_size)
        p = p + p9.theme(
            **self._theme_kwargs(
                figure_size=fig_size,
                panel_background=p9.element_rect(fill=self._bg_color, color=None),
                panel_border=p9.element_rect(
                    color=self._spine_color, size=0.5, fill=None
                ),
                panel_grid_major=p9.element_line(color="#E0E0E0", size=0.3),
                panel_grid_minor=p9.element_blank(),
                axis_text=p9.element_text(color=self._tick_color),
                axis_ticks_major_x=p9.element_line(color=self._tick_color, size=0.5),
                axis_ticks_major_y=p9.element_line(color=self._tick_color, size=0.5),
            )
        )

        p = self._register_fixed_panel(p)

        return p

    def plot_violin(
        self,
        column: str,
        group_by: Optional[str] = None,
        additional_columns: Optional[list[str]] = None,
    ) -> p9.ggplot:
        """Build violin plots for a numeric column, optionally grouped by a categorical.

        Args:
            column:   Numeric obs column or gene name (y-axis).
            group_by: Optional categorical column whose unique values define the
                      x-axis groups.  When ``None`` a single violin per facet
                      panel is shown, labelled by *column*.

        Honours the plotter's faceting (:meth:`facet` / :meth:`facet_2d`),
        :meth:`title`, :meth:`panel_size`, and :meth:`theme` configuration.

        The discrete fill palette is taken from :meth:`colormap_discrete`.

        Raises:
            RuntimeError: if no data source has been set.
            ValueError:   if *column* is not numeric, or if *group_by* is numeric.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_violin()")

        data = self._data
        expr, expr_name = data.get_column(column)

        is_numerical = (
            expr.dtype != "object" and expr.dtype != "category" and expr.dtype != "bool"
        )
        if not is_numerical:
            raise ValueError(
                f"plot_violin() is for numeric columns; {column!r} is not numeric"
            )

        df = pd.DataFrame({"value": expr})
        if additional_columns is not None:
            for column in additional_columns:
                col_data, _ = data.get_column(column)
                df[column] = col_data

        if group_by is not None:
            grp, grp_name = data.get_column(group_by)
            grp_is_num = (
                grp.dtype != "object"
                and grp.dtype != "category"
                and grp.dtype != "bool"
            )
            if grp_is_num:
                raise ValueError(
                    f"group_by={group_by!r} must be categorical, not numeric"
                )
            if grp.dtype == "category":
                cats = list(grp.cat.categories)
            else:
                cats = natsorted([c for c in grp.unique() if not pd.isna(c)])
            cats_str = [str(c) for c in cats]
            df["group"] = pd.Categorical(grp.astype(str), categories=cats_str)
        else:
            cats_str = [expr_name]
            df["group"] = pd.Categorical([expr_name] * len(df), categories=cats_str)

        facet_cols = []
        if self._facet_variable is not None:
            fv, _ = data.get_column(self._facet_variable)
            df["facet"] = pd.Categorical(fv.astype(str))
            facet_cols.append("facet")
        if self._facet_row_variable is not None:
            rv, _ = data.get_column(self._facet_row_variable)
            df["facet_row"] = pd.Categorical(rv.astype(str))
            facet_cols.append("facet_row")
        if self._facet_col_variable is not None:
            cv, _ = data.get_column(self._facet_col_variable)
            df["facet_col"] = pd.Categorical(cv.astype(str))
            facet_cols.append("facet_col")

        colors = self._colors_as_list(
            cats_str,
            column=group_by if group_by is not None else column,
            resolved_name=grp_name if group_by is not None else expr_name,
        )
        color_values = {c: colors[i % len(colors)] for i, c in enumerate(cats_str)}
        legend_title = (
            self._cat_colors_title
            if self._cat_colors_title is not None
            else (group_by if group_by is not None else expr_name)
        )

        p = (
            p9.ggplot(df, p9.aes(x="group", y="value", fill="group"))
            + p9.geom_violin()
            + p9.scale_fill_manual(
                values=color_values,
                name=legend_title,
                guide=None if group_by is None else p9.guide_legend(),
            )
            + p9.labs(
                x=group_by if group_by is not None else "",
                y=self._display_name(data, expr_name),
            )
        )

        p = self._apply_facet_layer(p)

        if self._title_override is not _UNSET:
            p = p + p9.labs(title=self._title_override)
        else:
            p = p + p9.labs(title=self._display_name(data, expr_name))

        if self.fig_size is None:
            if self._is_faceted():
                n_col, n_row = self._facet_grid_dims(df)
                fig_size = (6 * n_col, 5 * n_row)
            else:
                fig_size = (6, 5)
        else:
            fig_size = self.fig_size

        p = p + p9.theme_minimal(base_size=self.base_size)
        p = p + p9.theme(
            **self._theme_kwargs(
                figure_size=fig_size,
                panel_background=p9.element_rect(fill=self._bg_color, color=None),
                panel_border=p9.element_rect(
                    color=self._spine_color, size=0.5, fill=None
                ),
                panel_grid_major_x=p9.element_blank(),
                panel_grid_minor=p9.element_blank(),
                panel_grid_major_y=p9.element_line(color="#E0E0E0", size=0.3),
                axis_text=p9.element_text(color=self._tick_color),
                axis_ticks_major_y=p9.element_line(color=self._tick_color, size=0.5),
                axis_ticks_major_x=p9.element_blank(),
            )
        )

        p = self._register_fixed_panel(p)

        return p

    def plot_ridgeline(
        self,
        column: str,
        group_by: str,
        *,
        bw: Optional[float] = None,
        trim: bool = True,
        alpha: float = 0.9,
        row_height: float = 0.5,
        scales: str = "free_y",
    ) -> p9.ggplot:
        """Build a compact, one-row-per-category density plot.

        Args:
            column:     Numeric obs column or gene name (x-axis).
            group_by:   Categorical obs column; each value becomes its own
                        stacked row, labelled on the right.
            bw:         Bandwidth for the density estimate (``geom_density``'s
                        ``bw``); ``None`` uses the plotnine default.
            trim:       Trim each density curve to its group's observed range.
            alpha:      Fill transparency for the density curves.
            row_height: Height (inches) of each stacked row.
            scales:     Passed to ``facet_grid`` — ``"free_y"`` (default) lets
                        each row's peak fill its own panel height (shared
                        across facet columns within a row); ``"fixed"``
                        shares one density scale across all rows and columns.

        The discrete fill palette is taken from :meth:`colormap_discrete`.

        *group_by* always drives the stacked rows, so :meth:`facet_2d` (which
        would also claim the row axis) is not supported. :meth:`facet` *is*
        supported — it adds a second grid dimension, one column per value of
        the faceting variable (``facet_grid(group_by ~ facet)`` rather than
        its usual ``facet_wrap``), so ``n_col``/``dir``/etc. from
        :meth:`facet` are ignored here.

        Raises:
            RuntimeError: if no data source has been set.
            ValueError:   if *column* is not numeric, *group_by* is numeric,
                          or :meth:`facet_2d` is configured.
        """
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_ridgeline()")
        if self._facet_row_variable is not None or self._facet_col_variable is not None:
            raise ValueError(
                "plot_ridgeline() does not support .facet_2d() — group_by "
                "already drives the stacked rows; use .facet(variable) to "
                "add a column facet instead"
            )

        data = self._data
        expr, expr_name = data.get_column(column)

        is_numerical = (
            expr.dtype != "object" and expr.dtype != "category" and expr.dtype != "bool"
        )
        if not is_numerical:
            raise ValueError(
                f"plot_ridgeline() is for numeric columns; {column!r} is not numeric"
            )

        grp, grp_name = data.get_column(group_by)
        grp_is_num = (
            grp.dtype != "object" and grp.dtype != "category" and grp.dtype != "bool"
        )
        if grp_is_num:
            raise ValueError(f"group_by={group_by!r} must be categorical, not numeric")

        if grp.dtype == "category":
            cats = list(grp.cat.categories)
        else:
            cats = natsorted([c for c in grp.unique() if not pd.isna(c)])
        cats_str = [str(c) for c in cats]

        df = pd.DataFrame({"value": expr, "group": grp.reindex(expr.index).astype(str)})

        has_col_facet = self._facet_variable is not None
        if has_col_facet:
            fv, _ = data.get_column(self._facet_variable)
            if fv.dtype == "category":
                facet_cats = list(fv.cat.categories)
            else:
                facet_cats = natsorted([c for c in fv.dropna().unique()])
            facet_cats_str = [str(c) for c in facet_cats]
            df["facet_col"] = fv.reindex(expr.index).astype(str)

        df = df.dropna(subset=["value"])
        df["group"] = pd.Categorical(df["group"], categories=cats_str)
        if has_col_facet:
            df["facet_col"] = pd.Categorical(df["facet_col"], categories=facet_cats_str)

        colors = self._colors_as_list(cats_str, column=group_by, resolved_name=grp_name)
        color_values = {c: colors[i % len(colors)] for i, c in enumerate(cats_str)}
        legend_title = (
            self._cat_colors_title if self._cat_colors_title is not None else grp_name
        )

        density_kwargs = dict(alpha=alpha, color="#333333", size=0.3, trim=trim)
        if bw is not None:
            density_kwargs["bw"] = bw

        p = (
            p9.ggplot(df, p9.aes(x="value", fill="group"))
            + p9.geom_density(**density_kwargs)
            # No legend: the row labels already name each group; guide=None keeps
            # this scale purely for color assignment (name= is otherwise inert).
            + p9.scale_fill_manual(values=color_values, name=legend_title, guide=None)
            + p9.facet_grid(
                "group ~ facet_col" if has_col_facet else "group ~ .", scales=scales
            )
            + p9.labs(x=self._display_name(data, expr_name), y="Density")
        )

        if self._title_override is not _UNSET:
            p = p + p9.labs(title=self._title_override)
        else:
            p = p + p9.labs(title=self._display_name(data, expr_name))

        if self.fig_size is None:
            n_col = len(facet_cats_str) if has_col_facet else 1
            fig_size = (6 * n_col, row_height * len(cats_str) + 1.2)
        else:
            fig_size = self.fig_size

        p = p + p9.theme_minimal(base_size=self.base_size)
        p = p + p9.theme(
            # The row labels are deliberately horizontal here.  Nothing rotates
            # them back: this plot builds its own facet_grid and rejects
            # .facet_2d(), so _facet_theme_overwrites() is always empty.
            **self._theme_kwargs(
                figure_size=fig_size,
                panel_background=p9.element_rect(fill=self._bg_color, color=None),
                panel_border=p9.element_blank(),
                axis_line_x=p9.element_line(color=self._spine_color, size=0.5),
                axis_line_y=p9.element_line(color=self._spine_color, size=0.5),
                panel_spacing_y=0.006,
                panel_grid_major_x=p9.element_line(
                    color="#D8D8D8", size=0.3, linetype="dotted"
                ),
                panel_grid_major_y=p9.element_blank(),
                panel_grid_minor=p9.element_blank(),
                strip_background=p9.element_blank(),
                strip_text_y=p9.element_text(
                    angle=0, ha="left", color=self._tick_color
                ),
                axis_text_y=p9.element_blank(),
                axis_ticks_major_y=p9.element_blank(),
                axis_text_x=p9.element_text(color=self._tick_color),
                axis_ticks_major_x=p9.element_line(color=self._tick_color, size=0.5),
                plot_margin_right=0.06,
            )
        )

        p = self._register_fixed_panel(p)

        return p

    def plot_embedding_color(
        self,
        reference_embedding,
        *,
        corner_colors=_EMBEDDING_COLOR_DEFAULTS,
        gradient_region=None,
        cell_filter=None,
        outside_color: str = "#C0C0C0",
        show_legend: bool = False,
        show_gradient_region: bool = False,
        dot_size: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> p9.ggplot:
        """Plot cells in the current embedding colored by 2D position in another embedding.

        Each cell receives a color from a bilinear gradient defined by four corner
        colors at its normalized (x, y) position in the reference embedding.  This
        lets you see how the layout of one embedding corresponds to another.

        Args:
            reference_embedding: Embedding name (str) or EmbeddingData for color assignment.
            corner_colors:        4-tuple ``(top_left, top_right, bottom_left, bottom_right)``.
                                  Default: red / blue / yellow / green.
            gradient_region:      Optional ``(corner1, corner2)`` restricting which grid cells
                                  partake in the gradient.  Each corner is a grid label string
                                  (e.g. ``"A1"``) or an ``(x, y)`` float tuple in reference-
                                  embedding coordinates.  ``corner1`` is the top-left (>=)
                                  and ``corner2`` the bottom-right (<=), matching the
                                  ``focus_on`` region convention.  Cells outside the box get
                                  *outside_color*.
            cell_filter: Optional, restricting which cells get colored (others get
                                  *outside_color*).  Accepts the same spec grammar as
                                  ``set_filter``: a region ``(corner1, corner2)``, a list of
                                  such regions (a cell is colored if it falls inside *any* of
                                  them), or a callable ``fn(reference_data) -> bool mask``.
                                  Resolved in the reference embedding.
            outside_color:        Color for cells outside *gradient_region* (default ``"#C0C0C0"``).
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
            gradient_region=gradient_region,
            outside_color=outside_color,
            cell_filter=cell_filter,
        )
        if random_seed is not None:
            df = df.sample(frac=1.0, random_state=random_seed)

        # Draw highlighted cells above the gray outside_color background,
        # regardless of row/shuffle order (stable sort preserves relative
        # order within each group).
        if (df["color"] == outside_color).any():
            order = np.argsort(df["color"].values != outside_color, kind="stable")
            df = df.iloc[order]

        dot = dot_size if dot_size is not None else self._dot_size

        p = p9.ggplot(df, p9.aes("x", "y", color="color"))

        # Grid lines first so they render behind all other layers
        if self._grid_config is not None:
            p = self._add_grid_layers(p)

        # Boundary layer (behind scatter)
        if self._layer_borders and self._border_config is not None:
            p = self._add_border_layers(p)

        # Main scatter — identity scale reads hex color strings directly
        if self._layer_data:
            p = p + p9.geom_point(size=dot, alpha=self._dot_alpha)

        p = p + p9.scale_color_identity(guide=None)

        # Region outline overlay (in reference-embedding coordinates)
        if show_gradient_region and gradient_region is not None:
            from .transforms import _corner_to_bounds
            import numpy as _np

            if len(gradient_region) == 2:
                if isinstance(gradient_region[0], str) or isinstance(
                    gradient_region[1], str
                ):
                    c1 = _corner_to_bounds(gradient_region[0], ref_data)
                    c2 = _corner_to_bounds(gradient_region[1], ref_data)
                    xlo = min(c1[0], c1[1], c2[0], c2[1])
                    xhi = max(c1[0], c1[1], c2[0], c2[1])
                    ylo = min(c1[2], c1[3], c2[2], c2[3])
                    yhi = max(c1[2], c1[3], c2[2], c2[3])
                else:
                    (x0, y0), (x1, y1) = gradient_region
                    xlo, xhi = min(x0, x1), max(x0, x1)
                    ylo, yhi = min(y0, y1), max(y0, y1)
                # polygon order: tl → tr → br → bl
                corners = [(xlo, yhi), (xhi, yhi), (xhi, ylo), (xlo, ylo)]
            else:
                pts4 = sorted(
                    [_np.array(c, dtype=float) for c in gradient_region],
                    key=lambda p: -p[1],
                )
                top_two = sorted(pts4[:2], key=lambda pt: pt[0])
                bot_two = sorted(pts4[2:], key=lambda pt: pt[0])
                tl_, tr_ = top_two[0], top_two[1]
                bl_, br_ = bot_two[0], bot_two[1]
                corners = [tl_, tr_, br_, bl_]
            # closed polygon + corner markers
            cx = [c[0] for c in corners] + [corners[0][0]]
            cy = [c[1] for c in corners] + [corners[0][1]]
            gradient_region_path_df = pd.DataFrame({"x": cx, "y": cy})
            gradient_region_pts_df = pd.DataFrame(
                {"x": [c[0] for c in corners], "y": [c[1] for c in corners]}
            )
            p = p + p9.geom_path(
                data=gradient_region_path_df,
                mapping=p9.aes(x="x", y="y"),
                color="#000000",
                size=0.6,
                inherit_aes=False,
            )
            p = p + p9.geom_point(
                data=gradient_region_pts_df,
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
        p = self._apply_embedding_theme(p)
        p = p + p9.theme(**self._theme_kwargs(figure_size=(6, 5)))

        # Grid axis ticks
        p = self._apply_axis_ticks(p, grid_layers=False)

        # Fixed panel size
        p = self._register_fixed_panel(p)

        # 2D colour legend (to the right of the figure)
        if show_legend:
            _cfg = {
                "corner_colors": corner_colors,
                "ref_name": ref_name,
                "base_size": self.base_size,
            }
            p = _ensure_post_draw(p)
            p._post_draw_fns.append(
                lambda fig, _c=_cfg: _draw_embedding_color_legend(fig, **_c)
            )

        # Embedding label — runs after legend so tight-bbox is stable
        p = self._maybe_add_embedding_label(p, data)

        return p

    # ── internals ────────────────────────────────────────────────────────────

    def _register_fixed_panel(self, p: p9.ggplot) -> p9.ggplot:
        """Register a post-draw hook pinning the panel to ``_fixed_panel_size``.

        No-op when no fixed size is configured.
        """
        if self._fixed_panel_size is not None:
            w, h = self._fixed_panel_size
            p = _ensure_post_draw(p)
            p._post_draw_fns.append(
                lambda fig, _w=w, _h=h: _apply_fixed_panel(fig, _w, _h)
            )
        return p

    def _apply_embedding_theme(self, p: p9.ggplot) -> p9.ggplot:
        """Apply the shared embedding ``embedding_theme`` (spines / bg / colours).

        The per-plot ``p9.theme(figure_size=...)`` overrides stay at the call
        site since they differ across plot types.
        """
        return p + embedding_theme(
            base_size=self.base_size,
            show_spines=self._panel_border,
            bg_color=self._bg_color,
            spine_color=self._spine_color,
        )

    def _apply_axis_ticks(self, p: p9.ggplot, *, grid_layers: bool) -> p9.ggplot:
        """Apply grid / plain axis-tick labels after the theme.

        Two faithful variants:

        * ``grid_layers=False`` (scatter / embedding-colour): the grid layers are
          already added inside the builders, so only the tick labels are applied,
          gated on ``GridConfig.coords``.
        * ``grid_layers=True`` (density / moran): the grid overlay layers are
          added here as well, and tick labels follow unconditionally.
        """
        if grid_layers:
            if self._grid_config is not None:
                p = self._add_grid_layers(p)
                p = self._add_grid_axis_ticks(p)
            else:
                p = self._add_plain_axis_ticks(p)
        else:
            if self._grid_config is not None and self._grid_config.coords:
                p = self._add_grid_axis_ticks(p)
            elif self._grid_config is None:
                p = self._add_plain_axis_ticks(p)
        return p

    def _maybe_add_embedding_label(
        self, p: p9.ggplot, data: "EmbeddingData"
    ) -> p9.ggplot:
        """Add the embedding-name label when enabled. Always the final hook."""
        if self._embedding_label:
            p = self._add_embedding_label(p, data)
        return p

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
            lambda fig, _l=label, _fs=fontsize: _draw_embedding_label(
                fig, label=_l, fontsize=_fs
            )
        )
        return p

    def _palette_for(self, column=None, resolved_name: Optional[str] = None):
        """Return the discrete palette (list | dict | None) for a column.

        With a per-column mapping set (see :meth:`colormap_discrete`), look up
        the column spec as it was passed to the plot method (``(source, column)``
        tuples included), then the name it resolved to, then the ``""`` fallback
        entry.  Without such a mapping, the single flat palette applies to every
        column.
        """
        if self._cat_colors_by_column is None:
            return self._cat_colors
        for candidate in (column, resolved_name):
            if candidate is None:
                continue
            key = _palette_key(candidate)
            if key in self._cat_colors_by_column:
                return self._cat_colors_by_column[key]
        return self._cat_colors_by_column.get("")

    def _colors_as_list(
        self, cats: list, column=None, resolved_name: Optional[str] = None
    ) -> list:
        """Return an ordered color list for *cats* of a column.

        Handles both list (positional cycling) and dict (name → color) palette
        forms, picking the palette for the column when a per-column mapping is
        configured.  *column* is the spec as passed to the plot method,
        *resolved_name* what ``get_column`` resolved it to.
        """
        palette = self._palette_for(column, resolved_name)
        label = column if column is not None else resolved_name
        where = f" for column {label!r}" if label is not None else ""
        if isinstance(palette, dict):
            # Normalize keys to str so {True: 'red'} and {'True': 'red'} both work
            normalized = {str(k): v for k, v in palette.items()}
            missing = sorted([str(c) for c in cats if str(c) not in normalized])
            if missing:
                raise ValueError(
                    f"not enough colors: dict{where} is missing entries for: {missing}. Available {palette.keys()}"
                )
            return [normalized[str(c)] for c in cats]
        colors = palette or DEFAULT_COLORS_CATEGORIES.get(
            len(cats), DEFAULT_COLORS_CATEGORIES["any"]
        )
        if len(colors) < len(cats):
            raise ValueError(
                f"not enough colors: {len(colors)} provided{where} for "
                f"{len(cats)} categories"
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
        colors = self._resolve_border_colors()
        # colors are baked into the boundary df, so they are part of the cache key
        if (
            self._boundary_cache["df"] is None
            or self._boundary_cache.get("colors") != colors
        ):
            from .transforms import compute_boundaries

            bc = self._border_config
            boundary_data = self._data if bc.respect_filter else self._data.unfilter()
            self._boundary_cache["df"] = compute_boundaries(
                data=boundary_data,
                cell_type_column=self._cell_type_column,
                colors=colors,
                resolution=bc.resolution,
                blur=bc.blur,
                threshold=bc.threshold,
            )
            self._boundary_cache["colors"] = colors
        return self._boundary_cache["df"]

    def _display_name(self, data, expr_name: str) -> str:
        """Expand expr_name to 'alt_id (symbol)' when an alternative id column is set."""
        if data._alternative_id_column is not None:
            alt_id = data.alternative_id_for(expr_name)
            if alt_id is not None:
                return f"{alt_id} ({expr_name})"
        return expr_name

    def _border_categories(self) -> tuple[list, str]:
        """Ordered categories of the border cell type column, and its name."""
        bc = self._border_config
        data = self._data if bc.respect_filter else self._data.unfilter()
        cell_types, name = data.get_column(self._cell_type_column)
        cats = (
            list(cell_types.cat.categories)
            if hasattr(cell_types, "cat")
            else natsorted(cell_types.unique())
        )
        return cats, name

    def _resolve_border_colors(self) -> list:
        """Ordered border color list.

        Explicit ``with_borders(colors=...)`` wins.  Otherwise, if a per-column
        discrete palette is configured (:meth:`colormap_discrete`) and it has an
        entry for the cell type column (or a ``""`` fallback), the borders reuse
        it, so borders and dots agree on the color of a category.  Failing that,
        the separate ``DEFAULT_COLORS_BORDERS`` palette is used.
        """
        bc = self._border_config
        if bc.colors is not None:
            return list(bc.colors)
        if (
            self._cat_colors_by_column is not None
            and self._cell_type_column is not None
        ):
            cats, name = self._border_categories()
            if self._palette_for(self._cell_type_column, name) is not None:
                return list(
                    self._colors_as_list(
                        cats, column=self._cell_type_column, resolved_name=name
                    )
                )
        return list(DEFAULT_COLORS_BORDERS)

    def _border_cat_to_color(self) -> dict:
        """Return ordered {category: hex_color} mapping for the border palette."""
        cats, _ = self._border_categories()
        colors = self._resolve_border_colors()
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

        if bc.legend and self._cell_type_column is not None:
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
                    name=bc.legend_title
                    if bc.legend_title is not None
                    else self._cell_type_column,
                    guide=p9.guide_legend(
                        override_aes={
                            "alpha": bc.legend_dot_alpha,
                            "size": bc.legend_dot_size,
                        }
                    ),
                )
            )

        return p

    def _build_numerical(
        self,
        df: pd.DataFrame,
        expr_name: str,
        is_gene: bool = False,
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

        if self._anti_overplot_seed is not None:
            df_normal = df_normal.sample(
                frac=1.0, random_state=self._anti_overplot_seed
            )
        elif self._anti_overplot:
            df_normal = df_normal.sort_values(
                "expression", ascending=self._anti_overplot_ascending
            )

        df_normal["expression_plot"] = df_normal["expression"]

        p = p9.ggplot(df_normal, p9.aes("x", "y", color="expression_plot"))

        # Grid lines first so they render behind all other layers
        if self._grid_config is not None:
            p = self._add_grid_layers(p)

        # Boundary layer (behind scatter)
        if self._layer_borders and self._border_config is not None:
            p = self._add_border_layers(p)

        # Background layer (all cells, fixed colour, behind data)
        if self._background_enabled:
            p = self._add_background_layer(p, df)

        # Point layers, ordered low → high by expression so that the LAST layer
        # added (painted on top) reflects the chosen "on top" direction.
        # With ``ascending=True`` (default) the highest values end up on top, so
        # the clipped (>max_quantile) points are added last. With
        # ``ascending=False`` we want the *lowest* values on top, so the whole
        # stack is reversed — otherwise the clipped (highest) points would
        # stubbornly sit on top of everything, contradicting the requested
        # draw order. The seed (random) and enabled=False (dataset-order) modes
        # have no meaningful direction and keep the default low→high stack.
        point_layers = []
        if self._layer_zeros and len(df_zeros) > 0:
            point_layers.append(
                p9.geom_point(
                    data=df_zeros,
                    mapping=p9.aes("x", "y"),
                    color=self._zero_color,
                    size=self._zero_dot_size,
                    inherit_aes=False,
                )
            )
        if self._layer_data:
            point_layers.append(
                p9.geom_point(size=self._dot_size, alpha=self._dot_alpha)
            )
        if self._layer_data and len(df_above) > 0:
            point_layers.append(
                p9.geom_point(
                    data=df_above,
                    mapping=p9.aes("x", "y"),
                    color=self._upper_clip_color,
                    size=self._dot_size,
                    alpha=self._dot_alpha,
                    inherit_aes=False,
                )
            )

        if (
            self._anti_overplot
            and self._anti_overplot_seed is None
            and not self._anti_overplot_ascending
        ):
            point_layers.reverse()

        for layer in point_layers:
            p = p + layer

        # Color scale
        cmap_colors = self._get_cmap_colors()
        if is_gene:
            if len(expr_name) <= 15:
                default_cbar_name = expr_name + ": log2 expression"
            else:
                default_cbar_name = expr_name + ":\nlog2 expression"
        else:
            default_cbar_name = expr_name
        cbar_name = (
            self._cbar_title if self._cbar_title is not None else default_cbar_name
        )
        has_zeros = self._layer_zeros and len(df_zeros) > 0
        has_clips = len(df_above) > 0
        zero_val_str = "0" if abs(zero_val) < 1e-9 else f"{zero_val:.3g}"
        data_min = float(df["expression"].min())
        zero_label = (
            f"≤{zero_val_str}"
            if has_zeros and data_min < zero_val - 1e-9
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
                reverse=self._anti_overplot_ascending is False,
            ),
        )
        return p

    def _build_categorical(
        self,
        df: pd.DataFrame,
        expr_name: str,
        column=None,
    ) -> p9.ggplot:
        if df["expression"].dtype == "category":
            cats = list(df["expression"].cat.categories)
        else:
            cats = natsorted(df["expression"].unique())

        colors = self._colors_as_list(cats, column=column, resolved_name=expr_name)
        color_values = {str(c): colors[i % len(colors)] for i, c in enumerate(cats)}

        # Draw order: grouped by category (last category on top by default),
        # fully randomized with a seed, or left as original row order — see
        # anti_overplot().
        df = df.copy()
        if self._anti_overplot_seed is not None:
            df = df.sample(frac=1.0, random_state=self._anti_overplot_seed)
        elif self._anti_overplot:
            cat_order = {c: i for i, c in enumerate(cats)}
            df["_sort_key"] = df["expression"].map(cat_order)
            df = df.sort_values(
                "_sort_key", ascending=self._anti_overplot_ascending
            ).drop(columns=["_sort_key"])

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
        if self._layer_borders and self._border_config is not None:
            p = self._add_border_layers(p)

        # Background layer (all cells, fixed colour, behind data)
        if self._background_enabled:
            p = self._add_background_layer(p, df)

        # Scatter
        if self._layer_data:
            p = p + p9.geom_point(size=self._dot_size, alpha=self._dot_alpha)

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
                    alpha=self._dot_alpha,
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
            name=self._cat_colors_title
            if self._cat_colors_title is not None
            else expr_name,
            guide=p9.guide_legend(
                override_aes={
                    "size": self._legend_dot_size,
                    "shape": "o",
                    "alpha": self._legend_dot_alpha,
                },
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


def _normalize_counts(counts_df, factor):
    """Return a copy of *counts_df* with the ``count`` column divided by *factor*."""
    if factor is None or factor == 0:
        return counts_df
    result = counts_df.copy()
    result["count"] = result["count"] / factor
    return result


def _normalize_counts_per_facet(counts_df, norm_rows, facet_cols):
    """Divide each facet group's counts by that group's normalize_to count."""
    norm = norm_rows[facet_cols + ["count"]].rename(columns={"count": "_norm"})
    result = counts_df.merge(norm, on=facet_cols, how="left")
    result["count"] = result["count"] / result["_norm"]
    return result.drop(columns=["_norm"])
