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

    main_ax = fig.axes[0]
    pos = main_ax.get_position()  # Bbox in figure [0,1] coords

    # ── Layout: title | gap | bar (tick labels auto to the right) ────────────
    # The right margin (pos.x1 → 1.0) is already freed by suppressing the guide.
    title_half = 0.025   # half-width of title text area
    gap = 0.008
    bar_frac = 0.035     # colorbar bar width as fraction of figure

    bar_left = pos.x1 + 2 * title_half + gap

    # Safety: if tight, shrink main axes
    needed_right = bar_left + bar_frac + 0.01
    if needed_right > 0.99:
        shrink = needed_right - 0.99
        main_ax.set_position([pos.x0, pos.y0, pos.width - shrink, pos.height])
        pos = main_ax.get_position()
        bar_left = pos.x1 + 2 * title_half + gap

    cbar_ax = fig.add_axes([bar_left, pos.y0, bar_frac, pos.height])

    cb = mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap=cmap,
        norm=norm,
        extend=extend,
        extendrect=True,
        extendfrac=extendfrac,
        orientation="vertical",
        ticks=breaks,
    )
    cb.set_ticklabels(labels)
    cb.ax.tick_params(labelsize=9, length=3)
    cb.ax.yaxis.set_tick_params(which="both", labelleft=False, labelright=True)

    # ── Extension box labels ──────────────────────────────────────────────────
    # With extendfrac=f, each extension occupies f/(1+n_ext*f) of the total
    # axes height, where n_ext is number of extensions (1 or 2).
    n_ext = (1 if has_zeros else 0) + (1 if has_clips else 0)
    denom = 1 + n_ext * extendfrac
    ext_box_frac = extendfrac / denom  # each extension box as fraction of axes height

    # Whether both or only one extension is present changes which end is which.
    # With extend='both': bottom=min, top=max.
    # With extend='min': bottom only. With extend='max': top only.
    if has_zeros:
        y_zero = ext_box_frac / 2  # centre of bottom box in transAxes
        cb.ax.text(
            1.08, y_zero, "0",
            transform=cb.ax.transAxes,
            va="center", ha="left", fontsize=9, clip_on=False,
        )
    if has_clips:
        y_clip = 1.0 - ext_box_frac / 2  # centre of top box in transAxes
        cb.ax.text(
            1.08, y_clip, f">{labels[-1]}",
            transform=cb.ax.transAxes,
            va="center", ha="left", fontsize=9, clip_on=False,
        )

    # ── Title: vertical text to the LEFT of the bar ──────────────────────────
    title_cx = pos.x1 + title_half
    title_cy = pos.y0 + pos.height * 0.5
    fig.text(
        title_cx, title_cy, cbar_title,
        rotation=90, va="center", ha="center", fontsize=9,
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

    def __init__(self, ad_or_data=None, embedding: str = "umap"):
        self._data: Optional[EmbeddingData] = None
        self._cell_type_column: Optional[str] = None

        # dot appearance
        self._dot_size: float = 1
        self._show_spines: bool = True
        self._bg_color: str = "#FFFFFF"
        self._anti_overplot: bool = True
        self._flip_order: bool = False
        self._categorical_outliers: bool = True
        self._categorical_outlier_quantile: float = 0.95

        # colormap config (numerical)
        self._cmap = None  # None → default blue-magenta, or list of color strings
        self._max_quantile: float = 0.95
        self._upper_clip_color: str = "#FF0000"
        self._cbar_title: Optional[str] = None  # None → auto from gene name

        # zero handling
        self._zeros_behind: bool = True
        self._zero_color: str = "#D0D0D0"
        self._zero_dot_size: float = 5
        self._zero_value: float = 0.0

        # categorical colormap
        self._cat_colors: Optional[list] = None  # None → DEFAULT_COLORS

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
            grid_size = (
                self._data._grid_size if self._data is not None else 12
            )
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
        spines: Optional[bool] = None,
        bg_color: Optional[str] = None,
    ) -> "ScatterPlotter":
        """Configure visual appearance. Only supplied arguments are changed.

        Args:
            dot_size: Point size for the main scatter layer.
            spines:   Show/hide the panel border (True = show).
            bg_color: Background hex color (e.g. ``"#FFFFFF"``).
        """
        new = copy.copy(self)
        if dot_size is not None:
            new._dot_size = dot_size
        if spines is not None:
            new._show_spines = spines
        if bg_color is not None:
            new._bg_color = bg_color
        return new

    def flip_draw_order(self, value: bool = True) -> "ScatterPlotter":
        """Reverse categorical draw order (last category drawn on top when False)."""
        new = copy.copy(self)
        new._flip_order = value
        return new

    def outlier_replot(self, enabled: bool = True) -> "ScatterPlotter":
        """Enable or disable the categorical outlier replot pass (default: enabled)."""
        new = copy.copy(self)
        new._categorical_outliers = enabled
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
        behind: bool = True,
        color: str = "#D0D0D0",
        dot_size: Optional[float] = None,
    ) -> "ScatterPlotter":
        """Configure zero-value rendering."""
        new = copy.copy(self)
        new._zeros_behind = behind
        new._zero_color = color
        if dot_size is not None:
            new._zero_dot_size = dot_size
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

    # ── grid overlay ─────────────────────────────────────────────────────────

    def with_grid(
        self,
        *,
        labels: bool = False,
        coords: bool = False,
        vertical_letters: bool = False,
        grid_size: int = 12,
        color: str = "#777777",
    ) -> "ScatterPlotter":
        """
        labels: show "A1", "B2" text labels inside each grid cell
        coords: use grid coords as axis tick labels
        """
        if grid_size > 26:
            raise ValueError("grid_size max is 26")
        new = copy.copy(self)
        new._grid_config = GridConfig(
            labels=labels,
            coords=coords,
            vertical_letters=vertical_letters,
            grid_size=grid_size,
            color=color,
        )
        # Sync EmbeddingData grid settings if needed
        if new._data is not None and (
            new._data._grid_size != grid_size
            or new._data._grid_letters_on_vertical != vertical_letters
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
                grid_size=grid_size,
                grid_letters_on_vertical=vertical_letters,
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
            show_spines=self._show_spines, bg_color=self._bg_color
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

    def plot_density(self, bins: int = 200) -> p9.ggplot:
        """Build a 2D cell-density heatmap."""
        if self._data is None:
            raise RuntimeError("call .set_source() before .plot_density()")

        from .transforms import prepare_density_df

        df = prepare_density_df(self._data, bins=bins)
        coords = self._data.coordinates()
        clip_val = float(df["density"].quantile(0.99))

        cmap_colors = ["#BFBFFF", "#0000FF"]
        breaks = list(np.linspace(1, clip_val, 5))
        labels = [f"{b:.2f}" for b in breaks[:-1]] + [f">{clip_val:.2f}"]

        p = (
            p9.ggplot(df, p9.aes("x", "y", fill="density"))
            + p9.geom_tile(p9.aes(width="x_width", height="y_width"))
            + p9.scale_fill_gradientn(
                colors=cmap_colors,
                limits=(1, clip_val),
                breaks=breaks,
                labels=labels,
                na_value="#FFFFFF",
                name="density",
            )
        )

        # Boundary overlay
        if self._border_config is not None and self._cell_type_column is not None:
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
            show_spines=self._show_spines, bg_color=self._bg_color
        )
        p = p + p9.theme(figure_size=(6, 5))
        return p

    def plot_grid_histogram(
        self, column: str, min_cell_count: int = 10
    ) -> p9.ggplot:
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
        df_zeros = df[df["expression"] == zero_val]
        df_nonzero = df[df["expression"] != zero_val].copy()

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
        if self._border_config is not None and self._cell_type_column is not None:
            p = self._add_border_layers(p)

        # Zero underlay
        if self._zeros_behind and len(df_zeros) > 0:
            p = p + p9.geom_point(
                data=df_zeros,
                mapping=p9.aes("x", "y"),
                color=self._zero_color,
                size=self._zero_dot_size,
                inherit_aes=False,
            )

        # Main scatter (gradient range)
        p = p + p9.geom_point(size=self._dot_size)

        # Clipped values drawn on top in clip color
        if len(df_above) > 0:
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
        df = df.sort_values(
            "_sort_key", ascending=not self._flip_order
        ).drop(columns=["_sort_key"])

        p = p9.ggplot(df, p9.aes("x", "y", color="expression"))

        # Grid lines first so they render behind all other layers
        if self._grid_config is not None:
            p = self._add_grid_layers(p)

        # Boundary layer (behind scatter)
        if self._border_config is not None and self._cell_type_column is not None:
            p = self._add_border_layers(p)

        # Scatter
        p = p + p9.geom_point(size=self._dot_size)

        # Outlier replot
        if self._categorical_outliers:
            outlier_dfs = []
            for cat in cats:
                sdf = df[df["expression"] == cat]
                if len(sdf) == 0:
                    continue
                center_x, center_y = sdf["x"].mean(), sdf["y"].mean()
                dist = np.sqrt(
                    (sdf["x"] - center_x) ** 2 + (sdf["y"] - center_y) ** 2
                )
                thresh = dist.quantile(self._categorical_outlier_quantile)
                outlier_dfs.append(sdf[dist > thresh])
            if outlier_dfs:
                df_outliers = pd.concat(outlier_dfs)
                p = p + p9.geom_point(
                    data=df_outliers,
                    size=self._dot_size,
                    inherit_aes=True,
                )

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
                color=gc.color,
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
                axis_text_x=p9.element_text(size=10),
                axis_text_y=p9.element_text(size=10),
                axis_ticks_major_x=p9.element_line(color="#555555", size=0.5),
                axis_ticks_major_y=p9.element_line(color="#555555", size=0.5),
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
