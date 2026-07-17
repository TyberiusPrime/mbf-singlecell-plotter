"""Declarative, pipegraph-driven plotting.

Describe the plots you want as :class:`Plot` / :class:`PlotDensity` dataclasses,
collect them on a :class:`PlotBuilder` (which carries the shared data sources and
styling defaults), then turn the whole batch into pypipegraph2 jobs in one call::

    builder = PlotBuilder(
        base_h5ad="analysis.h5ad",
        additional_h5ads={"genes": "genes.h5ad"},
        column_sources={"my_score": compute_my_score},
        column_colors={"leiden": ["#1f77b4", "#ff7f0e", ...]},
    )
    builder.add_plot(Plot(column="S100A8"))
    builder.add_plot(Plot(column="leiden", do_grid_histogram=False))
    builder.add_plot(PlotDensity())
    jobs = builder.register_all(result_dir)

The builder resolves data sources, faceting, filters and colouring; each plot may
override any of the builder-level source defaults (``embedding``, ``transform``,
``layer``, ``alternative_id_column``, ``base_size``, ``panel_size``).
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas
import plotnine as p9
import pypipegraph2 as ppg

from .data import EmbeddingData
from .plots import ScatterPlotter

# A qualified type alias for the "colour by this" argument.  Either an obs/gene
# column name, or a ``(source_name, column)`` tuple selecting an alternative
# source.  When a tuple is used the ``filename`` must be given explicitly.
Column = Union[str, Tuple[str, str]]

# Default palette for :class:`PlotDensity` heatmaps (light grey → warm red).
DEFAULT_DENSITY_COLORS = [
    "#eFeFeF",
    "#ECDA9A",
    "#EFC47E",
    "#F3AD6A",
    "#F7945D",
    "#F97B57",
    "#F66356",
    "#EE4D5A",
]


def _default_transform(x):
    """Convert natural-log expression to log2 (the historical default)."""
    return x / np.log(2)


def _as_list(value) -> list:
    """Normalise ``str | list | None`` into a (possibly empty) list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


# ── plot descriptions ────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class _PlotBase:
    """Fields and helpers shared by every plot description.

    Declared keyword-only so subclasses can keep their own required/positional
    fields (e.g. ``Plot.column``) first in the constructor signature.

    Any of the source-override fields left as ``None`` fall back to the
    corresponding default on the :class:`PlotBuilder`.
    """

    # what to plot on (subset of the data)
    filter: Optional[Callable[["EmbeddingData"], "np.ndarray"]] = None
    hard_filter: Optional[Callable[["EmbeddingData"], "np.ndarray"]] = None

    # where to store the outputs
    filename: Optional[str] = None  # override the output filename (stem)
    subfolder: Optional[str] = None  # extra sub-directory, e.g. 'genes'

    # layout / style
    facet: Optional[Column] = None  # split into panels; names the sub-directory
    facet_args: Optional[dict] = None  # extra args to facet()/facet_2d()
    style: Optional[dict] = None  # extra style, composed on top of dot_size=1
    grey_border: Optional[bool] = None  # force the grey cell border on/off
    title: Optional[Union[str, Callable[[str], str]]] = None
    dpi: int = 150

    # source overrides (None → inherit from the PlotBuilder)
    embedding: Optional[str] = None
    transform: Optional[Callable[["np.ndarray"], "np.ndarray"]] = None
    layer: Optional[str] = None
    alternative_id_column: Optional[str] = None
    base_size: Optional[float] = None
    panel_size: Optional[Tuple[float, float]] = None

    def facet_name(self) -> str:
        assert self.facet is not None
        if isinstance(self.facet, str):
            return self.facet
        return f"{self.facet[0]}_vs_{self.facet[1]}"

    def get_facet_args(self) -> dict:
        return self.facet_args or {}

    def do_border(self) -> bool:
        if self.grey_border is not None:
            return self.grey_border
        return self.facet is not None

    def derived_columns_needed(self) -> List[str]:
        """Columns that must be resolvable for this plot (facet + border)."""
        res: List[str] = []
        if self.facet is not None:
            if isinstance(self.facet, str):
                res.append(self.facet)
            else:
                res.extend(self.facet)
        if self.do_border():
            res.append("constant")
        return res


@dataclass
class Plot(_PlotBase):
    """A scatter plot coloured by ``column`` (plus optional companion plots)."""

    column: Column = None  # what to colour by / plot; names the output file

    # which companion plots to also emit
    do_scatter: bool = True
    do_grid_histogram: bool = False  # per-grid-cell histogram
    do_global_histogram: bool = False  # one overall histogram
    do_global_relative_histogram: Optional[str] = None  # normalise_to column
    do_violin: Optional[Union[List[str], str]] = None  # violin per x-column
    do_facet_violin: Optional[List[Tuple[str, str]]] = None  # (x, facet) violins
    do_ridges: Optional[List[str]] = None  # ridgeline per split column

    # colouring / overplotting
    colors: Optional[Union[List[str], Dict[str, str]]] = None
    ascending: Optional[bool] = None
    anti_overplot_seed: Optional[int] = None

    def __post_init__(self):
        if self.column is None:
            raise ValueError("Plot.column is required")

    def derived_columns_needed(self) -> List[str]:
        res: List[str] = []
        res.append(self.column if isinstance(self.column, str) else self.column[1])
        res.extend(super().derived_columns_needed())
        res.extend(_as_list(self.do_violin))
        for x_column, facet_column in self.do_facet_violin or []:
            res.append(x_column)
            res.append(facet_column)
        return res


@dataclass
class PlotDensity(_PlotBase):
    """A 2D cell-density heatmap of the embedding (no colour-by column)."""

    bins: int = 12
    quantile: float = 0.95
    cmap_colors: List[str] = field(default_factory=lambda: list(DEFAULT_DENSITY_COLORS))
    include_counts: bool = True
    count_text_size: float = 7


# ── builder ──────────────────────────────────────────────────────────────────


class PlotBuilder:
    """Collects plot descriptions and turns them into pypipegraph2 jobs.

    Call :meth:`add_plot` as many times as needed, then :meth:`register_all`
    once to emit every job into the active pipegraph.
    """

    def __init__(
        self,
        base_h5ad: Union[str, Path | Tuple[Path, ppg.Job]],
        additional_h5ads: Optional[
            Dict[str, Union[str, Path, Tuple[Path, ppg.Job]]]
        ] = None,
        column_sources: Optional[
            Dict[str, Callable[[EmbeddingData], "pandas.Series"]]
        ] = None,
        column_colors: Optional[
            Dict[str | Tuple[str, str], Union[Dict[str, str], Sequence[str]]]
        ] = None,
        *,
        embedding: str = "umap",
        transform: Optional[
            Callable[["np.ndarray"], "np.ndarray"]
        ] = _default_transform,
        layer: str = "X",
        alternative_id_column: Optional[str] = "gene_name",
        base_size: float = 15,
        panel_size: Tuple[float, float] = (4, 4),
    ):
        self.base_h5ad, self.base_h5ad_deps = ppg.util.job_or_filename(base_h5ad)
        # Named .h5ad fallback sources (e.g. {"genes": Path(...)}).
        self.additional_h5ads = {
            name: ppg.util.job_or_filename(p)
            for name, p in (additional_h5ads or {}).items()
        }
        # Derived columns computed from a callable(EmbeddingData) -> Series.
        self.column_sources = dict(column_sources or {})
        # Discrete colour palettes keyed by column name.
        self.column_colors = dict(column_colors or {})

        # source / style defaults, individually overridable per Plot
        self.embedding = embedding
        self.transform = transform
        self.layer = layer
        self.alternative_id_column = alternative_id_column
        self.base_size = base_size
        self.panel_size = panel_size

        self.plots: List[_PlotBase] = []

        # dependency-job caches so shared sources/colours yield one invariant each
        self._build_invariant = None
        self._source_dep_cache: Dict[str, list] = {}
        self._color_dep_cache: dict = {}

    # -- public API -----------------------------------------------------------

    def add_plot(self, plot: _PlotBase) -> _PlotBase:
        """Queue a :class:`Plot` / :class:`PlotDensity` for later registration."""
        self.plots.append(plot)
        return plot

    def register_all(self, result_dir: Union[str, Path]) -> list:
        """Create every queued plot's job(s) under ``result_dir``. Returns them."""
        rd = Path(result_dir)
        jobs: list = []
        for plot in self.plots:
            jobs.extend(self._register(plot, rd))
        return jobs

    # -- plotter construction -------------------------------------------------

    def _resolve(self, plot: _PlotBase, name: str):
        """Per-plot override if set, else the builder-level default."""
        value = getattr(plot, name)
        return value if value is not None else getattr(self, name)

    def build_plotter(self, plot: _PlotBase) -> ScatterPlotter:
        """Assemble the configured :class:`ScatterPlotter` for ``plot``."""
        p = ScatterPlotter(base_size=self._resolve(plot, "base_size"))
        p = p.set_source(
            self.base_h5ad,
            self._resolve(plot, "embedding"),
            alternative_id_column=self._resolve(plot, "alternative_id_column"),
            transform=self._resolve(plot, "transform"),
            layer=self._resolve(plot, "layer"),
        )
        for name, (path, _deps) in self.additional_h5ads.items():
            p = p.add_alternative_source(path, name=name)
        for column in plot.derived_columns_needed():
            if column in self.column_sources:
                p = p.add_derived_source({column: self.column_sources[column]})
        if plot.filter:
            p = p.set_filter(plot.filter)
        if plot.hard_filter:
            p = p.hard_filter(plot.hard_filter)
        p = p.style(dot_size=1)
        if plot.style:
            p = p.style(**plot.style)

        if isinstance(plot, Plot):
            column = plot.column
            if isinstance(column, str) and "_" not in column:
                p = p.colormap(title="log2 expression")
            colors = plot.colors
            if colors is None:
                colors = self.column_colors.get(column)
            if colors is not None:
                p = p.colormap_discrete(colors, title=str(column))

        if plot.facet:
            if isinstance(plot.facet, str):
                p = p.facet(plot.facet, **plot.get_facet_args())
            else:
                p = p.facet_2d(*plot.facet, **plot.get_facet_args())
        if plot.do_border():
            # so we can turn it off or force it on, but by default it's on.
            p = p.with_borders(
                cell_type_column="constant", colors=["#707070"], legend=False, size=10
            )
        if isinstance(plot, Plot):
            if plot.anti_overplot_seed:
                p = p.anti_overplot(seed=plot.anti_overplot_seed)
            if plot.ascending is not None:
                p = p.anti_overplot(ascending=plot.ascending)
        if plot.title is not None:
            p = p.title(plot.title)

        p = p.panel_size(*self._resolve(plot, "panel_size"))
        return p

    # -- output layout --------------------------------------------------------

    def _out_dir(self, plot: _PlotBase, rd: Path) -> Path:
        base = rd / plot.facet_name() if plot.facet else rd
        return base / plot.subfolder if plot.subfolder else base

    def _filename_stem(self, plot: _PlotBase) -> str:
        if plot.filename:
            return plot.filename
        column = getattr(plot, "column", None)
        if isinstance(column, str):
            return column
        if isinstance(column, tuple):
            return column[1]
        return "density"

    # -- registration ---------------------------------------------------------

    def _register(self, plot: _PlotBase, rd: Path) -> list:
        if isinstance(plot, PlotDensity):
            return [self._register_density(plot, rd)]
        return self._register_plot(plot, rd)

    def _register_plot(self, plot: Plot, rd: Path) -> list:
        out_dir = self._out_dir(plot, rd)
        stem = self._filename_stem(plot)

        outputs: Dict[str, Path] = {}
        if plot.do_scatter:
            outputs["scatter"] = out_dir / f"{stem}.png"
        if plot.do_grid_histogram:
            outputs["histo"] = out_dir / f"{stem}_histogram.png"
        if plot.do_global_histogram:
            outputs["global_histo"] = out_dir / f"{stem}_overall_histogram.png"
        if plot.do_global_relative_histogram:
            outputs["global_histo_relative"] = (
                out_dir / f"{stem}_overall_histogram_relative.png"
            )
        for col in _as_list(plot.do_violin):
            outputs[f"violin_{col}"] = out_dir / f"{stem}_violin_{col}.png"
        for x, facet in plot.do_facet_violin or []:
            outputs[f"violin_{x}_facet_{facet}"] = (
                out_dir / f"{stem}_violin_{x}_facet_{facet}.png"
            )
        if not outputs:
            raise ValueError(f"Plot {stem!r} would produce no output files")

        def generate(
            output_filenames,
            plot=plot,
            column_colors=self.column_colors,
            build_plotter=self.build_plotter,
        ):
            for path in output_filenames.values():
                path.parent.mkdir(exist_ok=True, parents=True)
            p = build_plotter(plot)
            if "scatter" in output_filenames:
                p.plot(plot.column).save(output_filenames["scatter"], dpi=plot.dpi)
            if "histo" in output_filenames:
                p.plot_grid_histogram(plot.column, scale_by_count=True).save(
                    output_filenames["histo"]
                )
            if "global_histo" in output_filenames:
                p.plot_histogram(plot.column).save(output_filenames["global_histo"])
            if "global_histo_relative" in output_filenames:
                p.plot_histogram(
                    plot.column, normalize_to=plot.do_global_relative_histogram
                ).save(output_filenames["global_histo_relative"])
            for violin_x in _as_list(plot.do_violin):
                colors = column_colors.get(violin_x)
                # colors=None restores the default palette for this column
                pv = p.colormap_discrete(cmap_or_list_or_dict=colors, title=violin_x)
                po = pv.plot_violin(plot.column, violin_x)
                po += p9.theme(axis_text_x=p9.element_text(rotation=90))
                po.save(output_filenames[f"violin_{violin_x}"])
            for violin_x, facet_column in plot.do_facet_violin or []:
                colors = column_colors.get(violin_x)
                pv = p.colormap_discrete(cmap_or_list_or_dict=colors, title=violin_x)
                po = pv.plot_violin(plot.column, violin_x, [facet_column])
                po += p9.facet_wrap(facet_column, scales="free_x")
                po += p9.theme(axis_text_x=p9.element_text(rotation=90))
                po.save(output_filenames[f"violin_{violin_x}_facet_{facet_column}"])

        job = self._make_job(plot, outputs, generate)
        jobs = [job]

        if plot.do_ridges:
            jobs.append(self._register_ridges(plot, out_dir, stem))
        return jobs

    def _register_ridges(self, plot: Plot, out_dir: Path, stem: str):
        outputs = {
            split: out_dir / f"{stem}_{split}_ridge.png" for split in plot.do_ridges
        }
        # ridgelines split the data themselves, so drop any facet.
        ridge_plot = dataclasses.replace(plot, facet=None)

        def generate(output_filenames, plot=ridge_plot):
            p = self.build_plotter(plot)
            for split, out in output_filenames.items():
                out.parent.mkdir(exist_ok=True, parents=True)
                p.plot_ridgeline(plot.column, split, scales="fixed").save(out)

        return self._make_job(ridge_plot, outputs, generate, id_suffix="_ridges")

    def _register_density(self, plot: PlotDensity, rd: Path):
        out_dir = self._out_dir(plot, rd)
        stem = self._filename_stem(plot)
        outputs = {"scatter": out_dir / f"{stem}.png"}

        def generate(output_filenames, plot=plot):
            output_filenames["scatter"].parent.mkdir(exist_ok=True, parents=True)
            po = self.build_plotter(plot).plot_density(
                plot.bins,
                quantile=plot.quantile,
                cmap_colors=plot.cmap_colors,
                include_counts=plot.include_counts,
                count_text_size=plot.count_text_size,
            )
            po.save(output_filenames["scatter"], dpi=plot.dpi)

        return self._make_job(plot, outputs, generate)

    def _make_job(
        self, plot: _PlotBase, outputs: Dict[str, Path], generate, id_suffix=""
    ):
        job = ppg.MultiFileGeneratingJob(outputs, generate, depend_on_function=True)
        job.depends_on(self.base_h5ad_deps)
        for _path, deps in self.additional_h5ads.values():
            job.depends_on(deps)
        plot_id = str(sorted(outputs.values())[0]) + id_suffix
        job.depends_on(self._deps(plot, plot_id))
        return job

    # -- dependency invariants ------------------------------------------------

    def _deps(self, plot: _PlotBase, plot_id: str) -> list:
        if self._build_invariant is None:
            self._build_invariant = ppg.FunctionInvariant(
                "mbf_scp_build_plotter", PlotBuilder.build_plotter
            )
        res: list = [self._build_invariant]

        for column in plot.derived_columns_needed():
            if column in self.column_sources:
                res.extend(self._source_invariants(column))
            if column in self.column_colors:
                res.append(self._color_invariant(column))

        # everything on the plot that isn't a callable / handled above
        res.append(ppg.ParameterInvariant(plot_id + "_config", _param_config(plot)))
        for fld in ("filter", "hard_filter", "transform", "title"):
            value = getattr(plot, fld)
            if callable(value):
                res.append(ppg.FunctionInvariant(f"{plot_id}_{fld}", value))
        return res

    def _source_invariants(self, column: str) -> list:
        if column not in self._source_dep_cache:
            fn = self.column_sources[column]
            deps = [ppg.FunctionInvariant(f"mbf_scp_column_{column}", fn)]
            if hasattr(fn, "deps"):
                # a computed_column may carry its own upstream dependencies
                extra = fn.deps
                deps.extend(extra if isinstance(extra, (list, tuple)) else [extra])
            self._source_dep_cache[column] = deps
        return self._source_dep_cache[column]

    def _color_invariant(self, column: str):
        if column not in self._color_dep_cache:
            self._color_dep_cache[column] = ppg.ParameterInvariant(
                f"mbf_scp_color_{column}", self.column_colors[column]
            )
        return self._color_dep_cache[column]


def _param_config(plot: _PlotBase) -> dict:
    """Serialisable snapshot of a plot's non-callable fields for invariants."""
    out = {"__type__": type(plot).__name__}
    for fld in dataclasses.fields(plot):
        value = getattr(plot, fld.name)
        if callable(value):
            continue  # captured via a FunctionInvariant instead
        out[fld.name] = value
    return out
