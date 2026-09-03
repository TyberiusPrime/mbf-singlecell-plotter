"""Declarative, pipegraph-driven plotting.

:class:`PlotBuilder` and :class:`Plot` are *call recorders* for
:class:`~mbf_singlecell_plotter.plots.ScatterPlotter`.  They expose exactly the
methods ScatterPlotter exposes -- discovered by introspection, so nothing has to
be re-declared here -- but instead of executing a call they remember it.  When a
terminal (``scatter``, ``violin``, ...) is called the remembered script is
turned into a pypipegraph2 job on the spot, and the same recording is serialised
into that job's invariants.  Configuration and change-detection therefore cannot
drift apart.

Like ScatterPlotter itself, the recorders are immutable: every call returns an
altered copy, so a configured builder can be shared without any risk of one plot
leaking settings into the next.

::

    builder = (
        PlotBuilder(base_size=15, output_folder="results/plots")
        .set_source(Path("analysis.h5ad"), embedding="umap")
        .add_alternative_source(Path("genes.h5ad"), name="genes")
        .style(dot_size=1)
        .panel_size(4, 4)
    )

    p = builder.plot("S100A8").facet("condition")
    p.scatter()                       # -> S100A8_scatter.png, job created now
    p.violin("leiden")                # -> S100A8_violin_leiden.png
    p.unfacet().ridgeline("leiden")   # -> S100A8_ridgeline_leiden.png, unfaceted

    jobs = builder.jobs_              # everything created from this builder

Expression and embedding in different files
-------------------------------------------
Keep the *expression* file as the primary source and name the file holding the
coordinates with ``set_embedding_source`` -- then every column keeps its plain
name, and swapping either file is one edit::

    builder = (
        PlotBuilder(output_folder="results/plots")
        .set_source(Path("expression.h5ad"), embedding=None)
        .set_embedding_source(Path("coordinates.h5ad"), "umap")
    )
    builder.plot("S100A8").scatter()      # not ("coordinates", "S100A8")

Both files become dependencies of every job built from that builder.  The
embedding file is registered as an ordinary alternative source, so its ``obs``
columns (clusters, QC metrics, ...) resolve under their plain names too.

Interactive HTML exports
------------------------
``save_interactive_moran_grid`` / ``save_interactive_cluster_markers`` write
their own file rather than returning a figure, so they are neither
configuration nor a terminal.  They are recognised as *exports* and appear on
:class:`Plot` without the ``save_`` prefix; the output path is the job's output
file, and ``save_tsv=True`` simply makes the TSV a second output of the same
job::

    p = builder.plot("leiden")
    p.interactive_cluster_markers(save_tsv=True)   # .html + .tsv, one job

``plot_genes`` additionally declares a :class:`pypipegraph2.JobGeneratingJob`
that waits for the export, reads the marker genes back out of the TSV (which it
turns on) and plots each of them with *this* plot's configuration -- builder
calls and plot calls alike, only the column swapped.  The genes are only known
once the export has run, which is exactly what a job-generating job is for::

    p.interactive_moran_grid(plot_genes=True)      # one scatter per gene
    p.interactive_cluster_markers(                 # or spell it out
        plot_genes=lambda gene: gene.into("markers").scatter().violin("leiden")
    )

``plot_genes=True`` puts those scatters into a ``scatter/`` sub-directory of the
export's own directory -- the HTML stays visible instead of drowning in PNGs --
and points the export's ``gene_url`` at them, so clicking a gene in the HTML
shows its plot inline.  Pass ``gene_url=`` yourself to link somewhere else; a
callable ``plot_genes`` names its own files, so it links nowhere by default.

Signatures
----------
``add_signature`` is ordinary configuration, so the gene set travels in the
job's ParameterInvariant: editing the list re-runs the plots that use it.
Every terminal that plots a signature also writes ``<stem>_genes.tsv``, saying
which of its genes this dataset actually has -- one job, declared identically by
every terminal of that signature (its recipe covers the calls that decide gene
resolution and nothing else), so styling one plot differently does not fork the
file::

    sig = builder.add_signature("Myeloid", genes, method="mean").plot("Myeloid")
    sig.scatter()                  # Myeloid_scatter.png + Myeloid_genes.tsv
    sig.violin("leiden")           # ... and the very same TSV

``plot_genes`` repeats a plot for every gene of the set, into a sub-directory
named after the plot -- the score stays visible next to the folder::

    sig.scatter(plot_genes=True)   # Myeloid/LYZ_scatter.png, ...
    sig.violin("leiden", plot_genes=True)   # per-gene violins, same folder
    sig.scatter(plot_genes=lambda gene: gene.into("genes").histogram())

``True`` replays *this* terminal with the same arguments; a callable receives
each gene's Plot and names its own files, as with the exports.  A signature
keeps plotting when some of its genes are absent, so its fan-out has to skip
them -- which only the TSV can tell, hence a :class:`pypipegraph2.JobGeneratingJob`
waiting for it.  Genes named explicitly instead (``plot_genes=["LYZ", "CST3"]``,
which works for any column) are declared right away and plotted as named.

Semantics
---------
Replay order is ``builder calls -> plot calls``, exactly as if you had made
those calls in sequence on one ScatterPlotter.  Overriding is therefore just
calling again (``panel_size``, ``style``, ...); undoing uses the
``un*``/``without_*`` methods.  Configuration must precede the terminal it
should affect -- the terminal builds the job immediately, so nothing recorded
afterwards can reach it.

A terminal returns a copy of the plot carrying ``job_`` (the job just created)
and ``jobs_`` (every job created along this object's lineage), so the jobs can
be wired into further dependencies::

    p1 = builder.plot("Major").scatter().grid_histogram()
    p1.jobs_        # [scatter job, grid_histogram job]
    p1.job_         # the grid_histogram job

Declaring one file twice with the *same* configuration is not an error -- an
interactive graph that is adjusted and re-run redeclares everything, and that has
to go through untouched.  Two *differing* declarations of one file do collide:
fatally in a strict run mode, with a warning in the interactive ones, following
pypipegraph2's own policy on redefinition.

Anything a recorded call receives ends up in the job's invariants:  ``Path``\\ s
(and ``(Path, job)`` pairs) become file dependencies, callables become
:class:`FunctionInvariant`\\ s, everything else is JSON-encoded into a single
:class:`ParameterInvariant`.  Arguments that cannot be encoded deterministically
are rejected at record time rather than silently hashed by object identity.
"""

import copy
import functools
import hashlib
import inspect
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .data import _spec_str as _gene_key  # the TSV's spelling of a gene spec
from .plots import ScatterPlotter

__all__ = ["PlotBuilder", "Plot", "UnencodableArgument"]


class UnencodableArgument(TypeError):
    """A recorded argument cannot be turned into a deterministic invariant."""


def _require_ppg():
    try:
        import pypipegraph2 as ppg
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pypipegraph2 is required to create plot jobs. Install it, or use "
            "ScatterPlotter directly for interactive plotting."
        ) from e
    return ppg


def _current_graph():
    """The active pipegraph, or None -- without importing ppg if it is absent."""
    ppg = sys.modules.get("pypipegraph2")
    return getattr(ppg, "global_pipegraph", None) if ppg is not None else None


def _active_graph(what: str):
    """The live pipegraph as ``(module, graph)``, or a pointed error."""
    ppg = _require_ppg()
    graph = getattr(ppg, "global_pipegraph", None)
    if graph is None:
        raise RuntimeError(
            f"{what} creates a pipegraph job immediately and needs an active "
            "graph - call ppg.new() first, or use ScatterPlotter directly for "
            "interactive work."
        )
    return ppg, graph


# ── introspection of ScatterPlotter ──────────────────────────────────────────

# Methods are classified purely by their signature: '-> "ScatterPlotter"' is a
# configuration step we can record and replay, '-> p9.ggplot' is a terminal that
# produces one output file, and '-> None' taking (column, output_path) is an
# export -- a method that writes its own file (the interactive HTML views)
# instead of handing back a figure.  Adding any of them to plots.py needs no
# change here.


def _classify(cls) -> Tuple[dict, dict, dict]:
    config, terminal, export = {}, {}, {}
    for name, fn in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith("_"):
            continue
        signature = inspect.signature(fn)
        annotation = str(signature.return_annotation)
        if cls.__name__ in annotation:
            config[name] = fn
        elif "ggplot" in annotation:
            terminal[name] = fn
        elif annotation == "None" and list(signature.parameters)[1:3] == [
            "column",
            "output_path",
        ]:
            export[name] = fn
    return config, terminal, export


CONFIG_METHODS, TERMINAL_METHODS, EXPORT_METHODS = _classify(ScatterPlotter)

# Terminal-level keywords that shadow nothing in any terminal signature; see
# test_ppg2_core.py::test_reserved_output_kwargs_do_not_shadow_terminals.
RESERVED_OUTPUT_KWARGS = ("name", "filename", "dpi", "plot_genes")

# The same for exports -- `dpi` is missing on purpose: it is a real argument of
# the export methods (the resolution of the PNG embedded in the HTML), so it is
# passed through to the call rather than consumed here.
RESERVED_EXPORT_KWARGS = ("name", "filename", "plot_genes")

# The methods that take a data source.  A source has to be a Path (or a
# ``(Path, job)`` pair, or a job) so it can become a file dependency -- a plain
# string would look like any other argument and silently lose its invariant.
# Everything a signature needs to know which of its genes this dataset has --
# and nothing else.  The genes TSV is declared by *every* terminal that plots a
# signature, so its recipe has to be blind to styling, faceting and filtering:
# that is what lets .scatter() and .style(...).violin() declare one and the same
# file instead of colliding over it.
GENE_RESOLUTION_METHODS = (
    "set_source",
    "set_embedding_source",
    "add_alternative_source",
    "add_derived_source",
    "add_signature",
)

SOURCE_PARAMS = {
    "set_source": "ad_or_data",
    "add_alternative_source": "source",
    "set_embedding_source": "source",
}


def _output_name_for(terminal: str) -> str:
    """``plot`` -> ``scatter``, ``plot_violin`` -> ``violin``."""
    return "scatter" if terminal == "plot" else terminal[len("plot_") :]


def _export_name_for(export: str) -> str:
    """``save_interactive_moran_grid`` -> ``interactive_moran_grid``."""
    return export[len("save_") :] if export.startswith("save_") else export


def _first_param_is_column(fn) -> bool:
    params = list(inspect.signature(fn).parameters)
    return len(params) > 1 and params[1] == "column"


_SELF = object()  # placeholder for the bound `self` during signature checks
_OUTPUT = object()  # ... and for an export's output_path, which we supply


# ── recorded calls ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Call:
    method: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


def _replay(calls: Sequence[_Call], init_kwargs: dict) -> ScatterPlotter:
    """Apply a recorded script to a fresh ScatterPlotter."""
    plotter = ScatterPlotter(**init_kwargs)
    for call in calls:
        plotter = getattr(plotter, call.method)(*call.args, **call.kwargs)
    return plotter


# ── argument encoding ────────────────────────────────────────────────────────


def _is_job(value) -> bool:
    return hasattr(value, "job_id") and hasattr(value, "depends_on")


def _is_path_spec(value) -> bool:
    """A Path, a job, or a ``(path, job)`` pair as accepted by job_or_filename."""
    if isinstance(value, Path) or _is_job(value):
        return True
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], (str, Path))
        and _is_job(value[1])
    )


def _qualname(obj) -> str:
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if name is None:
        return repr(obj)
    return f"{module}.{name}" if module else name


def _default_resolve_path(value):
    """ppg-free fallback: keep the path, contribute no dependency."""
    if _is_path_spec(value) and isinstance(value, tuple):
        return Path(value[0]), None
    if _is_job(value):
        return value, None
    return Path(value), None


class _Walker:
    """Turn recorded calls into (JSON-safe encoding, replayable calls, deps).

    The encoding is what the job's ParameterInvariant hashes; the replayable
    calls are the same calls with path specs resolved to plain paths; deps are
    the jobs/invariants the walk discovered along the way.
    """

    MAX_DEPTH = 20

    def __init__(self, prefix: str, resolve_path=_default_resolve_path, ppg=None):
        self.prefix = prefix
        self.resolve_path = resolve_path
        self.ppg = ppg
        self.deps: List[Any] = []
        self._functions: List[Any] = []

    # -- public ------------------------------------------------------------

    def walk_calls(self, calls: Sequence[_Call]) -> Tuple[list, List[_Call]]:
        encoded, resolved = [], []
        for index, call in enumerate(calls):
            where = f"{self.prefix}[{index}].{call.method}"
            args_enc, args_res = [], []
            for position, value in enumerate(call.args):
                enc, res = self.walk(value, f"{where}() argument {position}")
                args_enc.append(enc)
                args_res.append(res)
            kwargs_enc, kwargs_res = {}, {}
            for key in sorted(call.kwargs):
                enc, res = self.walk(call.kwargs[key], f"{where}({key}=...)")
                kwargs_enc[key] = enc
                kwargs_res[key] = res
            encoded.append(
                {"method": call.method, "args": args_enc, "kwargs": kwargs_enc}
            )
            resolved.append(_Call(call.method, tuple(args_res), kwargs_res))
        return encoded, resolved

    def walk(self, value, where: str, depth: int = 0):
        """Return ``(encoded, resolved)`` for one argument."""
        if depth > self.MAX_DEPTH:
            raise UnencodableArgument(
                f"{where}: nested too deeply ({self.MAX_DEPTH} levels) - "
                "is the value self-referential?"
            )

        if value is None or isinstance(value, (bool, int, float, str)):
            return value, value

        if _is_path_spec(value):
            path, deps = self.resolve_path(value)
            if isinstance(deps, (list, tuple)):
                self.deps.extend(deps)
            elif deps is not None:
                self.deps.append(deps)
            return {"__path__": str(path)}, path

        if isinstance(value, np.generic):
            return value.item(), value
        if isinstance(value, np.ndarray):
            return {"__array__": value.tolist()}, value

        if isinstance(value, tuple):
            pairs = [
                self.walk(v, f"{where}[{i}]", depth + 1) for i, v in enumerate(value)
            ]
            return {"__tuple__": [e for e, _ in pairs]}, tuple(r for _, r in pairs)
        if isinstance(value, list):
            pairs = [
                self.walk(v, f"{where}[{i}]", depth + 1) for i, v in enumerate(value)
            ]
            return [e for e, _ in pairs], [r for _, r in pairs]
        if isinstance(value, (set, frozenset)):
            encoded = sorted(
                json.dumps(
                    self.walk(v, f"{where}{{...}}", depth + 1)[0], sort_keys=True
                )
                for v in value
            )
            return {"__set__": encoded}, value
        if isinstance(value, dict):
            items = []
            resolved = {}
            for key in sorted(value, key=repr):
                enc, res = self.walk(value[key], f"{where}[{key!r}]", depth + 1)
                key_enc, _ = self.walk(key, f"{where} key {key!r}", depth + 1)
                items.append([key_enc, enc])
                resolved[key] = res
            return {"__dict__": items}, resolved

        if isinstance(value, functools.partial):
            enc, _ = self.walk(
                {"args": list(value.args), "kwargs": value.keywords or {}},
                f"{where} (partial)",
                depth + 1,
            )
            self._functions.append(value.func)
            return {"__partial__": _qualname(value.func), "__bound__": enc}, value

        if isinstance(value, type):
            return {"__class__": _qualname(value)}, value

        if callable(value):
            self._functions.append(value)
            return {"__function__": _qualname(value)}, value

        # Last resort: an object is its type plus its (encodable) state.  This
        # covers plotnine's element_* objects without ever hashing an id().
        # An object exposing no state at all is *not* accepted -- it may be
        # keeping it somewhere we cannot see, and encoding it as its bare type
        # would silently make two different values look identical.
        state = getattr(value, "__dict__", None)
        if state is None:
            slots = {
                s: getattr(value, s)
                for s in getattr(type(value), "__slots__", ())
                if hasattr(value, s)
            }
            state = slots or None
        if state is not None:
            enc, _ = self.walk(
                dict(state), f"{where} (state of {type(value).__name__})", depth + 1
            )
            return {"__object__": _qualname(type(value)), "__state__": enc}, value

        raise UnencodableArgument(
            f"{where}: cannot encode {type(value).__name__} deterministically. "
            "Pass a value built from primitives, paths or functions, or move the "
            "logic into a function argument (which is tracked by its source)."
        )

    def function_invariants(self, prefix: Optional[str] = None) -> list:
        """FunctionInvariants for every callable seen, in encounter order.

        Names are built here rather than during the walk so a caller can pick a
        prefix derived from the finished encoding -- which is what keeps the
        invariant ids stable across runs (see :meth:`PlotBuilder._walk`).
        """
        if not self.ppg:
            return []
        prefix = self.prefix if prefix is None else prefix
        out = []
        for index, fn in enumerate(self._functions):
            out.append(
                self.ppg.FunctionInvariant(f"{prefix}::fn{index}::{_qualname(fn)}", fn)
            )
            extra = getattr(fn, "deps", None)
            if extra is not None:
                # a computed_column may carry its own upstream dependencies
                out.extend(extra if isinstance(extra, (list, tuple)) else [extra])
        return out


def _resolver(ppg):
    """Path resolution that also yields the matching file dependency.

    ``job_or_filename`` understands a filename or a job (whose first output is
    the file), but not a ``(path, job)`` pair -- which is the only way to name
    one particular file of a job that produces several, so it is resolved here.
    """

    def resolve(value):
        if isinstance(value, tuple):
            path, job = value
            return Path(path), [job]
        path, deps = ppg.util.job_or_filename(value)
        return Path(path), deps

    return resolve


# ── bookkeeping ──────────────────────────────────────────────────────────────


def _caller_location() -> str:
    """``demo.py:31`` for the first frame outside this module."""
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
    if frame is None:  # pragma: no cover - only reachable from a bare exec
        return "<unknown>"
    return f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"


@dataclass
class _Recipe:
    """One output file's replayable script, plus what it hashes to."""

    parameters: str
    calls: tuple
    init_kwargs: dict
    terminal: str
    args: tuple
    kwargs: dict
    dpi: int
    walker: "_Walker"
    builder_deps: list


@dataclass
class _Claim:
    """Who declared one output file, and what they declared."""

    fingerprint: str
    description: str


def _claim_output(graph, target: Path, fingerprint: str, description: str) -> None:
    """Reserve one output path, or explain who took it first.

    Re-declaring the *same* file with the *same* recipe is not a collision: it
    is what an interactive session does when a graph is adjusted and re-run, and
    pypipegraph2 deduplicates it silently.  Only a differing recipe is a real
    conflict, and there this follows pypipegraph2's own redefinition policy --
    fatal in a strict run mode, a warning in the interactive ones, where
    redefining a plot is the whole point.

    pypipegraph2 would eventually catch the strict case itself, through the
    redefined ParameterInvariant, but only with a diff of two JSON blobs.  The
    registry lives on the graph, so builders that share an output directory
    collide too.
    """
    registry = getattr(graph, "_mbf_singlecell_plotter_outputs", None)
    if registry is None:
        registry = {}
        graph._mbf_singlecell_plotter_outputs = registry
    key = str(target)
    previous = registry.get(key)
    if previous is not None and previous.fingerprint != fingerprint:
        message = (
            f"{key} is already produced by {previous.description}, with a "
            "different configuration.\nPass filename=... or name=... to "
            "disambiguate."
        )
        if graph.run_mode.is_strict():
            raise ValueError(message)
        warnings.warn(f"{message}\nRedefining it ({graph.run_mode}).", stacklevel=3)
    registry[key] = _Claim(fingerprint, description)


class _Session:
    """The jobs created from one builder lineage, shared by every copy of it."""

    def __init__(self):
        self.graph = None
        self.jobs: List[Any] = []

    def record(self, graph, job) -> None:
        if self.graph is not graph:  # ppg.new() -- the old jobs are history
            self.graph = graph
            self.jobs = []
        if not any(known is job for known in self.jobs):  # re-declared, not new
            self.jobs.append(job)

    def current(self, graph) -> list:
        return list(self.jobs) if self.graph is graph else []


# ── recorder base ────────────────────────────────────────────────────────────


class _Recorder:
    """Records ScatterPlotter configuration calls instead of performing them.

    Immutable, like ScatterPlotter: every recorded call returns a copy.
    """

    def __init__(self):
        self._calls: Tuple[_Call, ...] = ()
        self._walk_cache = None

    def _copy(self):
        new = copy.copy(self)
        new._walk_cache = None
        return new

    def _record(self, method: str, args: tuple, kwargs: dict):
        signature = inspect.signature(CONFIG_METHODS[method])
        try:
            signature.bind(_SELF, *args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{type(self).__name__}.{method}(): {e}") from None
        if method in SOURCE_PARAMS:
            self._check_source(method, signature, args, kwargs)
        new = self._copy()
        new._calls = self._calls + (_Call(method, tuple(args), dict(kwargs)),)
        return new

    def _check_source(self, method: str, signature, args: tuple, kwargs: dict):
        bound = signature.bind(_SELF, *args, **kwargs)
        source = bound.arguments.get(SOURCE_PARAMS[method])
        if source is not None and not _is_path_spec(source):
            raise TypeError(
                f"{type(self).__name__}.{method}(): the data source must be a "
                "pathlib.Path, a job, or a (path, job) pair so it can be tracked "
                f"as a file dependency - got {type(source).__name__}. In-memory "
                "AnnData objects and plain strings cannot be hashed."
            )

    def _init_kwargs_from(self, kwargs: dict) -> dict:
        signature = inspect.signature(ScatterPlotter.__init__)
        try:
            signature.bind(_SELF, **kwargs)
        except TypeError as e:
            raise TypeError(f"{type(self).__name__}(): {e}") from None
        if "ad_or_data" in kwargs:
            raise TypeError(
                f"{type(self).__name__}(): pass the data source to set_source() "
                "rather than to the constructor, so it can be tracked as a file "
                "dependency."
            )
        return dict(kwargs)


def _make_config_method(name: str, fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return self._record(name, args, kwargs)

    wrapper.__doc__ = (fn.__doc__ or "").rstrip() + (
        "\n\n        Recorded, not executed; replayed inside the pipegraph job."
        "\n        Returns an altered copy -- the receiver is left untouched."
    )
    return wrapper


def _check_no_shadowing(cls, names, kind: str):
    """A generated method must never silently replace one written here."""
    clashes = sorted(set(names) & set(vars(cls)))
    if clashes:  # pragma: no cover - a guard against a future plots.py change
        raise RuntimeError(
            f"ScatterPlotter now exposes {', '.join(clashes)} as a {kind}, "
            f"which would silently replace {cls.__name__}.{clashes[0]}. "
            "Rename the attribute defined in ppg2.py."
        )


def _install_config_methods(cls):
    _check_no_shadowing(cls, CONFIG_METHODS, "configuration method")
    for name, fn in CONFIG_METHODS.items():
        method = _make_config_method(name, fn)
        method.__qualname__ = f"{cls.__name__}.{name}"
        setattr(cls, name, method)
    return cls


# ── plot ─────────────────────────────────────────────────────────────────────


def _make_terminal_method(terminal: str, fn):
    default_name = _output_name_for(terminal)
    takes_column = _first_param_is_column(fn)
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(
        self, *args, name=None, filename=None, dpi=None, plot_genes=None, **kwargs
    ):
        if takes_column and "column" not in kwargs:
            if self.column is None:
                raise TypeError(
                    f"{type(self).__name__}.{default_name}(): this plot has no "
                    "column; pass column=... explicitly."
                )
            args = (self.column,) + tuple(args)
        try:
            signature.bind(_SELF, *args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{type(self).__name__}.{default_name}(): {e}") from None
        if name is None:
            name = "_".join(
                [default_name]
                + [
                    str(a)
                    for a in args[1 if takes_column else 0 :]
                    if isinstance(a, str)
                ]
            )
        if plot_genes is not None and not takes_column:
            raise TypeError(
                f"{type(self).__name__}.{default_name}(): plot_genes= needs a "
                "terminal that plots one column, and this one does not take one."
            )
        return self._build(
            terminal,
            tuple(args),
            dict(kwargs),
            name=name,
            filename=filename,
            dpi=dpi,
            plot_genes=plot_genes,
            gene_args=tuple(args[1:]) if takes_column else (),
        )

    wrapper.__doc__ = (fn.__doc__ or "").rstrip() + (
        "\n\n        Creates the pipegraph job for this file instead of building the"
        "\n        plot; the plot's column is supplied automatically. Returns a copy"
        "\n        of the plot with .job_ and .jobs_ set."
        "\n"
        "\n        Plotting a registered signature also writes ``<stem>_genes.tsv``,"
        "\n        listing which of its genes this data has."
        "\n"
        "\n        ``plot_genes`` repeats the plot for every gene of the set, into a"
        "\n        sub-directory named after this plot.  ``True`` replays *this*"
        "\n        terminal with the same arguments; a list of genes does the same"
        "\n        for a column that is no signature; a callable receives each"
        "\n        gene's Plot and names its own files::"
        "\n"
        "\n            sig.violin('leiden', plot_genes=True)"
        "\n            sig.scatter(plot_genes=lambda gene: gene.into('g').scatter())"
        "\n"
        "\n        A signature's genes are plotted by a JobGeneratingJob waiting"
        "\n        for the genes TSV, so genes this data lacks are skipped rather"
        "\n        than failing; genes named in a list are declared right away and"
        "\n        plotted as named."
    )
    return wrapper


def _install_terminal_methods(cls):
    names = {
        _output_name_for(terminal): fn for terminal, fn in TERMINAL_METHODS.items()
    }
    _check_no_shadowing(cls, names, "terminal")
    for terminal, fn in TERMINAL_METHODS.items():
        method = _make_terminal_method(terminal, fn)
        method.__qualname__ = f"{cls.__name__}.{_output_name_for(terminal)}"
        setattr(cls, _output_name_for(terminal), method)
    return cls


# ── exports (self-writing outputs: the interactive HTML views) ───────────────


# The default ``plot_genes`` hook writes one scatter per gene into a
# sub-directory named after the plot type, so the handful of HTML files stay
# visible next to a folder instead of being buried under hundreds of PNGs.
# The export links to exactly those files, hence the shared naming here.
GENE_PLOT_TERMINAL = "scatter"


def _default_gene_plot(plot: "Plot") -> None:
    """The ``plot_genes=True`` hook: one scatter per marker gene, in its own
    sub-directory (``scatter/``) below the export's own directory."""
    # .scatter() is installed by _install_terminal_methods
    plot.into(GENE_PLOT_TERMINAL).scatter()  # ty: ignore


def _default_signature_gene_plot(
    plot: "Plot", *, terminal, args, kwargs, name, dpi, folder
) -> None:
    """The terminal ``plot_genes=True`` hook: this very plot, per gene.

    Same terminal, same arguments, one sub-directory down -- so
    ``violin('leiden', plot_genes=True)`` yields per-gene violins rather than
    something the caller did not ask for.
    """
    plot.into(folder)._build(
        terminal,
        (plot.column,) + tuple(args),
        dict(kwargs),
        name=name,
        filename=None,  # the caller's filename names one file, not every gene's
        dpi=dpi,
    )


def _default_gene_url() -> str:
    """The ``gene_url`` template pointing at what :func:`_default_gene_plot` writes.

    Relative to the HTML file, which sits in the directory the sub-directory
    hangs off, so the link works from a copied result folder as well.
    """
    return f"{GENE_PLOT_TERMINAL}/{{gene}}_{GENE_PLOT_TERMINAL}.png"


def _present_genes(tsv_path, genes: Sequence) -> list:
    """The registered *genes* this dataset actually has, per the genes TSV.

    A signature keeps plotting when some of its genes are missing (that is what
    ``missing="warn"`` means), so its per-gene plots have to skip them too --
    and whether a gene resolves is only known once something has looked into
    the file.  The TSV is that look; the specs come from the recorded call, so
    ``(source, gene)`` routing survives the round trip.
    """
    import pandas as pd

    table = pd.read_csv(tsv_path, sep="\t")
    present = {
        str(gene) for gene, found in zip(table["gene"], table["present"]) if bool(found)
    }
    return [gene for gene in genes if _gene_key(gene) in present]


def _generate_signature_gene_plots(tsv_path, genes, template: "Plot", hook) -> list:
    """Replay *template* once per present gene, through *hook*."""
    found = _present_genes(tsv_path, genes)
    for gene in found:
        plot = template._copy()
        plot.column = gene
        plot._jobs = ()
        plot._job = None
        hook(plot)
    return found


def _marker_genes(tsv_path) -> list:
    """The marker genes of an export's TSV, deduplicated, best-ranked first."""
    import pandas as pd

    table = pd.read_csv(tsv_path, sep="\t")
    if "gene" not in table.columns:  # pragma: no cover - only a changed writer
        raise ValueError(
            f"{tsv_path} has no 'gene' column, so the marker genes cannot be "
            f"plotted (columns: {list(table.columns)})."
        )
    if "rank" in table.columns:
        table = table.sort_values("rank", kind="stable")
    return list(dict.fromkeys(table["gene"].dropna().astype(str)))


def _generate_gene_plots(tsv_path, template: "Plot", hook) -> list:
    """Replay *template* once per marker gene in *tsv_path*, through *hook*.

    Runs inside a JobGeneratingJob, i.e. only once the export has written its
    TSV -- which is why the genes cannot be known when the graph is declared.
    """
    genes = _marker_genes(tsv_path)
    for gene in genes:
        plot = template._copy()
        plot.column = gene
        plot._jobs = ()
        hook(plot)
    return genes


def _make_export_method(export: str, fn):
    default_name = _export_name_for(export)
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(self, *args, name=None, filename=None, plot_genes=None, **kwargs):
        column = kwargs.pop("column", None)
        if args:
            if len(args) > 1:
                raise TypeError(
                    f"{type(self).__name__}.{default_name}(): only the column may "
                    "be passed positionally - everything else is keyword-only, "
                    "and the output file comes from name=/filename=."
                )
            if column is not None:
                raise TypeError(
                    f"{type(self).__name__}.{default_name}(): got two values for "
                    "column."
                )
            column = args[0]
        if column is None:
            column = self.column
        if column is None:
            raise TypeError(
                f"{type(self).__name__}.{default_name}(): this plot has no "
                "column; pass column=... explicitly."
            )
        if "output_path" in kwargs:
            raise TypeError(
                f"{type(self).__name__}.{default_name}(): the output path is the "
                "job's output file - name it with filename=... or name=..., not "
                "with output_path=..."
            )
        try:
            signature.bind(_SELF, column, _OUTPUT, **kwargs)
        except TypeError as e:
            raise TypeError(f"{type(self).__name__}.{default_name}(): {e}") from None
        if not (
            plot_genes is None or isinstance(plot_genes, bool) or callable(plot_genes)
        ):
            raise TypeError(
                f"{type(self).__name__}.{default_name}(): plot_genes must be "
                "True/False or a callable taking the gene's Plot - got "
                f"{type(plot_genes).__name__}."
            )
        return self._build_export(
            export,
            column,
            dict(kwargs),
            name=default_name if name is None else name,
            filename=filename,
            plot_genes=plot_genes,
        )

    wrapper.__doc__ = (fn.__doc__ or "").rstrip() + (
        "\n\n        Creates the pipegraph job for this file instead of writing it"
        "\n        now; the plot's column is supplied automatically and the output"
        "\n        path is the job's output file (name it with name=/filename=)."
        "\n        With ``save_tsv=True`` the TSV is a second output of the same job."
        "\n"
        "\n        ``plot_genes`` adds a JobGeneratingJob that, once the export has"
        "\n        run, plots every marker gene of the TSV (which it turns on) with"
        "\n        this plot's own configuration -- builder calls and plot calls"
        "\n        alike, only the column swapped. ``True`` makes one scatter per"
        "\n        gene in a ``scatter/`` sub-directory and links the HTML's genes"
        "\n        to them (unless you pass your own ``gene_url``); a callable"
        "\n        receives each gene's Plot, may call any terminals on it, and"
        "\n        sets no links::"
        "\n"
        "\n            p.interactive_cluster_markers("
        "\n                plot_genes=lambda gene: gene.into('genes').scatter()"
        "\n            )"
        "\n"
        "\n        Returns a copy of the plot with .job_ and .jobs_ set."
    )
    return wrapper


def _install_export_methods(cls):
    names = {_export_name_for(export): fn for export, fn in EXPORT_METHODS.items()}
    _check_no_shadowing(cls, names, "export")
    for export, fn in EXPORT_METHODS.items():
        method = _make_export_method(export, fn)
        method.__qualname__ = f"{cls.__name__}.{_export_name_for(export)}"
        setattr(cls, _export_name_for(export), method)
    return cls


@_install_export_methods
@_install_terminal_methods
@_install_config_methods
class Plot(_Recorder):
    """A subject (usually one column) and the files derived from it.

    Created by :meth:`PlotBuilder.plot`, never directly.  Every configuration
    call returns an altered copy; every terminal call creates one job right
    away and returns a copy carrying it.
    """

    def __init__(
        self,
        builder: "PlotBuilder",
        column: Optional[Union[str, Tuple[str, str]]] = None,
        *,
        name: Optional[str] = None,
        into: Optional[Union[str, Path]] = None,
        dpi: Optional[int] = None,
        **plotter_kwargs,
    ):
        super().__init__()
        self._builder = builder
        self.column = column
        self._name = name
        self._into: Tuple[str, ...] = Path(into).parts if into is not None else ()
        self._dpi = dpi
        self._jobs: Tuple[Any, ...] = ()
        self._job: Any = None
        self.init_kwargs = self._init_kwargs_from(plotter_kwargs)

    # -- naming / layout ---------------------------------------------------

    @property
    def stem(self) -> str:
        """Filename stem: the explicit name, else the column."""
        if self._name is not None:
            return self._name
        if isinstance(self.column, str):
            return self.column
        if isinstance(self.column, tuple):
            return self.column[1]
        raise ValueError(
            "Plot needs a name: it has no column to derive one from "
            "(pass builder.plot(name='...'))."
        )

    def into(self, sub_directory: Union[str, Path]) -> "Plot":
        """Append a sub-directory below the builder's output directory."""
        new = self._copy()
        new._into = self._into + Path(sub_directory).parts
        return new

    def dpi(self, value: int) -> "Plot":
        """Set the default dpi for this plot's files."""
        new = self._copy()
        new._dpi = value
        return new

    # -- results -----------------------------------------------------------

    @property
    def jobs_(self) -> list:
        """Every job created along this object's lineage, in creation order."""
        return list(self._jobs)

    @property
    def job_(self):
        """The job for the file the most recent terminal named.

        The extras a terminal may create along the way -- a signature's genes
        TSV, the per-gene plots of ``plot_genes`` -- are in :attr:`jobs_`, not
        here: ``depends_on(plot.job_)`` waits for *this* plot's file.
        """
        if self._job is None:
            raise AttributeError(
                "no terminal has been called on this Plot yet, so there is no "
                "job - call .scatter(), .violin(), ... first."
            )
        return self._job

    # -- job creation ------------------------------------------------------

    def output(
        self,
        terminal: str,
        *args,
        name: Optional[str] = None,
        filename: Optional[str] = None,
        dpi: Optional[int] = None,
        **kwargs,
    ) -> "Plot":
        """Create a job for an arbitrary terminal ScatterPlotter method.

        The generated shorthands (``scatter``, ``violin``, ...) are thin
        wrappers around this; use it directly for anything without one.  Unlike
        the shorthands this does *not* inject the plot's column.
        """
        if terminal not in TERMINAL_METHODS:
            if terminal in EXPORT_METHODS or terminal in {
                _export_name_for(e) for e in EXPORT_METHODS
            }:
                raise ValueError(
                    f"{terminal!r} writes its own file rather than returning a "
                    f"figure - call .{_export_name_for(terminal)}() directly."
                )
            known = ", ".join(sorted(TERMINAL_METHODS))
            raise ValueError(f"{terminal!r} is not a ScatterPlotter terminal ({known})")
        signature = inspect.signature(TERMINAL_METHODS[terminal])
        try:
            signature.bind(_SELF, *args, **kwargs)
        except TypeError as e:
            raise TypeError(f"Plot.output({terminal!r}): {e}") from None
        if name is None:
            name = _output_name_for(terminal)
        return self._build(
            terminal, tuple(args), dict(kwargs), name=name, filename=filename, dpi=dpi
        )

    def _build(
        self,
        terminal,
        args,
        kwargs,
        *,
        name,
        filename,
        dpi,
        plot_genes=None,
        gene_args=(),
    ) -> "Plot":
        label = _output_name_for(terminal)
        ppg, graph = _active_graph(f"{type(self).__name__}.{label}()")
        self._require_source(label)
        signature = self._signature_call(self.column)
        genes, from_signature = self._genes_to_plot(plot_genes, signature, label)
        target = self._target(name, filename)
        recipe = self._encode(ppg, graph, terminal, args, kwargs, target, dpi)
        _claim_output(
            graph,
            target,
            recipe.parameters,
            f"{self._describe()}.{label}() at {_caller_location()}",
        )
        new = self._copy()
        # the genes TSV first: it says what the score is made of, and it is the
        # one file every terminal of this signature agrees on.
        tsv_job = None
        if signature is not None:
            tsv_job = self._build_signature_tsv(ppg, graph, signature)
            new._jobs = self._add_jobs(self._jobs, [tsv_job])
        else:
            new._jobs = self._jobs
        job = self._create_job(ppg, target, recipe)
        self._builder._session.record(graph, job)
        new._job = job
        new._jobs = self._add_jobs(new._jobs, [job])
        if genes is not None:
            hook = (
                plot_genes
                if callable(plot_genes)  # True is not, so it takes the default
                else functools.partial(
                    _default_signature_gene_plot,
                    terminal=terminal,
                    args=gene_args,
                    kwargs=kwargs,
                    name=name,
                    dpi=dpi,
                    folder=self.stem,
                )
            )
            if from_signature:
                created = [
                    self._create_signature_gene_job(
                        ppg, graph, tsv_job, target, genes, hook
                    )
                ]
            else:
                created = self._plot_each_gene(graph, genes, hook)
            new._jobs = self._add_jobs(new._jobs, created)
        return new

    def _create_signature_gene_job(self, ppg, graph, tsv_job, target, genes, hook):
        """A JobGeneratingJob that plots the genes of the signature this plot shows.

        Which of them the data has is what the genes TSV answers, so the plots
        can only be declared once it has run -- the same shape the interactive
        exports use for their marker genes.  The template is this very plot --
        builder script, plot script, ``into``, ``dpi`` -- with the column
        swapped per gene.
        """
        template = self._copy()
        template._jobs = ()
        template._job = None
        template._name = None  # each gene names its own files

        def callback(genes=tuple(genes), hook=hook, template=template, tsv=tsv_job):
            _generate_signature_gene_plots(Path(tsv.job_id), genes, template, hook)

        job = ppg.JobGeneratingJob(
            f"{target}::gene_plots", callback, depend_on_function=False
        )
        job.depends_on(tsv_job)
        self._builder._session.record(graph, job)
        return job

    @staticmethod
    def _add_jobs(jobs: tuple, new_jobs) -> tuple:
        """Append *new_jobs*, skipping the ones already there.

        A re-declaration hands back the very same job object (pypipegraph2
        deduplicates by id), and a second terminal on the same plot re-declares
        the signature's TSV -- which must not show up twice in ``jobs_``.
        """
        out = list(jobs)
        for job in new_jobs:
            if not any(known is job for known in out):
                out.append(job)
        return tuple(out)

    # -- signatures --------------------------------------------------------

    def _signature_call(self, column) -> Optional[_Call]:
        """The recorded ``add_signature`` call defining *column*, if any.

        Read off the script rather than out of the data: nothing may open the
        h5ad while the graph is being declared.
        """
        if not isinstance(column, str):
            return None
        found = None
        for call in self._builder._calls + self._calls:
            if call.method != "add_signature":
                continue
            name = call.args[0] if call.args else call.kwargs.get("name")
            if name == column:
                found = call  # a later definition wins, as on replay
        return found

    @staticmethod
    def _signature_genes(call: _Call) -> list:
        genes = call.args[1] if len(call.args) > 1 else call.kwargs["genes"]
        return list(dict.fromkeys(genes))

    def _genes_to_plot(self, plot_genes, signature, label):
        """``(genes, from_signature)`` -- or ``(None, False)`` for no fan-out.

        Genes named explicitly are plotted as named (and a typo fails loudly);
        a signature's genes are filtered by what the data has, which only its
        genes TSV can say.
        """
        if plot_genes is None or plot_genes is False:
            return None, False
        if not (plot_genes is True or callable(plot_genes)):
            if isinstance(plot_genes, (str, bytes)) or not isinstance(
                plot_genes, (list, tuple)
            ):
                raise TypeError(
                    f"{self._describe()}.{label}(): plot_genes must be True, a "
                    "list of genes, or a callable taking the gene's Plot - got "
                    f"{type(plot_genes).__name__}."
                )
            genes = list(dict.fromkeys(plot_genes))
            if not genes:
                raise ValueError(
                    f"{self._describe()}.{label}(): plot_genes= is an empty list."
                )
            return genes, False
        if signature is None:
            raise TypeError(
                f"{self._describe()}.{label}(): plot_genes= plots the genes of a "
                f"signature, and {self.column!r} is not one (add_signature(...) "
                "registers it). Pass the genes as a list to plot them anyway."
            )
        return self._signature_genes(signature), True

    def _plot_each_gene(self, graph, genes: list, hook) -> list:
        """Run *hook* once per gene, and collect the jobs it created.

        The template is this plot -- builder script, plot script, ``into``,
        ``dpi`` -- with only the column swapped, exactly as the exports do it.
        """
        session = self._builder._session
        before = len(session.current(graph))
        for gene in genes:
            plot = self._copy()
            plot.column = gene
            plot._name = None  # each gene names its own files
            plot._jobs = ()
            plot._job = None
            hook(plot)
        return session.current(graph)[before:]

    def _build_signature_tsv(self, ppg, graph, signature: _Call):
        """The ``<stem>_genes.tsv`` job: which genes of the set this data has.

        Declared by every terminal that plots the signature, so its recipe is
        built from the gene-resolution calls alone -- two terminals of one
        signature then declare the very same file with the very same recipe,
        which is a no-op rather than a collision.
        """
        name = signature.args[0] if signature.args else signature.kwargs["name"]
        target = self._target("genes", None, suffix=".tsv")
        calls = []
        for call in self._builder._calls + self._calls:
            if call.method == "add_signature":
                if call is signature:
                    calls.append(call)
                    break  # later signatures cannot change this one's genes
                continue  # ... and neither can the other signatures
            if call.method in GENE_RESOLUTION_METHODS:
                calls.append(call)
        walker = _Walker(str(target), _resolver(ppg), ppg)
        encoded, resolved = walker.walk_calls(calls)
        parameters = json.dumps({"calls": encoded, "signature": name}, sort_keys=True)
        _claim_output(
            graph,
            target,
            parameters,
            f"{self._describe()} signature genes at {_caller_location()}",
        )

        def write(output_filename, calls=tuple(resolved), signature=name):
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            _replay(calls, {}).signature_report(signature).to_csv(
                output_filename, sep="\t", index=False
            )

        job = ppg.FileGeneratingJob(target, write, depend_on_function=False)
        job.depends_on(list(walker.deps) + walker.function_invariants())
        job.depends_on(ppg.ParameterInvariant(str(target) + "_config", parameters))
        self._builder._session.record(graph, job)
        return job

    def _target(self, name: str, filename: Optional[str], suffix=".png") -> Path:
        builder = self._builder
        directory = builder.output_folder.joinpath(*builder._into, *self._into)
        return directory / (filename if filename else f"{self.stem}_{name}{suffix}")

    def _resolved_dpi(self) -> int:
        return self._dpi if self._dpi is not None else self._builder._dpi

    def _require_source(self, label: str) -> None:
        if not any(
            call.method == "set_source" for call in self._builder._calls + self._calls
        ):
            raise ValueError(
                f"{self._describe()}.{label}(): no data source - call "
                "set_source() on the builder or on the plot first."
            )

    def _encode(self, ppg, graph, terminal, args, kwargs, target: Path, dpi):
        """Everything this file needs, and the fingerprint it hashes to.

        Kept separate from job creation so the fingerprint is available before
        the output path is claimed -- a re-declaration is only a collision when
        the recipe actually differs.
        """
        builder = self._builder
        job_id = str(target)

        builder_encoded, builder_calls, builder_deps = builder._walk(ppg, graph)
        walker = _Walker(job_id, _resolver(ppg), ppg)
        plot_encoded, plot_calls = walker.walk_calls(self._calls)

        init_kwargs = {**builder.init_kwargs, **self.init_kwargs}
        init_encoded, _ = walker.walk(init_kwargs, f"{job_id} ScatterPlotter()")

        args_encoded, args_resolved = [], []
        for position, value in enumerate(args):
            enc, res = walker.walk(value, f"{job_id}: argument {position}")
            args_encoded.append(enc)
            args_resolved.append(res)
        kwargs_encoded, kwargs_resolved = {}, {}
        for key in sorted(kwargs):
            enc, res = walker.walk(kwargs[key], f"{job_id}: {key}=...")
            kwargs_encoded[key] = enc
            kwargs_resolved[key] = res

        if dpi is None:
            dpi = self._dpi if self._dpi is not None else builder._dpi

        return _Recipe(
            parameters=json.dumps(
                {
                    "builder": builder_encoded,
                    "init": init_encoded,
                    "plot": plot_encoded,
                    "terminal": terminal,
                    "args": args_encoded,
                    "kwargs": kwargs_encoded,
                    "dpi": dpi,
                },
                sort_keys=True,
            ),
            calls=tuple(builder_calls) + tuple(plot_calls),
            init_kwargs=init_kwargs,
            terminal=terminal,
            args=tuple(args_resolved),
            kwargs=kwargs_resolved,
            dpi=dpi,
            walker=walker,
            builder_deps=builder_deps,
        )

    def _create_job(self, ppg, target: Path, recipe: "_Recipe"):
        job_id = str(target)
        calls = recipe.calls
        init_kwargs = recipe.init_kwargs
        terminal = recipe.terminal
        args_resolved = recipe.args
        kwargs_resolved = recipe.kwargs
        dpi = recipe.dpi

        # every value the job needs is bound as a default argument: pypipegraph2
        # rejects job functions that reach into the enclosing scope.
        def generate(
            output_filename,
            args_resolved=args_resolved,
            calls=calls,
            dpi=dpi,
            init_kwargs=init_kwargs,
            kwargs_resolved=kwargs_resolved,
            terminal=terminal,
        ):
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            plotter = _replay(calls, init_kwargs)
            figure = getattr(plotter, terminal)(*args_resolved, **kwargs_resolved)
            figure.save(output_filename, dpi=dpi)

        job = ppg.FileGeneratingJob(target, generate, depend_on_function=False)
        job.depends_on(recipe.builder_deps)
        job.depends_on(list(recipe.walker.deps) + recipe.walker.function_invariants())
        job.depends_on(ppg.ParameterInvariant(job_id + "_config", recipe.parameters))
        # job.depends_on(ppg.FunctionInvariant("mbf_scp_replay", _replay))
        return job

    # -- exports -----------------------------------------------------------

    def _build_export(self, export, column, kwargs, *, name, filename, plot_genes):
        """One HTML export (plus its TSV), and optionally its per-gene plots."""
        label = _export_name_for(export)
        ppg, graph = _active_graph(f"{type(self).__name__}.{label}()")
        self._require_source(label)
        target = self._target(name, filename, suffix=".html")
        kwargs.setdefault("dpi", self._resolved_dpi())
        if plot_genes:
            kwargs["save_tsv"] = True  # the genes are read back from it
        if plot_genes is True:
            # the default hook's files are known up front, so the HTML can link
            # to them - and show them inline, they are right next to it.
            _accepted = inspect.signature(EXPORT_METHODS[export]).parameters
            if "gene_url" in _accepted and kwargs.get("gene_url") is None:
                kwargs["gene_url"] = _default_gene_url()
                if "gene_url_inline" in _accepted:
                    kwargs.setdefault("gene_url_inline", True)
        tsv = target.with_suffix(".tsv")
        if kwargs.get("save_tsv") and tsv == target:
            raise ValueError(
                f"{self._describe()}.{label}(): save_tsv writes {tsv}, which is "
                "the HTML file itself - give the output a non-.tsv filename."
            )
        recipe = self._encode(
            ppg, graph, export, (column,), kwargs, target, kwargs["dpi"]
        )
        outputs = [target] + ([tsv] if kwargs.get("save_tsv") else [])
        description = f"{self._describe()}.{label}() at {_caller_location()}"
        for output in outputs:
            _claim_output(graph, output, recipe.parameters, description)
        job = self._create_export_job(ppg, outputs, recipe)
        self._builder._session.record(graph, job)
        new = self._copy()
        new._job = job
        new._jobs = self._jobs + (job,)
        if plot_genes:
            hook = _default_gene_plot if plot_genes is True else plot_genes
            genes_job = self._create_gene_job(ppg, tsv, hook, job)
            self._builder._session.record(graph, genes_job)
            new._jobs = new._jobs + (genes_job,)
        return new

    def _create_export_job(self, ppg, outputs: list, recipe: "_Recipe"):
        calls = recipe.calls
        init_kwargs = recipe.init_kwargs
        export = recipe.terminal
        args_resolved = recipe.args
        kwargs_resolved = recipe.kwargs

        # as in _create_job: nothing may be read from the enclosing scope.
        def generate(
            output_filenames,
            args_resolved=args_resolved,
            calls=calls,
            export=export,
            init_kwargs=init_kwargs,
            kwargs_resolved=kwargs_resolved,
        ):
            target = Path(output_filenames[0])
            target.parent.mkdir(parents=True, exist_ok=True)
            plotter = _replay(calls, init_kwargs)
            getattr(plotter, export)(*args_resolved, target, **kwargs_resolved)

        job = ppg.MultiFileGeneratingJob(outputs, generate, depend_on_function=False)
        job.depends_on(recipe.builder_deps)
        job.depends_on(list(recipe.walker.deps) + recipe.walker.function_invariants())
        job.depends_on(
            ppg.ParameterInvariant(str(outputs[0]) + "_config", recipe.parameters)
        )
        return job

    def _create_gene_job(self, ppg, tsv: Path, hook, export_job):
        """A JobGeneratingJob that plots the export's marker genes.

        The genes only exist once the export has run, so the plots cannot be
        declared up front.  The template is this very plot -- builder script,
        plot script, ``into``, ``dpi`` -- with the column swapped per gene, and
        each generated plot goes through the ordinary job machinery, so it gets
        its own invariants and output claim.
        """
        template = self._copy()
        template._jobs = ()
        template._name = None  # each gene names its own files

        def callback(hook=hook, template=template, tsv=tsv):
            _generate_gene_plots(tsv, template, hook)

        job = ppg.JobGeneratingJob(
            f"{tsv}::gene_plots", callback, depend_on_function=False
        )
        job.depends_on(export_job)
        return job

    # -- misc --------------------------------------------------------------

    def _describe(self) -> str:
        if self.column is not None:
            return f"Plot({self.column!r})"
        return f"Plot(name={self._name!r})"

    def __repr__(self):
        return f"<{self._describe()}, {len(self._jobs)} jobs>"


# ── builder ──────────────────────────────────────────────────────────────────


@_install_config_methods
class PlotBuilder(_Recorder):
    """Shared configuration plus the plots that build on it.

    Immutable: every configuration call returns an altered copy, so one
    configured builder can safely seed any number of divergent plots::

        faceted = builder.facet("AgeGroup")   # builder itself is unchanged
        faceted.plot("GeneXY").scatter()
    """

    def __init__(
        self,
        *,
        output_folder: Union[str, Path],
        into: Optional[Union[str, Path]] = None,
        dpi: int = 150,
        **plotter_kwargs,
    ):
        super().__init__()
        self.output_folder = Path(output_folder)
        self._into: Tuple[str, ...] = Path(into).parts if into is not None else ()
        self._dpi = dpi
        self._session = _Session()
        self.init_kwargs = self._init_kwargs_from(plotter_kwargs)

    # -- layout ------------------------------------------------------------

    def into(self, sub_directory: Union[str, Path]) -> "PlotBuilder":
        """Append a sub-directory below the output folder."""
        new = self._copy()
        new._into = self._into + Path(sub_directory).parts
        return new

    def dpi(self, value: int) -> "PlotBuilder":
        """Set the default dpi for every plot built from here."""
        new = self._copy()
        new._dpi = value
        return new

    # -- plots -------------------------------------------------------------

    def plot(
        self,
        column: Optional[Union[str, Tuple[str, str]]] = None,
        *,
        name: Optional[str] = None,
        into: Optional[Union[str, Path]] = None,
        dpi: Optional[int] = None,
        **plotter_kwargs,
    ) -> Plot:
        """A :class:`Plot` for one column, carrying this builder's script.

        Nothing is created yet -- a job appears when a terminal (``scatter``,
        ``violin``, ...) is called on the result.
        """
        return Plot(self, column, name=name, into=into, dpi=dpi, **plotter_kwargs)

    @property
    def jobs_(self) -> list:
        """Every job created from this builder and its copies, in order."""
        return self._session.current(_current_graph())

    # -- invariants --------------------------------------------------------

    def _walk(self, ppg, graph):
        """Encode this builder's script once per graph.

        The FunctionInvariant names are prefixed with a digest of the finished
        encoding: stable across runs (unlike a counter) and distinct per
        builder configuration (unlike a fixed 'builder'), which matters now
        that a script can hold any number of divergent builder copies.
        """
        cached = self._walk_cache
        if cached is not None and cached[0] is graph:
            return cached[1]
        walker = _Walker("builder", _resolver(ppg), ppg)
        encoded, resolved = walker.walk_calls(self._calls)
        digest = hashlib.sha256(
            json.dumps(encoded, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        deps = list(walker.deps) + walker.function_invariants(f"builder_{digest}")
        result = (encoded, resolved, deps)
        self._walk_cache = (graph, result)
        return result

    def __repr__(self):
        return f"<PlotBuilder {str(self.output_folder)!r}, {len(self._calls)} calls>"
