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

# Methods are classified purely by return annotation: '-> "ScatterPlotter"' is a
# configuration step we can record and replay, '-> p9.ggplot' is a terminal that
# produces one output file.  Adding either to plots.py needs no change here.


def _classify(cls) -> Tuple[dict, dict]:
    config, terminal = {}, {}
    for name, fn in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith("_"):
            continue
        annotation = str(inspect.signature(fn).return_annotation)
        if cls.__name__ in annotation:
            config[name] = fn
        elif "ggplot" in annotation:
            terminal[name] = fn
    return config, terminal


CONFIG_METHODS, TERMINAL_METHODS = _classify(ScatterPlotter)

# Terminal-level keywords that shadow nothing in any terminal signature; see
# test_ppg2_core.py::test_reserved_output_kwargs_do_not_shadow_terminals.
RESERVED_OUTPUT_KWARGS = ("name", "filename", "dpi")

# The methods that take a data source.  A source has to be a Path (or a
# ``(Path, job)`` pair, or a job) so it can become a file dependency -- a plain
# string would look like any other argument and silently lose its invariant.
SOURCE_PARAMS = {
    "set_source": "ad_or_data",
    "add_alternative_source": "source",
    "set_embedding_source": "source",
}


def _output_name_for(terminal: str) -> str:
    """``plot`` -> ``scatter``, ``plot_violin`` -> ``violin``."""
    return "scatter" if terminal == "plot" else terminal[len("plot_") :]


def _first_param_is_column(fn) -> bool:
    params = list(inspect.signature(fn).parameters)
    return len(params) > 1 and params[1] == "column"


_SELF = object()  # placeholder for the bound `self` during signature checks


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
    def wrapper(self, *args, name=None, filename=None, dpi=None, **kwargs):
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
        return self._build(
            terminal,
            tuple(args),
            dict(kwargs),
            name=name,
            filename=filename,
            dpi=dpi,
        )

    wrapper.__doc__ = (fn.__doc__ or "").rstrip() + (
        "\n\n        Creates the pipegraph job for this file instead of building the"
        "\n        plot; the plot's column is supplied automatically. Returns a copy"
        "\n        of the plot with .job_ and .jobs_ set."
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
        """The job the most recent terminal created."""
        if not self._jobs:
            raise AttributeError(
                "no terminal has been called on this Plot yet, so there is no "
                "job - call .scatter(), .violin(), ... first."
            )
        return self._jobs[-1]

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

    def _build(self, terminal, args, kwargs, *, name, filename, dpi) -> "Plot":
        label = _output_name_for(terminal)
        ppg, graph = _active_graph(f"{type(self).__name__}.{label}()")
        self._require_source(label)
        target = self._target(name, filename)
        recipe = self._encode(ppg, graph, terminal, args, kwargs, target, dpi)
        _claim_output(
            graph,
            target,
            recipe.parameters,
            f"{self._describe()}.{label}() at {_caller_location()}",
        )
        job = self._create_job(ppg, target, recipe)
        self._builder._session.record(graph, job)
        new = self._copy()
        new._jobs = self._jobs + (job,)
        return new

    def _target(self, name: str, filename: Optional[str]) -> Path:
        builder = self._builder
        directory = builder.output_folder.joinpath(*builder._into, *self._into)
        return directory / (filename if filename else f"{self.stem}_{name}.png")

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
