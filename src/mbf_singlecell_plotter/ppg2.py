"""Declarative, pipegraph-driven plotting.

:class:`PlotBuilder` and :class:`Plot` are *call recorders* for
:class:`~mbf_singlecell_plotter.plots.ScatterPlotter`.  They expose exactly the
methods ScatterPlotter exposes -- discovered by introspection, so nothing has to
be re-declared here -- but instead of executing a call they remember it.  At
:meth:`PlotBuilder.register_all` time the remembered calls are replayed inside a
pypipegraph2 job, and the same recording is serialised into the job's
invariants.  Configuration and change-detection therefore cannot drift apart.

::

    builder = PlotBuilder(base_size=15)
    builder.set_source(Path("analysis.h5ad"), embedding="umap", transform=log2)
    builder.add_alternative_source(Path("genes.h5ad"), name="genes")
    builder.style(dot_size=1)
    builder.panel_size(4, 4)

    p = builder.add(Plot("S100A8"))
    p.facet("condition")
    p.scatter()                      # -> S100A8_scatter.png
    p.violin("leiden")               # -> S100A8_violin_leiden.png
    p.ridgeline("leiden").unfacet()  # -> S100A8_ridgeline_leiden.png

    jobs = builder.register_all("results/plots")

Semantics
---------
Replay order is ``builder calls -> plot calls -> output calls``, exactly as if
you had made those calls in sequence on one ScatterPlotter.  Overriding is
therefore just calling again (``panel_size``, ``style``, ...); undoing uses the
``un*``/``without_*`` methods.  Within a plot the *whole* plot script is applied
before any output script, regardless of where ``.scatter()`` was called.

Anything a recorded call receives ends up in the job's invariants:  ``Path``\\ s
(and ``(Path, job)`` pairs) become file dependencies, callables become
:class:`FunctionInvariant`\\ s, everything else is JSON-encoded into a single
:class:`ParameterInvariant`.  Arguments that cannot be encoded deterministically
are rejected at record time rather than silently hashed by object identity.
"""

import functools
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .plots import ScatterPlotter

__all__ = ["PlotBuilder", "Plot", "Output", "UnencodableArgument"]


class UnencodableArgument(TypeError):
    """A recorded argument cannot be turned into a deterministic invariant."""


def _require_ppg():
    try:
        import pypipegraph2 as ppg
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pypipegraph2 is required to register plot jobs. Install it, or use "
            "ScatterPlotter directly for interactive plotting."
        ) from e
    return ppg


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

# Output-level keywords that shadow nothing in any terminal signature; see
# test_ppg2_core.py::test_reserved_output_kwargs_do_not_shadow_terminals.
RESERVED_OUTPUT_KWARGS = ("name", "filename", "dpi")

# The two methods that take a data source.  A source has to be a Path (or a
# ``(Path, job)`` pair, or a job) so it can become a file dependency -- a plain
# string would look like any other argument and silently lose its invariant.
SOURCE_PARAMS = {"set_source": "ad_or_data", "add_alternative_source": "source"}


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
        self._functions: List[Tuple[str, Any]] = []

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
            self._register_function(value.func, where)
            return {"__partial__": _qualname(value.func), "__bound__": enc}, value

        if isinstance(value, type):
            return {"__class__": _qualname(value)}, value

        if callable(value):
            self._register_function(value, where)
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

    def function_invariants(self) -> list:
        """FunctionInvariants for every callable seen, in encounter order."""
        if not self.ppg:
            return []
        out = []
        for name, fn in self._functions:
            out.append(self.ppg.FunctionInvariant(name, fn))
            extra = getattr(fn, "deps", None)
            if extra is not None:
                # a computed_column may carry its own upstream dependencies
                out.extend(extra if isinstance(extra, (list, tuple)) else [extra])
        return out

    # -- internal ----------------------------------------------------------

    def _register_function(self, fn, where: str):
        index = len(self._functions)
        self._functions.append((f"{self.prefix}::fn{index}::{_qualname(fn)}", fn))


# ── recorder base ────────────────────────────────────────────────────────────


class _Recorder:
    """Records ScatterPlotter configuration calls instead of performing them."""

    def __init__(self):
        self._calls: List[_Call] = []

    def _record(self, method: str, args: tuple, kwargs: dict):
        signature = inspect.signature(CONFIG_METHODS[method])
        try:
            signature.bind(_SELF, *args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{type(self).__name__}.{method}(): {e}") from None
        if method in SOURCE_PARAMS:
            self._check_source(method, signature, args, kwargs)
        self._calls.append(_Call(method, tuple(args), dict(kwargs)))
        return self

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
        return dict(kwargs)


def _make_config_method(name: str, fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return self._record(name, args, kwargs)

    wrapper.__doc__ = (fn.__doc__ or "").rstrip() + (
        "\n\n        Recorded, not executed; replayed inside the pipegraph job."
    )
    return wrapper


def _install_config_methods(cls):
    for name, fn in CONFIG_METHODS.items():
        method = _make_config_method(name, fn)
        method.__qualname__ = f"{cls.__name__}.{name}"
        setattr(cls, name, method)
    return cls


# ── outputs ──────────────────────────────────────────────────────────────────


@_install_config_methods
class Output(_Recorder):
    """One output file: a terminal ScatterPlotter call plus its own tweaks.

    Configuration recorded here applies on top of the owning plot's script and
    affects this file only -- that is how ``p.ridgeline("leiden").unfacet()``
    drops a facet for the ridgeline without touching the sibling scatter.
    """

    def __init__(
        self,
        terminal: str,
        args: tuple,
        kwargs: dict,
        *,
        name: str,
        filename: Optional[str] = None,
        dpi: Optional[int] = None,
    ):
        super().__init__()
        self.terminal = terminal
        self.terminal_args = args
        self.terminal_kwargs = kwargs
        self.name = name
        self.filename = filename
        self._dpi = dpi

    def dpi(self, value: int) -> "Output":
        """Override the owning plot's dpi for this file."""
        self._dpi = value
        return self

    def file_name(self, stem: str) -> str:
        if self.filename:
            return self.filename
        return f"{stem}_{self.name}.png"

    def __repr__(self):
        return f"<Output {self.name} ({self.terminal})>"


def _make_output_method(terminal: str, fn):
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
        return self._add_output(
            Output(
                terminal,
                tuple(args),
                dict(kwargs),
                name=name,
                filename=filename,
                dpi=dpi,
            )
        )

    wrapper.__doc__ = (fn.__doc__ or "").rstrip() + (
        "\n\n        Records an output file instead of building the plot. The plot's "
        "\n        column is supplied automatically. Returns the Output, which accepts "
        "\n        further ScatterPlotter calls scoped to this file alone."
    )
    return wrapper


def _install_output_methods(cls):
    for terminal, fn in TERMINAL_METHODS.items():
        method = _make_output_method(terminal, fn)
        name = _output_name_for(terminal)
        method.__qualname__ = f"{cls.__name__}.{name}"
        setattr(cls, name, method)
    return cls


# ── plot ─────────────────────────────────────────────────────────────────────


@_install_output_methods
@_install_config_methods
class Plot(_Recorder):
    """A subject (usually one column) and the output files derived from it.

    All outputs of one Plot share a single pipegraph job.
    """

    def __init__(
        self,
        column: Optional[Union[str, Tuple[str, str]]] = None,
        *,
        name: Optional[str] = None,
        into: Optional[Union[str, Path]] = None,
        dpi: Optional[int] = None,
        **plotter_kwargs,
    ):
        super().__init__()
        self.column = column
        self._name = name
        self._into: List[str] = []
        self._dpi = dpi
        self._outputs: List[Output] = []
        self.init_kwargs = self._init_kwargs_from(plotter_kwargs)
        if into is not None:
            self.into(into)

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
            "(pass Plot(name='...'))."
        )

    def into(self, sub_directory: Union[str, Path]) -> "Plot":
        """Append a sub-directory below the builder's output directory."""
        self._into.extend(Path(sub_directory).parts)
        return self

    def dpi(self, value: int) -> "Plot":
        """Set the default dpi for this plot's outputs."""
        self._dpi = value
        return self

    # -- outputs -----------------------------------------------------------

    def output(
        self,
        terminal: str,
        *args,
        name: Optional[str] = None,
        filename: Optional[str] = None,
        dpi: Optional[int] = None,
        **kwargs,
    ) -> Output:
        """Record an arbitrary terminal ScatterPlotter method as an output.

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
        return self._add_output(
            Output(
                terminal,
                tuple(args),
                dict(kwargs),
                name=name,
                filename=filename,
                dpi=dpi,
            )
        )

    def _add_output(self, output: Output) -> Output:
        existing = {o.name for o in self._outputs}
        if output.name in existing:
            raise ValueError(
                f"Plot {self._name or self.column!r} already has an output named "
                f"{output.name!r}; pass name=... to disambiguate."
            )
        self._outputs.append(output)
        return output

    @property
    def outputs(self) -> List[Output]:
        return list(self._outputs)

    def __repr__(self):
        subject = self._name or self.column
        return f"<Plot {subject!r} ({len(self._outputs)} outputs)>"


# ── builder ──────────────────────────────────────────────────────────────────


@_install_config_methods
class PlotBuilder(_Recorder):
    """Shared configuration plus the plots that build on it."""

    def __init__(
        self,
        *,
        into: Optional[Union[str, Path]] = None,
        dpi: int = 150,
        **plotter_kwargs,
    ):
        super().__init__()
        self._into: List[str] = []
        self._dpi = dpi
        self._plots: List[Plot] = []
        self.init_kwargs = self._init_kwargs_from(plotter_kwargs)
        if into is not None:
            self.into(into)

    # -- collection --------------------------------------------------------

    def into(self, sub_directory: Union[str, Path]) -> "PlotBuilder":
        """Append a sub-directory below the ``register_all`` result directory."""
        self._into.extend(Path(sub_directory).parts)
        return self

    def add(self, plot: Plot) -> Plot:
        """Queue a :class:`Plot`; returns it so you can keep configuring."""
        if not isinstance(plot, Plot):
            raise TypeError(f"expected a Plot, got {type(plot).__name__}")
        self._plots.append(plot)
        return plot

    @property
    def plots(self) -> List[Plot]:
        return list(self._plots)

    # -- registration ------------------------------------------------------

    def register_all(self, result_dir: Union[str, Path]) -> list:
        """Create one job per queued plot below ``result_dir``. Returns them."""
        ppg = _require_ppg()
        result_dir = Path(result_dir)
        self._builder_walk(ppg)  # once; shared by every plot
        return [self._register(plot, result_dir, ppg) for plot in self._plots]

    def _builder_walk(self, ppg):
        cached = getattr(self, "_builder_cache", None)
        if cached is not None and cached[0] == len(self._calls):
            return cached[1]
        walker = _Walker("builder", _resolver(ppg), ppg)
        encoded, resolved = walker.walk_calls(self._calls)
        deps = list(walker.deps) + walker.function_invariants()
        self._builder_cache = (len(self._calls), (encoded, resolved, deps))
        return self._builder_cache[1]

    def _out_dir(self, plot: Plot, result_dir: Path) -> Path:
        return result_dir.joinpath(*self._into, *plot._into)

    def _register(self, plot: Plot, result_dir: Path, ppg):
        if not plot._outputs:
            raise ValueError(f"{plot!r} declares no outputs")

        out_dir = self._out_dir(plot, result_dir)
        stem = plot.stem
        files = {o.name: out_dir / o.file_name(stem) for o in plot._outputs}
        job_id = str(sorted(files.values())[0])

        builder_encoded, builder_calls, builder_deps = self._builder_walk(ppg)
        walker = _Walker(job_id, _resolver(ppg), ppg)
        plot_encoded, plot_calls = walker.walk_calls(plot._calls)

        init_kwargs = {**self.init_kwargs, **plot.init_kwargs}
        init_encoded, _ = walker.walk(init_kwargs, f"{job_id} ScatterPlotter()")

        recipes = {}
        outputs_encoded = {}
        for output in plot._outputs:
            output_calls_encoded, output_calls = walker.walk_calls(output._calls)
            terminal_args, resolved_args = [], []
            for position, value in enumerate(output.terminal_args):
                enc, res = walker.walk(
                    value, f"{job_id} {output.name}: argument {position}"
                )
                terminal_args.append(enc)
                resolved_args.append(res)
            terminal_kwargs, resolved_kwargs = {}, {}
            for key in sorted(output.terminal_kwargs):
                enc, res = walker.walk(
                    output.terminal_kwargs[key], f"{job_id} {output.name}: {key}=..."
                )
                terminal_kwargs[key] = enc
                resolved_kwargs[key] = res
            dpi = (
                output._dpi
                if output._dpi is not None
                else (plot._dpi if plot._dpi is not None else self._dpi)
            )
            outputs_encoded[output.name] = {
                "terminal": output.terminal,
                "args": terminal_args,
                "kwargs": terminal_kwargs,
                "calls": output_calls_encoded,
                "dpi": dpi,
                "file": files[output.name].name,
            }
            recipes[output.name] = (
                tuple(builder_calls) + tuple(plot_calls) + tuple(output_calls),
                output.terminal,
                tuple(resolved_args),
                resolved_kwargs,
                dpi,
            )

        parameters = json.dumps(
            {
                "builder": builder_encoded,
                "init": init_encoded,
                "plot": plot_encoded,
                "outputs": outputs_encoded,
            },
            sort_keys=True,
        )

        def generate(output_filenames, recipes=recipes, init_kwargs=init_kwargs):
            for name, (calls, terminal, args, kwargs, dpi) in recipes.items():
                target = output_filenames[name]
                target.parent.mkdir(parents=True, exist_ok=True)
                plotter = _replay(calls, init_kwargs)
                figure = getattr(plotter, terminal)(*args, **kwargs)
                figure.save(target, dpi=dpi)

        job = ppg.MultiFileGeneratingJob(
            files, generate, depend_on_function=True, resources=ppg.Resources.RunsHere
        )
        job.depends_on(builder_deps)
        job.depends_on(list(walker.deps) + walker.function_invariants())
        job.depends_on(ppg.ParameterInvariant(job_id + "_config", parameters))
        job.depends_on(ppg.FunctionInvariant("mbf_scp_replay", _replay))
        return job


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
