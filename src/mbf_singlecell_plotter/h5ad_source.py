"""h5ad-inspect backed data source — fast selective reads from .h5ad files."""

import io
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Any
from collections.abc import Sequence
from numpy.typing import NDArray

import numpy as np
import pandas as pd

H5AD_TIMEOUT = 60

#: Oldest h5ad-inspect whose CLI this module speaks.  0.2.0 replaced the flat
#: ``export <thing>`` surface with verbs (``list`` / ``describe`` / ``get`` /
#: ``write``); nothing here works against 0.1.x.
MIN_H5AD_INSPECT_VERSION = (0, 2)

_INSTALL_HINT = (
    "Installation options:\n"
    "  • Nix devShell: add h5ad_inspect.packages.${system}.h5ad-inspect to packages\n"
    "  • Cargo:        cargo install --git "
    "https://github.com/TyberiusPrime/h5ad_inspect\n"
)

# ── availability ─────────────────────────────────────────────────────────────

# Version probe results, keyed by the resolved binary path so a PATH change
# within one process is picked up while repeat calls stay free.
_version_cache: dict[str, Optional[tuple[int, ...]]] = {}


def _parse_version(text: str) -> Optional[tuple[int, ...]]:
    """Pull a ``(major, minor[, patch])`` tuple out of ``--version`` output."""
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def _probe_version(binary: str) -> Optional[tuple[int, ...]]:
    """Return the version *binary* reports, or None if it reports none.

    Only a clean ``--version`` (exit 0 with a parseable version on stdout)
    counts.  This is deliberately a *positive* test: 0.1.x had no ``--version``
    at all — it prints its usage to stderr and exits 1 — and its error messages
    are too varied to recognise by their text.  So anything that is not an
    affirmative 0.2+ answer is treated as "unusable", without guessing why.
    """
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            timeout=H5AD_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _parse_version(result.stdout.decode(errors="replace"))


def _h5ad_inspect_version() -> Optional[tuple[int, ...]]:
    """Version of the ``h5ad-inspect`` on PATH, or None if absent/unusable."""
    binary = shutil.which("h5ad-inspect")
    if binary is None:
        return None
    if binary not in _version_cache:
        _version_cache[binary] = _probe_version(binary)
    return _version_cache[binary]


def is_h5ad_inspect_available() -> bool:
    """Return True if a *usable* ``h5ad-inspect`` is on PATH.

    Usable means present **and** at least
    :data:`MIN_H5AD_INSPECT_VERSION`.  A 0.1.x binary does not provide the
    interface this module speaks, so it reports False — callers using this as
    a feature test then correctly fall back to loading the file with
    ``anndata`` instead of failing part-way through a plot.
    """
    version = _h5ad_inspect_version()
    return version is not None and version >= MIN_H5AD_INSPECT_VERSION


def _require_h5ad_inspect() -> None:
    """Raise a descriptive RuntimeError unless a usable h5ad-inspect is present."""
    minimum = ".".join(str(part) for part in MIN_H5AD_INSPECT_VERSION)
    binary = shutil.which("h5ad-inspect")
    if binary is None:
        raise RuntimeError(
            "h5ad-inspect is not available on PATH.\n"
            "\n"
            "h5ad-inspect is required to load .h5ad files by filename.\n"
            f"{_INSTALL_HINT}"
        )

    version = _h5ad_inspect_version()
    if version is None:
        raise RuntimeError(
            f"h5ad-inspect at {binary} does not report a version, so it is "
            f"older than the required {minimum} (which is where --version was "
            "added).\n"
            "\n"
            "0.2.0 replaced the old `export <thing>` commands with verbs "
            "(list / describe / get / write); this library only speaks the "
            "newer interface.\n"
            f"{_INSTALL_HINT}"
        )
    if version < MIN_H5AD_INSPECT_VERSION:
        found = ".".join(str(part) for part in version)
        raise RuntimeError(
            f"h5ad-inspect at {binary} is version {found}, but at least "
            f"{minimum} is required.\n"
            "\n"
            "0.2.0 replaced the old `export <thing>` commands with verbs "
            "(list / describe / get / write); this library only speaks the "
            "newer interface.\n"
            f"{_INSTALL_HINT}"
        )


# ── low-level helpers ─────────────────────────────────────────────────────────


def _run_inspect(path: Path, *args: str, flags: Sequence[str] = ()) -> bytes:
    """Run ``h5ad-inspect <flags…> -- <path> <args…>`` and return stdout bytes.

    h5ad-inspect 0.2 accepts flags either after the positionals or, as used
    here, before a ``--`` separator.  The separator form is the robust one:
    everything after ``--`` is a positional even when it starts with a dash,
    so a file, column or gene whose name looks like a flag is still reachable.
    """
    argv = ["h5ad-inspect", *flags, "--", str(path), *args]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            check=True,
            timeout=H5AD_TIMEOUT,  # if it's that slow, something is seriously wrong. Best case your file is CSR and gigantic
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"h5ad-inspect failed on {path!r} with args {args!r} "
            f"and flags {tuple(flags)!r}:\n"
            f"stdout:\n{e.stdout.decode()}\n"
            f"stderr:\n{e.stderr.decode()}"
        ) from e
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"h5ad-inspect timed out on {path!r} with args {args!r}:\n"
            f"mbf_singlecell_plotter.h5ad_source.H5AD_TIMEOUT was set to {H5AD_TIMEOUT}.\n"
            f"Increase or fix the underlying perfomance problem"
        )


def _run_lines(path: Path, *args: str, flags: Sequence[str] = ()) -> list[str]:
    """Run h5ad-inspect and return a list of non-empty output lines."""
    raw = _run_inspect(path, *args, flags=flags).decode().strip()
    return [line for line in raw.split("\n") if line] if raw else []


def _col_encoding(path: Path, group: str, key: str) -> tuple[str, Optional[list[str]]]:
    """Return ``(encoding, categories)`` for an obs/var column.

    Uses ``h5ad-inspect describe column <group> <key>``, which emits a JSON
    object such as ``{"encoding":"categorical","categories":[...]}``,
    ``{"encoding":"bool"}``, or ``{"encoding":"numeric"}``.  A column that is
    not in the file is an error there, but callers reach this only after
    ``get column`` has already succeeded for the same key.

    ``encoding`` is one of ``'categorical'``, ``'bool'``, ``'numeric'``;
    ``categories`` is the ordered category list for categorical columns,
    otherwise ``None``.
    """
    raw = _run_inspect(path, "describe", "column", group, key).decode().strip()
    if not raw:
        return "numeric", None
    info = json.loads(raw)
    return info.get("encoding", "numeric"), info.get("categories")


def _parse_series(
    lines: list[str], name: str, index: pd.Index, encoding: str, categories
) -> pd.Series:
    """Parse text lines from a ``get`` subcommand into a typed Series."""
    if not lines:
        return pd.Series([], index=index, name=name, dtype=object)

    if encoding == "categorical":
        return pd.Series(pd.Categorical(lines, categories), index=index, name=name)

    if encoding == "bool":
        lower = [ln.lower() for ln in lines]
        return pd.Series([ln == "true" for ln in lower], index=index, name=name)

    # numeric (float/int) — fall back to strings when the column isn't numeric
    try:
        return pd.Series(pd.to_numeric(lines), index=index, name=name)
    except (ValueError, TypeError):
        return pd.Series(lines, index=index, name=name, dtype=object)


def _read_obsm(path: Path, key: str, n_cells: int) -> np.ndarray:
    """Read an obsm entry via ``h5ad-inspect get embedding --format binary``.

    The binary stream is little-endian float64, row-major
    (n_cells × n_components); we reshape using ``n_cells`` (the known
    dimension) so callers receive a 2-D array.
    """
    raw = _run_inspect(path, "get", "embedding", key, flags=("--format", "binary"))
    arr = np.frombuffer(raw, dtype="<f8").copy()
    return arr.reshape(n_cells, -1)


def _layer_flags(layer: str) -> tuple[()] | tuple[str, str]:
    """Return the ``--layer`` CLI flag for *layer*, empty for the default ``'X'``.

    ``h5ad-inspect`` reads ``.X`` by default and any named layer via
    ``--layer <key>`` (which resolves to ``layers/<key>`` in the file).  The
    sentinel ``'X'`` therefore maps to *no* flag rather than ``--layer X`` —
    matching how ``AnnData`` treats ``'X'`` as ``.X`` rather than
    ``.layers['X']``.
    """
    return () if layer in (None, "X") else ("--layer", layer)


def _load_matrix_csr(path: Path, layer: str = "X"):
    """Load the full matrix for *layer* as a scipy CSR sparse matrix in one shot.

    Uses ``h5ad-inspect write npz_csr [--layer <key>]``, which streams a NumPy
    ``.npz`` archive (``csr_data`` / ``csr_indices`` / ``csr_indptr`` /
    ``csr_shape``) to stdout.  The resulting columns are in the file's native
    gene order, i.e. the same order as :meth:`H5adFacade.var_names`
    (``get index var``) — which mirrors real ``AnnData``.  *layer* defaults
    to ``'X'`` (the ``.X`` matrix); any other key reads ``layers/<key>``.
    """
    from scipy import sparse as sp

    raw = _run_inspect(path, "write", "npz_csr", flags=_layer_flags(layer))
    parts = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            key = name[:-4] if name.endswith(".npy") else name
            parts[key] = np.load(io.BytesIO(z.read(name)))
    return sp.csr_matrix(
        (parts["csr_data"], parts["csr_indices"], parts["csr_indptr"]),
        shape=tuple(int(v) for v in parts["csr_shape"]),
    )


# ── AnnData-compatible facade classes ─────────────────────────────────────────


class _ColProxy:
    """Shared base for _ObsProxy and _VarProxy — lazy column fetching."""

    def __init__(self, path: Path, h5_group: str, row_index: pd.Index) -> None:
        self._path = path
        self._h5_group = h5_group  # "obs" or "var"
        self._index = row_index
        self._available: Optional[set[str]] = None
        self._cache: dict[str, pd.Series] = {}
        self._shape: Optional[tuple[int, int]] = None

    def _available_columns(self) -> set[str]:
        if self._available is None:
            self._available = set(_run_lines(self._path, "list", self._h5_group))
        return self._available

    @property
    def columns(self) -> pd.Index:
        return pd.Index(sorted(self._available_columns()))

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape is None:
            s = _run_lines(self._path, "describe", "shape", self._h5_group)
            n_rows = None
            n_columns = None
            for line in s:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if parts[0] == f"n_{self._h5_group}":
                        n_rows = int(parts[1])
                    elif parts[0] == "n_columns":
                        n_columns = int(parts[1])
                    else:
                        raise ValueError(
                            f"Unexpected result in describe shape "
                            f"{self._h5_group} call: {s}"
                        )
            if n_rows is None or n_columns is None:
                raise ValueError(
                    f"describe shape {self._h5_group} call did not contain "
                    f"expected keys: {s}"
                )
            self._shape = (n_rows, n_columns)
        return self._shape

    def __contains__(self, key: str) -> bool:
        return key in self._available_columns()

    def __getitem__(self, key: str) -> pd.Series:
        if key not in self._cache:
            lines = _run_lines(self._path, "get", "column", self._h5_group, key)
            encoding, categories = _col_encoding(self._path, self._h5_group, key)
            self._cache[key] = _parse_series(
                lines, key, self._index, encoding, categories
            )
        return self._cache[key]

    def to_df(self):
        """Convert into a true pandas Dataframe"""
        return pd.DataFrame({col: self[col] for col in self.columns})


class _ObsProxy(_ColProxy):
    """Mimics ``AnnData.obs`` — lazily fetches obs columns via h5ad-inspect."""

    def __init__(self, path: Path, obs_names: pd.Index) -> None:
        super().__init__(path, "obs", obs_names)

    @property
    def index(self) -> pd.Index:
        return self._index


class _VarProxy(_ColProxy):
    """Mimics ``AnnData.var`` — lazily fetches var columns via h5ad-inspect."""

    def __init__(self, path: Path, var_index: pd.Index) -> None:
        super().__init__(path, "var", var_index)

    @property
    def index(self) -> pd.Index:
        return self._index


class _ObsmProxy:
    """Mimics ``AnnData.obsm`` — reads embedding arrays via ``get embedding``."""

    def __init__(self, path: Path, n_cells: int) -> None:
        self._path = path
        self._n_cells = n_cells
        self._keys_list: Optional[list[str]] = None
        self._cache: dict[str, NDArray[Any]] = {}

    def _list_keys(self) -> list[str]:
        if self._keys_list is None:
            self._keys_list = _run_lines(self._path, "list", "obsm")
        return self._keys_list

    def keys(self) -> list[str]:
        return self._list_keys()

    def __contains__(self, key: str) -> bool:
        return key in self._list_keys()

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            if key not in self:
                raise KeyError(f"obsm key {key!r} not found in {self._path!r}")
            self._cache[key] = _read_obsm(self._path, key, self._n_cells)
        return self._cache[key]


class _XProxy:
    """Mimics ``AnnData.X`` / ``AnnData.layers[key]`` for single-gene column access.

    Bound to a single matrix *layer* (``'X'`` for ``.X``, any other key for a
    named layer).  Only ``X[:, int]`` and ``X[:, str]`` are supported here —
    they fetch one gene column lazily via
    ``get vector var <gene> --format binary [--layer <key>]`` and cache it.
    ``get vector var`` is the matrix column for one gene, i.e. one value per
    cell.  This keeps the common "colour by one gene" plotting path cheap (a
    single small subprocess payload).

    For bulk access over many genes (e.g. Moran's I, which needs the whole
    matrix row-sliced per bin) use :meth:`H5adFacade.get_X_csr`, which loads
    the full matrix in one shot via ``write npz_csr``.
    """

    def __init__(self, path: Path, var_names: pd.Index, layer: str = "X") -> None:
        self._path = path
        self._var_names = var_names
        self._layer = layer
        self._col_cache: dict[str, NDArray] = {}

    def _column(self, gene: str) -> np.ndarray:
        """Full ``(n_cells,)`` expression vector for one gene, cached by name."""
        if gene not in self._col_cache:
            raw = _run_inspect(
                self._path,
                "get",
                "vector",
                "var",
                gene,
                flags=("--format", "binary", *_layer_flags(self._layer)),
            )
            self._col_cache[gene] = np.frombuffer(raw, dtype="<f8").copy()
        return self._col_cache[gene]

    def __getitem__(self, idx):
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and idx[0] == slice(None)
            and isinstance(idx[1], (int, np.integer))
        ):
            return self._column(str(self._var_names[idx[1]]))
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and idx[0] == slice(None)
            and isinstance(idx[1], str)
        ):
            return self._column(idx[1])
        raise NotImplementedError(
            "H5adFacade.X only supports [:, int] / [:, str] single-column "
            f"indexing; got {idx!r}. For bulk/multi-gene access (e.g. Moran's "
            "I) use H5adFacade.get_X_csr()."
        )


# ── main facade ───────────────────────────────────────────────────────────────


class H5adFacade:
    """
    Minimal AnnData-compatible object backed by h5ad-inspect.

    Implements the subset of the AnnData interface used by EmbeddingData:

    * ``obs_names`` / ``var_names``  — cell / gene indices
    * ``obs``                        — lazy obs-column access (numeric, bool, categorical)
    * ``var``                        — lazy var-column access + gene index
    * ``obsm``                       — embedding arrays via h5ad-inspect ``--format binary``
    * ``X``                          — gene expression via h5ad-inspect ``--format binary``
    """

    def __init__(self, path: Path) -> None:
        _require_h5ad_inspect()
        self._path = Path(path).resolve()
        self._obs_names: Optional[pd.Index] = None
        self._var_names: Optional[pd.Index] = None
        self._shape: Optional[tuple[int, int]] = None
        self._obs: Optional[_ObsProxy] = None
        self._var: Optional[_VarProxy] = None
        self._obsm_proxy: Optional[_ObsmProxy] = None
        self._matrix_proxies: dict[str, _XProxy] = {}  # layer key → _XProxy
        self._x_csr_cache: dict[str, Any] = {}  # layer key → CSR matrix

    @property
    def obs_names(self) -> pd.Index:
        if self._obs_names is None:
            self._obs_names = pd.Index(_run_lines(self._path, "get", "index", "obs"))
        return self._obs_names

    @property
    def var_names(self) -> pd.Index:
        # ``get index var`` returns the gene IDs in matrix order — the same
        # order the ``X`` matrix columns and ``var`` columns are stored in, and
        # the order a real ``AnnData`` exposes as ``var.index``.
        try:
            if self._var_names is None:
                self._var_names = pd.Index(
                    _run_lines(self._path, "get", "index", "var")
                )
        except RuntimeError:
            # A file without a var index at all: tolerate it only when var
            # carries no columns either, otherwise the misalignment is real.
            var_columns = _run_lines(self._path, "list", "var")
            if len(var_columns) == 0:
                self._var_names = pd.Index([])
            else:
                raise
        return self._var_names

    @property
    def obs(self) -> _ObsProxy:
        if self._obs is None:
            self._obs = _ObsProxy(self._path, self.obs_names)
        return self._obs

    @property
    def var(self) -> _VarProxy:
        if self._var is None:
            self._var = _VarProxy(self._path, self.var_names)
        return self._var

    @property
    def obsm(self) -> _ObsmProxy:
        if self._obsm_proxy is None:
            self._obsm_proxy = _ObsmProxy(self._path, len(self.obs_names))
        return self._obsm_proxy

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape is None:
            s = _run_lines(self._path, "describe", "shape")
            n_obs = None
            n_var = None
            for line in s:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if parts[0] == "n_obs":
                        n_obs = int(parts[1])
                    elif parts[0] == "n_var":
                        n_var = int(parts[1])
                    else:
                        raise ValueError(
                            f"Unexpected result in describe shape call: {s}"
                        )
            if n_obs is not None and n_var is not None:
                self._shape = (n_obs, n_var)
            else:
                raise ValueError(
                    f"describe shape call did not contain n_obs or n_var: {s}"
                )

        return self._shape

    @property
    def n_obs(self) -> int:
        return self.shape[0]

    def n_var(self) -> int:
        return self.shape[1]

    def matrix(self, layer: str = "X") -> _XProxy:
        """Return a single-column accessor for *layer* (``'X'`` → ``.X``).

        The proxy is cached per layer so repeated column reads from the same
        layer share one cache.  ``layer='X'`` reads ``.X``; any other key reads
        the named ``layers/<key>`` matrix.
        """
        if layer is None:
            layer = "X"
        proxy = self._matrix_proxies.get(layer)
        if proxy is None:
            proxy = _XProxy(self._path, self.var_names, layer)
            self._matrix_proxies[layer] = proxy
        return proxy

    @property
    def X(self) -> _XProxy:
        return self.matrix("X")

    def get_X_csr(self, layer: str = "X"):
        """Return the full matrix for *layer* as a scipy CSR sparse matrix.

        Loaded once per layer via ``h5ad-inspect write npz_csr [--layer
        <key>]`` and cached.  CSR is row-major, so ``X[row_array]`` slicing
        (the access pattern used by :func:`compute_grid_moran`) is cheap.
        Columns follow :attr:`var_names`.  *layer* defaults to ``'X'``.
        """
        if layer is None:
            layer = "X"
        if layer not in self._x_csr_cache:
            self._x_csr_cache[layer] = _load_matrix_csr(self._path, layer)
        return self._x_csr_cache[layer]
