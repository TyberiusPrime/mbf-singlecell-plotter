"""h5ad-inspect backed data source — fast selective reads from .h5ad files."""

import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── availability ─────────────────────────────────────────────────────────────


def is_h5ad_inspect_available() -> bool:
    """Return True if the ``h5ad-inspect`` binary is found on PATH."""
    return shutil.which("h5ad-inspect") is not None


def _require_h5ad_inspect() -> None:
    """Raise a descriptive RuntimeError when h5ad-inspect is not on PATH."""
    if not is_h5ad_inspect_available():
        raise RuntimeError(
            "h5ad-inspect is not available on PATH.\n"
            "\n"
            "h5ad-inspect is required to load .h5ad files by filename.\n"
            "Installation options:\n"
            "  • Nix devShell: add h5ad_inspect.packages.${system}.h5ad-inspect to packages\n"
            "  • Cargo:        cargo install --git "
            "https://github.com/TyberiusPrime/h5ad_inspect\n"
        )


# ── low-level helpers ─────────────────────────────────────────────────────────


def _run_inspect(path: Path, *args: str) -> bytes:
    """Run ``h5ad-inspect <path> <args…>`` and return stdout bytes."""
    result = subprocess.run(
        ["h5ad-inspect", path, *args],
        capture_output=True,
        check=True,
    )
    return result.stdout


def _run_lines(path: Path, *args: str) -> list:
    """Run h5ad-inspect and return a list of non-empty output lines."""
    raw = _run_inspect(path, *args).decode().strip()
    return [line for line in raw.split("\n") if line] if raw else []


def _col_encoding(path: Path, group: str, key: str) -> tuple:
    """Return ``(encoding, categories)`` for an obs/var column.

    Uses ``h5ad-inspect export <group>_encoding <key>``, which emits a JSON
    object such as ``{"encoding":"categorical","categories":[...]}``,
    ``{"encoding":"bool"}``, or ``{"encoding":"numeric"}`` (a missing column
    also resolves to ``numeric``).

    ``encoding`` is one of ``'categorical'``, ``'bool'``, ``'numeric'``;
    ``categories`` is the ordered category list for categorical columns,
    otherwise ``None``.
    """
    raw = _run_inspect(path, "export", f"{group}_encoding", key).decode().strip()
    if not raw:
        return "numeric", None
    info = json.loads(raw)
    return info.get("encoding", "numeric"), info.get("categories")


def _parse_series(
    lines: list, name: str, index: pd.Index, encoding: str, categories
) -> pd.Series:
    """Parse text lines from an ``export`` subcommand into a typed Series."""
    if not lines:
        return pd.Series([], index=index, name=name, dtype=object)

    if encoding == "categorical":
        return pd.Series(
            pd.Categorical(lines, categories), index=index, name=name
        )

    if encoding == "bool":
        lower = [ln.lower() for ln in lines]
        return pd.Series([ln == "true" for ln in lower], index=index, name=name)

    # numeric (float/int) — fall back to strings when the column isn't numeric
    try:
        return pd.Series(pd.to_numeric(lines), index=index, name=name)
    except (ValueError, TypeError):
        return pd.Series(lines, index=index, name=name, dtype=object)


def _read_obsm(path: Path, key: str, n_cells: int) -> np.ndarray:
    """Read an obsm entry via ``h5ad-inspect export --binary obsm``.

    The binary stream is little-endian float64, row-major
    (n_cells × n_components); we reshape using ``n_cells`` (the known
    dimension) so callers receive a 2-D array.
    """
    raw = _run_inspect(path, "export", "--binary", "obsm", key)
    arr = np.frombuffer(raw, dtype="<f8").copy()
    return arr.reshape(n_cells, -1)


def _layer_args(layer: str) -> tuple:
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

    Uses ``h5ad-inspect export [--layer <key>] matrix_csr``, which emits a zip
    archive of numpy arrays (``csr_data`` / ``csr_indices`` / ``csr_indptr`` /
    ``csr_shape``).  The resulting columns are in the file's native gene
    order, i.e. the same order as :meth:`H5adFacade.var_names`
    (``export var _index``) — which mirrors real ``AnnData``.  *layer* defaults
    to ``'X'`` (the ``.X`` matrix); any other key reads ``layers/<key>``.
    """
    from scipy import sparse as sp

    raw = _run_inspect(path, "export", *_layer_args(layer), "matrix_csr")
    parts: dict = {}
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
        self._available: Optional[set] = None
        self._cache: dict = {}

    def _available_columns(self) -> set:
        if self._available is None:
            self._available = set(_run_lines(self._path, self._h5_group))
        return self._available

    @property
    def columns(self) -> pd.Index:
        return pd.Index(sorted(self._available_columns()))

    def __contains__(self, key: str) -> bool:
        return key in self._available_columns()

    def __getitem__(self, key: str) -> pd.Series:
        if key not in self._cache:
            lines = _run_lines(self._path, "export", self._h5_group, key)
            encoding, categories = _col_encoding(self._path, self._h5_group, key)
            self._cache[key] = _parse_series(
                lines, key, self._index, encoding, categories
            )
        return self._cache[key]


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
    """Mimics ``AnnData.obsm`` — reads embedding arrays via h5ad-inspect ``--binary``."""

    def __init__(self, path: Path, n_cells: int) -> None:
        self._path = path
        self._n_cells = n_cells
        self._keys_list: Optional[list] = None
        self._cache: dict = {}

    def _list_keys(self) -> list:
        if self._keys_list is None:
            self._keys_list = _run_lines(self._path, "obsm")
        return self._keys_list

    def keys(self) -> list:
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
    ``export --binary [--layer <key>] column <gene>`` and cache it.  This keeps
    the common "colour by one gene" plotting path cheap (a single small
    subprocess payload).

    For bulk access over many genes (e.g. Moran's I, which needs the whole
    matrix row-sliced per bin) use :meth:`H5adFacade.get_X_csr`, which loads
    the full matrix in one shot via ``export matrix_csr``.
    """

    def __init__(self, path: Path, var_names: pd.Index, layer: str = "X") -> None:
        self._path = path
        self._var_names = var_names
        self._layer = layer
        self._col_cache: dict = {}

    def _column(self, gene: str) -> np.ndarray:
        """Full ``(n_cells,)`` expression vector for one gene, cached by name."""
        if gene not in self._col_cache:
            raw = _run_inspect(
                self._path, "export", "--binary", *_layer_args(self._layer),
                "column", gene,
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
    * ``obsm``                       — embedding arrays via h5ad-inspect ``--binary``
    * ``X``                          — gene expression via h5ad-inspect ``--binary``
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path).resolve()
        self._obs_names: Optional[pd.Index] = None
        self._var_names: Optional[pd.Index] = None
        self._obs: Optional[_ObsProxy] = None
        self._var: Optional[_VarProxy] = None
        self._obsm_proxy: Optional[_ObsmProxy] = None
        self._matrix_proxies: dict = {}  # layer key → _XProxy
        self._x_csr_cache: dict = {}  # layer key → CSR matrix

    @property
    def obs_names(self) -> pd.Index:
        if self._obs_names is None:
            self._obs_names = pd.Index(_run_lines(self._path, "export", "obs_index"))
        return self._obs_names

    @property
    def var_names(self) -> pd.Index:
        # ``export var _index`` returns the file's native gene order — the same
        # order the ``X`` matrix columns and ``var`` columns are stored in, and
        # the order a real ``AnnData`` exposes as ``var.index``.  (``export
        # var_index`` instead returns the names *sorted*, which would misalign
        # ``X`` and ``var`` against the indices.)
        if self._var_names is None:
            self._var_names = pd.Index(
                _run_lines(self._path, "export", "var", "_index")
            )
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

        Loaded once per layer via ``h5ad-inspect export [--layer <key>]
        matrix_csr`` and cached.  CSR is row-major, so ``X[row_array]`` slicing
        (the access pattern used by :func:`compute_grid_moran`) is cheap.
        Columns follow :attr:`var_names`.  *layer* defaults to ``'X'``.
        """
        if layer is None:
            layer = "X"
        if layer not in self._x_csr_cache:
            self._x_csr_cache[layer] = _load_matrix_csr(self._path, layer)
        return self._x_csr_cache[layer]
