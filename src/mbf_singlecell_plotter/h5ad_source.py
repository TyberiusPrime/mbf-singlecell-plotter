"""h5ad-inspect backed data source — fast selective reads from .h5ad files."""

import shutil
import subprocess
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


def _run_inspect(path: str, *args: str) -> bytes:
    """Run ``h5ad-inspect <path> <args…>`` and return stdout bytes."""
    result = subprocess.run(
        ["h5ad-inspect", path, *args],
        capture_output=True,
        check=True,
    )
    return result.stdout


def _run_lines(path: str, *args: str) -> list:
    """Run h5ad-inspect and return a list of non-empty output lines."""
    raw = _run_inspect(path, *args).decode().strip()
    return [line for line in raw.split("\n") if line] if raw else []


def _read_obsm(path: Path, key: str) -> np.ndarray:
    """Read an obsm entry directly from the .h5ad file via h5py."""
    import h5py

    with h5py.File(path, "r") as f:
        if "obsm" not in f:
            raise KeyError(f"No 'obsm' group in {path}")
        obsm = f["obsm"]
        if key not in obsm:
            raise KeyError(f"obsm key {key!r} not found in {path!r}")
        item = obsm[key]
        if isinstance(item, h5py.Dataset):
            return np.array(item)
        elif isinstance(item, h5py.Group):
            # DataFrame stored as a group — honour declared column order
            col_order = list(
                item.attrs.get("column-order", sorted(item.keys()))
            )
            return np.column_stack([np.array(item[c]) for c in col_order])
    raise KeyError(f"obsm key {key!r} not found")  # pragma: no cover


def _obs_col_encoding(h5_path: Path, key: str) -> str:
    """Return the h5ad encoding type for an obs column: 'categorical', 'bool', or 'numeric'."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        obs_group = f.get("obs")
        if obs_group is None or key not in obs_group:
            return "numeric"
        item = obs_group[key]
        if isinstance(item, h5py.Group):
            enc = item.attrs.get("encoding-type", b"")
            if isinstance(enc, bytes):
                enc = enc.decode()
            if enc == "categorical":
                return "categorical"
        elif isinstance(item, h5py.Dataset):
            if item.dtype.kind == "b":
                return "bool"
    return "numeric"


def _parse_obs_series(
    lines: list, name: str, index: pd.Index, encoding: str
) -> pd.Series:
    """Parse text lines from ``export obs <col>`` using the known encoding type."""
    if not lines:
        return pd.Series([], index=index, name=name, dtype=object)

    if encoding == "categorical":
        return pd.Series(pd.Categorical(lines), index=index, name=name)

    if encoding == "bool":
        lower = [ln.lower() for ln in lines]
        return pd.Series(
            [ln == "true" for ln in lower], index=index, name=name
        )

    # numeric (float/int)
    values = pd.to_numeric(lines)
    return pd.Series(values, index=index, name=name)


# ── AnnData-compatible facade classes ─────────────────────────────────────────


class _ObsProxy:
    """Mimics ``AnnData.obs`` — lazily fetches columns via h5ad-inspect."""

    def __init__(self, path: str, h5_path: Path, obs_names: pd.Index) -> None:
        self._path = path
        self._h5_path = h5_path
        self._obs_names = obs_names
        self._available: Optional[set] = None
        self._cache: dict = {}

    def _available_columns(self) -> set:
        if self._available is None:
            self._available = set(_run_lines(self._path, "obs"))
        return self._available

    @property
    def index(self) -> pd.Index:
        return self._obs_names

    def __contains__(self, key: str) -> bool:
        return key in self._available_columns()

    def __getitem__(self, key: str) -> pd.Series:
        if key not in self._cache:
            lines = _run_lines(self._path, "export", "obs", key)
            encoding = _obs_col_encoding(self._h5_path, key)
            self._cache[key] = _parse_obs_series(lines, key, self._obs_names, encoding)
        return self._cache[key]


class _VarProxy:
    """Mimics ``AnnData.var`` — provides ``index`` and an empty ``columns``."""

    def __init__(self, var_index: pd.Index) -> None:
        self._index = var_index

    @property
    def index(self) -> pd.Index:
        return self._index

    @property
    def columns(self) -> pd.Index:
        # h5ad-inspect does not expose var columns — alternative_id_column
        # lookups are therefore skipped by EmbeddingData automatically.
        return pd.Index([])

    def __getitem__(self, key):  # pragma: no cover
        raise KeyError(f"var column {key!r} not available via h5ad-inspect")


class _ObsmProxy:
    """Mimics ``AnnData.obsm`` — reads arrays via h5py."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._keys_list: Optional[list] = None
        self._cache: dict = {}

    def _list_keys(self) -> list:
        if self._keys_list is None:
            self._keys_list = _run_lines(str(self._path), "obsm")
        return self._keys_list

    def keys(self) -> list:
        return self._list_keys()

    def __contains__(self, key: str) -> bool:
        return key in self._list_keys()

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = _read_obsm(self._path, key)
        return self._cache[key]


class _XProxy:
    """Mimics ``AnnData.X`` — fetches gene-expression columns via ``--binary``."""

    def __init__(self, path: str, var_names: pd.Index) -> None:
        self._path = path
        self._var_names = var_names

    def __getitem__(self, idx):
        # EmbeddingData.get_column calls ad.X[:, int_index]
        if isinstance(idx, tuple) and len(idx) == 2:
            _rows, col = idx
            if isinstance(col, (int, np.integer)):
                gene = str(self._var_names[col])
                raw = _run_inspect(
                    self._path, "export", "--binary", "column", gene
                )
                # little-endian float64 bytes → numpy array
                return np.frombuffer(raw, dtype="<f8").copy()
        raise NotImplementedError(
            f"H5adFacade.X only supports [:, int] indexing; got {idx!r}"
        )


# ── main facade ───────────────────────────────────────────────────────────────


class _H5adFacade:
    """
    Minimal AnnData-compatible object backed by h5ad-inspect and h5py.

    Implements the subset of the AnnData interface used by EmbeddingData:

    * ``obs_names`` / ``var_names``  — cell / gene indices
    * ``obs``                        — lazy column access (numeric, bool, categorical)
    * ``var``                        — gene index + empty columns
    * ``obsm``                       — embedding arrays via h5py
    * ``X``                          — gene expression via h5ad-inspect ``--binary``
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._str_path = str(self._path)
        self._obs_names: Optional[pd.Index] = None
        self._var_names: Optional[pd.Index] = None
        self._obs: Optional[_ObsProxy] = None
        self._var: Optional[_VarProxy] = None
        self._obsm_proxy: Optional[_ObsmProxy] = None
        self._x_proxy: Optional[_XProxy] = None

    @property
    def obs_names(self) -> pd.Index:
        if self._obs_names is None:
            lines = _run_lines(self._str_path, "export", "obs_index")
            self._obs_names = pd.Index(lines)
        return self._obs_names

    @property
    def var_names(self) -> pd.Index:
        if self._var_names is None:
            lines = _run_lines(self._str_path, "export", "var_index")
            self._var_names = pd.Index(lines)
        return self._var_names

    @property
    def obs(self) -> _ObsProxy:
        if self._obs is None:
            self._obs = _ObsProxy(self._str_path, self._path, self.obs_names)
        return self._obs

    @property
    def var(self) -> _VarProxy:
        if self._var is None:
            self._var = _VarProxy(self.var_names)
        return self._var

    @property
    def obsm(self) -> _ObsmProxy:
        if self._obsm_proxy is None:
            self._obsm_proxy = _ObsmProxy(self._path)
        return self._obsm_proxy

    @property
    def X(self) -> _XProxy:
        if self._x_proxy is None:
            self._x_proxy = _XProxy(self._str_path, self.var_names)
        return self._x_proxy
