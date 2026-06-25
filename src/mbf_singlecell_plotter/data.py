"""Layer 1: data access (AnnData → DataFrames). No plotting."""

import copy
import collections
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from natsort import natsorted

from .util import map_to_integers, unmap

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _parse_grid_label(label: str, gs: int, letters_on_vertical: bool) -> tuple:
    """Parse a grid label string → (col_idx, row_from_top), both 0-indexed.

    Default orientation (letters_on_vertical=False): format '{letter}{number}',
    e.g. 'G3' — letter is the column (A=0), number is the row (1=top).

    Vertical-letters orientation (letters_on_vertical=True): format '{number}{letter}',
    e.g. '3G' — number is the column (1=0), letter is the row (A=top).

    Raises ValueError with a descriptive message on invalid input.
    """
    valid_letters = _LETTERS[:gs]
    s = label.strip()
    if not letters_on_vertical:
        # expect: one letter then 1-2 digits
        if len(s) < 2 or not s[0].isalpha() or not s[1:].isdigit():
            raise ValueError(
                f"grid label must be letter+number, e.g. 'A1' (grid_size={gs}), got {label!r}"
            )
        letter, number = s[0].upper(), int(s[1:])
        col_idx = _LETTERS.index(letter) if letter in valid_letters else -1
        if col_idx < 0:
            raise ValueError(
                f"grid label column must be A..{valid_letters[-1]} (grid_size={gs}), got {letter!r}"
            )
        if not (1 <= number <= gs):
            raise ValueError(
                f"grid label row must be 1..{gs} (grid_size={gs}), got {number}"
            )
        return col_idx, number - 1
    else:
        # expect: 1-2 digits then one letter
        if len(s) < 2 or not s[-1].isalpha() or not s[:-1].isdigit():
            raise ValueError(
                f"grid label must be number+letter, e.g. '1A' (grid_size={gs}), got {label!r}"
            )
        letter, number = s[-1].upper(), int(s[:-1])
        if letter not in valid_letters:
            raise ValueError(
                f"grid label row must be A..{valid_letters[-1]} (grid_size={gs}), got {letter!r}"
            )
        if not (1 <= number <= gs):
            raise ValueError(
                f"grid label column must be 1..{gs} (grid_size={gs}), got {number}"
            )
        col_idx = number - 1
        row_from_top = _LETTERS.index(letter)
        return col_idx, row_from_top


class ColumnData(NamedTuple):
    """Return type of :meth:`EmbeddingData.get_column`."""

    series: pd.Series
    name: str


class AlternativeSource(NamedTuple):
    """A registered fallback source for :meth:`EmbeddingData.get_column`.

    ``name`` is ``None`` for sources that only participate in the automatic
    fallback search; a non-None name additionally enables explicit tuple
    routing via ``get_column((name, column))``.
    """

    name: Optional[str]
    ad: object


class DerivedSource(NamedTuple):
    """A source whose columns are computed on demand from other sources.

    ``columns`` maps a column name to a callable that receives the owning
    :class:`EmbeddingData` and returns a :class:`pandas.Series` indexed by the
    primary ``obs_names`` (so it can pull from the primary source or any
    registered alternative via ``get_column(...)`` and combine the results).
    Like :class:`AlternativeSource`, a non-None ``name`` additionally enables
    explicit tuple routing via ``get_column((name, column))``.
    """

    name: Optional[str]
    columns: Dict[str, Callable[["EmbeddingData"], "pd.Series"]]


class EmbeddingData:
    """Wraps an AnnData + embedding choice. Pure data extraction, no plotting."""

    def __init__(
        self,
        ad,
        embedding,
        alternative_id_column: Optional[str] = None,
        alternative_sources: Optional[list] = None,
        derived_sources: Optional[list] = None,
        grid_size: int = 12,
        grid_letters_on_vertical: bool = False,
        filter_fn: Optional[Callable[["EmbeddingData"], "pd.Series | np.ndarray"]] = None,
    ):
        self.ad = ad
        if grid_size > 26:
            raise ValueError("grid_size max is 26")
        self._grid_size = grid_size
        self._grid_letters_on_vertical = grid_letters_on_vertical
        self._alternative_id_column = alternative_id_column
        self._has_name_and_id = ad.var.index.str.contains(" ").any()
        self._alternative_sources = self._normalize_alternative_items(
            alternative_sources or []
        )
        # derived names share the namespace with alternative names
        alt_names = {a.name for a in self._alternative_sources if a.name is not None}
        self._derived_sources = self._normalize_derived_items(
            derived_sources or [], existing_names=alt_names
        )

        # Resolve embedding — check tuple BEFORE string concatenation
        if isinstance(embedding, tuple):
            if len(embedding) != 3:
                raise ValueError(
                    "Tuple embedding must be ('key', col1, col2), e.g. ('pca', 0, 1)"
                )
            key_raw, c1, c2 = embedding
            if key_raw in ad.obsm:
                key = key_raw
            elif "X_" + key_raw in ad.obsm:
                key = "X_" + key_raw
            else:
                raise KeyError(
                    f"Embedding {key_raw!r} not found in ad.obsm. Available: "
                    + ", ".join(sorted(ad.obsm.keys()))
                )
            self._embedding = key
            self._embedding_cols: Optional[tuple] = (c1, c2)
        elif isinstance(embedding, str):
            if embedding in ad.obsm:
                self._embedding = embedding
            elif "X_" + embedding in ad.obsm:
                self._embedding = "X_" + embedding
            else:
                raise KeyError(
                    f"Embedding {embedding!r} not found in ad.obsm. Available: "
                    + ", ".join(sorted(ad.obsm.keys()))
                )
            self._embedding_cols = None
        else:
            raise ValueError(
                f"embedding must be a string or 3-tuple, got {type(embedding)}"
            )
        self._focus: Optional[tuple] = None  # (x_min, x_max, y_min, y_max)
        self._filter: Optional[
            Callable[["EmbeddingData"], "pd.Series | np.ndarray"]
        ] = filter_fn
        self._filter_cache: Optional[np.ndarray] = None

    @property
    def embedding(self) -> str:
        return self._embedding

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def has_focus(self) -> bool:
        return self._focus is not None

    @property
    def alternative_sources(self) -> list:
        """List of :class:`AlternativeSource` fallbacks consulted by :meth:`get_column`."""
        return list(self._alternative_sources)

    @property
    def derived_sources(self) -> list:
        """List of :class:`DerivedSource` computed-column sources consulted by :meth:`get_column`."""
        return list(self._derived_sources)

    @staticmethod
    def _coerce_ad(source):
        """Normalise an alternative-source argument to an AnnData-like object.

        Accepts an ``AnnData``, an :class:`H5adFacade`, a path (``str``/
        ``pathlib.Path``) to an ``.h5ad`` file, or another ``EmbeddingData``
        (its wrapped ``.ad`` is reused).  Paths require ``h5ad-inspect`` on
        ``PATH`` and are wrapped in :class:`H5adFacade` for lazy, on-demand
        column reads.
        """
        from .h5ad_source import _require_h5ad_inspect, H5adFacade

        if isinstance(source, dict):
            raise TypeError(
                "An alternative source must be an AnnData, H5adFacade, .h5ad "
                "path, or EmbeddingData — got a dict. Use add_derived_source() "
                "to register computed {column: callable} columns."
            )
        if isinstance(source, EmbeddingData):
            return source.ad
        if isinstance(source, (str, Path)):
            _require_h5ad_inspect()
            return H5adFacade(Path(source))
        # Assume AnnData or any AnnData-compatible facade (e.g. H5adFacade)
        return source

    def _normalize_alternative_items(self, items, existing_names=None):
        """Coerce a sequence of alternative-source specs into AlternativeSource records.

        Each item may be:

        * an :class:`AlternativeSource` (name + ad preserved)
        * a ``(name, source)`` 2-tuple (``name`` must be a ``str``)
        * a bare source (AnnData / H5adFacade / path / EmbeddingData) → unnamed

        Non-None names must be unique (checked against *existing_names* too);
        duplicates raise ``ValueError``.
        """
        seen = set(existing_names or [])
        result = []
        for item in items:
            if isinstance(item, AlternativeSource):
                name, src = item.name, item.ad
            elif (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
            ):
                name, src = item
            else:
                name, src = None, item
            if name is not None:
                if name in seen:
                    raise ValueError(
                        f"Duplicate alternative source name {name!r}"
                    )
                seen.add(name)
            result.append(AlternativeSource(name, self._coerce_ad(src)))
        return result

    def add_alternative_source(self, source, name=None) -> "EmbeddingData":
        """Return a copy with an additional fallback source appended.

        ``source`` may be an ``AnnData``, an :class:`H5adFacade`, an ``.h5ad``
        path, or another ``EmbeddingData``.  Sources are consulted in
        registration order when :meth:`get_column` cannot resolve a name in the
        primary source; the first hit wins and is reindexed onto the primary
        ``obs_names``.

        If *name* is given, the source can additionally be addressed
        explicitly via ``get_column((name, column))`` — which resolves
        *column* from that specific source only.  Names must be unique among
        registered alternatives.
        """
        existing = {a.name for a in self._alternative_sources if a.name is not None}
        item = (name, source) if name is not None else source
        new = copy.copy(self)
        new._alternative_sources = self._alternative_sources + (
            self._normalize_alternative_items([item], existing_names=existing)
        )
        return new

    def _normalize_derived_items(self, items, existing_names=None):
        """Coerce a sequence of derived-source specs into DerivedSource records.

        Each item may be:

        * a :class:`DerivedSource` (name + columns preserved)
        * a ``(name, columns_dict)`` 2-tuple (``name`` must be a ``str``)
        * a bare ``{column: callable}`` dict (→ unnamed)

        Every column value must be callable.  Non-None names must be unique
        (checked against *existing_names* too — the shared alternative/derived
        namespace); duplicates raise ``ValueError``.
        """
        seen = set(existing_names or [])
        result = []
        for item in items:
            if isinstance(item, DerivedSource):
                name, cols = item.name, dict(item.columns)
            elif (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], dict)
            ):
                name, cols = item[0], dict(item[1])
            elif isinstance(item, dict):
                name, cols = None, dict(item)
            else:
                raise TypeError(
                    "Derived source must be a {column: callable} dict, a "
                    f"(name, dict) tuple, or a DerivedSource; got {item!r}"
                )
            for col, fn in cols.items():
                if not callable(fn):
                    raise TypeError(
                        f"Derived column {col!r} must map to a callable, "
                        f"got {type(fn).__name__}"
                    )
            if name is not None:
                if name in seen:
                    raise ValueError(f"Duplicate source name {name!r}")
                seen.add(name)
            result.append(DerivedSource(name, cols))
        return result

    def add_derived_source(self, derived, name=None) -> "EmbeddingData":
        """Return a copy with an additional computed (derived) source appended.

        *derived* is a ``{column_name: callable}`` mapping.  Each callable
        receives this :class:`EmbeddingData` and must return a
        :class:`pandas.Series` indexed by the primary ``obs_names`` — so it can
        pull from the primary source or any registered alternative via
        ``get_column(...)`` and combine the results.  Columns are computed on
        demand, once per :meth:`get_column` call (no caching).

        Derived columns participate in :meth:`get_column` lookups two ways:

        * by plain string — checked after the primary source but *before*
          alternative sources (so an explicit derived column wins over an
          accidentally same-named column in a fallback source);
        * explicitly via ``get_column((name, column_name))`` when *name* is
          given.

        *name* must be unique among all alternative and derived sources.  The
        result is reindexed onto the primary ``obs_names`` for consistency with
        other sources.
        """
        existing = {a.name for a in self._alternative_sources if a.name is not None}
        existing |= {d.name for d in self._derived_sources if d.name is not None}
        item = (name, derived) if name is not None else derived
        new = copy.copy(self)
        new._derived_sources = self._derived_sources + (
            self._normalize_derived_items([item], existing_names=existing)
        )
        return new

    # ── viewport ────────────────────────────────────────────────────────────

    def focus_on(
        self,
        *args,
        x: tuple = None,
        y: tuple = None,
    ) -> "EmbeddingData":
        """Return a new EmbeddingData restricted to the given coordinate window.

        Accepts either two grid label strings::

            data.focus_on("A1", "C5")

        or explicit coordinate ranges (keyword-only)::

            data.focus_on(x=(x_min, x_max), y=(y_min, y_max))

        Grid labels use the same format as :meth:`grid_coordinate` (e.g. ``"G3"``
        for the default orientation, ``"3G"`` for vertical-letters).  Swapped
        corners are silently corrected.
        """
        if args:
            if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
                return self._focus_on_grid(args[0], args[1])
            raise TypeError(
                f"positional arguments must be two grid label strings "
                f"(e.g. focus_on('A1', 'C5')), got {args!r}"
            )
        new = copy.copy(self)
        new._focus = (x[0], x[1], y[0], y[1])
        return new

    def _focus_on_grid(self, cell_min: str, cell_max: str) -> "EmbeddingData":
        """Restrict viewport to the rectangle from cell_min (top-left) to cell_max (bottom-right).

        Internal implementation called by :meth:`focus_on` when given string arguments.
        Uses the same label format as grid_coordinate(): e.g. 'G3' for default
        orientation, '3G' for vertical-letters orientation.  The focus spans
        from the left/top edge of cell_min to the right/bottom edge of cell_max,
        resolved in the original (unfocused) coordinate space.
        """
        gs = self._grid_size
        glv = self._grid_letters_on_vertical
        col_min, row_min = _parse_grid_label(cell_min, gs, glv)
        col_max, row_max = _parse_grid_label(cell_max, gs, glv)

        if col_min > col_max:
            col_min, col_max = col_max, col_min
        if row_min > row_max:
            row_min, row_max = row_max, row_min

        x_min_d, x_max_d, y_min_d, y_max_d = self.full_bounds()
        cell_w = (x_max_d - x_min_d) / gs
        cell_h = (y_max_d - y_min_d) / gs

        return self.focus_on(
            x=(x_min_d + col_min * cell_w, x_min_d + (col_max + 1) * cell_w),
            y=(y_max_d - (row_max + 1) * cell_h, y_max_d - row_min * cell_h),
        )

    def unfocus(self) -> "EmbeddingData":
        """Return a new EmbeddingData with no focus restriction."""
        new = copy.copy(self)
        new._focus = None
        return new

    # ── cell filtering ──────────────────────────────────────────────────────

    @property
    def has_filter(self) -> bool:
        return self._filter is not None

    def set_filter(
        self,
        filter_fn: Optional[Callable[["EmbeddingData"], "pd.Series | np.ndarray"]],
    ) -> "EmbeddingData":
        """Return a copy that keeps only the cells selected by *filter_fn*.

        *filter_fn* is a callable that receives this :class:`EmbeddingData`
        (with the filter *disabled*, so it sees the full dataset — avoiding
        recursion) and returns a boolean vector (array or ``Series``) of length
        ``n_obs`` marking the cells to keep.

        The filter is evaluated lazily, every time :meth:`coordinates` or
        :meth:`get_column` is called — there is no caching.  It restricts the
        cells returned by those two accessors, but **not** the coordinate
        bounds: :meth:`bounds` / :meth:`full_bounds` always reflect the full
        dataset so the embedding frame stays stable.

        Pass ``None`` (or call :meth:`unfilter`) to remove an existing filter.
        """
        new = copy.copy(self)
        new._filter = filter_fn
        new._filter_cache = None
        return new

    def unfilter(self) -> "EmbeddingData":
        """Return a new EmbeddingData with any cell filter removed."""
        return self.set_filter(None)

    def _filter_mask(self) -> "Optional[np.ndarray]":
        """Evaluate the active filter against the full data, caching the result.

        Returns a boolean ``ndarray`` of length ``n_obs`` (in ``obs_names``
        order) marking the cells to keep, or ``None`` when no filter is set.

        The mask is computed on first use and cached on the instance; it stays
        valid until the next :meth:`set_filter` / :meth:`unfilter` call (which
        resets the cache).

        The filter callable is handed an *unfiltered* shallow copy of this
        :class:`EmbeddingData` so that its own :meth:`get_column` /
        :meth:`coordinates` calls return the complete dataset — this prevents
        the recursion that would otherwise occur when a filter calls
        :meth:`get_column` to derive its mask.
        """
        if self._filter is None:
            return None
        if self._filter_cache is None:
            unfiltered = copy.copy(self)
            unfiltered._filter = None
            mask = self._filter(unfiltered)
            if isinstance(mask, pd.Series):
                mask = mask.reindex(self.ad.obs_names).fillna(False).values
            else:
                mask = np.asarray(mask)
            if mask.dtype != bool:
                mask = mask.astype(bool)
            if len(mask) != self.ad.n_obs:
                raise ValueError(
                    f"filter must return a boolean vector of length "
                    f"{self.ad.n_obs} (n_obs), got length {len(mask)}"
                )
            self._filter_cache = mask
        return self._filter_cache


    def bounds(self) -> tuple:
        """Return (x_min, x_max, y_min, y_max) — from focus if set, else full data range.

        Always reflects the *full* dataset: an active cell filter
        (:meth:`set_filter`) is ignored so the embedding frame stays stable.
        """
        if self._focus is not None:
            return self._focus
        coords = self._full_coordinates()
        return (
            float(coords["x"].min()),
            float(coords["x"].max()),
            float(coords["y"].min()),
            float(coords["y"].max()),
        )

    # ── data accessors ──────────────────────────────────────────────────────

    def get_column(self, name) -> ColumnData:
        """Return ColumnData(series, column_name) for an obs column or gene.

        Resolution:

        * If *name* is a ``(source_name, column)`` tuple, resolve *column*
          from the source (alternative or derived) registered under
          *source_name* (see :meth:`add_alternative_source` /
          :meth:`add_derived_source`).  The result is reindexed onto the
          primary ``obs_names``.  ``KeyError`` is raised if no source is
          registered under that name or it does not contain *column*.

        * Otherwise *name* is a string: the primary source is consulted first,
          then each registered *derived* source (columns computed on demand),
          then each alternative source in registration order.  The first hit
          is reindexed to the primary ``obs_names`` (extra cells dropped,
          missing cells → NaN).

        When a cell filter is active (:meth:`set_filter`), the returned series
        is restricted to the kept cells.  The filter is evaluated lazily (on
        first use) and cached until the next :meth:`set_filter` call.
        """
        result = self._get_column_raw(name)
        mask = self._filter_mask()
        if mask is not None:
            result = ColumnData(result.series[mask], result.name)
        return result

    def _get_column_raw(self, name) -> ColumnData:
        """Resolve *name* to a full-length ColumnData, ignoring any cell filter.

        Internal helper behind :meth:`get_column`; the public wrapper applies
        the active filter mask to the result.  See :meth:`get_column` for the
        documented resolution order.
        """
        # Explicit routing to a named source.
        if isinstance(name, tuple):
            if len(name) != 2:
                raise KeyError(
                    "Tuple column lookup must be (source_name, column), "
                    f"got {name!r}"
                )
            src_name, col = name
            for d in self._derived_sources:
                if d.name == src_name:
                    return ColumnData(
                        self._compute_derived(col, d).reindex(self.ad.obs_names),
                        col,
                    )
            for alt in self._alternative_sources:
                if alt.name == src_name:
                    resolved = self._resolve_column_from(alt.ad, col)
                    return ColumnData(
                        resolved.series.reindex(self.ad.obs_names), resolved.name
                    )
            registered = [
                s.name
                for s in (self._derived_sources + self._alternative_sources)
                if s.name is not None
            ]
            raise KeyError(
                f"No source named {src_name!r}. "
                f"Registered names: {registered!r}"
            )

        # Plain string: primary first, then derived, then alternative fallback.
        try:
            return self._resolve_column_from(self.ad, name)
        except KeyError:
            pass

        for d in self._derived_sources:
            if name in d.columns:
                return ColumnData(
                    self._compute_derived(name, d).reindex(self.ad.obs_names), name
                )

        primary_index = self.ad.obs_names
        for alt in self._alternative_sources:
            try:
                resolved = self._resolve_column_from(alt.ad, name)
            except KeyError:
                continue
            return ColumnData(resolved.series.reindex(primary_index), resolved.name)

        raise KeyError(
            f"Column or gene {name!r} not found in primary source"
            + (
                f" or any of {len(self._derived_sources)} derived source(s)"
                if self._derived_sources
                else ""
            )
            + (
                f" or any of {len(self._alternative_sources)} alternative source(s)"
                if self._alternative_sources
                else ""
            )
        )

    def _compute_derived(self, col, derived) -> pd.Series:
        """Run the callable for *col* of *derived*, returning its pandas Series.

        The callable receives an *unfiltered* view of this
        :class:`EmbeddingData` so that any :meth:`get_column` / coordinates
        calls it makes operate on the complete dataset — derived columns are
        defined over the full ``obs_names``, and the cell filter is applied
        only at the public :meth:`get_column` boundary.

        Raises ``KeyError`` if *col* is not a registered column of this derived
        source, and ``TypeError`` if the callable does not return a Series.
        """
        try:
            fn = derived.columns[col]
        except KeyError:
            raise KeyError(
                f"Derived source {derived.name!r} has no column {col!r}; "
                f"available: {list(derived.columns)}"
            )
        unfiltered = copy.copy(self)
        unfiltered._filter = None
        unfiltered._filter_cache = None
        series = fn(unfiltered)
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"Derived column {col!r} callable must return a pandas Series, "
                f"got {type(series).__name__}"
            )
        return series

    def _resolve_column_from(self, ad, name: str) -> ColumnData:
        """Resolve *name* against a single AnnData-like source.

        Same resolution order as documented for :meth:`get_column`, applied to
        the given ``ad`` (primary or alternative).  Works for ``AnnData`` and
        :class:`H5adFacade` alike since only the shared interface is used.
        Raises ``KeyError`` if *name* is not found in this source.
        """
        if name in ad.obs:
            return ColumnData(ad.obs[name], name)

        def _extract(mask, col_name=None):
            idx = np.nonzero(mask)[0][0]
            if col_name is None:
                col_name = ad.var.index[idx]
            col = ad.X[:, idx]
            if sp.issparse(col):
                col = col.toarray().ravel()
            else:
                col = np.asarray(col).ravel()
            return ColumnData(pd.Series(col, index=ad.obs_names), col_name)

        if name in ad.var.index:
            return _extract(ad.var.index == name, name)
        if self._alternative_id_column is not None and self._alternative_id_column in ad.var.columns:
            alt_hits = ad.var[self._alternative_id_column] == name
            if alt_hits.sum() == 1:
                return _extract(alt_hits)
        if ad.var.index.str.contains(" ").any():
            name_hits = ad.var.index.str.startswith(name + " ")
            if name_hits.sum() == 1:
                return _extract(name_hits)
            id_hits = ad.var.index.str.endswith(" " + name)
            if id_hits.sum() == 1:
                return _extract(id_hits)
        raise KeyError(f"Column or gene {name!r} not found")

    def alternative_id_for(self, gene_name: str) -> Optional[str]:
        """Return the alternative-id value for *gene_name*, or ``None``.

        Searches the primary source first, then each registered alternative
        source, looking up ``var[_alternative_id_column]`` for the row whose
        index equals *gene_name*.  Returns ``None`` when no
        ``_alternative_id_column`` is configured, the column is absent in a
        source, the gene is not found, or its alternative id is missing.
        """
        if self._alternative_id_column is None:
            return None
        col = self._alternative_id_column
        for ad in [self.ad, *(s.ad for s in self._alternative_sources)]:
            if col in ad.var.columns and gene_name in ad.var.index:
                series = ad.var[col]
                if gene_name not in series.index:
                    continue
                val = series[gene_name]
                if isinstance(val, pd.Series):  # duplicate gene symbols
                    val = val.iloc[0]
                if val is None or pd.isna(val):
                    continue
                return val
        return None

    def _classify_in(self, ad, name: str) -> Optional[bool]:
        """Classify *name* against a single source without extracting data.

        Mirrors :meth:`_resolve_column_from`'s resolution order.  Returns
        ``True`` if *name* is a feature (``var`` row), ``False`` if it is an
        ``obs`` column, or ``None`` if it is not found in this source.
        """
        if name in ad.obs:
            return False
        if name in ad.var.index:
            return True
        if (
            self._alternative_id_column is not None
            and self._alternative_id_column in ad.var.columns
            and bool((ad.var[self._alternative_id_column] == name).sum() == 1)
        ):
            return True
        if ad.var.index.str.contains(" ").any():
            if bool((ad.var.index.str.startswith(name + " ")).sum() == 1):
                return True
            if bool((ad.var.index.str.endswith(" " + name)).sum() == 1):
                return True
        return None

    def is_gene(self, name: Union[str, tuple]) -> bool:
        """Return ``True`` if *name* resolves to a feature (``var`` row) rather
        than an ``obs`` column.

        Accepts the same *name* forms as :meth:`get_column` — a plain string or
        a ``(source_name, column)`` tuple — and mirrors its resolution order so
        that a gene sourced from an alternative source, or addressed via an
        alternative id / ``"symbol id"`` pattern, is still recognised.
        """
        # Explicit routing to a named source.
        if isinstance(name, tuple):
            src_name, col = name
            for d in self._derived_sources:
                if d.name == src_name:
                    return False  # derived columns are computed, not genes
            for alt in self._alternative_sources:
                if alt.name == src_name:
                    return self._classify_in(alt.ad, col) is True
            return False  # unknown source — get_column would have raised

        # Plain string: primary first, then derived, then alternative fallback.
        result = self._classify_in(self.ad, name)
        if result is not None:
            return result
        for d in self._derived_sources:
            if name in d.columns:
                return False  # derived column wins before alt sources
        for alt in self._alternative_sources:
            result = self._classify_in(alt.ad, name)
            if result is not None:
                return result
        return False

    def coordinates(self) -> pd.DataFrame:
        """Return DataFrame with x, y columns, indexed by obs index.

        When a cell filter is active (:meth:`set_filter`), only the kept cells
        are returned (the filter is evaluated lazily on first use and cached
        until the next :meth:`set_filter` call).  Use
        :meth:`_full_coordinates` to ignore the filter.
        """
        coords = self._full_coordinates()
        mask = self._filter_mask()
        if mask is not None:
            coords = coords[mask]
        return coords

    def _full_coordinates(self) -> pd.DataFrame:
        """Return x, y DataFrame for *every* cell, ignoring the active filter.

        Index is the primary ``obs_names``.  Used by bounds/grid helpers so the
        embedding frame always reflects the complete dataset.
        """
        if self._embedding_cols is not None:
            c1, c2 = self._embedding_cols
            arr = self.ad.obsm[self._embedding][:, [c1, c2]]
        else:
            arr = self.ad.obsm[self._embedding][:, :2]
        return (
            pd.DataFrame(arr, columns=["x", "y"])
            .assign(index=self.ad.obs.index)
            .set_index("index")
        )

    def get_X_csr(self):
        """Return the primary source's ``X`` as a scipy CSR sparse matrix.

        This is the bulk-access entry point used by analyses that touch many
        genes at once (e.g. :func:`~mbf_singlecell_plotter.transforms.compute_grid_moran`,
        which row-slices ``X`` per embedding bin).  CSR is row-major, so
        ``X[row_array]`` slicing is cheap.

        For :class:`H5adFacade` sources the whole matrix is loaded in one
        ``h5ad-inspect`` call (``export matrix_csr``) and cached.  For real
        ``AnnData`` (or anything exposing a ``get_X_csr`` method) it forwards
        directly, converting dense/sparse matrices to CSR as needed.
        """
        ad = self.ad
        getter = getattr(ad, "get_X_csr", None)
        if getter is not None:
            return getter()
        from scipy import sparse as sp

        X = ad.X
        if sp.issparse(X):
            return X.tocsr()
        return sp.csr_matrix(np.asarray(X))

    # ── grid helpers ────────────────────────────────────────────────────────

    def point_to_grid(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        x: float,
        y: float,
    ) -> tuple:
        """Map a single point to a (letter, number) or (number, letter) grid cell."""
        x_step = (x_max - x_min) / self._grid_size
        y_step = (y_max - y_min) / self._grid_size
        assert x <= x_max, "x outside of x_max range"
        assert y <= y_max, "y outside of y_max range"
        x_index = min(
            int(round((x - x_min) / x_step)), self._grid_size - 1
        )
        y_index = min(
            int(round((y - y_min) / y_step)), self._grid_size - 1
        )
        letters = _LETTERS[: self._grid_size]
        non_letters = list(range(1, self._grid_size + 1))
        if self._grid_letters_on_vertical:
            letters_rev = letters[::-1]
            letter = letters_rev[y_index]
            number = non_letters[x_index]
            return (number, letter)
        else:
            letter = letters[x_index]
            number = non_letters[self._grid_size - 1 - y_index]
            return (letter, number)

    def grid_coordinate(self, x: float, y: float) -> str:
        """Return grid label (e.g. 'A1') for embedding coordinates."""
        x_min, x_max, y_min, y_max = self.bounds()
        parts = self.point_to_grid(x_min, x_max, y_min, y_max, x, y)
        if self._grid_letters_on_vertical:
            return f"{parts[1]}{parts[0]}"
        return f"{parts[0]}{parts[1]}"

    def grid_coordinates(self) -> pd.Series:
        """Return a Series of grid labels for all cells (vectorised)."""
        coords = self.coordinates()
        x_min, x_max, y_min, y_max = self.bounds()
        x_step = (x_max - x_min) / self._grid_size
        y_step = (y_max - y_min) / self._grid_size
        x_idx = np.clip(
            ((coords["x"] - x_min) / x_step).round().astype(int),
            0,
            self._grid_size - 1,
        )
        y_idx = np.clip(
            ((coords["y"] - y_min) / y_step).round().astype(int),
            0,
            self._grid_size - 1,
        )
        letters = list(_LETTERS[: self._grid_size])
        if self._grid_letters_on_vertical:
            letters_rev = list(reversed(letters))
            letter_col = [letters_rev[i] for i in y_idx]
            number_col = [i + 1 for i in x_idx]
            labels = [f"{l}{n}" for l, n in zip(letter_col, number_col)]
        else:
            letter_col = [letters[i] for i in x_idx]
            number_col = [self._grid_size - i for i in y_idx]
            labels = [f"{l}{n}" for l, n in zip(letter_col, number_col)]
        return pd.Series(labels, index=coords.index)

    def full_bounds(self) -> tuple:
        """Return (x_min, x_max, y_min, y_max) from the full data range.

        Ignores both the focus window and any active cell filter, so the bounds
        always reflect the complete dataset.
        """
        coords = self._full_coordinates()
        return (
            float(coords["x"].min()),
            float(coords["x"].max()),
            float(coords["y"].min()),
            float(coords["y"].max()),
        )

    def grid_labels(self) -> tuple:
        """Return (x_positions, y_positions, x_labels, y_labels) for grid axis ticks.

        Always computed in the original (unfocused) coordinate space so that
        labels reflect the correct grid cell when a focus/zoom is active.
        """
        x_min, x_max, y_min, y_max = self.full_bounds()
        gs = self._grid_size
        cell_w = (x_max - x_min) / gs
        cell_h = (y_max - y_min) / gs

        # Centers of each grid cell — direct arithmetic, no rounding issues
        x_positions = np.array([x_min + (i + 0.5) * cell_w for i in range(gs)])
        y_positions = np.array([y_min + (i + 0.5) * cell_h for i in range(gs)])

        letters = list(_LETTERS[:gs])
        if self._grid_letters_on_vertical:
            # x-axis: numbers 1..gs; y-axis: letters (A at top = max y)
            x_labels = list(range(1, gs + 1))
            y_labels = letters[::-1]
        else:
            # x-axis: letters A..Z; y-axis: numbers (1 at top = max y)
            x_labels = letters
            y_labels = list(range(gs, 0, -1))

        return x_positions, y_positions, x_labels, y_labels

    def cluster_centers(self, cluster_column: str) -> pd.DataFrame:
        """Return DataFrame with x, y, grid for each category in cluster_column."""
        col_data, col_name = self.get_column(cluster_column)
        if pd.api.types.is_numeric_dtype(col_data) and not isinstance(
            col_data.dtype, pd.CategoricalDtype
        ):
            raise ValueError(
                f"Column '{cluster_column}' contains numeric data. "
                "This function only works with categorical data."
            )
        coords = self.coordinates()
        if self._focus is not None:
            x_min, x_max, y_min, y_max = self._focus
            mask = (
                (coords["x"] >= x_min) & (coords["x"] <= x_max)
                & (coords["y"] >= y_min) & (coords["y"] <= y_max)
            )
            coords = coords[mask]
        merged = coords.copy()
        merged["category"] = col_data.loc[coords.index]
        centers = merged.groupby("category", observed=True).agg(
            {"x": "median", "y": "median"}
        )
        centers["grid"] = centers.apply(
            lambda row: self.grid_coordinate(row["x"], row["y"]), axis=1
        )
        centers.index.name = col_name
        return centers

    def moran_markers(
        self,
        n_bins: int = 40,
        min_cells: int = 3,
        k: int = 20,
        min_moran: float = 0.2,
    ) -> dict:
        """Identify marker genes per UMAP region using Moran's I spatial autocorrelation.

        Bins cells into an ``n_bins × n_bins`` grid, computes Moran's I for every
        gene across occupied bins, and returns the top-*k* spatially coherent genes
        for each region.

        Args:
            n_bins:    Grid resolution per axis (default 40).
            min_cells: Minimum cells per bin (default 3).
            k:         Maximum marker genes per region (default 20).
            min_moran: Minimum Moran's I to qualify as a marker (default 0.2).

        Returns:
            Dict mapping ``(xi, yi)`` bin-index tuple → list of gene names.
        """
        from .transforms import compute_grid_moran, marker_genes_by_region

        gene_df = compute_grid_moran(self, n_bins=n_bins, min_cells=min_cells)
        return marker_genes_by_region(gene_df, k=k, min_moran=min_moran)

    def grid_local_histogram(
        self,
        key: str,
        min_cells: int = 10,
    ) -> pd.DataFrame:
        """Return DataFrame of grid-local category histograms."""
        expr, _ = self.get_column(key)
        if isinstance(expr.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(
            expr
        ):
            # categorical or bool — both are handled as discrete categories below
            pass
        elif pd.api.types.is_numeric_dtype(expr):
            raise ValueError("category types only")
        coords = self.coordinates()
        x_min, x_max, y_min, y_max = self.bounds()

        x_grid = np.linspace(x_min, x_max + 0.1, self._grid_size + 1)
        y_grid = np.linspace(y_min, y_max + 0.1, self._grid_size + 1)
        x_bins = np.digitize(coords["x"].values, x_grid) - 1
        y_bins = np.digitize(coords["y"].values, y_grid) - 1
        valid = (
            (x_bins >= 0)
            & (x_bins < len(x_grid) - 1)
            & (y_bins >= 0)
            & (y_bins < len(y_grid) - 1)
        )
        assert all(valid)
        # Coerce bool → str so the histogram "category" values are plain labels
        # (mirrors how ScatterPlotter.plot renders bool columns).
        if pd.api.types.is_bool_dtype(expr):
            expr = expr.astype(str)
        try:
            df_cells = pd.DataFrame(
                {
                    "x_bin": x_bins[valid],
                    "y_bin": y_bins[valid],
                    "category": expr.loc[coords.index[valid]].values,
                }
            )
        except ValueError as e:
            raise ValueError("Make sure your obs.keys are distinct!", e)

        histogram: dict = {
            "x": [],
            "y": [],
            "category": [],
            "frequency": [],
            "total": [],
        }
        for (ix, iy), sub in df_cells.groupby(["x_bin", "y_bin"]):
            if len(sub) >= min_cells:
                freqs = sub["category"].value_counts(normalize=True)
                for cat, freq in freqs.items():
                    histogram["x"].append(ix)
                    histogram["y"].append(iy)
                    histogram["category"].append(cat)
                    histogram["frequency"].append(freq)
                    histogram["total"].append(len(sub))
        return pd.DataFrame(histogram)
