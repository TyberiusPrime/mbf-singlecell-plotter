"""Layer 1: data access (AnnData → DataFrames). No plotting."""

import copy
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Union, Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy.sparse as sp


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _source_matrix(ad, layer: str):
    """Return the expression matrix of *ad* for *layer*.

    ``layer='X'`` (or ``None``) yields ``ad.X``.  For any other key, an
    :class:`~mbf_singlecell_plotter.h5ad_source.H5adFacade` (which exposes a
    ``matrix(layer)`` method) is asked for its layer-bound accessor, while a
    real ``AnnData`` is indexed as ``ad.layers[layer]``.  The result supports
    ``matrix[:, idx]`` single-column indexing.
    """
    if layer in (None, "X"):
        return ad.X
    getter = getattr(ad, "matrix", None)
    if getter is not None:
        return getter(layer)
    return ad.layers[layer]


def has_x(ad, layer: str = "X") -> bool:
    """Return whether *ad* has a readable matrix for *layer*.

    Cheap when *ad* keeps its matrix in memory (a real ``AnnData``): a
    missing ``.X``/layer is simply ``None``, no I/O involved. For a lazy
    source like :class:`~mbf_singlecell_plotter.h5ad_source.H5adFacade`
    (whose proxies are always non-``None``) this reads one probe column via
    ``h5ad-inspect``, so only call it reactively — to confirm the cause of an
    extraction failure that already happened — not before every column
    lookup.
    """
    matrix = _source_matrix(ad, layer)
    if matrix is None:
        return False
    try:
        matrix[:, 0]
        return True
    except RuntimeError as exc:
        if "no X matrix" in str(exc):
            return False
        raise


def _parse_grid_label(
    label: str, gs: int, letters_on_vertical: bool
) -> tuple[int, int]:
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

    ``layer`` selects which expression matrix feature columns are read from:
    the sentinel ``'X'`` (default) reads ``.X``; any other key reads
    ``.layers[layer]`` (for :class:`H5adFacade` sources, via the
    ``h5ad-inspect --layer`` flag).  ``transform``, when given, is a callable
    applied to each feature column read from this source's matrix (e.g. to
    convert natural-log values to log2) — it does **not** touch ``obs`` columns.
    """

    name: Optional[str]
    ad: object
    layer: str = "X"
    transform: Optional[Callable[["np.ndarray"], "np.ndarray"]] = None


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


def computed_column(name: Optional[str] = None):
    """Decorator: register an :class:`EmbeddingData` method as a
    :meth:`get_column` fallback.

    A method decorated with this becomes resolvable *by name* through
    :meth:`EmbeddingData.get_column`: when the requested name matches no
    ``obs`` column or gene in the primary source, the tagged method is invoked
    and its returned :class:`pandas.Series` is used.  The lookup name defaults
    to the method's ``__name__``; pass *name* to override.

    At lookup time the method is run on an *unfiltered* view of the data and
    the active cell filter is applied once, at the public :meth:`get_column`
    boundary — exactly like a derived-source callable — so the method should
    compute over the full dataset, not a pre-filtered subset.  Its result is
    reindexed onto the primary ``obs_names`` (extra indices dropped, missing
    cells → ``NaN``).

    Tagged columns sit *between* the primary obs/gene lookup and the
    registered derived/alternative sources, so an explicit ``obs`` column or
    gene of the same name always wins.
    """

    def decorator(fn):
        fn._computed_column_name = name or fn.__name__
        return fn

    return decorator


def _collect_computed_columns(cls):
    """Class decorator: gather every ``@computed_column`` method into a registry.

    Builds ``cls._computed_columns`` — ``{column_name: method_attr_name}`` — by
    scanning the full MRO so subclasses inherit (and may override) tagged
    methods.  Tagging is detected via the ``_computed_column_name`` attribute
    set by :func:`computed_column`.  Applied to :class:`EmbeddingData` here,
    and re-run from :meth:`EmbeddingData.__init_subclass__` for subclasses.
    """
    registry: Dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        for attr, value in vars(klass).items():
            col_name = getattr(value, "_computed_column_name", None)
            if col_name is not None:
                registry[col_name] = attr
    cls._computed_columns = registry
    return cls


@_collect_computed_columns
class EmbeddingData:
    """Wraps an AnnData + embedding choice. Pure data extraction, no plotting.

    The embedding array normally comes from the primary source's ``obsm``.
    To read it from a *named alternative source* instead (e.g. gene
    expression from the primary ``ad`` but the UMAP coordinates from a second
    ``ad2``), pass a source-routed tuple as ``embedding``::

        data = EmbeddingData(ad, ("ad2", "umap"))
        data = data.add_alternative_source(ad2, name="ad2")

    The 2-tuple ``(source_name, key)`` parallels the ``(source_name, column)``
    routing used by :meth:`get_column`; the coordinates are reindexed onto the
    primary ``obs_names`` exactly like alternative-source columns.
    """

    # Populated by :func:`_collect_computed_columns` (and re-populated for
    # subclasses by :meth:`__init_subclass__`); declared here so type checkers
    # and readers can see it. Maps ``@computed_column`` lookup name → method attr.
    _computed_columns: Dict[str, str] = {}

    def __init__(
        self,
        ad,
        embedding,
        alternative_id_column: Optional[str] = None,
        alternative_sources: Optional[list["EmbeddingData"]] = None,
        derived_sources: Optional[
            list[dict[str, Callable[["EmbeddingData"], pd.Series]]]
        ] = None,
        grid_size: int = 12,
        grid_letters_on_vertical: bool = False,
        filter_fn: Optional[
            Callable[["EmbeddingData"], "pd.Series | np.ndarray"]
        ] = None,
        layer: str = "X",
        transform: Optional[Callable[["np.ndarray"], "np.ndarray"]] = None,
    ):
        self.ad = ad
        if grid_size > 26:
            raise ValueError("grid_size max is 26")
        if not isinstance(layer, str):
            raise TypeError(
                f"layer must be a string ('X' for .X, else a layer key), "
                f"got {type(layer).__name__}"
            )
        if transform is not None and not callable(transform):
            raise TypeError(
                f"transform must be a callable or None, got {type(transform).__name__}"
            )
        self._layer = layer
        self._transform = transform
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

        # Resolve embedding.  Accepted forms:
        #   "umap"                        primary source .obsm key
        #   ("pca", 0, 1)                 primary source .obsm key + two columns
        #   (source_name, "umap")         named alternative source .obsm key
        #   (source_name, ("pca", 0, 1))  named alternative source + two columns
        #
        # The source-routed 2-tuple forms are resolved LAZILY: the named
        # alternative source is usually registered via add_alternative_source()
        # *after* construction, so its .obsm keys cannot be validated here.
        # Validation (and the obsm read) happens on first coordinates()/embedding
        # access — see _embedding_source_ad() / _resolve_obsm_key().
        self._embedding_source: Optional[str] = None
        self._embedding_raw_key: Optional[str] = None
        if isinstance(embedding, tuple):
            if len(embedding) == 2 and isinstance(embedding[0], str):
                # Source-routed embedding — pull the array from a named
                # alternative source instead of the primary .obsm.
                source_name, inner = embedding
                if isinstance(inner, str):
                    self._embedding_source = source_name
                    self._embedding_raw_key = inner
                    self._embedding = None
                    self._embedding_cols: Optional[tuple[int, int]] = None
                elif isinstance(inner, tuple):
                    if (
                        len(inner) != 3
                        or not isinstance(inner[0], str)
                        or not isinstance(inner[1], int)
                        or not isinstance(inner[2], int)
                    ):
                        raise ValueError(
                            "Source-routed tuple embedding must be "
                            "(source_name, (key, col1, col2)), e.g. "
                            "('ad2', ('pca', 0, 1)); got " + repr(inner)
                        )
                    key_raw, c1, c2 = inner
                    self._embedding_source = source_name
                    self._embedding_raw_key = key_raw
                    self._embedding = None
                    self._embedding_cols = (c1, c2)
                else:
                    raise ValueError(
                        "Source-routed embedding must be (source_name, key) or "
                        "(source_name, (key, col1, col2)); the second element "
                        "must be a str or 3-tuple, got "
                        f"{type(inner).__name__}"
                    )
            elif len(embedding) == 3:
                # Primary source — check tuple BEFORE string concatenation.
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
                self._embedding_cols = (c1, c2)
            else:
                raise ValueError(
                    "Tuple embedding must be (key, col1, col2) for the primary "
                    "source, or (source_name, key) / "
                    "(source_name, (key, col1, col2)) to pull the embedding from "
                    "a named alternative source"
                )
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
                "embedding must be a string, a (key, col1, col2) tuple, or a "
                "(source_name, key) / (source_name, (key, col1, col2)) tuple "
                "to source the embedding from a named alternative; got "
                f"{type(embedding).__name__}"
            )
        self._focus: Optional[tuple[float, float, float, float]] = (
            None  # (x_min, x_max, y_min, y_max)
        )
        self._filter: Optional[
            Callable[["EmbeddingData"], "pd.Series | np.ndarray"]
        ] = filter_fn
        self._filter_cache: Optional[np.ndarray] = None
        # When True, the active filter is a *hard* restrict: bounds() and
        # full_bounds() follow the masked subset (a fresh grid is spanned over
        # just the kept cells).  When False the filter is soft — it subsets the
        # cells shown but leaves the embedding frame at the full-data extent.
        self._filter_hard: bool = False

    def __init_subclass__(cls, **kwargs):
        """Rebuild the ``@computed_column`` registry for subclasses.

        Subclasses may add or override ``@computed_column`` methods; this hooks
        their creation so :attr:`_computed_columns` always reflects the full
        MRO (the base class registry is built by :func:`_collect_computed_columns`).
        """
        super().__init_subclass__(**kwargs)
        _collect_computed_columns(cls)

    def _replace(self, **changes) -> "EmbeddingData":
        """Return a shallow copy with the specified private attributes replaced.

        Mirrors :func:`dataclasses.replace`: ``changes`` maps attribute names
        (without leading underscore) to new values.  The returned object is a
        shallow copy — mutable shared state like ``self.ad`` is **not** copied.

        Example::

            data2 = data._replace(grid_size=16, grid_letters_on_vertical=True)
        """
        new = copy.copy(self)
        for k, v in changes.items():
            setattr(new, "_" + k, v)
        return new

    @property
    def embedding(self) -> str:
        """The resolved obsm key backing this embedding (e.g. ``"X_umap"``).

        For a source-routed embedding — ``EmbeddingData(ad, (source, key))`` —
        the key is resolved lazily against the named alternative source on
        first access (sources are registered after construction).  Raises
        ``KeyError`` if that source is not registered or lacks the key.
        """
        if self._embedding is not None:
            return self._embedding
        # Source-routed — resolve the obsm key lazily.
        return self._resolved_source_embedding()[1]

    @property
    def embedding_source(self) -> Optional[str]:
        """Name of the alternative source backing the embedding, or ``None``.

        ``None`` means the embedding array is read from the primary source's
        ``obsm``.  A non-``None`` value is the registered name of the
        alternative source supplying the embedding (set via
        ``embedding=(name, ...)``).
        """
        return self._embedding_source

    def _embedding_source_ad(self):
        """Return the AnnData-like object the embedding array is read from.

        The primary ``self.ad`` by default; for a source-routed embedding the
        named alternative source's ``ad`` (resolved lazily — alternatives are
        registered after construction).  Raises ``KeyError`` if the named
        source is missing.  Derived sources cannot host an embedding (they
        yield computed ``Series``, not arrays), so only alternative sources
        are consulted.
        """
        if self._embedding_source is None:
            return self.ad
        for alt in self._alternative_sources:
            if alt.name == self._embedding_source:
                return alt.ad
        registered = [s.name for s in self._alternative_sources if s.name is not None]
        raise KeyError(
            f"Embedding source {self._embedding_source!r} is not a registered "
            f"alternative source. Registered names: {registered!r}"
        )

    def _resolved_source_embedding(self):
        """Resolve a source-routed embedding to ``(source_ad, obsm_key)``.

        Only valid when :attr:`_embedding_source` is not ``None`` (the
        source-routed case); the primary case reads :attr:`_embedding` directly.
        Raises ``KeyError`` if the named source is unregistered or lacks the
        key.  The raw key is guaranteed non-``None`` by construction (see
        :meth:`__init__`).
        """
        src_ad = self._embedding_source_ad()
        assert self._embedding_raw_key is not None  # set iff source-routed
        return src_ad, self._resolve_obsm_key(src_ad, self._embedding_raw_key)

    @staticmethod
    def _resolve_obsm_key(ad, key_raw: str) -> str:
        """Resolve a raw embedding key against ``ad.obsm`` (``"umap"`` → ``"X_umap"``).

        Accepts an exact ``obsm`` key or a bare name prefixed with ``"X_"``.
        Raises ``KeyError`` listing the available keys if neither matches.
        """
        if key_raw in ad.obsm:
            return key_raw
        if "X_" + key_raw in ad.obsm:
            return "X_" + key_raw
        raise KeyError(
            f"Embedding {key_raw!r} not found in source obsm. Available: "
            + ", ".join(sorted(ad.obsm.keys()))
        )

    @property
    def layer(self) -> str:
        """Matrix layer read from the primary source (``'X'`` → ``.X``)."""
        return self._layer

    @property
    def transform(self) -> Optional[Callable[["np.ndarray"], "np.ndarray"]]:
        """Callable applied to primary-source feature columns, or ``None``."""
        return self._transform

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def has_focus(self) -> bool:
        return self._focus is not None

    @property
    def alternative_sources(self) -> list[AlternativeSource]:
        """List of :class:`AlternativeSource` fallbacks consulted by :meth:`get_column`."""
        return list(self._alternative_sources)

    @property
    def derived_sources(self) -> list[DerivedSource]:
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
            layer, transform = "X", None
            if isinstance(item, AlternativeSource):
                name, src = item.name, item.ad
                layer, transform = item.layer, item.transform
            elif (
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            ):
                name, src = item
            else:
                name, src = None, item
            if name is not None:
                if name in seen:
                    raise ValueError(f"Duplicate alternative source name {name!r}")
                seen.add(name)
            result.append(
                AlternativeSource(name, self._coerce_ad(src), layer, transform)
            )
        return result

    def add_alternative_source(
        self,
        source: Any,
        name=None,
        layer="X",
        transform=None,
    ) -> "EmbeddingData":
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

        *layer* selects which expression matrix feature columns are read from
        (``'X'`` → ``.X``, else ``.layers[layer]``).  *transform*, when given,
        is a callable applied to each feature column read from this source (not
        to ``obs`` columns) — e.g. ``lambda x: x / np.log(2)`` to convert
        natural-log values to log2.
        """
        existing = {a.name for a in self._alternative_sources if a.name is not None}
        new = copy.copy(self)
        new._alternative_sources = self._alternative_sources + (
            self._normalize_alternative_items(
                [AlternativeSource(name, source, layer, transform)],
                existing_names=existing,
            )
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

    def focus_on(self, region) -> "EmbeddingData":
        """Return a new EmbeddingData whose viewport is restricted to *region*.

        *region* is a 2-corner ``(corner1, corner2)`` box.  Each corner is either
        a grid-label string (e.g. ``"A1"`` for the default orientation, ``"1A"``
        for vertical-letters) or an ``(x, y)`` coordinate pair::

            data.focus_on(("A1", "C5"))                 # grid labels
            data.focus_on(((x_min, y_min), (x_max, y_max)))  # raw coordinates

        Grid labels are resolved in the original (unfocused) coordinate space and
        span the full extent of the named cells; swapped corners are silently
        corrected.  This is a *soft* viewport (the grid is re-spanned over the
        window but every cell is still returned) — use :meth:`hard_filter` to
        actually restrict the data and re-span the grid over just the subset.
        """
        from .transforms import _region_to_bbox

        new = copy.copy(self)
        new._focus = _region_to_bbox(region, self)
        return new

    def unfocus(self) -> "EmbeddingData":
        """Return a new EmbeddingData with no focus restriction."""
        new = copy.copy(self)
        new._focus = None
        return new

    # ── cell filtering ──────────────────────────────────────────────────────

    @property
    def has_filter(self) -> bool:
        return self._filter is not None

    @property
    def has_hard_filter(self) -> bool:
        """True when a *hard* filter (bounds-following restrict) is active."""
        return self._filter is not None and self._filter_hard

    def _apply_filter(self, spec, *, hard: bool) -> "EmbeddingData":
        """Shared implementation of :meth:`set_filter` / :meth:`hard_filter`.

        *spec* is either ``None`` (clear), a callable ``fn(EmbeddingData) -> bool
        mask``, or a region / list of regions (see the class-level region grammar
        used by :meth:`focus_on`).  Regions are converted to a membership filter
        via :func:`~mbf_singlecell_plotter.transforms._regions_to_mask_fn`.

        There is a single filter slot: setting either kind replaces the other.
        """
        from .transforms import _regions_to_mask_fn

        new = copy.copy(self)
        new._filter = None if spec is None else _regions_to_mask_fn(spec)
        new._filter_hard = hard and spec is not None
        new._filter_cache = None
        return new

    def set_filter(self, spec) -> "EmbeddingData":
        """Return a copy that keeps only the cells selected by *spec* (soft filter).

        *spec* is either a callable that receives this :class:`EmbeddingData`
        (with the filter *disabled*, so it sees the full dataset — avoiding
        recursion) and returns a boolean vector (array or ``Series``) of length
        ``n_obs`` marking the cells to keep, or a region / list of regions in the
        same grammar as :meth:`focus_on` (each region a ``(corner1, corner2)`` box
        of grid labels or ``(x, y)`` pairs), which is converted to a membership
        filter.

        The filter is evaluated lazily and cached until the next filter call.  It
        restricts the cells returned by :meth:`coordinates` / :meth:`get_column`,
        but **not** the coordinate bounds: :meth:`bounds` / :meth:`full_bounds`
        keep reflecting the full dataset so the embedding frame stays stable.  Use
        :meth:`hard_filter` if you want the bounds (and grid) to follow the subset.

        Pass ``None`` (or call :meth:`unfilter`) to remove an existing filter.
        """
        return self._apply_filter(spec, hard=False)

    def hard_filter(self, spec) -> "EmbeddingData":
        """Return a copy restricted to *spec*, with the bounds following the subset.

        Accepts the same *spec* grammar as :meth:`set_filter` (a callable, or a
        region / list of regions).  Unlike the soft :meth:`set_filter`, a hard
        filter makes :meth:`bounds` and :meth:`full_bounds` reflect only the kept
        cells, so a fresh grid is spanned over the subset and downstream analyses
        (grid Moran's I, interactive per-cell views) recompute over the smaller
        cells — the "hard focus" behaviour.

        Pass ``None`` (or call :meth:`unfilter`) to remove an existing filter.
        """
        return self._apply_filter(spec, hard=True)

    def unfilter(self) -> "EmbeddingData":
        """Return a new EmbeddingData with any cell filter removed."""
        return self._apply_filter(None, hard=False)

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
            # Disable the hard flag too, so region grid-labels and any bounds()
            # the filter callable consults resolve against the true full extent
            # (and never recurse back into this half-computed mask).
            unfiltered._filter_hard = False
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

    def bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) — from focus if set, else data range.

        A *soft* cell filter (:meth:`set_filter`) is ignored so the embedding
        frame stays stable.  A *hard* filter (:meth:`hard_filter`) is honoured:
        the bounds shrink to the kept subset so a fresh grid is spanned over it.

        Cells lacking an embedding coordinate (see :meth:`_finite_coordinates`)
        are excluded so a source-routed embedding with dropped cells doesn't
        turn the range into ``NaN``.
        """
        if self._focus is not None:
            return self._focus
        coords = (
            self._finite_coordinates()
            if self._filter_hard
            else self._finite_full_coordinates()
        )
        if coords.empty:
            raise ValueError(
                "No cells have a finite embedding coordinate; cannot compute bounds."
            )
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

        * Otherwise *name* is a string: the primary source is consulted first
          (``obs`` columns, then genes), then any tagged ``@computed_column``
          method on this object (e.g. ``n_genes_per_cell`` / ``per_cell_sum``),
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
                    f"Tuple column lookup must be (source_name, column), got {name!r}"
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
                    resolved = self._resolve_column_from(
                        alt.ad, col, alt.layer, alt.transform
                    )
                    return ColumnData(
                        resolved.series.reindex(self.ad.obs_names), resolved.name
                    )
            registered = [
                s.name
                for s in (self._derived_sources + self._alternative_sources)
                if s.name is not None
            ]
            raise KeyError(
                f"No source named {src_name!r}. Registered names: {registered!r}"
            )

        # Plain string: primary first, then derived, then alternative fallback.
        try:
            return self._resolve_column_from(
                self.ad, name, self._layer, self._transform
            )
        except KeyError as exc:
            # Ambiguous matches (e.g. duplicate alternative ids) must bubble up
            # rather than being silently swallowed and re-raised as "not found".
            if "Duplicate" in str(exc):
                raise
            pass
        except (TypeError, RuntimeError) as exc:
            # name is a var (gene) in the primary source, but its matrix isn't
            # readable (ad.X is None, or an H5adFacade backing file has no X
            # dataset). has_x() re-checks so an unrelated bug doesn't get
            # misreported as this — deliberately reactive, only on failure.
            if has_x(self.ad, self._layer):
                raise
            raise ValueError(self._no_x_message(name)) from exc

        # Tagged @computed_column methods on this data object (built-in QC
        # metrics like n_genes_per_cell / per_cell_sum): consulted after the
        # primary obs/gene lookup fails, before derived/alternative sources.
        # Run on an unfiltered view so the cell filter is applied exactly once,
        # at the public get_column boundary — same contract as derived sources.
        if name in self._computed_columns:
            attr = self._computed_columns[name]
            unfiltered = copy.copy(self)
            unfiltered._filter = None
            unfiltered._filter_hard = False
            unfiltered._filter_cache = None
            series = getattr(unfiltered, attr)()
            if not isinstance(series, pd.Series):
                raise TypeError(
                    f"@computed_column method {attr!r} must return a pandas "
                    f"Series, got {type(series).__name__}"
                )
            return ColumnData(series.reindex(self.ad.obs_names), name)

        for d in self._derived_sources:
            if name in d.columns:
                return ColumnData(
                    self._compute_derived(name, d).reindex(self.ad.obs_names), name
                )

        primary_index = self.ad.obs_names
        for alt in self._alternative_sources:
            try:
                resolved = self._resolve_column_from(
                    alt.ad, name, alt.layer, alt.transform
                )
            except KeyError:
                continue
            return ColumnData(resolved.series.reindex(primary_index), resolved.name)

        raise KeyError(
            f"Column or gene {name!r} not found in primary source"
            + (
                f" or any of {len(self._computed_columns)} computed column(s)"
                if self._computed_columns
                else ""
            )
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
            + ". Gene id columns considered: var_index"
            + (
                f", {self._alternative_id_column}"
                if self._alternative_id_column is not None
                else ""
            )
        )

    def _no_x_message(self, name: str) -> str:
        """Diagnostic for a gene present in the primary ``var`` but whose
        matrix isn't readable — points the user at tuple routing / a named
        alternative source instead of the raw crash from the matrix layer.
        """
        msg = (
            f"No X (matrix) on the data source, but a var that listed {name!r}. "
            "You might want to use the (data_source, 'column') syntax, "
        )
        if not self._alternative_sources:
            return msg + "but you need to add an alternative source."
        named = [alt for alt in self._alternative_sources if alt.name is not None]
        if not named:
            return msg + "but none of the alternative sources had a name."
        with_x = [alt.name for alt in named if has_x(alt.ad, alt.layer)]
        return msg + f"Alternative sources with X are: {with_x!r}."

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
        unfiltered._filter_hard = False
        unfiltered._filter_cache = None
        series = fn(unfiltered)
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"Derived column {col!r} callable must return a pandas Series, "
                f"got {type(series).__name__}"
            )
        return series

    def _resolve_column_from(
        self, ad, name: str, layer: str = "X", transform=None
    ) -> ColumnData:
        """Resolve *name* against a single AnnData-like source.

        Same resolution order as documented for :meth:`get_column`, applied to
        the given ``ad`` (primary or alternative).  Works for ``AnnData`` and
        :class:`H5adFacade` alike since only the shared interface is used.
        Feature (gene) columns are read from *layer* (``'X'`` → ``.X``) and, if
        *transform* is given, passed through it; ``obs`` columns are returned
        untouched.  Raises ``KeyError`` if *name* is not found in this source.
        """
        if name in ad.obs:
            return ColumnData(ad.obs[name], name)

        def _extract(mask, col_name=None):
            idx = np.nonzero(mask)[0][0]
            if col_name is None:
                col_name = ad.var.index[idx]
            col = _source_matrix(ad, layer)[:, idx]
            if sp.issparse(col):
                col = col.toarray().ravel()
            else:
                col = np.asarray(col).ravel()
            if transform is not None:
                col = np.asarray(transform(col)).ravel()
            return ColumnData(pd.Series(col, index=ad.obs_names), col_name)

        if name in ad.var.index:
            return _extract(ad.var.index == name, name)
        if (
            self._alternative_id_column is not None
            and self._alternative_id_column in ad.var.columns
        ):
            alt_hits = ad.var[self._alternative_id_column] == name
            if alt_hits.sum() == 1:
                return _extract(alt_hits)
            if alt_hits.sum() > 1:
                raise KeyError(
                    f"Duplicate (n={alt_hits.sum()} in alternative-id column {self._alternative_id_column} \
                cannot resolve {name!r}"
                )
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
            if col in ad.var.columns:
                if gene_name in ad.var.index:
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

    def is_gene(self, name: Union[str, tuple[str, str]]) -> bool:
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

        For a source-routed embedding whose source drops some primary cells,
        those cells appear here as ``NaN`` coordinates (the reindex keeps them
        so the index stays aligned with :meth:`get_column` results).  Consumers
        that place cells in space (grid labels, histograms, Moran's I, …)
        should use :meth:`_finite_coordinates` to drop them.
        """
        coords = self._full_coordinates()
        mask = self._filter_mask()
        if mask is not None:
            coords = coords[mask]
        return coords

    def _finite_coordinates(self) -> pd.DataFrame:
        """Return :meth:`coordinates` with cells lacking an embedding dropped.

        A source-routed embedding reindexes onto the primary ``obs_names``, so
        cells absent from the embedding source become ``NaN`` coordinates.
        Spatial consumers (grid labels, histograms, Moran's I, boundaries) cannot
        place ``NaN`` cells and must drop them — this helper centralises that.
        The index is the subset of ``obs_names`` that has a finite embedding.
        No-op when the embedding is fully aligned with the primary source.
        """
        coords = self.coordinates()
        return coords.dropna(subset=["x", "y"])

    def _finite_full_coordinates(self) -> pd.DataFrame:
        """Return :meth:`_full_coordinates` with cells lacking an embedding dropped.

        Same rationale as :meth:`_finite_coordinates`, but ignoring the active
        cell filter — used by :meth:`bounds`/:meth:`full_bounds` so the frame
        is spanned only over cells that actually have a position.
        """
        return self._full_coordinates().dropna(subset=["x", "y"])

    def _full_coordinates(self) -> pd.DataFrame:
        """Return x, y DataFrame for *every* cell, ignoring the active filter.

        Index is the primary ``obs_names``.  Used by bounds/grid helpers so the
        embedding frame always reflects the complete dataset.  For a
        source-routed embedding, the array is read from the named alternative
        source and reindexed onto the primary ``obs_names`` — mirroring
        :meth:`get_column`'s alignment behaviour (extra cells dropped, primary
        cells absent from the source → ``NaN`` coordinates).
        """
        src_ad = self._embedding_source_ad()
        if self._embedding_source is None:
            key = self._embedding
        else:
            src_ad, key = self._resolved_source_embedding()
        if self._embedding_cols is not None:
            c1, c2 = self._embedding_cols
            arr = src_ad.obsm[key][:, [c1, c2]]
        else:
            arr = src_ad.obsm[key][:, :2]
        df = (
            pd.DataFrame(arr, columns=np.array(["x", "y"]))
            .assign(index=src_ad.obs.index)
            .set_index("index")
        )
        # Align the alternative source's embedding onto the primary obs_names
        # (no-op for the primary source / perfectly-aligned alternatives).
        if self._embedding_source is not None:
            df = df.reindex(self.ad.obs_names)
        return df

    def get_X_csr(self, layer=None):
        """Return the primary source's ``X`` as a scipy CSR sparse matrix.

        This is the bulk-access entry point used by analyses that touch many
        genes at once (e.g. :func:`~mbf_singlecell_plotter.transforms.compute_grid_moran`,
        which row-slices ``X`` per embedding bin).  CSR is row-major, so
        ``X[row_array]`` slicing is cheap.

        For :class:`H5adFacade` sources the whole matrix is loaded in one
        ``h5ad-inspect`` call (``export [--layer <key>] matrix_csr``) and
        cached.  For real ``AnnData`` (or anything exposing a ``get_X_csr``
        method) it forwards directly, converting dense/sparse matrices to CSR
        as needed.  Reads the primary source's configured :attr:`layer`; the
        primary :attr:`transform` is **not** applied here (bulk analyses such
        as Moran's I operate on the raw matrix).
        """
        ad = self.ad
        if layer is None:
            layer = self._layer
        else:
            pass
        getter = getattr(ad, "get_X_csr", None)
        if getter is not None:
            return getter() if layer == "X" else getter(layer)
        from scipy import sparse as sp

        X = _source_matrix(ad, layer)
        if sp.issparse(X):
            return X.tocsr()
        return sp.csr_matrix(np.asarray(X))

    def _get_filtered_X_csr(self):
        X = self.get_X_csr()
        mask = self._filter_mask()
        index = self.ad.obs_names
        if mask is not None:
            X = X[mask]
            index = index[mask]
        return X, index

    @computed_column()
    def per_cell_sum(self) -> pd.Series:
        X, index = self._get_filtered_X_csr()
        sums = X.sum(axis=1).A1
        return pd.Series(sums, index=index, name="per_cell_sum")

    @computed_column()
    def per_cell_missing_count(self) -> pd.Series:
        X, index = self._get_filtered_X_csr()
        missing_per_row = X.shape[1] - X.getnnz(axis=1)
        return pd.Series(missing_per_row, index)

    @computed_column()
    def n_genes_per_cell(self) -> pd.Series:
        """Return the number of genes with nonzero expression, per cell.

        Counts stored nonzero entries per row of :meth:`get_X_csr`. Honors
        the active cell filter (:meth:`set_filter` / :meth:`hard_filter`),
        matching :meth:`coordinates`.
        """
        X, index = self._get_filtered_X_csr()
        # that's only correct if 0 is missing
        # return pd.Series(np.diff(X.indptr), index=index, name="n_genes")
        mask = (X.data != 0) & np.isfinite(X.data)

        # Count valid stored entries per row
        counts = np.zeros(X.shape[0], dtype=np.int64)
        np.add.at(counts, np.repeat(np.arange(X.shape[0]), np.diff(X.indptr)), mask)

        return pd.Series(counts, index=index, name="n_genes")

    @computed_column()
    def n_cells_per_gene(self) -> pd.Series:
        """Return the number of cells with nonzero expression, per gene.

        Counts stored nonzero entries per column of :meth:`get_X_csr`.
        Honors the active cell filter (:meth:`set_filter` / :meth:`hard_filter`),
        matching :meth:`coordinates`.
        """
        X, _ = self._get_filtered_X_csr()
        # counts = np.bincount(X.indices, minlength=X.shape[1])
        # return pd.Series(counts, index=self.ad.var.index, name="n_cells")
        mask = (X.data != 0) & np.isfinite(X.data)
        counts = np.bincount(
            X.indices[mask],
            minlength=X.shape[1],
        )

        return pd.Series(counts, index=self.ad.var.index, name="n_cells")

    # ── grid helpers ────────────────────────────────────────────────────────

    def point_to_grid(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        x: float,
        y: float,
    ) -> tuple[str, str]:
        """Map a single point to a (letter, number) or (number, letter) grid cell."""
        x_step = (x_max - x_min) / self._grid_size
        y_step = (y_max - y_min) / self._grid_size
        if (x < x_min) or (x > x_max):
            raise ValueError(f"x={x} is outside of x_min={x_min}..x_max={x_max} range")
        if (y < y_min) or (y > y_max):
            raise ValueError(f"y={y} is outside of y_min={y_min}..y_max={y_max} range")
        x_index = min(int(round((x - x_min) / x_step)), self._grid_size - 1)
        y_index = min(int(round((y - y_min) / y_step)), self._grid_size - 1)
        letters = _LETTERS[: self._grid_size]
        non_letters = list(range(1, self._grid_size + 1))
        if self._grid_letters_on_vertical:
            letters_rev = letters[::-1]
            letter = letters_rev[y_index]
            number = non_letters[x_index]
            return (str(number), letter)
        else:
            letter = letters[x_index]
            number = non_letters[self._grid_size - 1 - y_index]
            return (letter, str(number))

    def grid_coordinate(self, x: float, y: float) -> str:
        """Return grid label (e.g. 'A1') for embedding coordinates."""
        x_min, x_max, y_min, y_max = self.bounds()
        parts = self.point_to_grid(x_min, x_max, y_min, y_max, x, y)
        if self._grid_letters_on_vertical:
            return f"{parts[1]}{parts[0]}"
        return f"{parts[0]}{parts[1]}"

    def grid_coordinates(self) -> pd.Series:
        """Return a Series of grid labels for all cells (vectorised).

        Cells lacking an embedding coordinate (see :meth:`_finite_coordinates`)
        are omitted — the returned index is a subset of ``obs_names`` rather
        than raising on the ``NaN`` → int cast.
        """
        coords = self._finite_coordinates()
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
            labels = [f"{ltr}{n}" for ltr, n in zip(letter_col, number_col)]
        else:
            letter_col = [letters[i] for i in x_idx]
            number_col = [self._grid_size - i for i in y_idx]
            labels = [f"{ltr}{n}" for ltr, n in zip(letter_col, number_col)]
        return pd.Series(labels, index=coords.index)

    def full_bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) from the data range, ignoring focus.

        The focus window and any *soft* cell filter are ignored.  A *hard* filter
        (:meth:`hard_filter`) is honoured, so the bounds reflect the kept subset
        (the grid is spanned over just the restricted cells).

        Cells lacking an embedding coordinate (see :meth:`_finite_coordinates`)
        are excluded so a source-routed embedding with dropped cells doesn't
        turn the range into ``NaN``.
        """
        coords = (
            self._finite_coordinates()
            if self._filter_hard
            else self._finite_full_coordinates()
        )
        if coords.empty:
            raise ValueError(
                "No cells have a finite embedding coordinate; cannot compute bounds."
            )
        return (
            float(coords["x"].min()),
            float(coords["x"].max()),
            float(coords["y"].min()),
            float(coords["y"].max()),
        )

    def grid_labels(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str], list[str]]:
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
            x_labels = list([str(x) for x in range(1, gs + 1)])
            y_labels = letters[::-1]
        else:
            # x-axis: letters A..Z; y-axis: numbers (1 at top = max y)
            x_labels = letters
            y_labels = list([str(x) for x in range(gs, 0, -1)])

        return x_positions, y_positions, x_labels, y_labels

    def cluster_centers(self, cluster_column: str) -> pd.DataFrame:
        """Return DataFrame with x, y, grid for each category in cluster_column.

        Cells lacking an embedding coordinate (see :meth:`_finite_coordinates`)
        are excluded before computing per-category medians, so a category
        with no positioned cells is dropped rather than producing a ``NaN``
        center that :meth:`grid_coordinate` cannot place.
        """
        col_data, col_name = self.get_column(cluster_column)
        if pd.api.types.is_numeric_dtype(col_data) and not isinstance(
            col_data.dtype, pd.CategoricalDtype
        ):
            raise ValueError(
                f"Column '{cluster_column}' contains numeric data. "
                "This function only works with categorical data."
            )
        coords = self._finite_coordinates()
        if self._focus is not None:
            x_min, x_max, y_min, y_max = self._focus
            mask = (
                (coords["x"] >= x_min)
                & (coords["x"] <= x_max)
                & (coords["y"] >= y_min)
                & (coords["y"] <= y_max)
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
    ) -> dict[tuple[int, int], list[str]]:
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
        facet: Optional[pd.Series] = None,
        facet_row: Optional[pd.Series] = None,
        facet_col: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Return DataFrame of grid-local category histograms.

        When *facet* (or *facet_row*/*facet_col*) is a :class:`pandas.Series`
        aligned to the embedding's cell index, the histogram is computed
        separately per facet group over the same shared grid, and an
        ordered-Categorical ``facet`` (or ``facet_row``/``facet_col``) column
        is added — ready for ``plotnine.facet_wrap``/``facet_grid``.
        """
        from .transforms import _facet_categories

        expr, _ = self.get_column(key)
        if isinstance(expr.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(
            expr
        ):
            # categorical or bool — both are handled as discrete categories below
            pass
        elif pd.api.types.is_numeric_dtype(expr):
            raise ValueError("category types only")
        # Drop cells lacking an embedding coordinate (see
        # :meth:`_finite_coordinates`) — a NaN coordinate can't be digitized
        # into a bin and would otherwise trip the `assert all(valid)` below.
        coords = self._finite_coordinates()
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

        valid_index = coords.index[valid]
        cell_data: dict[str, Any] = {
            "x_bin": x_bins[valid],
            "y_bin": y_bins[valid],
            "category": expr.loc[valid_index].values,
        }
        facet_cols = []
        facet_cats: dict[str, list] = {}
        for col_name, series in (
            ("facet", facet),
            ("facet_row", facet_row),
            ("facet_col", facet_col),
        ):
            if series is not None:
                aligned = series.reindex(valid_index)
                facet_cats[col_name] = [str(v) for v in _facet_categories(aligned)]
                cell_data[col_name] = aligned.astype(str).values
                facet_cols.append(col_name)

        try:
            df_cells = pd.DataFrame(cell_data)
        except ValueError as e:
            raise ValueError(f"Make sure your obs.keys are distinct!. Was: {e}")

        histogram: dict[str, list[Any]] = {
            "x": [],
            "y": [],
            "category": [],
            "frequency": [],
            "total": [],
        }
        for col_name in facet_cols:
            histogram[col_name] = []

        group_cols = facet_cols + ["x_bin", "y_bin"]
        for group_key, sub in df_cells.groupby(group_cols):  # ty: ignore
            if len(sub) < min_cells:
                continue
            if facet_cols:
                *facet_vals, ix, iy = group_key
            else:
                ix, iy = group_key
                facet_vals = []
            freqs = sub["category"].value_counts(normalize=True)
            for cat, freq in freqs.items():
                histogram["x"].append(ix)
                histogram["y"].append(iy)
                histogram["category"].append(cat)
                histogram["frequency"].append(freq)
                histogram["total"].append(len(sub))
                for col_name, val in zip(facet_cols, facet_vals):
                    histogram[col_name].append(val)

        df = pd.DataFrame(histogram)
        for col_name in facet_cols:
            df[col_name] = pd.Categorical(df[col_name], categories=facet_cats[col_name])
        return df
