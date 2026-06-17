"""Layer 1: data access (AnnData → DataFrames). No plotting."""

import copy
import collections
from pathlib import Path
from typing import NamedTuple, Optional

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


class EmbeddingData:
    """Wraps an AnnData + embedding choice. Pure data extraction, no plotting."""

    def __init__(
        self,
        ad,
        embedding,
        alternative_id_column: Optional[str] = None,
        alternative_sources: Optional[list] = None,
        grid_size: int = 12,
        grid_letters_on_vertical: bool = False,
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

    def bounds(self) -> tuple:
        """Return (x_min, x_max, y_min, y_max) — from focus if set, else full data range."""
        if self._focus is not None:
            return self._focus
        coords = self.coordinates()
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
          from the alternative source registered under *source_name* (see
          :meth:`add_alternative_source`).  The result is reindexed onto the
          primary ``obs_names``.  ``KeyError`` is raised if no alternative is
          registered under that name or it does not contain *column*.

        * Otherwise *name* is a string: the primary source is consulted first
          and, on a miss, each registered alternative is tried in registration
          order (the existing fallback search).  The first hit is reindexed to
          the primary ``obs_names`` (extra cells dropped, missing cells → NaN).
        """
        # Explicit routing to a named alternative source.
        if isinstance(name, tuple):
            if len(name) != 2:
                raise KeyError(
                    "Tuple column lookup must be (source_name, column), "
                    f"got {name!r}"
                )
            src_name, col = name
            for alt in self._alternative_sources:
                if alt.name == src_name:
                    resolved = self._resolve_column_from(alt.ad, col)
                    return ColumnData(
                        resolved.series.reindex(self.ad.obs_names), resolved.name
                    )
            registered = [
                a.name for a in self._alternative_sources if a.name is not None
            ]
            raise KeyError(
                f"No alternative source named {src_name!r}. "
                f"Registered names: {registered!r}"
            )

        # Plain string: primary first, then fallback chain.
        try:
            return self._resolve_column_from(self.ad, name)
        except KeyError:
            pass

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
                f" or any of {len(self._alternative_sources)} alternative source(s)"
                if self._alternative_sources
                else ""
            )
        )

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

    def coordinates(self) -> pd.DataFrame:
        """Return DataFrame with x, y columns, indexed by obs index."""
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
        """Return (x_min, x_max, y_min, y_max) from the full data range, ignoring focus."""
        coords = self.coordinates()
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
        if pd.api.types.is_numeric_dtype(expr) and not isinstance(
            expr.dtype, pd.CategoricalDtype
        ):
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
