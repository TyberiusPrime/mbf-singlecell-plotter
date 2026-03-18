"""Layer 1: data access (AnnData → DataFrames). No plotting."""

import copy
import collections
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from natsort import natsorted

from .util import map_to_integers, unmap

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ColumnData(NamedTuple):
    """Return type of :meth:`EmbeddingData.get_column`."""

    series: pd.Series
    name: str


class EmbeddingData:
    """Wraps an AnnData + embedding choice. Pure data extraction, no plotting."""

    def __init__(
        self,
        ad,
        embedding,
        alternative_id_column: Optional[str] = None,
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

    # ── viewport ────────────────────────────────────────────────────────────

    def focus_on(
        self,
        *,
        x: tuple,
        y: tuple,
    ) -> "EmbeddingData":
        """Return a new EmbeddingData restricted to the given coordinate window.

        Args:
            x: (x_min, x_max)
            y: (y_min, y_max)
        """
        new = copy.copy(self)
        new._focus = (x[0], x[1], y[0], y[1])
        return new

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

    def get_column(self, name: str) -> ColumnData:
        """Return ColumnData(series, column_name) for an obs column or gene."""
        ad = self.ad
        if name in ad.obs:
            return ColumnData(ad.obs[name], name)
        if name in ad.var.index:
            pdf = ad[:, ad.var.index == name].to_df()
            return ColumnData(pdf[name], name)
        if self._alternative_id_column is not None and self._alternative_id_column in ad.var.columns:
            alt_hits = ad.var[self._alternative_id_column] == name
            if alt_hits.sum() == 1:
                pdf = ad[:, alt_hits].to_df()
                col = pdf.columns[0]
                return ColumnData(pdf[col], col)
        if self._has_name_and_id:
            name_hits = ad.var.index.str.startswith(name + " ")
            if name_hits.sum() == 1:
                pdf = ad[:, name_hits].to_df()
                col = pdf.columns[0]
                return ColumnData(pdf[col], col)
            id_hits = ad.var.index.str.endswith(" " + name)
            if id_hits.sum() == 1:
                pdf = ad[:, id_hits].to_df()
                col = pdf.columns[0]
                return ColumnData(pdf[col], col)
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
