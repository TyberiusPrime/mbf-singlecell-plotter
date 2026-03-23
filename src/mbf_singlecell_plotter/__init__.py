"""mbf-singlecell-plotter — public API."""

from .util import map_to_integers, unmap
from .theme import DEFAULT_COLORS_BORDERS, DEFAULT_COLORS_CATEGORIES, embedding_theme
from .data import EmbeddingData, ColumnData
from .transforms import prepare_density_df, prepare_scatter_df, compute_boundaries, prepare_embedding_color_df
from .plots import ScatterPlotter, BorderConfig, GridConfig
from .colorbar import sc_guide_colorbar

__all__ = [
    # data
    "EmbeddingData",
    "ColumnData",
    # plotting
    "ScatterPlotter",
    "BorderConfig",
    "GridConfig",
    # transforms
    "prepare_density_df",
    "prepare_scatter_df",
    "compute_boundaries",
    "prepare_embedding_color_df",
    # theme / colors
    "DEFAULT_COLORS_BORDERS",
    "DEFAULT_COLORS_CATEGORIES",
    "embedding_theme",
    # colorbar
    "sc_guide_colorbar",
    # utilities
    "map_to_integers",
    "unmap",
]
