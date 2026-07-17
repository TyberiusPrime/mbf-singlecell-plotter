import pypipegraph2 as ppg2

from dataclasses import dataclass
from pathlib import Path
from .data import EmbeddingData
from typing import Dict, Callable, Sequence, Tuple, Optional, List
import pandas as pd


class PlotBuilder:
    def __init__(
        self,
        base_h5ad: Path,
        additional_h5ads: Dict[str, Path],
        column_sources: Dict[str, Callable[[EmbeddingData], pd.Series]],
        column_colors: Dict[str, Dict[str, str] | Sequence[str]],
    ): 
        self.plots = []


    def add_plot(self,
                 plot: Plot)
        ...


@dataclass
class Plot:
    # arguments on the data to plot
    column: str | Tuple[str, str]  # what to colour by / plot; names the output file
    filter: Optional[Callable[["EmbeddingData"], "pd.Series | np.ndarray"]] = None
    hard_filter: Optional[Callable[["EmbeddingData"], "pd.Series | np.ndarray"]] = None

    # where to store
    filename: Optional[str] = None  # override the output filename
    subfolder: Optional[str] = None  # subfolder for the plot, e.g. 'genes'

    # which plots to do.
    do_scatter: Optional[bool] = True  # emit a scatter plot
    do_grid_histogram: Optional[bool] = False  # also emit a grid histogram (off for continuous data)
    do_violin: Optional[List[str] | str] = None  # also emit violet plot(s)
    do_ridges: Optional[List[str]] = None
    do_global_histogram: bool = False  # an overall histogram
    do_global_relative_histogram: Optional[str] = None
    do_facet_violin: Optional[List[Tuple[str, str]]] = (
        None  #  also emit violet plot facteded
    )

    # plot style options
    facet: Optional[str | Tuple[str, str]] = (
        None  # split into panels; names the sub-directory
    )
    facet_args: Optional[dict] = None  # extra args to facet/facet_2d
    style: Optional[dict] = None  # extra style, composed on top of dot_size=1
    colors: Optional[List[str]] = None
    grey_border: Optional[bool] = None
    title: Optional[str | Callable[[str], str]] = None
    ascending: Optional[bool] = None
    dpi: int = 150
    anti_overplot_seed: Optional[int] = None


    def facet_name(self):
        assert self.facet is not None
        if isinstance(self.facet, str):
            return self.facet
        else:
            return f"{self.facet[0]}_vs_{self.facet[1]}"

    def get_facet_args(self):
        if self.facet_args is None:
            return {}
        else:
            return self.facet_args

    def derived_columns_needed(self):
        res = []
        if isinstance(self.column, str):
            res.append(self.column)
        else:
            res.append(self.column[1])
        if self.facet is not None:
            if isinstance(self.facet, str):
                res.append(self.facet)
            else:
                res.extend(self.facet)
        if self.do_border():
            res.append("constant")
        if self.violin is not None:
            if isinstance(self.violin, str):
                res.append(self.violin)
            else:
                res.extend(self.violin)
        if self.facet_violin is not None:
            for x_column, facet_column in self.facet_violin:
                res.append(x_column)
                res.append(facet_column)
        return res

    def do_border(self):
        if self.grey_border:
            return True
        elif self.facet is not None:
            return True
        return False

