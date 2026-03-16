import numpy as np
import pandas as pd


def map_to_integers(series, upper, min=None, max=None):
    """Map values into 0...upper-1."""
    min = series.min() if min is None else min
    max = series.max() if max is None else max
    zero_to_one = (series - min) / (max - min)
    scaled = zero_to_one * (upper - 1)
    return scaled.astype(int)


def unmap(series, org_series, res):
    """Inverse of map_to_integers."""
    zero_to_one = series / (res - 1)
    mult = zero_to_one * (org_series.max() - org_series.min())
    shifted = mult + org_series.min()
    return shifted
