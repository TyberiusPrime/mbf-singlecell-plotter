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


def within_grid(embedding_data, top_left, bottom_right):
    """Return boolean series, cells within grid rect"""
    coords = embedding_data.grid_coordinates()
    top = top_left[0]
    left = int(top_left[1:])
    bottom = bottom_right[0]
    right = int(bottom_right[1:])
    return ~pd.isnull(coords) & (
        (coords.str[0] >= top)
        & (coords.str[1:].astype(int) >= left)
        & (coords.str[0] <= bottom)
        & (coords.str[1:].astype(int) <= right)
    )
