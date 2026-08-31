"""Simple baseline recommenders used in the early course sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def popularity_ranking(
    ratings: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    item_col: str = "item_id",
    rating_col: str = "rating",
    min_count: int = 1,
) -> pd.DataFrame:
    """Rank items by interaction count, then mean rating.

    Parameters
    ----------
    ratings:
        A pandas DataFrame or iterable of row mappings.
    item_col:
        Item identifier column.
    rating_col:
        Explicit rating column.
    min_count:
        Minimum number of ratings required for an item.

    Returns
    -------
    pandas.DataFrame
        Columns: item identifier, ``rating_count``, ``mean_rating``.
    """
    frame = ratings.copy() if isinstance(ratings, pd.DataFrame) else pd.DataFrame(ratings)
    required = {item_col, rating_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    result = (
        frame.groupby(item_col, as_index=False)[rating_col]
        .agg(rating_count="count", mean_rating="mean")
        .query("rating_count >= @min_count")
        .sort_values(
            ["rating_count", "mean_rating", item_col],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return result
