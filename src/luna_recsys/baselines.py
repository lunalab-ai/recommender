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


def mean_rating_recommendations(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    genre: str | None = None,
    min_ratings: int = 50,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return a deterministic mean-rating baseline for the MovieLens lesson app."""
    rating_columns = {"movie_id", "rating"}
    movie_columns = {"movie_id", "title"}
    if missing := rating_columns.difference(ratings.columns):
        raise ValueError(f"Missing rating columns: {sorted(missing)}")
    if missing := movie_columns.difference(movies.columns):
        raise ValueError(f"Missing movie columns: {sorted(missing)}")
    if min_ratings < 1:
        raise ValueError("min_ratings must be at least 1")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if genre is not None and genre not in movies.columns:
        raise ValueError(f"Unknown genre column: {genre}")

    summary = (
        ratings.groupby("movie_id", as_index=False)["rating"]
        .agg(rating_count="count", mean_rating="mean")
        .query("rating_count >= @min_ratings")
    )
    movie_view = movies
    if genre is not None:
        movie_view = movies.loc[movies[genre].eq(1)]

    result = summary.merge(movie_view[["movie_id", "title"]], on="movie_id", how="inner")
    return (
        result.sort_values(
            ["mean_rating", "rating_count", "title", "movie_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .head(top_n)
        .loc[:, ["movie_id", "title", "mean_rating", "rating_count"]]
        .assign(mean_rating=lambda frame: frame["mean_rating"].round(3))
        .reset_index(drop=True)
    )
