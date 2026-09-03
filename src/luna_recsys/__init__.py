"""Reusable teaching utilities for the recommender-systems course."""

from .baselines import mean_rating_recommendations, popularity_ranking
from .data import (
    MovieLens100K,
    extract_movielens_100k_archive,
    load_movielens_100k,
    validate_movielens_100k,
)

__all__ = [
    "MovieLens100K",
    "extract_movielens_100k_archive",
    "load_movielens_100k",
    "mean_rating_recommendations",
    "popularity_ranking",
    "validate_movielens_100k",
]
__version__ = "0.1.0"
