"""Reusable teaching utilities for the recommender-systems course."""

from .baselines import mean_rating_recommendations, popularity_ranking
from .data import (
    MovieLens100K,
    extract_movielens_100k_archive,
    load_movielens_100k,
    validate_movielens_100k,
)
from .datasets import PreparedMovieLens, prepare_movielens, synthetic_movielens
from .evaluation import evaluate_means, split_ratings
from .rating_models import MeanRatingPredictor

__all__ = [
    "PreparedMovieLens", "prepare_movielens", "synthetic_movielens",
    "MeanRatingPredictor", "split_ratings", "evaluate_means",
    "MovieLens100K",
    "extract_movielens_100k_archive",
    "load_movielens_100k",
    "mean_rating_recommendations",
    "popularity_ranking",
    "validate_movielens_100k",
]
__version__ = "0.1.0"
