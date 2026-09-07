"""Shared holdout evaluation for explicit rating predictors, never count scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from .rating_models import MeanRatingPredictor, validate_ratings


@dataclass(frozen=True)
class RatingSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    method: str


def split_ratings(ratings: pd.DataFrame, *, test_size: float = 0.25, seed: int = 42) -> RatingSplit:
    """Split observation positions once, stratifying users when feasible.

    Tiny data may not support stratification. In that case use a seeded random
    holdout and return its explicit method; never silently call it stratified.
    This is a random-observation exercise, not a temporal production evaluation.
    """
    validate_ratings(ratings)
    if not 0 < test_size < 1 or len(ratings) < 2:
        raise ValueError("Need at least two ratings and 0 < test_size < 1")
    counts = ratings["user_id"].value_counts()
    ntest = math.ceil(len(ratings) * test_size)
    if ntest >= len(ratings):
        raise ValueError("test_size leaves no training observations")
    feasible = counts.min() >= 2 and min(ntest, len(ratings) - ntest) >= len(counts)
    positions = np.arange(len(ratings))
    train, test = train_test_split(
        positions,
        test_size=test_size,
        random_state=seed,
        stratify=ratings["user_id"] if feasible else None,
    )
    return RatingSplit(
        ratings.iloc[train].copy(),
        ratings.iloc[test].copy(),
        "user-stratified" if feasible else "random-small-data",
    )


def evaluate_means(
    split: RatingSplit, users: pd.DataFrame, *, group_col: str = "sex", min_group_ratings: int = 1
) -> pd.DataFrame:
    """Fit three methods on identical train rows and score identical test rows.

    Returned fallback counts help distinguish group predictions from substitutes.
    No rounding is performed before RMSE. Lower held-out error is better; an
    improvement is not promised and does not establish recommendation satisfaction.
    """
    validate_ratings(split.test)
    rows = []
    for mode, label in [
        ("global", "전체 평균"),
        ("movie", "영화 평균"),
        ("group", "집단별 영화 평균"),
    ]:
        model = MeanRatingPredictor(
            mode, group_col=group_col, min_group_ratings=min_group_ratings
        ).fit(split.train, users)
        details = model.predict_details(split.test[["user_id", "movie_id"]])
        rows.append(
            {
                "method": label,
                "rmse": float(root_mean_squared_error(split.test["rating"], details["prediction"])),
                "test_count": len(split.test),
                "group_used": int(details["level"].eq("group").sum()),
                "movie_used": int(details["level"].eq("movie").sum()),
                "global_used": int(details["level"].eq("global").sum()),
            }
        )
    return pd.DataFrame(rows)
