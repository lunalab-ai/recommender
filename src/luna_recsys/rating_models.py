"""Training-only mean predictors for repeated use across course sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Check the observation schema without printing private rating rows."""
    required = {"user_id", "movie_id", "rating"}
    if missing := required.difference(ratings.columns):
        raise ValueError(f"Missing rating columns: {sorted(missing)}")
    if ratings.empty or ratings[list(required)].isna().any().any():
        raise ValueError("Ratings must be nonempty without missing identifiers or values")
    if (
        not pd.api.types.is_numeric_dtype(ratings["rating"])
        or not ratings["rating"].between(1, 5).all()
    ):
        raise ValueError("Ratings must be finite numbers between 1 and 5")


class MeanRatingPredictor:
    """Predict by global, movie, or group/movie training means.

    ``MeanRatingPredictor('group', group_col='occupation').fit(train, users)``
    learns once; ``predict(test)`` accepts identifiers, not test ratings.
    Group fallback: sufficiently supported group/movie mean -> movie -> global.
    The object snapshots training statistics, so later frame edits cannot leak in.
    """

    def __init__(self, mode: str = "movie", *, group_col: str = "sex", min_group_ratings: int = 1):
        if mode not in {"global", "movie", "group"}:
            raise ValueError("mode must be global, movie or group")
        if not isinstance(min_group_ratings, int) or min_group_ratings < 1:
            raise ValueError("min_group_ratings must be a positive integer")
        self.mode, self.group_col, self.min_group_ratings = mode, group_col, min_group_ratings

    def fit(self, ratings: pd.DataFrame, users: pd.DataFrame | None = None) -> MeanRatingPredictor:
        """Learn all statistics only from the supplied training observations."""
        validate_ratings(ratings)
        # Validate before changing learned state, so a failed refit is not partial.
        if self.mode == "group":
            if users is None or not {"user_id", self.group_col}.issubset(users.columns):
                raise ValueError("Users must include user_id and the selected group column")
            if (
                users["user_id"].duplicated().any()
                or users[["user_id", self.group_col]].isna().any().any()
            ):
                raise ValueError("Users must have unique nonmissing IDs and group values")
            if not ratings["user_id"].isin(users["user_id"]).all():
                raise ValueError("Training users lack group metadata")
            groups = users.set_index("user_id")[self.group_col].copy()
            joined = ratings.assign(_group=ratings["user_id"].map(groups))
            summary = joined.groupby(["_group", "movie_id"])["rating"].agg(["mean", "count"])
            self.user_groups_ = groups
            self.group_means_ = summary.loc[
                summary["count"].ge(self.min_group_ratings), "mean"
            ].copy()
        self.global_mean_ = float(ratings["rating"].mean())
        self.movie_means_ = ratings.groupby("movie_id")["rating"].mean().copy()
        return self

    def predict_details(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Return predictions and the actual source of each estimate, in input order."""
        if not hasattr(self, "global_mean_"):
            raise ValueError("Call fit before predict")
        if not {"user_id", "movie_id"}.issubset(pairs.columns):
            raise ValueError("Prediction pairs need user_id and movie_id")
        values = np.full(len(pairs), self.global_mean_, dtype=float)
        levels = np.full(len(pairs), "global", dtype=object)
        if self.mode != "global":
            movie_values = pairs["movie_id"].map(self.movie_means_).to_numpy(dtype=float)
            known = np.isfinite(movie_values)
            values[known], levels[known] = movie_values[known], "movie"
        if self.mode == "group":
            keys = pd.MultiIndex.from_arrays(
                [pairs["user_id"].map(self.user_groups_), pairs["movie_id"]]
            )
            group_values = self.group_means_.reindex(keys).to_numpy(dtype=float)
            known = np.isfinite(group_values)
            values[known], levels[known] = group_values[known], "group"
        return pd.DataFrame({"prediction": values, "level": levels}, index=pairs.index)

    def predict(self, pairs: pd.DataFrame) -> pd.Series:
        """Predict a 1–5 rating, preserving row order and even duplicate indices."""
        return self.predict_details(pairs)["prediction"]
