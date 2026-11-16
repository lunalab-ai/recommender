"""Train, align, blend and inspect comparable rating predictions."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cf_synthesis import EvidenceCF
from .data import download_movielens_100k, load_movielens_100k
from .mf_sgd import MFSGD

PAIR = ["user_id", "movie_id"]


def temporal_split(ratings):
    """Global chronology, with same-second batches kept together."""
    ordered = ratings.sort_values(["timestamp", *PAIR]).reset_index(drop=True)
    if ordered.duplicated(PAIR).any() or ordered.timestamp.isna().any():
        raise ValueError("Unique user–movie rows with a recorded time are required.")
    first = ordered.timestamp.iloc[int(len(ordered) * .7)]
    second = ordered.timestamp.iloc[int(len(ordered) * .85)]
    parts = {"train": ordered[ordered.timestamp < first],
             "validation": ordered[(ordered.timestamp >= first) & (ordered.timestamp < second)],
             "test": ordered[ordered.timestamp >= second]}
    if any(part.empty for part in parts.values()):
        raise ValueError("All three chronological splits must be nonempty.")
    return parts


def align_predictions(base, cf, mf):
    """Refuse duplicates, missing and extra keys before merging by ID."""
    for frame in (base, cf, mf):
        if frame[PAIR].isna().any().any() or frame.duplicated(PAIR).any():
            raise ValueError("A user–movie key must be present and unique.")
    expected = set(base[PAIR].itertuples(index=False, name=None))
    for frame in (cf, mf):
        actual = set(frame[PAIR].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError(f"Keys differ: {len(expected-actual)} missing / {len(actual-expected)} extra.")
    return base.merge(cf, on=PAIR, how="left", validate="one_to_one").merge(
        mf, on=PAIR, how="left", validate="one_to_one"
    )


def blend_scores(rows, weight_mf):
    """Same-scale rating prediction: one MF share and its CF complement."""
    if not 0 <= weight_mf <= 1:
        raise ValueError("MF share must be between 0 and 1.")
    cf = rows["score_cf"].to_numpy(float)
    mf = rows["score_mf"].to_numpy(float)
    if not (np.isfinite(cf).all() and np.isfinite(mf).all()):
        raise ValueError("Blend requires two finite scores on each row.")
    return weight_mf * mf + (1 - weight_mf) * cf


def rmse(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise ValueError("RMSE requires matching nonempty one-dimensional arrays.")
    if not (np.isfinite(y).all() and np.isfinite(prediction).all()):
        raise ValueError("RMSE requires finite labels and predictions.")
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def select_weight(validation, grid=None, *, blend_fn=blend_scores):
    """Score only validation labels. Break exact ties toward lower MF share."""
    if grid is None:
        grid = np.linspace(0, 1, 101)
    curve = pd.DataFrame([
        (float(weight), rmse(validation.rating, blend_fn(validation, weight)))
        for weight in grid
    ], columns=["weight_mf", "validation_rmse"])
    best = curve.sort_values(["validation_rmse", "weight_mf"]).iloc[0]
    return float(best.weight_mf), curve


def predict_pair_tables(cf, mf, pairs):
    """Predict every key. Report support and apply one shared global fallback."""
    keys = pairs[PAIR].reset_index(drop=True)
    cf_details = cf.predict_details(keys)
    cf_supported = cf_details.basis.eq("cf").to_numpy()
    cf_predictions = np.where(cf_supported, cf_details.prediction.to_numpy(), cf.global_mean_)
    mf_supported = (keys.user_id.isin(mf.user_index_) & keys.movie_id.isin(mf.item_index_)).to_numpy()
    mf_predictions = np.array([
        mf.predict(uid, mid) if supported else cf.global_mean_
        for (uid, mid), supported in zip(keys.itertuples(index=False, name=None), mf_supported)
    ])
    return (keys.assign(score_cf=np.clip(cf_predictions, 1, 5), supported_cf=cf_supported),
            keys.assign(score_mf=np.clip(mf_predictions, 1, 5), supported_mf=mf_supported))


def fit_predictors(train):
    cf = EvidenceCF(k=37, min_common=3, min_neighbors=2, centered=True).fit(train)
    mf = MFSGD(n_factors=12, epochs=12, learning_rate=.02, regularization=.035,
               seed=20261117).fit(train)
    return cf, mf


def load_ratings(cache: Path):
    return load_movielens_100k(download_movielens_100k(cache), expect_full=True)


def model_report(test, chosen_weight, *, blend_fn=blend_scores, other_name="CF"):
    values = [(other_name, 0.), ("교재 출발값", .8), ("MF", 1.), ("검증 선택", chosen_weight)]
    return pd.DataFrame(
        [(name, weight, rmse(test.rating, blend_fn(test, weight)), len(test)) for name, weight in values],
        columns=["setting", "weight_mf", "test_rmse", "n_test"],
    )


def pick_catalog(train, movies, cf, mf, *, align_fn=align_predictions):
    """A demo viewer from training only and unrated popular candidates."""
    uid = int(train.groupby("user_id").size().sort_index().idxmax())
    observed = set(train.loc[train.user_id.eq(uid), "movie_id"])
    popular = train.groupby("movie_id").size().sort_values(ascending=False).index
    unseen = [mid for mid in popular if mid not in observed][:70]
    pairs = movies[movies.movie_id.isin(unseen)][["movie_id", "title"]].assign(user_id=uid)
    cf_table, mf_table = predict_pair_tables(cf, mf, pairs)
    candidates = align_fn(pairs, cf_table, mf_table)
    lower = candidates.nlargest(5, "score_cf")
    upper = candidates.nlargest(5, "score_mf")
    return pd.concat([lower, upper]).drop_duplicates(PAIR).reset_index(drop=True)


def rank_catalog(candidates, weight, *, blend_fn=blend_scores):
    scores = candidates.assign(
        cf_part=(1-weight)*candidates.score_cf,
        mf_part=weight*candidates.score_mf,
        combined=blend_fn(candidates, weight),
    )
    scores = scores.sort_values(["combined", "movie_id"], ascending=[False, True]).reset_index(drop=True)
    return scores.assign(rank=np.arange(1, len(scores) + 1))


def fit_deep_scores(train, validation_keys, test_keys):
    """Chapter 10-1. Same train and rows, with the earlier rating MLP architecture."""
    from .neural_recommenders import build_mlp
    from tensorflow import keras

    users = {value: idx for idx, value in enumerate(sorted(train.user_id.unique()), 1)}
    movies = {value: idx for idx, value in enumerate(sorted(train.movie_id.unique()), 1)}
    center = float(train.rating.mean())
    model = build_mlp(len(users) + 1, len(movies) + 1, k=8, hidden=(16, 8), seed=20261117)

    def arrays(rows):
        user = rows.user_id.map(users).fillna(0).to_numpy(np.int32)
        movie = rows.movie_id.map(movies).fillna(0).to_numpy(np.int32)
        return user.reshape(-1, 1), movie.reshape(-1, 1)

    optimizer = keras.optimizers.Adam(learning_rate=.005)
    model.compile(optimizer=optimizer, loss="mse")
    inputs = arrays(train)
    y = (train.rating.to_numpy(np.float32) - center).reshape(-1, 1)
    rng = np.random.default_rng(20261117)
    for _ in range(5):
        order = rng.permutation(len(train))
        for start in range(0, len(train), 512):
            batch = order[start:start + 512]
            model.train_on_batch([x[batch] for x in inputs], y[batch])
    output = []
    for keys in (validation_keys, test_keys):
        user, movie = arrays(keys)
        score = model([user, movie], training=False).numpy().ravel() + center
        score = np.where((user.ravel() > 0) & (movie.ravel() > 0), score, center)
        output.append(keys[PAIR].assign(score_cf=np.clip(score, 1, 5)))
    return output
