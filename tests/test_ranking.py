"""Independent hand calculations, leakage checks and common candidate rules."""
import numpy as np
import pandas as pd
import pytest

from luna_recsys.data import MOVIELENS_100K_GENRES
from luna_recsys.ranking import FourMethodRecommender, METHODS, evaluate_rankings, ranking_metrics


def fixture():
    movies = pd.DataFrame({"movie_id": [1, 2, 3, 4, 5], "title": list("ABCDE")})
    for genre in MOVIELENS_100K_GENRES:
        movies[genre] = 0
    movies.loc[[0, 2], "Action"] = 1
    movies.loc[[1, 3], "Comedy"] = 1
    users = pd.DataFrame({"user_id": [1, 2, 3], "sex": ["F", "F", "M"]})
    train = pd.DataFrame([(1, 1, 5), (1, 2, 1), (2, 3, 4), (2, 4, 3), (3, 1, 2)],
                         columns=["user_id", "movie_id", "rating"])
    return train, users, movies


def test_hand_metric_oracle():
    # At ranks 1 and 3: DCG=1+1/2, IDCG=1+1/log2(3).
    m = ranking_metrics([10, 20, 30], {10, 30}, k=3)
    assert m["precision"] == pytest.approx(2/3)
    assert m["recall"] == 1
    assert m["ndcg"] == pytest.approx(1.5/(1+1/np.log2(3)))
    assert ranking_metrics([], {10}, k=3)["ndcg"] == 0
    with pytest.raises(ValueError):
        ranking_metrics([10, 10], {10})


def test_content_profile_and_unseen_candidates():
    train, users, movies = fixture()
    model = FourMethodRecommender().fit(train, users, movies)
    result = model.recommend(1, "content", k=10)
    assert result.movie_id.tolist() == [3, 4, 5]
    np.testing.assert_allclose(result.score, [1, 0, 0])
    for method in METHODS:
        assert set(model.recommend(1, method, 10).movie_id) == {3, 4, 5}
    assert result.iloc[-1].basis == "zero-genres"


def test_learning_snapshot_and_empty_profile_fallback():
    train, users, movies = fixture()
    model = FourMethodRecommender().fit(train, users, movies)
    before = model.score_catalog(1, "content")
    train["rating"] = 1
    movies["Action"] = 0
    pd.testing.assert_frame_equal(before, model.score_catalog(1, "content"))
    cold = model.recommend(999, "content", 5)
    assert cold.basis.str.startswith("empty-profile/").all()
    assert cold.movie_id.tolist() == model.recommend(999, "mean", 5).movie_id.tolist()
    # User 3 has train observations but none meeting the like threshold.
    assert model.recommend(3, "content").basis.str.startswith("empty-profile/").all()


def test_group_support_fallback_and_count_id_ties():
    train, users, movies = fixture()
    model = FourMethodRecommender(min_group_ratings=2).fit(train, users, movies)
    group = model.score_catalog(1, "group").set_index("movie_id")
    assert group.loc[1, "score"] == 3.5  # F group n=1: use movie mean (5+2)/2.
    assert group.loc[1, "basis"] == "movie"
    assert group.loc[5, "score"] == 3.0  # No train movie: global 15/5.
    assert group.loc[5, "basis"] == "global"
    assert model.recommend(999, "count", 5).movie_id.tolist() == [1, 2, 3, 4, 5]


def test_evaluation_label_changes_cannot_change_model_scores():
    train, users, movies = fixture()
    model = FourMethodRecommender().fit(train, users, movies)
    test = pd.DataFrame([(1, 3, 5), (1, 4, 2), (2, 1, 1), (3, 2, 5)],
                        columns=train.columns)
    before = {method: model.score_catalog(1, method) for method in METHODS}
    summary, details, metadata = evaluate_rankings(model, test, k=2)
    assert len(summary) == 4 and len(details) == 8
    assert metadata["evaluated_users"] == 2 and metadata["excluded_no_positive"] == 1
    changed = test.copy()
    changed["rating"] = 6 - changed.rating
    evaluate_rankings(model, changed, k=2)
    for method in METHODS:
        pd.testing.assert_frame_equal(before[method], model.score_catalog(1, method))
    with pytest.raises(ValueError, match="overlap"):
        evaluate_rankings(model, train)


def test_empty_candidate_list_is_explicit():
    train, users, movies = fixture()
    model = FourMethodRecommender().fit(train, users, movies)
    model.seen_[1] = set(movies.movie_id)
    assert model.recommend(1).empty
