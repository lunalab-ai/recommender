from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from luna_recsys.baselines import (
    baseline_recommendations,
    mean_rating_recommendations,
    popularity_ranking,
)
from luna_recsys.data import (
    download_movielens_100k,
    extract_movielens_100k_archive,
    load_movielens_100k,
)
from luna_recsys.datasets import prepare_movielens, synthetic_movielens
from luna_recsys.demo_app import recommend_for_app
from luna_recsys.evaluation import evaluate_means, split_ratings
from luna_recsys.rating_models import MeanRatingPredictor


def test_predictor_fallback_order_and_input_order():
    ratings = pd.DataFrame({'user_id':[1,2,1,2], 'movie_id':[1,1,2,2], 'rating':[5,1,3,1]})
    users = pd.DataFrame({'user_id':[1,2], 'sex':['F','M']})
    model = MeanRatingPredictor('group').fit(ratings, users)
    pairs = pd.DataFrame({'user_id':[1,99,1,2], 'movie_id':[1,1,99,2]}, index=[7,7,2,9])
    result = model.predict_details(pairs)
    assert result.prediction.tolist() == [5,3,2.5,1]
    assert result.level.tolist() == ['group','movie','global','group']
    assert result.index.tolist() == [7,7,2,9]
    # Editing frames after fit or supplying test labels cannot change learned means.
    ratings['rating'] = 1
    users['sex'] = 'unknown'
    assert model.predict(pairs.assign(rating=5)).tolist() == [5,3,2.5,1]


def test_predictor_support_filter_and_failed_fit():
    ratings = pd.DataFrame({'user_id':[1,2,1,2], 'movie_id':[1,1,2,2], 'rating':[5,1,3,1]})
    users = pd.DataFrame({'user_id':[1,2], 'sex':['F','M']})
    model = MeanRatingPredictor('group', min_group_ratings=2).fit(ratings, users)
    result = model.predict_details(ratings)
    assert result.prediction.tolist() == [3,3,2,2]
    assert result.level.eq('movie').all()
    with pytest.raises(ValueError, match='unique'):
        model.fit(ratings, pd.concat([users, users]))
    assert model.predict(ratings).tolist() == [3,3,2,2]
    with pytest.raises(ValueError, match='fit'):
        MeanRatingPredictor().predict(ratings)
    with pytest.raises(ValueError, match='between'):
        MeanRatingPredictor().fit(ratings.assign(rating=float('inf')))


def test_split_is_reproducible_and_observations_are_disjoint():
    data = synthetic_movielens()
    ratings = data.ratings.assign(observation=range(len(data.ratings)))
    ratings.index = [0] * len(ratings)  # position-based splitting must still work
    split = split_ratings(ratings)
    again = split_ratings(ratings)
    assert split.method == 'user-stratified'
    assert split.train.observation.tolist() == again.train.observation.tolist()
    assert set(split.train.observation).isdisjoint(split.test.observation)
    assert len(split.train) + len(split.test) == len(ratings)
    result = evaluate_means(split, data.users)
    assert result.test_count.eq(len(split.test)).all()
    assert result[['group_used','movie_used','global_used']].sum(axis=1).eq(len(split.test)).all()


def test_small_data_split_is_labeled_and_evaluation_matches_hand_calculation():
    from luna_recsys.evaluation import RatingSplit
    tiny = pd.DataFrame({'user_id':[1,2], 'movie_id':[1,2], 'rating':[5,1]})
    assert split_ratings(tiny).method == 'random-small-data'
    train = pd.DataFrame({'user_id':[1,2], 'movie_id':[1,1], 'rating':[5,1]})
    test = pd.DataFrame({'user_id':[1,2], 'movie_id':[1,1], 'rating':[4,2]})
    users = pd.DataFrame({'user_id':[1,2], 'sex':['F','M']})
    result = evaluate_means(RatingSplit(train,test,'manual'), users)
    assert result.rmse.tolist() == pytest.approx([1,1,1])


def test_group_listing_and_fallback_are_explicit():
    data = synthetic_movielens()
    grouped = baseline_recommendations(data.ratings, data.movies, users=data.users,
        method='group',group_col='occupation',group_value='student',min_ratings=1,top_n=4)
    assert len(grouped) == 4 and grouped.basis.eq('occupation=student').all()
    fallback = baseline_recommendations(data.ratings, data.movies, users=data.users,
        method='group',group_col='occupation',group_value='missing',min_ratings=1,top_n=4)
    assert len(fallback) == 4 and fallback.basis.str.contains('전체').all()
    assert baseline_recommendations(data.ratings, data.movies, min_ratings=99999).empty
    count = baseline_recommendations(data.ratings, data.movies, method='count')
    assert count.rating_count.is_monotonic_decreasing


def test_dataset_auto_failure_is_visible_and_explicit_local_errors_are_not_hidden(tmp_path, monkeypatch):
    import luna_recsys.datasets as datasets
    def fail(*args, **kwargs):
        raise OSError('offline')
    monkeypatch.setattr(datasets, 'download_movielens_100k', fail)
    prepared = prepare_movielens(tmp_path)
    assert prepared.mode == 'synthetic' and '실패' in prepared.description
    assert len(prepared.data.ratings) == 960
    assert prepared.data.ratings.equals(synthetic_movielens().ratings)
    with pytest.raises(FileNotFoundError):
        prepare_movielens(local_dir=tmp_path/'missing')
    with pytest.raises(ValueError):
        prepare_movielens(mode='typo')


def test_dataset_offline_mode_never_downloads(tmp_path, monkeypatch):
    import luna_recsys.datasets as datasets
    def forbidden(*args, **kwargs):
        raise AssertionError('offline mode must not download')
    monkeypatch.setattr(datasets, 'download_movielens_100k', forbidden)
    assert prepare_movielens(tmp_path, mode='synthetic').mode == 'synthetic'


def test_popularity_ranking_orders_by_count_then_mean() -> None:
    ratings = pd.DataFrame(
        {
            "item_id": ["B", "A", "A", "B", "C"],
            "rating": [5.0, 4.0, 5.0, 3.0, 5.0],
        }
    )
    result = popularity_ranking(ratings)
    assert result["item_id"].tolist() == ["A", "B", "C"]
    assert result.loc[0, "rating_count"] == 2
    assert result.loc[0, "mean_rating"] == pytest.approx(4.5)


def test_popularity_ranking_validates_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        popularity_ranking([{"item_id": "A"}])


def _movie_row(movie_id: int, title: str, *, action: int = 0, comedy: int = 0) -> str:
    genres = [0] * 19
    genres[1] = action
    genres[5] = comedy
    values = [movie_id, title, "01-Jan-1995", "", "https://example.invalid", *genres]
    return "|".join(map(str, values))


def _write_tiny_movielens(root: Path) -> Path:
    data_dir = root / "ml-100k"
    data_dir.mkdir(parents=True)
    (data_dir / "u.user").write_text(
        "1|20|F|student|00000\n2|21|M|student|00000\n", encoding="latin-1"
    )
    (data_dir / "u.item").write_text(
        _movie_row(1, "Alpha", action=1)
        + "\n"
        + _movie_row(2, "Beta", comedy=1)
        + "\n"
        + _movie_row(3, "Gamma", action=1)
        + "\n",
        encoding="latin-1",
    )
    (data_dir / "u.data").write_text(
        "1\t1\t5\t1\n2\t1\t4\t2\n1\t2\t5\t3\n1\t3\t4\t4\n2\t3\t5\t5\n",
        encoding="latin-1",
    )
    return data_dir


def test_load_movielens_100k_reads_three_formats(tmp_path: Path) -> None:
    data_dir = _write_tiny_movielens(tmp_path)
    dataset = load_movielens_100k(data_dir)
    assert dataset.users.shape == (2, 5)
    assert dataset.movies.shape == (3, 24)
    assert dataset.ratings.shape == (5, 4)
    assert dataset.movies.loc[0, "title"] == "Alpha"


def test_bundled_ml100k_schema_fallback_is_valid() -> None:
    sample_dir = Path(__file__).parents[1] / "data" / "sample" / "ml100k_tiny"
    dataset = load_movielens_100k(sample_dir)
    assert dataset.users.shape == (10, 5)
    assert dataset.movies.shape == (12, 24)
    assert dataset.ratings.shape == (60, 4)


def test_load_movielens_100k_validates_foreign_keys(tmp_path: Path) -> None:
    data_dir = _write_tiny_movielens(tmp_path)
    with (data_dir / "u.data").open("a", encoding="latin-1") as handle:
        handle.write("999\t1\t5\t6\n")
    with pytest.raises(ValueError, match="unknown users"):
        load_movielens_100k(data_dir)


def test_download_movielens_100k_extracts_only_lesson_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_data = _write_tiny_movielens(source)
    archive_path = tmp_path / "ml-100k.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for filename in ("u.user", "u.item", "u.data"):
            archive.write(source_data / filename, f"ml-100k/{filename}")
        archive.writestr("ml-100k/not-for-this-lesson.txt", "do not extract")
    digest = hashlib.md5(archive_path.read_bytes(), usedforsecurity=False).hexdigest()

    output = download_movielens_100k(
        tmp_path / "cache", url=archive_path.as_uri(), expected_md5=digest
    )
    assert sorted(path.name for path in output.iterdir()) == ["u.data", "u.item", "u.user"]


def test_uploaded_movielens_archive_rejects_wrong_checksum(tmp_path: Path) -> None:
    archive_path = tmp_path / "ml-100k.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("ml-100k/u.user", "1|20|F|student|00000\n")
        archive.writestr("ml-100k/u.item", "placeholder\n")
        archive.writestr("ml-100k/u.data", "1\t1\t5\t1\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        extract_movielens_100k_archive(
            archive_path,
            tmp_path / "cache",
            expected_md5="0" * 32,
        )


def test_download_uses_verified_mirror_after_tls_failure(tmp_path, monkeypatch):
    import io
    import urllib.error
    from luna_recsys import data

    files = _write_tiny_movielens(tmp_path / "source")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in data.MOVIELENS_100K_FILES:
            archive.write(files / name, f"ml-100k/{name}")
    payload = buffer.getvalue()
    expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    monkeypatch.setattr(data, "MOVIELENS_100K_MD5", expected)
    monkeypatch.setattr(data, "MOVIELENS_100K_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        if url == data.MOVIELENS_100K_URL:
            raise urllib.error.URLError("certificate has expired")
        return io.BytesIO(payload)

    monkeypatch.setattr(data.urllib.request, "urlopen", fetch)
    result = download_movielens_100k(tmp_path / "cache", expected_md5=expected)
    assert calls == [data.MOVIELENS_100K_URL, data.MOVIELENS_100K_MIRROR]
    assert len(load_movielens_100k(result).ratings) > 0
    monkeypatch.setattr(data, "MOVIELENS_100K_SHA256", "0" * 64)
    with pytest.raises(OSError, match="SHA-256 mismatch"):
        download_movielens_100k(tmp_path / "bad-cache", expected_md5=expected)
    assert not (tmp_path / "bad-cache/ml-100k/u.data").exists()


def test_real_mode_never_silently_returns_synthetic(tmp_path, monkeypatch):
    from luna_recsys import datasets

    def unavailable(*args, **kwargs):
        raise OSError("all sources unavailable")

    monkeypatch.setattr(datasets, "download_movielens_100k", unavailable)
    with pytest.raises(OSError, match="all sources unavailable"):
        prepare_movielens(tmp_path, mode="real")


def test_uploaded_movielens_archive_rejects_missing_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "ml-100k.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("ml-100k/u.user", "1|20|F|student|00000\n")
        archive.writestr("ml-100k/u.data", "1\t1\t5\t1\n")
    digest = hashlib.md5(archive_path.read_bytes(), usedforsecurity=False).hexdigest()

    with pytest.raises(ValueError, match="missing files"):
        extract_movielens_100k_archive(
            archive_path,
            tmp_path / "cache",
            expected_md5=digest,
        )


def test_mean_rating_recommendations_filters_and_breaks_ties() -> None:
    ratings = pd.DataFrame(
        {
            "movie_id": [1, 1, 2, 3, 3],
            "rating": [5, 4, 5, 4, 5],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "title": ["Alpha", "Beta", "Gamma"],
            "Action": [1, 0, 1],
        }
    )
    result = mean_rating_recommendations(ratings, movies, genre="Action", min_ratings=2, top_n=2)
    assert result["title"].tolist() == ["Alpha", "Gamma"]
    assert result["mean_rating"].tolist() == [4.5, 4.5]


def test_recommend_for_app_uses_korean_output_labels() -> None:
    ratings = pd.DataFrame({"movie_id": [1, 1], "rating": [5, 4]})
    movies = pd.DataFrame({"movie_id": [1], "title": ["Alpha"]})
    result = recommend_for_app(ratings, movies, "전체", 1, 5)
    assert result.columns.tolist() == ["영화 ID", "영화 제목", "평균 평점", "평점 수"]
