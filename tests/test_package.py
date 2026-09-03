from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from luna_recsys.baselines import mean_rating_recommendations, popularity_ranking
from luna_recsys.data import (
    download_movielens_100k,
    extract_movielens_100k_archive,
    load_movielens_100k,
)
from luna_recsys.demo_app import recommend_for_app


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
    result = mean_rating_recommendations(
        ratings, movies, genre="Action", min_ratings=2, top_n=2
    )
    assert result["title"].tolist() == ["Alpha", "Gamma"]
    assert result["mean_rating"].tolist() == [4.5, 4.5]


def test_recommend_for_app_uses_korean_output_labels() -> None:
    ratings = pd.DataFrame({"movie_id": [1, 1], "rating": [5, 4]})
    movies = pd.DataFrame({"movie_id": [1], "title": ["Alpha"]})
    result = recommend_for_app(ratings, movies, "전체", 1, 5)
    assert result.columns.tolist() == ["영화 ID", "영화 제목", "평균 평점", "평점 수"]
