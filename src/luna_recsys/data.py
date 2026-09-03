"""Dataset helpers for local and Colab teaching environments."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_100K_MD5 = "0e33842e24a9c977be4e0107933c0723"
MOVIELENS_100K_FILES = ("u.user", "u.item", "u.data")
MOVIELENS_100K_USER_COLUMNS = ("user_id", "age", "sex", "occupation", "zip_code")
MOVIELENS_100K_GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
MOVIELENS_100K_ITEM_COLUMNS = (
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    *MOVIELENS_100K_GENRES,
)
MOVIELENS_100K_RATING_COLUMNS = ("user_id", "movie_id", "rating", "timestamp")


@dataclass(frozen=True)
class MovieLens100K:
    """The three MovieLens 100K tables used in the early lessons."""

    users: pd.DataFrame
    movies: pd.DataFrame
    ratings: pd.DataFrame


def course_root() -> Path:
    """Return the repository root when running from an editable checkout."""
    return Path(__file__).resolve().parents[2]


def data_path(filename: str, *, local_only: bool = False) -> Path:
    """Resolve a course data file without hard-coding a user's home directory.

    The ``RECOMMENDER_DATA_DIR`` environment variable takes precedence. Otherwise,
    local-only files are searched under ``data/local`` and bundled samples under
    ``data/sample``.
    """
    custom_dir = os.getenv("RECOMMENDER_DATA_DIR")
    if custom_dir:
        candidate = Path(custom_dir).expanduser() / filename
    else:
        subdir = "local" if local_only else "sample"
        candidate = course_root() / "data" / subdir / filename
    if not candidate.exists():
        raise FileNotFoundError(
            f"Data file not found: {candidate}. See data/README.md and data/registry.yml."
        )
    return candidate


def _movielens_directory(path: str | Path) -> Path:
    root = Path(path)
    direct = root
    nested = root / "ml-100k"
    for candidate in (direct, nested):
        if all((candidate / filename).is_file() for filename in MOVIELENS_100K_FILES):
            return candidate
    raise FileNotFoundError(
        f"MovieLens 100K files were not found under {root}. "
        f"Expected: {', '.join(MOVIELENS_100K_FILES)}"
    )


def load_movielens_100k(path: str | Path, *, expect_full: bool = False) -> MovieLens100K:
    """Read ``u.user``, ``u.item`` and ``u.data`` from a MovieLens 100K folder."""
    data_dir = _movielens_directory(path)
    users = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=MOVIELENS_100K_USER_COLUMNS,
        encoding="latin-1",
    )
    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=MOVIELENS_100K_ITEM_COLUMNS,
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=MOVIELENS_100K_RATING_COLUMNS,
        encoding="latin-1",
    )
    dataset = MovieLens100K(users=users, movies=movies, ratings=ratings)
    validate_movielens_100k(dataset, expect_full=expect_full)
    return dataset


def validate_movielens_100k(dataset: MovieLens100K, *, expect_full: bool = False) -> None:
    """Raise ``ValueError`` when the three MovieLens tables are inconsistent."""
    users, movies, ratings = dataset.users, dataset.movies, dataset.ratings
    if users["user_id"].duplicated().any():
        raise ValueError("u.user contains duplicate user_id values")
    if movies["movie_id"].duplicated().any():
        raise ValueError("u.item contains duplicate movie_id values")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("u.data ratings must be between 1 and 5")

    unknown_users = set(ratings["user_id"]).difference(users["user_id"])
    unknown_movies = set(ratings["movie_id"]).difference(movies["movie_id"])
    if unknown_users:
        raise ValueError(f"u.data references unknown users: {sorted(unknown_users)[:5]}")
    if unknown_movies:
        raise ValueError(f"u.data references unknown movies: {sorted(unknown_movies)[:5]}")

    if expect_full:
        actual = (len(users), len(movies), len(ratings))
        expected = (943, 1682, 100000)
        if actual != expected:
            raise ValueError(
                "Unexpected MovieLens 100K table sizes: "
                f"users={actual[0]}, movies={actual[1]}, ratings={actual[2]}"
            )


def extract_movielens_100k_archive(
    archive_path: str | Path,
    cache_dir: str | Path,
    *,
    expected_md5: str = MOVIELENS_100K_MD5,
) -> Path:
    """Verify a MovieLens archive and extract only the three lesson files."""
    archive_file = Path(archive_path)
    digest = hashlib.md5(archive_file.read_bytes(), usedforsecurity=False).hexdigest()
    if digest.lower() != expected_md5.lower():
        raise ValueError(
            f"MovieLens archive checksum mismatch: expected {expected_md5}, got {digest}"
        )

    expected_members = {f"ml-100k/{filename}" for filename in MOVIELENS_100K_FILES}
    destination = Path(cache_dir) / "ml-100k"
    with zipfile.ZipFile(archive_file) as archive:
        names = archive.namelist()
        missing = expected_members.difference(names)
        if missing:
            raise ValueError(f"MovieLens archive is missing files: {sorted(missing)}")
        duplicates = sorted(member for member in expected_members if names.count(member) != 1)
        if duplicates:
            raise ValueError(f"MovieLens archive has duplicate files: {duplicates}")

        destination.mkdir(parents=True, exist_ok=True)
        for member in sorted(expected_members):
            target = destination / Path(member).name
            with archive.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
    return destination


def download_movielens_100k(
    cache_dir: str | Path,
    *,
    url: str = MOVIELENS_100K_URL,
    expected_md5: str = MOVIELENS_100K_MD5,
) -> Path:
    """Download and safely extract the three lesson files from the official archive.

    Existing complete files are reused. The archive is verified before extraction and
    only the three explicitly allowlisted members are written to ``cache_dir``.
    """
    cache = Path(cache_dir)
    destination = cache / "ml-100k"
    if all((destination / filename).is_file() for filename in MOVIELENS_100K_FILES):
        return destination

    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".zip", delete=False) as handle:
        archive_path = Path(handle.name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, archive_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        destination = extract_movielens_100k_archive(
            archive_path,
            cache,
            expected_md5=expected_md5,
        )
    finally:
        archive_path.unlink(missing_ok=True)
    return destination
