"""One-call dataset preparation, with explicit provenance and an offline alternative."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MOVIELENS_100K_GENRES, MovieLens100K, download_movielens_100k, load_movielens_100k


@dataclass(frozen=True)
class PreparedMovieLens:
    """Tables plus the actual data mode; fallback results are never labeled MovieLens."""

    data: MovieLens100K
    mode: str
    description: str


def synthetic_movielens(seed: int = 2026) -> MovieLens100K:
    """Create independent fictional tables (CC0); no rows are derived from MovieLens.

    This deterministic alternative is also available in an installed wheel, so it
    does not depend on a repository-relative sample folder.
    """
    rng = np.random.default_rng(seed)
    users = pd.DataFrame(
        {
            "user_id": range(1, 81),
            "age": [20 + i % 30 for i in range(80)],
            "sex": ["F" if i % 2 == 0 else "M" for i in range(80)],
            "occupation": [["student", "engineer", "artist", "educator"][i % 4] for i in range(80)],
            "zip_code": ["00000"] * 80,
        }
    )
    movies = pd.DataFrame(
        {"movie_id": range(1, 19), "title": [f"Fictional Movie {i:02}" for i in range(1, 19)]}
    )
    for i, genre in enumerate(MOVIELENS_100K_GENRES):
        movies[genre] = [int(i > 0 and (m + i) % 4 == 0) for m in range(18)]
    rows = []
    for user in range(1, 81):
        for movie in rng.choice(np.arange(1, 19), size=12, replace=False):
            preference = 0.7 if movie % 4 == user % 4 else -0.2
            rating = int(np.clip(np.rint(3.2 + preference + rng.normal(0, 1)), 1, 5))
            rows.append((user, int(movie), rating, len(rows) + 1))
    return MovieLens100K(
        users, movies, pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    )


def prepare_movielens(
    cache_dir: str | Path = "data/local",
    *,
    local_dir: str | Path | None = None,
    mode: str = "auto",
    timeout: float = 20,
) -> PreparedMovieLens:
    """Load existing local files, or download once, or use labeled fictional data.

    Example: ``prepared = prepare_movielens(); ratings = prepared.data.ratings``.
    ``local_dir`` is an explicit existing copy, never a URL or mandatory upload.
    An invalid explicit copy raises an error instead of silently replacing it.
    In auto mode an unavailable/invalid download falls back without disabling TLS.
    Use ``mode="real"`` to require real data and raise if all download sources fail.
    """
    if mode not in {"auto", "real", "synthetic"}:
        raise ValueError("mode must be auto, real or synthetic")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if mode == "synthetic":
        return PreparedMovieLens(
            synthetic_movielens(),
            "synthetic",
            "독자적 합성 데이터 · 실제 MovieLens 결과가 아닙니다.",
        )
    if local_dir is not None:
        data = load_movielens_100k(local_dir, expect_full=True)
        return PreparedMovieLens(
            data, "local", "기존 로컬 MovieLens 사본 · 세 표의 크기와 ID 연결 확인"
        )
    try:
        directory = download_movielens_100k(cache_dir, timeout=timeout)
        data = load_movielens_100k(directory, expect_full=True)
        return PreparedMovieLens(data, "movielens", "MovieLens 100K · 다운로드 또는 기존 캐시")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if mode == "real":
            raise OSError(f"실제 MovieLens 데이터를 준비하지 못했습니다. 합성 데이터로 전환하지 않습니다. {exc}") from exc
        return PreparedMovieLens(
            synthetic_movielens(),
            "synthetic",
            f"자동 다운로드/캐시 확인 실패 ({type(exc).__name__}). 독자적 합성 데이터로 계속합니다. 실제 MovieLens 결과가 아닙니다.",
        )
