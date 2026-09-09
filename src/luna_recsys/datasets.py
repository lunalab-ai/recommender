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
    """준비된 데이터와 실제 모드를 함께 전달하는 dataclass.

    data: MovieLens100K(users/movies/ratings DataFrame). mode: movielens는
    다운로드/캐시, local은 명시한 전체 로컬 사본, synthetic은 독자 합성 자료.
    description: 화면에 보여 줄 설명 문자열. frozen=True는 속성 재할당을
    막지만 내부 DataFrame을 자동으로 불변으로 만들지는 않는다."""

    data: MovieLens100K
    mode: str
    description: str


def synthetic_movielens(seed: int = 2026) -> MovieLens100K:
    """독자적인 CC0 가상 MovieLens 형식 자료를 메모리에 생성한다.

    seed: 난수 정수, 기본2026. 반환 MovieLens100K는80명·18편·960평점이다.
    실제 MovieLens의 사용자·영화·행을 추출하거나 변환한 데이터가 아니다.
    파일이나 네트워크가 필요 없고 입력 데이터 변경도 없다.
    synthetic_movielens(seed=7).ratings로 생성 평점 표를 얻는다."""
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
    """캐시/다운로드/명시한 사본을 준비하고 데이터 종류까지 반환한다.

    cache_dir: 폴더 문자열 또는 Path, 기본 data/local. 다운로드 파일 저장 위치.
    local_dir: 기본None. 이미 가진 전체3파일의 폴더를 명시할 때만 사용한다.
    mode: auto(기본)는 다운로드 실패 때 합성 대안을 명확히 표시한다. real은
    실제 자료 준비 실패를 오류로 전달한다. synthetic은 다운로드 없이 가상
    자료를 생성한다. 명시한 잘못된 local_dir는 합성으로 대체하지 않는다.
    timeout: 네트워크 요청 제한 초, 기본20, 양수. 전체 함수 총시간 상한은 아니다.

    반환 PreparedMovieLens의 data.users/data.movies/data.ratings는 표이며
    mode/description은 출처를 설명한다. 실제 전체 자료는943명/1682편/100000평점.
    폴더 생성과 다운로드/캐시 사용이 부작용이다. 실제 비교 실습은 반드시
    prepare_movielens(\"data/local\", mode=\"real\")로 호출한다.
    설정 오류는 ValueError, real 다운로드 실패는 OSError 등으로 보고한다."""
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
