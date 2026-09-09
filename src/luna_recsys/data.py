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
MOVIELENS_100K_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
MOVIELENS_100K_MIRROR = (
    "https://media.githubusercontent.com/media/dgraph-io/dgraph-benchmarks/"
    "0399f1c120208d3e78431eb7bf4ecafdcee15d8d/movielens/conv100k/ml-100k.zip"
)
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
    """사용자·영화·평점 세 표를 묶는 dataclass.

    users: user_id/age/sex/occupation/zip_code의 DataFrame.
    movies: movie_id/title/날짜/URL과19개 장르0·1열의 DataFrame.
    ratings: user_id/movie_id/rating/timestamp의 DataFrame.
    생성자 자체는 읽기/검사를 수행하지 않는다. load_movielens_100k가 파일을
    읽고 validate_movielens_100k가 검사한다. 개인 속성/원본 평점을 공개 출력하지 않는다."""

    users: pd.DataFrame
    movies: pd.DataFrame
    ratings: pd.DataFrame


def course_root() -> Path:
    """인자 없이 편집 가능한 checkout의 저장소 루트 Path를 반환한다.

    현재 모듈 위치에서 계산하며 파일 생성/다운로드를 하지 않는다. wheel로
    설치한 경우 같은 위치에 저장소 샘플이 있다고 보장하지 않으므로 학생은
    prepare_movielens의 다운로드/독자 합성 생성기를 사용한다."""
    return Path(__file__).resolve().parents[2]


def data_path(filename: str, *, local_only: bool = False) -> Path:
    """filename의 데이터 파일 Path를 확인하여 반환한다.

    filename: 상대 파일 이름 문자열. local_only: 기본False이면 data/sample,
    True이면 data/local에서 찾는다. RECOMMENDER_DATA_DIR 환경변수가 있으면
    그 폴더가 우선한다. 파일이 없으면 FileNotFoundError이며 자동 다운로드를
    하지 않는다. data_path(\"ratings_tiny.csv\")처럼 사용한다."""
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
    """path 또는 path/ml-100k에서 필수3파일이 모두 있는 폴더 Path를 찾는다.

    path: 문자열 또는 Path. u.user/u.item/u.data의 존재를 검사하고 없으면
    FileNotFoundError. 파일 내용이나 크기는 이 함수에서 검증하지 않는다."""
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
    """path의 세 파일을 읽고 검사한 MovieLens100K를 반환한다.

    path: u.user/u.item/u.data 또는 ml-100k 하위 폴더를 가진 폴더 경로.
    expect_full: 기본False이면 작은 형식 예제도 허용, True이면943명·1682편·
    100000평점 크기까지 요구한다. user/item은 |, data는 탭 구분자이며 헤더가
    없으므로 정해진 열 이름을 사용한다. latin-1로 읽는다. 원본 파일은 수정하지
    않는다. 파일 부재/형식/참조ID/평점 범위 오류를 전달한다.
    예: data = load_movielens_100k(\"data/local/ml-100k\", expect_full=True)."""
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
    """dataset 세 표의 키·범위·참조 관계를 검사하며 반환값은 None이다.

    dataset: MovieLens100K. expect_full: 기본False, True이면 공식 전체 크기
    943/1682/100000도 요구한다. 중복 사용자/영화ID, 중복 사용자–영화 평점,
    잘못된1–5점 범위나 참조할 수 없는 ID를 거부한다. 파일 쓰기/표 수정은 없고
    유효하지 않으면 ValueError. 성공은 데이터 사용 조건의 법적 인증이 아니다."""
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
    """archive_path ZIP을 검증해 필요한3파일만 cache_dir 아래에 꺼낸다.

    archive_path/cache_dir: 문자열 또는 Path. expected_md5: 기본 공식ZIP의
    MD5 문자열(MOVIELENS_100K_MD5). 일치하는 archive에서 u.user/u.item/u.data만
    사용하며 다른 멤버를 임의 경로로 풀지 않는다. 반환은 로딩할 데이터 폴더 Path.
    디렉터리 생성/파일 쓰기가 발생한다. 잘못된 checksum·필수 멤버 누락·ZIP
    오류는 예외로 전달한다. 학생 수동 업로드도 같은 검증 원리를 사용한다."""
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
    timeout: float = 60,
) -> Path:
    """cache_dir에서 재사용하거나 검증된 ZIP을 받아 데이터 폴더 Path를 반환한다.

    cache_dir: 저장 폴더 문자열/Path. url: 기본 GroupLens 공식ZIP 주소.
    expected_md5: 기본 공식MD5, timeout: 요청 제한 초(기본60). 테스트용 URL과
    검증값 변경은 가능한 API지만 일반 실습은 기본 주소/해시를 유지한다.
    기존 필수3파일이 있으면 재사용한다. 공식 경로 실패 시 공식기본설정에서만
    고정 HTTPS 미러를 시도하고 SHA256/MD5를 확인한다. TLS 검증을 끄지 않는다.
    다운로드/임시파일/캐시 쓰기가 발생한다. 모든 경로 실패는 OSError이며
    반환 후 전체 데이터 크기/키는 load_movielens_100k(expect_full=True)로 검사한다."""
    cache = Path(cache_dir)
    destination = cache / "ml-100k"
    if all((destination / filename).is_file() for filename in MOVIELENS_100K_FILES):
        return destination

    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".zip", delete=False) as handle:
        archive_path = Path(handle.name)
    try:
        urls = [url]
        standard = url == MOVIELENS_100K_URL and expected_md5 == MOVIELENS_100K_MD5
        if standard:
            urls.append(MOVIELENS_100K_MIRROR)
        failures = []
        for source_url in urls:
            try:
                with urllib.request.urlopen(source_url, timeout=timeout) as response, archive_path.open("wb") as out:
                    shutil.copyfileobj(response, out)
                if standard and hashlib.sha256(archive_path.read_bytes()).hexdigest() != MOVIELENS_100K_SHA256:
                    raise ValueError("MovieLens archive SHA-256 mismatch")
                destination = extract_movielens_100k_archive(
                    archive_path, cache, expected_md5=expected_md5,
                )
                break
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                if not standard:
                    raise
                failures.append(f"{source_url}: {exc}")
        else:
            raise OSError("MovieLens download failed on all verified sources: " + "; ".join(failures))
    finally:
        archive_path.unlink(missing_ok=True)
    return destination
