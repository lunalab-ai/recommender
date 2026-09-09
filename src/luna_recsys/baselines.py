"""Simple baseline recommenders used in the early course sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def popularity_ranking(
    ratings: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    item_col: str = "item_id",
    rating_col: str = "rating",
    min_count: int = 1,
) -> pd.DataFrame:
    """Rank items by interaction count, then mean rating.

    Parameters
    ----------
    ratings:
        A pandas DataFrame or iterable of row mappings.
    item_col:
        Item identifier column.
    rating_col:
        Explicit rating column.
    min_count:
        Minimum number of ratings required for an item.

    Returns
    -------
    pandas.DataFrame
        Columns: item identifier, ``rating_count``, ``mean_rating``.
    """
    frame = ratings.copy() if isinstance(ratings, pd.DataFrame) else pd.DataFrame(ratings)
    required = {item_col, rating_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    result = (
        frame.groupby(item_col, as_index=False)[rating_col]
        .agg(rating_count="count", mean_rating="mean")
        .query("rating_count >= @min_count")
        .sort_values(
            ["rating_count", "mean_rating", item_col],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return result


def mean_rating_recommendations(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    genre: str | None = None,
    min_ratings: int = 50,
    top_n: int = 10,
) -> pd.DataFrame:
    """영화 평균 평점의 상위 목록을 반환하는 W01B 함수.

    ratings: movie_id/rating 열의 DataFrame. movies: movie_id/title과 필요한
    장르 열의 DataFrame. genre: 기본None이면 전체 장르, 문자열이면 해당 열1만.
    min_ratings: 최소 관측 수, 기본50, 1이상. top_n: 최대 목록 길이, 기본10.

    영화별 개수/평균 집계→최소 표본 필터→영화 제목/장르 연결→평균 내림차순,
    개수 내림차순, 제목/ID 오름차순으로 정렬한다. 출력은 최대top_n행의
    movie_id/title/mean_rating/rating_count 표이며 마지막 표시에서만 평균을
    소수3자리로 반올림한다. 조건에 맞는 영화가 없으면 빈 표다. 사용자 이력
    제외/학습평가분할은 수행하지 않는다. 입력 표를 수정하지 않는다. 잘못된
    열/장르/최소값은 ValueError. W02B 공통 순위평가 API와 동점/후보 규칙이 다르다."""
    rating_columns = {"movie_id", "rating"}
    movie_columns = {"movie_id", "title"}
    if missing := rating_columns.difference(ratings.columns):
        raise ValueError(f"Missing rating columns: {sorted(missing)}")
    if missing := movie_columns.difference(movies.columns):
        raise ValueError(f"Missing movie columns: {sorted(missing)}")
    if min_ratings < 1:
        raise ValueError("min_ratings must be at least 1")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if genre is not None and genre not in movies.columns:
        raise ValueError(f"Unknown genre column: {genre}")

    summary = (
        ratings.groupby("movie_id", as_index=False)["rating"]
        .agg(rating_count="count", mean_rating="mean")
        .query("rating_count >= @min_ratings")
    )
    movie_view = movies
    if genre is not None:
        movie_view = movies.loc[movies[genre].eq(1)]

    result = summary.merge(movie_view[["movie_id", "title"]], on="movie_id", how="inner")
    return (
        result.sort_values(
            ["mean_rating", "rating_count", "title", "movie_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .head(top_n)
        .loc[:, ["movie_id", "title", "mean_rating", "rating_count"]]
        .assign(mean_rating=lambda frame: frame["mean_rating"].round(3))
        .reset_index(drop=True)
    )


def baseline_recommendations(
    ratings: pd.DataFrame, movies: pd.DataFrame, *, method: str = "mean",
    users: pd.DataFrame | None = None, group_col: str = "sex", group_value: str = "F",
    genre: str | None = None, min_ratings: int = 5, top_n: int = 10,
) -> pd.DataFrame:
    """W02A 앱의 평점수/영화평균/집단평균 목록을 공통 형식으로 반환한다.

    ratings: user_id/movie_id/rating DataFrame. movies: movie_id/title/장르 표.
    method: mean(기본)/count/group. users: 기본None, group에서 사용자속성필수.
    group_col: 집단 열 기본sex; group_value: 선택집단 기본F, 문자열로 비교.
    genre: 기본None(전체), 지정한 장르0/1열로 필터. min_ratings: 기본5, top_n:
    기본10, 모두 양수. 입력 표는 변경하지 않는다.

    반환 최대top_n행은 movie_id/title/mean_rating/rating_count/basis 열.
    count는 개수→평균 내림차순→ID, mean/group은 mean_rating_recommendations의
    동점 규칙을 따른다. group은 선택집단 관측만 집계하고 조건에 맞는 목록이
    전혀 없으면 전체 사용자 영화평균 목록으로 전환해 basis에 표시한다.
    개별 평점의 group→movie→global 대체와 다르다. 학습 이력 제외/holdout은
    이 함수 자체에서 하지 않으며 W02B 성능표는 FourMethodRecommender로 계산한다."""
    if method not in {"count", "mean", "group"}:
        raise ValueError("method must be count, mean or group")
    if min_ratings < 1 or top_n < 1:
        raise ValueError("min_ratings and top_n must be positive")
    selected = ratings
    basis = "전체 사용자"
    if method == "group":
        if users is None or group_col not in users or users["user_id"].duplicated().any():
            raise ValueError("Group recommendation requires unique user metadata")
        ids = users.loc[users[group_col].astype(str).eq(str(group_value)), "user_id"]
        selected = ratings.loc[ratings["user_id"].isin(ids)]
        basis = f"{group_col}={group_value}"
    if method == "count":
        result = popularity_ranking(selected, item_col="movie_id", min_count=min_ratings)
        view = movies
        if genre is not None:
            if genre not in movies:
                raise ValueError(f"Unknown genre: {genre}")
            view = movies.loc[movies[genre].eq(1)]
        result = result.merge(view[["movie_id", "title"]], on="movie_id", validate="one_to_one")
        result = result.sort_values(["rating_count", "mean_rating", "movie_id"], ascending=[False, False, True]).head(top_n)
        result = result[["movie_id", "title", "mean_rating", "rating_count"]].copy()
        result["mean_rating"] = result["mean_rating"].round(3)
    else:
        result = mean_rating_recommendations(selected, movies, genre=genre, min_ratings=min_ratings, top_n=top_n)
    if method == "group" and result.empty:
        result = mean_rating_recommendations(ratings, movies, genre=genre, min_ratings=min_ratings, top_n=top_n)
        basis = "집단 표본 부족 → 전체 사용자 평균"
    return result.assign(basis=basis).reset_index(drop=True)
