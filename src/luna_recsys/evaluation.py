"""Shared holdout evaluation for explicit rating predictors, never count scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from .rating_models import MeanRatingPredictor, validate_ratings


@dataclass(frozen=True)
class RatingSplit:
    """분할 결과를 묶는 dataclass.

    train/test: 원본 관측에서 선택한 독립 DataFrame 사본. method는
    user-stratified 또는 random-small-data 문자열. frozen 속성이 내부표를
    불변으로 만들지는 않으므로 평가 중 train/test를 덮어쓰지 않는다."""
    train: pd.DataFrame
    test: pd.DataFrame
    method: str


def split_ratings(ratings: pd.DataFrame, *, test_size: float = 0.25, seed: int = 42) -> RatingSplit:
    """ratings의 행 위치를 한 번 나누어 RatingSplit(train,test,method)를 반환한다.

    ratings: user_id/movie_id/rating을 포함한 관측 DataFrame, 최소2행.
    test_size: 평가 비율0과1사이(기본0.25), seed: 난수 정수(기본42).
    가능하면 user_id로 층화한다. 너무 작은 표의 대체는 random-small-data로
    명시한다. 출력은 원본 인덱스/열을 보존하는 사본이며 원본을 변경하지 않는다.
    MovieLens100K의 기본 설정은75000/25000이다. 같은 사용자가 양쪽에 존재할
    수 있는 무작위 관측평가이며 시간분할/새 사용자평가가 아니다.
    중복 관측쌍은 분할 전 전체 로더에서 검사한다. 비율/행 수 오류는 ValueError."""
    validate_ratings(ratings)
    if not 0 < test_size < 1 or len(ratings) < 2:
        raise ValueError("Need at least two ratings and 0 < test_size < 1")
    counts = ratings["user_id"].value_counts()
    ntest = math.ceil(len(ratings) * test_size)
    if ntest >= len(ratings):
        raise ValueError("test_size leaves no training observations")
    feasible = counts.min() >= 2 and min(ntest, len(ratings) - ntest) >= len(counts)
    positions = np.arange(len(ratings))
    train, test = train_test_split(
        positions,
        test_size=test_size,
        random_state=seed,
        stratify=ratings["user_id"] if feasible else None,
    )
    return RatingSplit(
        ratings.iloc[train].copy(),
        ratings.iloc[test].copy(),
        "user-stratified" if feasible else "random-small-data",
    )


def evaluate_means(
    split: RatingSplit, users: pd.DataFrame, *, group_col: str = "sex", min_group_ratings: int = 1
) -> pd.DataFrame:
    """동일한 split에서 세 평균 회귀모형을 학습·평가해 DataFrame을 반환한다.

    split: RatingSplit의 train/test. users: 사용자ID와 집단속성 표.
    group_col: 기본sex, min_group_ratings: 기본1인 최소 집단×영화 표본.
    전체/영화/집단모형을 각각train에fit하고 test의ID로 예측한 뒤 test평점과
    RMSE를 계산한다. 출력 열 method/rmse/test_count/group_used/movie_used/
    global_used; 대체 건수의 합은 각 행의test_count다. 반올림 전에 평가하며
    입력이나분할을변경하지않는다. 낮은RMSE가 순위만족도 개선을보증하지않는다."""
    validate_ratings(split.test)
    rows = []
    for mode, label in [
        ("global", "전체 평균"),
        ("movie", "영화 평균"),
        ("group", "집단별 영화 평균"),
    ]:
        model = MeanRatingPredictor(
            mode, group_col=group_col, min_group_ratings=min_group_ratings
        ).fit(split.train, users)
        details = model.predict_details(split.test[["user_id", "movie_id"]])
        rows.append(
            {
                "method": label,
                "rmse": float(root_mean_squared_error(split.test["rating"], details["prediction"])),
                "test_count": len(split.test),
                "group_used": int(details["level"].eq("group").sum()),
                "movie_used": int(details["level"].eq("movie").sum()),
                "global_used": int(details["level"].eq("global").sum()),
            }
        )
    return pd.DataFrame(rows)
