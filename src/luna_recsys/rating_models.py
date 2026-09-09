"""Training-only mean predictors for repeated use across course sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_ratings(ratings: pd.DataFrame) -> None:
    """평점 학습 표의 필수 열과 값 범위를 검사한다.

    ratings: pandas DataFrame. user_id/movie_id/rating 열, 1행 이상,
    세 열에 결측 없음, rating은 유한한 숫자 1–5여야 한다. 추가 열은 허용.
    반환 None; 성공해도 표를 바꾸거나 출력하지 않는다. 위반은 ValueError.
    이 함수만으로 ID 참조 관계·중복 관측을 검사하지는 않는다. 원본 파일은
    validate_movielens_100k, 순위 비교는 FourMethodRecommender.fit도 확인한다."""
    required = {"user_id", "movie_id", "rating"}
    if missing := required.difference(ratings.columns):
        raise ValueError(f"Missing rating columns: {sorted(missing)}")
    if ratings.empty or ratings[list(required)].isna().any().any():
        raise ValueError("Ratings must be nonempty without missing identifiers or values")
    if (
        not pd.api.types.is_numeric_dtype(ratings["rating"])
        or not ratings["rating"].between(1, 5).all()
    ):
        raise ValueError("Ratings must be finite numbers between 1 and 5")


class MeanRatingPredictor:
    """관측 평점을 이용해 범주별 상수 회귀모형을 학습하는 클래스.

    Parameters
    ----------
    mode : str, default 'movie'
        'global': 모든 행에 같은 학습 전체 평균. 'movie': 영화별 학습 평균.
        'group': 사용자의 집단과 영화가 같은 학습 행의 평균.
    group_col : str, default 'sex'
        mode='group'일 때 users에서 읽는 집단 속성 열. 'occupation'도 가능.
    min_group_ratings : int, default 1
        집단×영화 평균의 최소 학습 표본 수. 부족하면 영화 평균으로 대체.

    학습은 L(theta)=sum((rating-theta)**2)를 최소화하는 theta=평균을
    계산하는 것이다. 정답 rating을 쓰는 지도학습이며 경사하강법은 불필요하다.
    생성자는 설정만 보관하고 fit이 통계 모수를 추정한다.

    Attributes after fit
    --------------------
    global_mean_ : float, 학습 전체 평균.
    movie_means_ : Series, movie_id가 인덱스인 영화별 평균.
    user_groups_ : Series, group 모드의 user_id→집단 매핑.
    group_means_ : Series, (집단, movie_id) 다중 인덱스 평균; 최소 표본 충족만.
    밑줄 접미사는 학습 후 생기는 속성이라는 수업 코드의 명명 관례다.

    Example
    -------
    model = MeanRatingPredictor('group', group_col='occupation', min_group_ratings=2)
    model.fit(train, users)
    y_hat = model.predict(test[['user_id', 'movie_id']])

    집단 평균→영화 평균→전체 평균 순으로 대체하므로 새 집단/영화도 처리한다.
    원본 표를 나중에 수정해도 이미 학습한 통계는 바뀌지 않는다. 재학습하려면
    fit을 다시 호출한다. predict는 test 정답을 계산에 쓰지 않는다."""

    def __init__(self, mode: str = "movie", *, group_col: str = "sex", min_group_ratings: int = 1):
        """mode/group_col/min_group_ratings 설정을 검사해 보관한다.

        mode는 global/movie/group, group_col은 집단 열 이름(기본sex),
        min_group_ratings는 1 이상 정수(기본1)다. 데이터 인자를 받지 않고
        평균도 계산하지 않는다. 반환값 None; 잘못된 설정은 ValueError.
        객체를 만드는 것과 학습하는 것을 분리해 같은 설정을 재사용한다."""
        if mode not in {"global", "movie", "group"}:
            raise ValueError("mode must be global, movie or group")
        if not isinstance(min_group_ratings, int) or min_group_ratings < 1:
            raise ValueError("min_group_ratings must be a positive integer")
        self.mode, self.group_col, self.min_group_ratings = mode, group_col, min_group_ratings

    def fit(self, ratings: pd.DataFrame, users: pd.DataFrame | None = None) -> MeanRatingPredictor:
        """ratings의 정답 평점으로 평균 모수를 추정하고 self를 반환한다.

        ratings: 학습 N행 DataFrame(user_id/movie_id/rating 필수). test를 넣지
        않는다. users: 기본 None; group 모드에서는 user_id와 group_col을 가진
        사용자 메타데이터가 필수다. ID는 유일하고 결측이 없어야 하며 ratings의
        모든 사용자에 집단 정보가 있어야 한다. 다른 모드에서는 users를 무시한다.

        1. 전체 rating 평균을 global_mean_에 저장한다.
        2. movie_id별 rating 평균을 movie_means_에 저장한다.
        3. group 모드는 user_id로 속성을 연결해 (집단,movie_id)별 평균/개수를
           구하고 min_group_ratings 이상인 평균만 group_means_에 저장한다.
        실제 코드는 입력 검사를 먼저 끝낸 뒤 이 통계를 갱신한다. 입력 표는
        수정하지 않는다. 반환 self이므로 MeanRatingPredictor().fit(train)처럼
        연결 호출할 수 있다. 잘못된 표/집단 매핑에는 ValueError가 발생한다."""
        validate_ratings(ratings)
        # Validate before changing learned state, so a failed refit is not partial.
        if self.mode == "group":
            if users is None or not {"user_id", self.group_col}.issubset(users.columns):
                raise ValueError("Users must include user_id and the selected group column")
            if (
                users["user_id"].duplicated().any()
                or users[["user_id", self.group_col]].isna().any().any()
            ):
                raise ValueError("Users must have unique nonmissing IDs and group values")
            if not ratings["user_id"].isin(users["user_id"]).all():
                raise ValueError("Training users lack group metadata")
            groups = users.set_index("user_id")[self.group_col].copy()
            joined = ratings.assign(_group=ratings["user_id"].map(groups))
            summary = joined.groupby(["_group", "movie_id"])["rating"].agg(["mean", "count"])
            self.user_groups_ = groups
            self.group_means_ = summary.loc[
                summary["count"].ge(self.min_group_ratings), "mean"
            ].copy()
        self.global_mean_ = float(ratings["rating"].mean())
        self.movie_means_ = ratings.groupby("movie_id")["rating"].mean().copy()
        return self

    def predict_details(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """pairs의 각 사용자–영화 쌍에 예측값과 실제 사용 기준을 반환한다.

        pairs: DataFrame, user_id/movie_id 필수. rating 열은 필요 없으며 있어도
        사용하지 않는다. 반환은 같은 행 수·순서·인덱스의 DataFrame이며
        prediction(float,1–5)과 level('group'/'movie'/'global') 열을 갖는다.
        학습이 된 경우 빈 입력도 빈 출력을 반환한다. 원자료나 학습 상태는 수정하지
        않는다. fit 전 또는 필수 열 누락은 ValueError.

        먼저 전체 평균으로 배열을 채운다. 영화 평균이 있으면 덮어쓰고, group
        모드에서 최소 표본을 충족하는 집단×영화 평균이 있으면 다시 덮어쓴다.
        이 순서가 집단→영화→전체 대체 규칙을 구현한다. 평균은 반올림하지 않는다.
        level을 세면 집단 모형이 실제로 집단 평균을 쓴 비율을 알 수 있다."""
        if not hasattr(self, "global_mean_"):
            raise ValueError("Call fit before predict")
        if not {"user_id", "movie_id"}.issubset(pairs.columns):
            raise ValueError("Prediction pairs need user_id and movie_id")
        values = np.full(len(pairs), self.global_mean_, dtype=float)
        levels = np.full(len(pairs), "global", dtype=object)
        if self.mode != "global":
            movie_values = pairs["movie_id"].map(self.movie_means_).to_numpy(dtype=float)
            known = np.isfinite(movie_values)
            values[known], levels[known] = movie_values[known], "movie"
        if self.mode == "group":
            keys = pd.MultiIndex.from_arrays(
                [pairs["user_id"].map(self.user_groups_), pairs["movie_id"]]
            )
            group_values = self.group_means_.reindex(keys).to_numpy(dtype=float)
            known = np.isfinite(group_values)
            values[known], levels[known] = group_values[known], "group"
        return pd.DataFrame({"prediction": values, "level": levels}, index=pairs.index)

    def predict(self, pairs: pd.DataFrame) -> pd.Series:
        """pairs(user_id/movie_id 표)의 평점 예측 Series를 반환한다.

        predict_details(pairs)의 prediction 열만 꺼낸 편의 메소드다. 출력 길이는
        입력 행 수이고 순서와 중복 인덱스까지 보존한다. 학습 후 여러 번 호출할 수
        있고 상태를 갱신하지 않는다. test 정답과의 RMSE/MAE는 호출자가 별도로
        계산한다. fit 전/필수 열 누락은 predict_details와 같은 ValueError."""
        return self.predict_details(pairs)["prediction"]
