"""W04A: 목표 영화의 평가자 중 이웃을 고르는 사용자 CF. 원평점 코사인 유지."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from .collaborative import UserCF
from .rating_models import validate_ratings


class NeighborCF(UserCF):
    """k명/임계값 이웃과 선택적 사용자 평균 보정. MovieLens100K 교육용.

    k: 0=전체, 양의 정수=최대 이웃 수. centered: 평점 대신 관측평균 편차 집계.
    threshold: 0~1, similarity > threshold인 이웃만 사용. 두 제한을 함께 적용한다.
    fit에는 train만 전달한다. similarity는 W03A와 같은 전체 차원 원평점 코사인.
    같은 유사도는 사용자 ID 오름차순. 목표 영화 미관측자와 자신은 제외한다.
    예: NeighborCF(k=2, centered=True).fit(toy_ratings()).explain(1,4).
    파일/네트워크 접근 없음. 설정/입력 오류 ValueError. fit이 학습 상태를 갱신한다.
    """

    def __init__(self, k: int = 30, *, centered: bool = False, threshold: float = 0.0):
        super().__init__()
        self.k, self.centered, self.threshold = k, centered, threshold
        self._check_settings()

    def _check_settings(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 0:
            raise ValueError('k must be an integer >= 0; 0 means all')
        if not isinstance(self.centered, bool):
            raise ValueError('centered must be bool')
        if not isinstance(self.threshold, (float, int)) or not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be finite in [0,1]')

    def fit(self, ratings: pd.DataFrame) -> NeighborCF:
        """중복 없는 train 관측(user_id/movie_id/rating,1~5점)으로 학습하고 self 반환.

        원평점/관측 마스크는 보존한다. 사용자 평균은 NaN 제외 행 평균이다.
        inherited rating_matrix_, similarity_, seen_에 user_means_를 추가한다.
        원본 입력은 변경하지 않는다. 평균/유사도에 validation/test를 넣지 않는다.
        """
        self._check_settings()
        super().fit(ratings)
        self.user_means_ = self.rating_matrix_.mean(axis=1)
        self._order = np.argsort(-self._weights, axis=1, kind='stable')
        return self

    def _eligible(self, u: int, columns: np.ndarray) -> tuple:
        order = self._order[u]
        weights = self._weights[u, order]
        ratings = self.rating_matrix_.to_numpy()[np.ix_(order, columns)]
        keep = np.isfinite(ratings) & (weights[:, None] > self.threshold)
        if self.k:
            keep &= keep.cumsum(axis=0) <= self.k
        return order, ratings, np.where(keep, weights[:, None], 0.0)

    def predict_details(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """쌍 입력의 행 순서/인덱스를 보존한 예측과 근거 DataFrame을 반환한다.

        필수 user_id/movie_id; rating 열은 무시. 빈 표도 허용. 학습 상태 불변.
        prediction은1~5 clip, raw_prediction은 clip 전, n_contributors는 실제
        사용한 평가자 수, weight_sum은 정규화 전 합. basis=cf 또는 대체 종류.
        centered의 알려진 사용자 근거 부족→user-mean. 나머지는 movie-mean→
        global-mean. fit 전/잘못된 설정/결측ID는 ValueError. 추천 수 N과 k는 별개.
        """
        self._require_fit()
        self._check_settings()
        if not {'user_id', 'movie_id'}.issubset(pairs.columns) or pairs[['user_id','movie_id']].isna().any().any():
            raise ValueError('pairs require nonmissing user_id and movie_id')
        raw = pairs.movie_id.map(self.movie_means_).fillna(self.global_mean_).to_numpy(float, copy=True)
        basis = np.where(pairs.movie_id.isin(self.movie_means_.index), 'movie-mean', 'global-mean').astype(object)
        counts, totals = np.zeros(len(pairs), int), np.zeros(len(pairs))
        users, movies = pairs.user_id.to_numpy(), pairs.movie_id.to_numpy()
        for uid in pd.unique(users):
            u = self._user_pos.get(uid)
            if u is None:
                continue
            positions = np.flatnonzero(users == uid)
            if self.centered:
                raw[positions], basis[positions] = self.user_means_.iloc[u], 'user-mean'
            positions = positions[np.array([mid in self._movie_pos for mid in movies[positions]], dtype=bool)]
            if not len(positions):
                continue
            columns = np.array([self._movie_pos[mid] for mid in movies[positions]])
            order, values, weights = self._eligible(u, columns)
            if self.centered:
                values = values - self.user_means_.to_numpy()[order, None]
            totals[positions] = weights.sum(axis=0)
            counts[positions] = (weights > 0).sum(axis=0)
            supported = totals[positions] > 1e-12
            p = positions[supported]
            numerator = (np.nan_to_num(values) * weights).sum(axis=0)
            raw[p] = numerator[supported] / totals[p]
            if self.centered:
                raw[p] += self.user_means_.iloc[u]
            basis[p] = 'cf'
        return pd.DataFrame(dict(prediction=np.clip(raw,1,5), raw_prediction=raw,
            basis=basis, n_contributors=counts, weight_sum=totals), index=pairs.index)

    def explain(self, user_id: int, movie_id: int) -> pd.DataFrame:
        """실제 선택된 전체 이웃 근거 표. user_id/similarity/rating/user_mean/
        deviation/weight/contribution 열. weight합=1. 원평점: contribution합=
        raw_prediction, 보정: 대상 평균+contribution합=raw_prediction. 대체면 빈 표.
        입력 정수 ID, 출력은 유사도 내림차순/ID오름차순. 상태 변경 없음.
        """
        self._require_fit()
        self._check_settings()
        columns = ['user_id','similarity','rating','user_mean','deviation','weight','contribution']
        u, i = self._user_pos.get(user_id), self._movie_pos.get(movie_id)
        if u is None or i is None:
            return pd.DataFrame(columns=columns)
        order, values, weights = self._eligible(u, np.array([i]))
        keep = weights[:, 0] > 0
        sims, ratings = weights[keep, 0], values[keep, 0]
        if sims.sum() <= 1e-12:
            return pd.DataFrame(columns=columns)
        means = self.user_means_.to_numpy()[order[keep]]
        normalized = sims / sims.sum()
        return pd.DataFrame(dict(user_id=self.rating_matrix_.index[order[keep]],
            similarity=sims, rating=ratings, user_mean=means, deviation=ratings-means,
            weight=normalized, contribution=normalized*(ratings-means if self.centered else ratings)))


def validation_sweep(train: pd.DataFrame, validation: pd.DataFrame,
                     k_values=(1,5,10,20,30,40,60,0)) -> pd.DataFrame:
    """train에서만 학습, validation에서 k×원평점/보정 2종 RMSE를 계산한다.

    각 표는 고유한 user_id/movie_id/rating 관측; 쌍 중복/교집합은 ValueError.
    test 인자는 없다. 반환 k/centered/rmse/coverage/mean_neighbors 표를 낮은
    RMSE, k, centered 순으로 정렬한다. coverage는 cf 예측의 비율(0~1).
    평가 점수는 clip 후 계산한다. 파일 저장/입력 변경 없음. 예: table.iloc[0].
    """
    for frame in (train, validation):
        validate_ratings(frame)
        if frame.duplicated(['user_id','movie_id']).any():
            raise ValueError('Duplicate user/movie pairs')
    left = pd.MultiIndex.from_frame(train[['user_id','movie_id']])
    right = pd.MultiIndex.from_frame(validation[['user_id','movie_id']])
    if len(left.intersection(right)):
        raise ValueError('train and validation pairs overlap')
    model = NeighborCF().fit(train)
    rows = []
    for k in k_values:
        for centered in (False, True):
            model.k, model.centered = k, centered
            result = model.predict_details(validation[['user_id','movie_id']])
            rows.append(dict(k=k, centered=centered,
                rmse=float(root_mean_squared_error(validation.rating,result.prediction)),
                coverage=float(result.basis.eq('cf').mean()),
                mean_neighbors=float(result.n_contributors.mean())))
    if not rows:
        raise ValueError('k_values cannot be empty')
    return pd.DataFrame(rows).sort_values(['rmse','k','centered'],ignore_index=True)
