"""W03A: 관측 마스크와 예측 근거를 드러내는 기본 사용자 기반 CF.

교재 3.3의 전체 사용자 코사인 가중평균을 독립 구현한다. Pearson은 공통
관측의 평균을 쓰는 선택 비교다. 이웃 수 제한이나 평균 편차 보정은 하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .rating_models import validate_ratings


def toy_ratings() -> pd.DataFrame:
    """직접 설계한 5명×5편의 관측 18행을 새 DataFrame으로 반환한다.

    인자 없음. 열은 user_id(int), movie_id(int), rating(float,1–5).
    없는 행은 미관측이며 실제 0점이 아니다. MovieLens의 부분집합이 아니다.
    사용 예: toy_ratings().pivot(index='user_id', columns='movie_id', values='rating').
    파일/네트워크 접근이나 기존 객체 변경은 없다.
    """
    matrix = [[5, 3, 1, None, None], [4, 2, 1, 5, 2],
              [1, 3, 5, 1, 5], [5, 3, None, 4, None],
              [None, None, None, 2, 4]]
    return pd.DataFrame([(u+1, i+1, float(r)) for u, row in enumerate(matrix)
                         for i, r in enumerate(row) if r is not None],
                        columns=['user_id', 'movie_id', 'rating'])


def binary_jaccard(a: set, b: set) -> float:
    """두 행동 집합 a,b의 교집합/합집합 유사도를 float로 반환한다.

    set 또는 frozenset만 허용한다. 두 집합이 비면 근거 없음인 np.nan.
    예: binary_jaccard({1,2},{2,3}) == 1/3. 집합을 수정하지 않는다.
    scipy.spatial.distance.jaccard의 거리 반환과 구별한다.
    """
    if not isinstance(a, (set, frozenset)) or not isinstance(b, (set, frozenset)):
        raise TypeError('Use sets of observed positive actions, not raw rating arrays')
    union = a | b
    return len(a & b) / len(union) if union else float('nan')


class UserCF:
    """모든 유효 평가자의 양의 유사도를 사용하는 원평점 가중평균 CF.

    metric='cosine'(기본) 또는 'pearson'; min_common=3은 Pearson에만 적용한다.
    생성자는 설정만 저장한다. fit은 train의 평점/관측 마스크/유사도/평균과
    빠른 조회용 예측 배열을 준비한다. SGD나 별도 손실 최적화는 없다.
    학습 후 rating_matrix_, similarity_, common_counts_, seen_, global_mean_,
    movie_means_를 확인할 수 있다. 원본 입력 표는 수정하지 않는다.

    예: model = UserCF().fit(toy_ratings())
        model.predict(pd.DataFrame({'user_id':[1], 'movie_id':[4]}))
    Pearson의 음수·정의 불가 유사도는 예측 가중치로 쓰지 않는다.
    근거 부족은 영화 평균→전체 평균으로 대체하며 basis에 남긴다.
    """

    def __init__(self, metric: str = 'cosine', *, min_common: int = 3):
        """metric과 Pearson 최소 공통 관측 수(정수 2 이상)를 검증·저장한다.

        반환 None. 데이터·파일·네트워크 접근 없음. 설정 오류는 ValueError.
        min_common은 코사인 계산을 제한하지 않는다.
        """
        if metric not in {'cosine', 'pearson'}:
            raise ValueError('metric must be cosine or pearson')
        if isinstance(min_common, bool) or not isinstance(min_common, int) or min_common < 2:
            raise ValueError('min_common must be an integer >= 2')
        self.metric, self.min_common = metric, min_common

    def fit(self, ratings: pd.DataFrame) -> UserCF:
        """train 관측 표로 학습 상태를 만들고 self를 반환한다.

        ratings: user_id/movie_id/rating 필수, 중복 쌍 없음, 결측 없음,
        rating은 유한한 1–5점. 사용자/영화 ID는 정수다. test를 넣지 않는다.
        출력 행렬의 사용자·영화 순서는 ID 오름차순이다. 유사도는 U×U,
        평점은 U×I이며 원래 NaN을 보존한다. 예상 메모리는 O(U²+UI)라서
        MovieLens100K 규모의 교육용이며 대규모 서비스용 구현은 아니다.
        상태는 모든 계산 성공 후 갱신한다. 유효하지 않은 입력은 ValueError.
        """
        validate_ratings(ratings)
        if ratings.duplicated(['user_id', 'movie_id']).any():
            raise ValueError('Duplicate user/movie pairs must be resolved before fit')
        for key in ['user_id', 'movie_id']:
            if not pd.api.types.is_integer_dtype(ratings[key]):
                raise ValueError('User and movie IDs must be integers')
        matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
        matrix = matrix.sort_index().sort_index(axis=1).astype(float)
        mask = matrix.notna().to_numpy(dtype=float)
        values = matrix.fillna(0).to_numpy()
        common = mask @ mask.T
        if self.metric == 'cosine':
            similarity = cosine_similarity(values)
        else:
            # sums[u,v]는 u가 두 사용자의 공통 항목에 준 평점의 합이다.
            count = np.maximum(common, 1)
            sums = values @ mask.T
            squares = (values**2) @ mask.T
            centered_product = values @ values.T - sums * sums.T / count
            variance = np.maximum(squares - sums**2 / count, 0)
            denominator = np.sqrt(variance * variance.T)
            valid = (common >= self.min_common) & (denominator > 1e-12)
            similarity = np.divide(centered_product, denominator,
                                   out=np.full_like(common, np.nan), where=valid)
            similarity = np.clip(similarity, -1, 1)
        weights = np.maximum(np.nan_to_num(similarity), 0)
        np.fill_diagonal(weights, 0)  # 자기 자신의 알려진 평점도 예측 근거에서 제외한다.
        numerator = weights @ values
        denominator = weights @ mask
        counts = (weights > 0).astype(float) @ mask
        global_mean = float(ratings.rating.mean())
        means = matrix.mean(axis=0)
        predictions = np.broadcast_to(means.to_numpy(), matrix.shape).copy()
        supported = denominator > 1e-12
        np.divide(numerator, denominator, out=predictions, where=supported)
        # 양의 가중평균의 이론적 범위는 1–5이며 부동소수점 끝자리만 보정한다.
        predictions = np.clip(predictions, 1, 5)
        self.rating_matrix_ = matrix
        self.similarity_ = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)
        self.common_counts_ = pd.DataFrame(common.astype(int), index=matrix.index, columns=matrix.index)
        self.global_mean_, self.movie_means_ = global_mean, means
        self.seen_ = {u: set(group.movie_id) for u, group in ratings.groupby('user_id')}
        self._weights, self._predictions = weights, predictions
        self._denominator, self._counts, self._supported = denominator, counts, supported
        self._user_pos = {u: n for n, u in enumerate(matrix.index)}
        self._movie_pos = {i: n for n, i in enumerate(matrix.columns)}
        return self

    def _require_fit(self) -> None:
        if not hasattr(self, 'rating_matrix_'):
            raise ValueError('Call fit first')

    def predict_details(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """사용자/영화 쌍의 예측과 근거를 입력과 같은 행 순서·인덱스로 반환한다.

        pairs: user_id/movie_id 필수 DataFrame; rating 등 추가 열은 무시한다.
        출력 열: prediction(1–5점), basis('cf','movie-mean','global-mean'),
        n_contributors(양의 가중치 평가자 수), weight_sum(유사도 합).
        빈 입력도 허용한다. 미등록 사용자→영화 평균, 미등록 영화→전체 평균.
        fit 전/필수 열 부재/결측 ID는 ValueError. 학습 상태나 입력은 변경하지 않는다.
        """
        self._require_fit()
        if not {'user_id', 'movie_id'}.issubset(pairs.columns):
            raise ValueError('pairs require user_id and movie_id')
        if pairs[['user_id', 'movie_id']].isna().any().any():
            raise ValueError('pairs cannot contain missing IDs')
        n = len(pairs)
        predictions = np.full(n, self.global_mean_)
        basis = np.full(n, 'global-mean', dtype=object)
        counts = np.zeros(n, dtype=int)
        totals = np.zeros(n)
        for row, (uid, mid) in enumerate(pairs[['user_id', 'movie_id']].itertuples(index=False, name=None)):
            i, u = self._movie_pos.get(mid), self._user_pos.get(uid)
            if i is not None:
                predictions[row], basis[row] = self.movie_means_.iloc[i], 'movie-mean'
                if u is not None:
                    predictions[row] = self._predictions[u, i]
                    counts[row], totals[row] = self._counts[u, i], self._denominator[u, i]
                    if self._supported[u, i]:
                        basis[row] = 'cf'
        return pd.DataFrame({'prediction': predictions, 'basis': basis,
                             'n_contributors': counts, 'weight_sum': totals}, index=pairs.index)

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """predict_details와 같은 쌍 입력에서 예측 평점만 (N,) float 배열로 반환한다.

        출력은 입력 행 순서이며 rating 열은 읽지 않는다. 상태 변경 없음.
        사용 예: model.predict(test[['user_id','movie_id']]).
        """
        return self.predict_details(pairs)['prediction'].to_numpy()

    def explain(self, user_id: int, movie_id: int) -> pd.DataFrame:
        """한 예측에 기여한 평가자의 가중치·평점·기여도를 반환한다.

        user_id/movie_id는 정수. 출력은 유사도 내림차순·user_id 오름차순의
        user_id, common_items, similarity, rating, weight, contribution 표.
        weight는 해당 영화의 유사도 합으로 정규화하며 contribution=weight*rating.
        모든 행의 contribution 합이 CF 예측값이다. 대체 예측이면 빈 표다.
        전체 계산 근거를 반환하며, 화면에서 앞부분만 보여줄 때 생략을 표시해야 한다.
        원본/학습 상태는 변경하지 않고 fit 전에는 ValueError.
        """
        self._require_fit()
        columns = ['user_id', 'common_items', 'similarity', 'rating', 'weight', 'contribution']
        u, i = self._user_pos.get(user_id), self._movie_pos.get(movie_id)
        if u is None or i is None or not self._supported[u, i]:
            return pd.DataFrame(columns=columns)
        observed = self.rating_matrix_.iloc[:, i].notna().to_numpy()
        keep = observed & (self._weights[u] > 0)
        similarities = self._weights[u, keep]
        ratings = self.rating_matrix_.iloc[:, i].to_numpy()[keep]
        weight = similarities / similarities.sum()
        return pd.DataFrame({'user_id': self.rating_matrix_.index[keep],
                             'common_items': self.common_counts_.iloc[u].to_numpy()[keep],
                             'similarity': similarities, 'rating': ratings,
                             'weight': weight, 'contribution': weight * ratings}).sort_values(
                                 ['similarity', 'user_id'], ascending=[False, True], ignore_index=True)

    def recommend(self, user_id: int, movies: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """학습 이력을 제외한 카탈로그에서 예측 평점이 높은 영화를 반환한다.

        user_id: 정수. movies: 유일하고 결측 없는 movie_id와 title의 DataFrame.
        top_n: 1 이상 정수, 기본10. 출력은 최대top_n행, movie_id/title과
        predict_details의 4열. 예측 내림차순, 동점 movie_id 오름차순.
        미등록 사용자도 평균 대체로 반환하며 basis로 표시한다. 상태/입력 변경 없음.
        """
        self._require_fit()
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError('top_n must be a positive integer')
        if not {'movie_id', 'title'}.issubset(movies.columns):
            raise ValueError('movies require movie_id and title')
        if movies.movie_id.duplicated().any() or movies[['movie_id', 'title']].isna().any().any():
            raise ValueError('catalog must have unique nonmissing movie IDs and titles')
        candidate = movies.loc[~movies.movie_id.isin(self.seen_.get(user_id, set())),
                               ['movie_id', 'title']].reset_index(drop=True).copy()
        details = self.predict_details(candidate.assign(user_id=user_id))
        result = pd.concat([candidate, details], axis=1)
        return result.sort_values(['prediction', 'movie_id'], ascending=[False, True]).head(top_n).reset_index(drop=True)
