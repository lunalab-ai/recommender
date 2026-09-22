"""W04B: 신뢰도 조건·두 CF 축·정량 평가를 연결하는 교육용 공통 모듈."""
from __future__ import annotations

from copy import copy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .collaborative import UserCF
from .ranking import ranking_metrics
from .rating_models import validate_ratings


class EvidenceCF(UserCF):
    """전체 차원 코사인 CF. axis='user' 또는 'item', k=0은 전체 이웃.

    min_common(기본1)은 이웃 쌍의 공통 평가 수 하한, min_neighbors(1)는
    목표 예측의 유효 이웃 수 하한이다. 둘 다 >= 비교다. centered=False는
    원평점, True는 사용자 평균 편차(사용자 축만). clip=True는 1~5 제한.
    모든 이웃은 양의 유사도이며 자기 자신은 제외한다. 동점은 ID순이다.
    생성자는 설정만 저장. fit(train)이 평균/관측/유사도를 만든다. SGD 없음.
    예: EvidenceCF(k=2, centered=True).fit(toy_ratings()).explain(1, 4).
    MovieLens100K 규모의 교육용 밀집 행렬 구현으로 대규모 서비스용이 아니다.
    """

    def __init__(self, axis: str = 'user', k: int = 30, *, min_common: int = 1,
                 min_neighbors: int = 1, centered: bool = False, clip: bool = True):
        super().__init__()
        self.axis, self.k = axis, k
        self.min_common, self.min_neighbors = min_common, min_neighbors
        self.centered, self.clip = centered, clip
        self._validate_settings()

    def _validate_settings(self) -> None:
        if self.axis not in {'user', 'item'}:
            raise ValueError('axis must be user or item')
        for name, low in [('k', 0), ('min_common', 0), ('min_neighbors', 1)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
                raise ValueError(f'{name} must be an integer >= {low}')
        if type(self.centered) is not bool or type(self.clip) is not bool:
            raise ValueError('centered and clip must be bool')
        if self.axis == 'item' and self.centered:
            raise ValueError('This lesson uses mean centering only for user CF')

    def fit(self, ratings: pd.DataFrame) -> EvidenceCF:
        """중복 없는 user_id/movie_id/rating 관측으로 학습하고 self 반환.

        정수ID·유한한1~5점 표를 받는다. 원본 표/NaN 마스크는 보존한다.
        similarity_/common_counts_는 사용자 축 U×U, 아이템 축 I×I다.
        user_means_는 train 관측 평균. seen_는 train 영화 집합이다.
        파일·다운로드 부작용 없음. 검증/테스트 평점은 넣지 않는다.
        """
        self._validate_settings()
        super().fit(ratings)
        self._ratings = self.rating_matrix_.to_numpy(copy=True)
        self.user_means_ = self.rating_matrix_.mean(axis=1)
        if self.axis == 'item':
            values = np.nan_to_num(self._ratings).T
            mask = np.isfinite(self._ratings).astype(float).T
            ids = self.rating_matrix_.columns
            self.similarity_ = pd.DataFrame(cosine_similarity(values), index=ids, columns=ids)
            self.common_counts_ = pd.DataFrame((mask @ mask.T).astype(int), index=ids, columns=ids)
        self._sim = self.similarity_.to_numpy(copy=True)
        np.fill_diagonal(self._sim, 0)
        self._common = self.common_counts_.to_numpy(copy=True)
        self._order = np.argsort(-self._sim, axis=1, kind='stable')
        # 부모 클래스의 전체 이웃 예측 캐시는 새 조건부 예측에서 사용하지 않는다.
        for name in ('_predictions', '_denominator', '_counts', '_supported', '_weights'):
            delattr(self, name)
        return self

    def configured(self, **settings) -> EvidenceCF:
        """학습 상태를 공유하는 설정 복사본. 원래 설정과 배열은 변경하지 않는다.

        k/min_common/min_neighbors/centered/clip만 변경 가능. 축 변경은 별도 fit.
        예: model.configured(min_common=3, min_neighbors=2).predict(pairs).
        """
        self._require_fit()
        if set(settings) - {'k', 'min_common', 'min_neighbors', 'centered', 'clip'}:
            raise ValueError('Only prediction settings may change; axis requires a new model')
        result = copy(self)
        for name, value in settings.items():
            setattr(result, name, value)
        result._validate_settings()
        return result

    def _terms(self, u: int, columns: np.ndarray) -> tuple:
        # 행은 목표 영화, 열은 유사도순 후보 이웃. 두 축을 같은 집계식에 연결한다.
        if self.axis == 'user':
            order = np.broadcast_to(self._order[u], (len(columns), len(self._order[u])))
            ratings = self._ratings[order, columns[:, None]]
            sims = np.broadcast_to(self._sim[u, self._order[u]], ratings.shape)
            common = np.broadcast_to(self._common[u, self._order[u]], ratings.shape)
            reference = np.broadcast_to(self.user_means_.to_numpy()[self._order[u]], ratings.shape)
            ids = self.rating_matrix_.index.to_numpy()[order]
        else:
            order = self._order[columns]
            ratings = self._ratings[u, order]
            sims = self._sim[columns[:, None], order]
            common = self._common[columns[:, None], order]
            reference = np.zeros_like(ratings)
            ids = self.rating_matrix_.columns.to_numpy()[order]
        keep = np.isfinite(ratings) & (sims > 1e-12) & (common >= self.min_common)
        if self.k:
            keep &= keep.cumsum(axis=1) <= self.k
        weights = np.where(keep, sims, 0)
        values = ratings - reference if self.centered else ratings
        return ratings, values, weights, common, ids, reference

    def predict_details(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """ID쌍 표→같은 인덱스/행순서의 예측 및 근거 표. rating열은 무시.

        prediction은 clip 선택 적용 후, raw_prediction은 적용 전 평점이다.
        eligible_neighbors는 k 적용 후 확보 수, n_contributors는 실제 사용 수
        (최소 이웃 조건 실패 시0), weight_sum은 실제 사용한 유사도 합이다.
        basis='cf' 또는 user-mean/movie-mean/global-mean. 근거 부족 시
        보정 User-CF는 사용자 평균, 나머지는 영화 평균→전체 평균으로 대체.
        빈 표 허용. fit 전/ID누락/설정 오류는 ValueError. 상태 변경 없음.
        """
        self._require_fit()
        self._validate_settings()
        if not {'user_id', 'movie_id'}.issubset(pairs) or pairs[['user_id', 'movie_id']].isna().any().any():
            raise ValueError('pairs require nonmissing user_id and movie_id')
        raw = pairs.movie_id.map(self.movie_means_).fillna(self.global_mean_).to_numpy(float, copy=True)
        basis = np.where(pairs.movie_id.isin(self.movie_means_.index), 'movie-mean', 'global-mean').astype(object)
        available, used = np.zeros(len(pairs), int), np.zeros(len(pairs), int)
        total = np.zeros(len(pairs))
        users, movies = pairs.user_id.to_numpy(), pairs.movie_id.to_numpy()
        for uid in pd.unique(users):
            u = self._user_pos.get(uid)
            if u is None:
                continue
            positions = np.flatnonzero(users == uid)
            if self.centered:
                raw[positions], basis[positions] = self.user_means_.iloc[u], 'user-mean'
            positions = positions[np.array([m in self._movie_pos for m in movies[positions]], dtype=bool)]
            if not len(positions):
                continue
            columns = np.array([self._movie_pos[m] for m in movies[positions]])
            _, values, weights, _, _, _ = self._terms(u, columns)
            denominator = weights.sum(axis=1)
            count = (weights > 0).sum(axis=1)
            available[positions] = count
            supported = (count >= self.min_neighbors) & (denominator > 1e-12)
            pos = positions[supported]
            numerator = (np.nan_to_num(values) * weights).sum(axis=1)
            raw[pos] = numerator[supported] / denominator[supported]
            if self.centered:
                raw[pos] += self.user_means_.iloc[u]
            used[pos], total[pos], basis[pos] = count[supported], denominator[supported], 'cf'
        return pd.DataFrame(dict(prediction=np.clip(raw, 1, 5) if self.clip else raw.copy(),
            raw_prediction=raw, basis=basis, eligible_neighbors=available,
            n_contributors=used, weight_sum=total), index=pairs.index)

    def explain(self, user_id: int, movie_id: int) -> pd.DataFrame:
        """선택된 전체 이웃의 ID/공통수/평점/기준평균/가중치/기여도 표.

        이웃ID는 axis에 따라 사용자 또는 영화다. 유사도순·동점ID순이다.
        contribution 합(+보정 시 대상 평균)=raw_prediction. 평균 대체는 빈 표.
        실제 한 예측의 계산표이며 파일/네트워크/학습상태 변경이 없다.
        """
        cols = ['neighbor_id', 'common_ratings', 'similarity', 'rating', 'reference_mean', 'value', 'weight', 'contribution']
        detail = self.predict_details(pd.DataFrame({'user_id': [user_id], 'movie_id': [movie_id]})).iloc[0]
        if detail.basis != 'cf':
            return pd.DataFrame(columns=cols)
        terms = self._terms(self._user_pos[user_id], np.array([self._movie_pos[movie_id]]))
        ratings, values, weights, common, ids, means = [x[0] for x in terms]
        keep = weights > 0
        normalized = weights[keep] / weights.sum()
        return pd.DataFrame(dict(neighbor_id=ids[keep], common_ratings=common[keep],
            similarity=weights[keep], rating=ratings[keep], reference_mean=means[keep] if self.centered else 0,
            value=values[keep], weight=normalized, contribution=values[keep]*normalized))


def rating_report(model: EvidenceCF, heldout: pd.DataFrame) -> dict:
    """미사용 관측 표의 MAE/MSE/RMSE와 CF 근거 비율을 반환. 재학습 없음.

    train과 관측쌍 중복, 평가표 중복, 빈 표는 거부한다. 분모는 관측 행 수.
    clip 비활성도 그대로 평가한다. 예: rating_report(model, validation).
    """
    model._require_fit()
    validate_ratings(heldout)
    if heldout.duplicated(['user_id', 'movie_id']).any():
        raise ValueError('Duplicate evaluation pairs')
    for uid, rows in heldout.groupby('user_id'):
        if model.seen_.get(uid, set()).intersection(rows.movie_id):
            raise ValueError('Train and evaluation pairs overlap')
    result = model.predict_details(heldout[['user_id', 'movie_id']])
    error = heldout.rating.to_numpy() - result.prediction.to_numpy()
    return dict(mae=float(np.abs(error).mean()), mse=float((error**2).mean()),
        rmse=float(np.sqrt((error**2).mean())), cf_coverage=float(result.basis.eq('cf').mean()),
        mean_neighbors=float(result.n_contributors.mean()),
        clipped_fraction=float((np.abs(result.prediction-result.raw_prediction)>1e-12).mean()),
        rating_count=len(heldout))


def evaluate_cf(model: EvidenceCF, heldout: pd.DataFrame, movies: pd.DataFrame, *,
                top_n: int = 10, ranking_users: list[int] | None = None) -> dict:
    """평점 지표와 고정 후보의 순위 지표를 묶는다. model은 이미 fit되어야 함.

    heldout: 미사용 user_id/movie_id/rating 표. movies: 유일한 ID/title 목록.
    순위는 전체 카탈로그에서 train 이력만 제외, 관련성은 heldout>=4다.
    ranking_users=None이면 모든 평가 사용자; 고정 표본을 주면 해당 집단만
    순위 평가한다. 관련영화 없는 사용자는 제외 수를 보고한다. P/R/F1/NDCG는
    사용자별 값의 단순평균. catalog_coverage는 고유 추천영화/카탈로그다.
    cf_user_coverage는 평가 사용자 중 CF 근거가 있는 후보를 하나라도 가진
    비율이다. 미관측은 비선호가 아닌 미확인이다. 상태/데이터 변경 없음.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError('top_n must be a positive integer')
    result = rating_report(model, heldout)
    if movies.movie_id.duplicated().any() or movies.empty or not heldout.movie_id.isin(movies.movie_id).all():
        raise ValueError('Need a unique catalog containing all evaluation items')
    users = sorted(heldout.user_id.unique()) if ranking_users is None else list(ranking_users)
    if len(users) != len(set(users)) or not set(users).issubset(set(heldout.user_id)):
        raise ValueError('ranking_users must be unique evaluation user IDs')
    rows, union, supported_users = [], set(), 0
    for uid in users:
        truth = set(heldout.loc[(heldout.user_id == uid) & heldout.rating.ge(4), 'movie_id'])
        if not truth:
            continue
        candidates = movies.loc[~movies.movie_id.isin(model.seen_.get(uid, set())), ['movie_id','title']].reset_index(drop=True)
        details = model.predict_details(candidates.assign(user_id=uid))
        supported_users += int(details.basis.eq('cf').any())
        top = pd.concat([candidates, details], axis=1).sort_values(
            ['prediction','movie_id'], ascending=[False,True]).head(top_n)
        metrics = ranking_metrics(top.movie_id.tolist(), truth, k=top_n)
        p, r = metrics['precision'], metrics['recall']
        rows.append(dict(**metrics, f1=2*p*r/(p+r) if p+r else 0))
        union.update(top.movie_id)
    if not rows:
        raise ValueError('No users with relevant heldout items in the ranking cohort')
    average = pd.DataFrame(rows)[['precision','recall','f1','ndcg']].mean().to_dict()
    return dict(**result, **average, catalog_coverage=len(union)/len(movies),
        cf_user_coverage=supported_users/len(rows), ranking_users=len(rows),
        excluded_no_positive=len(users)-len(rows), ranking_cohort=len(users),
        catalog_size=len(movies), top_n=top_n)
