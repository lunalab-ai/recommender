"""MF 검증과 선택: test를 받지 않는 학습/탐색 API. W06A 교육용.

기존 MFSGD와 sgd_step을 재사용한다. 0 채우기 없이 관측 행으로 계산하며
평가 평점의 gradient 사용을 막고 최저 validation RMSE 상태를 복원한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
import numpy as np
import pandas as pd
from .mf_sgd import MFSGD, sgd_step
from .evaluation import split_ratings


def _ratings(rows: pd.DataFrame) -> None:
    required = ['user_id', 'movie_id', 'rating']
    if not set(required).issubset(rows.columns) or rows.empty:
        raise ValueError('user_id/movie_id/rating을 포함한 비어 있지 않은 관측표가 필요합니다.')
    if rows[required].isna().any().any() or not np.isfinite(rows.rating.to_numpy(float)).all():
        raise ValueError('결측 ID 또는 유한하지 않은 평점이 있습니다.')
    if rows.duplicated(['user_id', 'movie_id']).any():
        raise ValueError('사용자–영화 쌍을 먼저 중복 없이 집계하세요.')


def _pairs(rows: pd.DataFrame) -> set:
    return set(rows[['user_id', 'movie_id']].itertuples(index=False, name=None))


@dataclass(frozen=True)
class ThreeWaySplit:
    """관측 단위 train/validation/test 사본과 두 분할의 방법 이름.

    무작위 보류 평점 평가이며 미래 또는 새 사용자 평가가 아니다.
    frozen은 속성 교체를 막지만 DataFrame 내부 변경까지 막지는 않는다.
    """
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    methods: tuple[str, str]


def split_three_way(ratings: pd.DataFrame, *, seed: int = 20261006) -> ThreeWaySplit:
    """약 60/20/20 관측 분리. 원본 불변, 중복 쌍 거부, 작은 표는 반올림.

    먼저 20% test를 보류한 뒤 나머지의 25%를 validation으로 둔다.
    기존 split_ratings를 재사용하여 가능한 경우 사용자 층화한다.
    최소 5행. 반환 methods는 각 단계의 실제 층화/무작위 대체 방법이다.
    예: split_three_way(ratings).train. 파일·네트워크 없음.
    """
    _ratings(ratings)
    if len(ratings) < 5:
        raise ValueError('세 집합 분리를 위해 최소 5개 관측이 필요합니다.')
    first = split_ratings(ratings, test_size=.2, seed=seed)
    second = split_ratings(first.train, test_size=.25, seed=seed+1)
    out = ThreeWaySplit(second.train, second.test, first.test, (first.method, second.method))
    sets = [_pairs(x) for x in (out.train, out.validation, out.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == _pairs(ratings)
    return out


def predict_rows(model: MFSGD, rows: pd.DataFrame) -> np.ndarray:
    """관측표 ID의 벡터화 예측. 순서 보존, clipping 없음, 모델 변경 없음.

    미지 ID의 편향/상호작용은 0이며 알려진 쪽 편향은 유지한다.
    반환 shape=(len(rows),). 예: predict_rows(fitted_model, validation).
    """
    if not hasattr(model, 'P_'):
        raise ValueError('학습한 MFSGD가 필요합니다.')
    _ratings(rows)
    u = rows.user_id.map(model.user_index_).fillna(-1).to_numpy(int)
    i = rows.movie_id.map(model.item_index_).fillna(-1).to_numpy(int)
    ku, ki = u >= 0, i >= 0
    pred = np.full(len(rows), model.mean_, dtype=float)
    pred[ku] += model.user_bias_[u[ku]]
    pred[ki] += model.item_bias_[i[ki]]
    both = ku & ki
    pred[both] += np.sum(model.P_[u[both]] * model.Q_[i[both]], axis=1)
    return pred


def rating_metrics(actual, predicted) -> dict:
    """같은 길이의 유한한 1D 배열 → n/RMSE/MAE. 0점도 포함한다.

    반올림·clipping을 하지 않는다. 빈 배열/모양 불일치는 ValueError.
    예: rating_metrics([5,3,1,4], [4,3,2,3]).
    """
    y, pred = np.asarray(actual, float), np.asarray(predicted, float)
    if y.ndim != 1 or y.shape != pred.shape or y.size == 0:
        raise ValueError('같은 길이의 비어 있지 않은 1D 배열이 필요합니다.')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('유한한 실제값과 예측값이 필요합니다.')
    e = y-pred
    return {'n': len(y), 'rmse': float(np.sqrt(np.mean(e*e))), 'mae': float(np.mean(abs(e)))}


def evaluate_mf(model: MFSGD, rows: pd.DataFrame) -> dict:
    """학습된 모델을 주어진 관측에 채점. 행을 버리지 않고 cold-start 건수 표시.

    train/validation/test 역할은 호출자가 명시한다. 모델은 변경하지 않는다.
    """
    result = rating_metrics(rows.rating.to_numpy(), predict_rows(model, rows))
    unknown_u = ~rows.user_id.isin(model.user_index_)
    unknown_i = ~rows.movie_id.isin(model.item_index_)
    result.update(unknown_user=int(unknown_u.sum()), unknown_item=int(unknown_i.sum()),
                  fallback_rows=int((unknown_u | unknown_i).sum()))
    return result


@dataclass
class ValidatedMF:
    """model은 선택 epoch로 복원, history는 실행한 모든 epoch의 기록.

    best_epoch: 초기 상태를 포함한 최저 검증 RMSE의 첫 epoch.
    seconds는 현재 실행의 경과 시간이며 재현할 수치 목표가 아니다.
    """
    model: MFSGD
    history: pd.DataFrame
    best_epoch: int
    best_validation_rmse: float
    seconds: float


def fit_validated(train: pd.DataFrame, validation: pd.DataFrame, *, n_factors: int = 8,
                  learning_rate: float = .02, regularization: float = .02,
                  epochs: int = 20, seed: int = 20261006) -> ValidatedMF:
    """train에만 SGD; validation은 관찰/최적 상태 선택에만 사용. test 인수 없음.

    K/학습률/정규화는 MFSGD와 같다. epochs는 양의 정수 상한.
    epoch 0도 후보이며 동점은 먼저 나온 상태를 선택한다. 조기 중단 대신
    상한까지 곡선을 기록하고 마지막에 최적 상태를 복원한다.
    반환 ValidatedMF. 입력표 불변, 파일/네트워크 없음.
    예: fit_validated(train, validation, n_factors=2, epochs=10).
    """
    _ratings(train); _ratings(validation)
    if _pairs(train) & _pairs(validation):
        raise ValueError('학습과 검증에 같은 사용자–영화 쌍이 있습니다.')
    if isinstance(epochs, bool) or int(epochs) != epochs or epochs < 1:
        raise ValueError('epochs는 양의 정수여야 합니다.')
    start = perf_counter()
    model = MFSGD(n_factors=n_factors, learning_rate=learning_rate,
                  regularization=regularization, epochs=0, seed=seed).fit(train)
    rng = np.random.default_rng(seed)
    # MFSGD.fit와 같은 난수 소비: 초기 P/Q 다음에 관측 순서 생성.
    rng.normal(0, model.init_scale, model.P_.shape)
    rng.normal(0, model.init_scale, model.Q_.shape)
    u = train.user_id.map(model.user_index_).to_numpy(int)
    i = train.movie_id.map(model.item_index_).to_numpy(int)
    y = train.rating.to_numpy(float)
    names = ('P_', 'Q_', 'user_bias_', 'item_bias_')
    history = []; best_score = np.inf; best_epoch = 0; snapshot = None

    def record(epoch):
        nonlocal best_score, best_epoch, snapshot
        tr = evaluate_mf(model, train); va = evaluate_mf(model, validation)
        norms = (np.sum(model.P_[u]**2,axis=1)+np.sum(model.Q_[i]**2,axis=1)
                 +model.user_bias_[u]**2+model.item_bias_[i]**2)
        objective = .5*tr['rmse']**2 + .5*regularization*np.mean(norms)
        if not np.isfinite(objective):
            raise FloatingPointError('발산했습니다. 학습률을 낮추세요.')
        history.append({'epoch': epoch, 'train_rmse': tr['rmse'],
                        'validation_rmse': va['rmse'], 'objective': float(objective)})
        if va['rmse'] < best_score:
            best_score, best_epoch = va['rmse'], epoch
            snapshot = {name: getattr(model, name).copy() for name in names}

    record(0)
    for epoch in range(1, int(epochs)+1):
        for row in rng.permutation(len(train)):
            a, b = u[row], i[row]
            step = sgd_step(model.P_[a], model.Q_[b], y[row], mean=model.mean_,
                            user_bias=model.user_bias_[a], item_bias=model.item_bias_[b],
                            learning_rate=learning_rate, regularization=regularization)
            model.P_[a], model.Q_[b] = step['p_new'], step['q_new']
            model.user_bias_[a], model.item_bias_[b] = step['user_bias_new'], step['item_bias_new']
        record(epoch)
    for name, value in snapshot.items():
        setattr(model, name, value)
    model.epochs = int(epochs)
    model.history_ = pd.DataFrame(history)
    model.best_epoch_ = best_epoch
    return ValidatedMF(model, model.history_.copy(), best_epoch, best_score, perf_counter()-start)


def search_mf(train: pd.DataFrame, validation: pd.DataFrame, *, factors=(2, 8),
              regularizations=(.02, .2), learning_rate=.02, epochs=20,
              seed=20261006) -> tuple[ValidatedMF, pd.DataFrame]:
    """작은 Cartesian grid를 동일 train/validation에서 비교. test를 받지 않는다.

    후보별 best epoch에서의 train/validation RMSE, 설정, 실행시간 표와
    최소 validation RMSE 후보를 반환한다. 동점은 입력 후보 순서로 선택.
    기본 4회 학습. seed/초기화 규모 고정, K가 다르면 난수 배열도 달라진다.
    """
    rows = []; winner = None
    for k, reg in product(factors, regularizations):
        result = fit_validated(train, validation, n_factors=k, regularization=reg,
                               learning_rate=learning_rate, epochs=epochs, seed=seed)
        row = result.history.loc[result.history.epoch == result.best_epoch].iloc[0]
        rows.append({'K': int(k), 'regularization': float(reg), 'learning_rate': learning_rate,
                     'best_epoch': result.best_epoch, 'train_rmse': float(row.train_rmse),
                     'validation_rmse': result.best_validation_rmse, 'seconds': result.seconds})
        if winner is None or result.best_validation_rmse < winner.best_validation_rmse:
            winner = result
    if winner is None:
        raise ValueError('최소 한 개의 후보가 필요합니다.')
    return winner, pd.DataFrame(rows)


def svd_reconstruct(matrix, rank: int) -> dict:
    """유한한 완전 2D 행렬의 절단 SVD. NaN을 자동으로 0으로 바꾸지 않는다.

    rank는 1..min(shape). 반환 U/s/Vt, reconstruction, frobenius_error,
    discarded_energy. 행렬 입력을 수정하지 않는다. 예: svd_reconstruct([[5,1],[1,5]],1).
    """
    a = np.asarray(matrix, float)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError('유한한 값으로 채워진 2차원 행렬이 필요합니다. 결측 처리 정책을 먼저 정하세요.')
    if isinstance(rank, bool) or int(rank) != rank or not 1 <= rank <= min(a.shape):
        raise ValueError('rank는 1부터 행/열 수 중 작은 값까지의 정수입니다.')
    k = int(rank)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    reconstructed = (u[:, :k]*s[:k])@vt[:k]
    return {'U': u, 's': s, 'Vt': vt, 'reconstruction': reconstructed,
            'frobenius_error': float(np.linalg.norm(a-reconstructed)),
            'discarded_energy': float(np.sum(s[k:]**2))}
