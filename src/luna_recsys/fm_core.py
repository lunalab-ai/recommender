"""2-way FM의 순전파·희소 표현·한 관측 SGD를 확인하는 교육용 함수.

전체 FM 학습기는 아니다. 절편은 주어진 상수로 고정하며 손실은 한 관측의
0.5*(y-prediction)**2, 정규화는 0이다. 입력은 변경하지 않는다. 예측은
clipping하지 않는다. 데이터 다운로드/파일 쓰기/네트워크 부작용은 없다.
효율식의 산술 핵심은 O(n*k), 희소 핵심은 O(nnz(x)*k)이다. 안전한 입력
검증을 위한 전체 모수 스캔의 비용은 이 산술 복잡도와 별도다.
"""
from __future__ import annotations

import numpy as np


def _parameters(w, V, bias):
    w = np.asarray(w, dtype=float)
    V = np.asarray(V, dtype=float)
    if w.ndim != 1 or not w.size or V.ndim != 2 or V.shape[0] != w.size or V.shape[1] < 1:
        raise ValueError('w는 (n,), V는 (n,k), n과 k는 양수여야 합니다.')
    if np.ndim(bias) != 0 or not np.isfinite(bias) or not np.isfinite(w).all() or not np.isfinite(V).all():
        raise ValueError('모수와 고정 절편은 유한한 실수여야 합니다.')
    return w, V, float(bias)


def _input(x, n):
    x = np.asarray(x, dtype=float)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise ValueError('x는 모수의 n과 같은 길이의 유한한 1차원 벡터여야 합니다.')
    return x


def toy_parameters() -> dict:
    """새 배열의 합성 모수를 반환한다. 학습 결과나 실제 장르 축이 아니다.

    names: 특성 열 순서 5개; w:(5,), V:(5,2), bias:1.0.
    user/item 원핫 블록과 수업용 연속 맥락 한 열을 사용한다.
    매 호출마다 복사하므로 다른 호출이나 앱 입력을 오염시키지 않는다.
    """
    return {'names': ['user_A', 'user_B', 'item_P', 'item_Q', 'context'],
            'w': np.array([.5, .2, -.5, -.1, .25]),
            'V': np.array([[1., 1.], [.3, -.2], [1., 0.], [-.1, .4], [0., 1.]]),
            'bias': 1.0}


def encode_observation(user: str = 'A', item: str = 'P', context: float = 1.) -> np.ndarray:
    """합성 사례를 (5,) x로 만든다. y는 입력하지 않는다.

    user는 A/B, item은 P/Q, context는 유한 실수(수업용 무단위 합성값).
    열은 user_A,user_B,item_P,item_Q,context 순서다. 알 수 없는 ID는
    거부한다. 이는 신규 ID를 이미 학습한 것처럼 예측하지 않기 위함이다.
    예: encode_observation('A','P',1) -> [1,0,1,0,1].
    """
    users, items = {'A': 0, 'B': 1}, {'P': 2, 'Q': 3}
    if user not in users or item not in items:
        raise ValueError('이 합성 예제에 정의된 사용자 A/B와 아이템 P/Q만 사용하세요.')
    if np.ndim(context) != 0 or not np.isfinite(context):
        raise ValueError('맥락값은 유한한 실수여야 합니다.')
    x = np.zeros(5)
    x[users[user]], x[items[item]], x[4] = 1., 1., float(context)
    return x


def to_sparse(x) -> tuple[np.ndarray, np.ndarray]:
    """유한 1D x를 오름차순 0기반 비영 indices/values로 반환한다.

    명시적인 0은 저장하지 않으며 음수·연속값은 보존한다. 빈 벡터도
    가능하다. 예: [0,1,0,.5] -> (array([1,3]),array([1.,.5])).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('희소화할 x는 유한한 1차원 벡터여야 합니다.')
    idx = np.flatnonzero(x)
    return idx, x[idx].copy()


def _sparse(indices, values, n):
    idx = np.asarray(indices)
    values = np.asarray(values, dtype=float)
    if idx.size == 0:
        idx = np.array([], dtype=int)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or values.shape != idx.shape:
        raise ValueError('indices는 1차원 정수, values는 같은 길이여야 합니다.')
    if len(np.unique(idx)) != idx.size or np.any(idx < 0) or np.any(idx >= n):
        raise ValueError('인덱스 중복·범위 초과를 허용하지 않습니다(0기반).')
    if not np.isfinite(values).all() or np.any(values == 0):
        raise ValueError('values에는 유한한 비영 값만 저장하세요.')
    return idx.astype(int, copy=False), values


def from_sparse(indices, values, n_features: int) -> np.ndarray:
    """0기반 비영 index/value를 밀집 x로 복원한다. 중복은 합치지 않고 거부.

    n_features는 양의 정수. 반환은 새 (n_features,) float 배열이다.
    교재 1기반 번호는 호출 전에 명시적으로 1을 빼야 한다.
    """
    if isinstance(n_features, bool) or not isinstance(n_features, (int, np.integer)) or n_features < 1:
        raise ValueError('n_features는 양의 정수여야 합니다.')
    idx, values = _sparse(indices, values, n_features)
    x = np.zeros(n_features)
    x[idx] = values
    return x


def fm_score(x, w, V, bias: float = 0.) -> dict:
    """효율식으로 2-way FM을 분해한다. x,w:(n,), V:(n,k), bias:고정 실수.

    반환 dict: prediction/bias/linear/interaction은 실수, linear_terms:(n,),
    weighted_factors:(n,k), factor_sums/factor_self/factor_interactions:(k,).
    factor_self는 sum((v_if*x_i)**2)이며 sum(v_if**2*x_i)가 아니다.
    예: toy_parameters와 encode_observation의 기본값 -> prediction 3.25.
    """
    w, V, bias = _parameters(w, V, bias)
    x = _input(x, len(w))
    weighted = V * x[:, None]
    sums = weighted.sum(axis=0)
    self_terms = np.square(weighted).sum(axis=0)
    factors = .5 * (np.square(sums) - self_terms)
    linear_terms = w * x
    linear, interaction = float(linear_terms.sum()), float(factors.sum())
    prediction = bias + linear + interaction
    if not np.isfinite(prediction):
        raise ValueError('계산 범위를 넘었습니다. 입력/모수의 크기를 줄이세요.')
    return {'prediction': prediction, 'bias': bias, 'linear': linear,
            'interaction': interaction, 'linear_terms': linear_terms,
            'weighted_factors': weighted, 'factor_sums': sums,
            'factor_self': self_terms, 'factor_interactions': factors}


def pair_contributions(x, V) -> list[dict]:
    """활성 i<j 쌍을 인덱스 순서로 반환한다. 0인 내적도 생략하지 않는다.

    V:(n,k), x:(n,). 각 행은 i,j,dot,x_product,contribution을 갖는다.
    s개의 활성 특성은 s*(s-1)/2행이다. 자기항/역순 쌍은 없다.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] < 1 or V.shape[1] < 1 or not np.isfinite(V).all():
        raise ValueError('V는 유한한 (n,k) 배열이어야 합니다.')
    x = _input(x, V.shape[0])
    active = np.flatnonzero(x)
    return [{'i': int(i), 'j': int(j), 'dot': float(V[i] @ V[j]),
             'x_product': float(x[i]*x[j]),
             'contribution': float((V[i] @ V[j])*x[i]*x[j])}
            for pos, i in enumerate(active) for j in active[pos+1:]]


def fm_score_naive(x, w, V, bias: float = 0.) -> float:
    """i<j를 직접 열거하는 독립 검산용 예측. 반환 실수, 입력 불변.

    효율식의 합을 호출하지 않고 각 내적/입력 곱을 직접 더한다.
    """
    w, V, bias = _parameters(w, V, bias)
    x = _input(x, len(w))
    interactions = sum(float(V[i] @ V[j])*x[i]*x[j]
                       for i in range(len(x)) for j in range(i+1, len(x)))
    return float(bias + w @ x + interactions)


def fm_score_sparse(indices, values, w, V, bias: float = 0.) -> float:
    """비영 항만으로 효율식을 계산한다. index는 Python 0기반 정수.

    전체 x를 할당하지 않는다. 안전성 검증의 전체 모수 스캔과 구별하면
    산술 핵심은 O(k*s), s=len(indices)다. 반환 실수, 0값은 저장하지 않는다.
    """
    w, V, bias = _parameters(w, V, bias)
    idx, values = _sparse(indices, values, len(w))
    weighted = V[idx] * values[:, None]
    interaction = .5*np.sum(weighted.sum(axis=0)**2 - (weighted**2).sum(axis=0))
    return float(bias + w[idx] @ values + interaction)


def prediction_gradients(x, V) -> dict:
    """예측 yhat의 w/V 미분을 반환한다. 손실 미분과 구별한다.

    x:(n,), V:(n,k). 반환 w:(n,), V:(n,k), sums:(k,).
    V 미분은 x_i*(S_f-v_if*x_i), S_f=sum_j(v_jf*x_j).
    편향은 고정하므로 갱신 대상에 포함하지 않는다.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or min(V.shape, default=0) < 1 or not np.isfinite(V).all():
        raise ValueError('V는 유한한 (n,k) 배열이어야 합니다.')
    x = _input(x, V.shape[0])
    weighted = V*x[:, None]
    sums = weighted.sum(axis=0)
    return {'w': x.copy(), 'V': x[:, None]*(sums-weighted), 'sums': sums}


def fm_sgd_step(x, y: float, w, V, bias: float = 0., learning_rate: float = .1) -> dict:
    """고정 bias, 정규화0, L=.5*(y-yhat)^2의 한 관측 동시 갱신.

    x,w:(n,), V:(n,k), y:유한 실수, learning_rate:양의 유한 실수.
    모든 미분/잔차는 같은 갱신 전 상태에서 계산한다. 반환은 새 w/V,
    예측 전후·잔차 전후·손실 전후·예측 미분·손실 미분 사전이다.
    원본 배열 변경/학습 상태 보관/다운로드 없음. 손실 감소는 보장하지 않는다.
    """
    w, V, bias = _parameters(w, V, bias)
    x = _input(x, len(w))
    if np.ndim(y) != 0 or not np.isfinite(y) or np.ndim(learning_rate) != 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('y는 유한 실수, 학습률은 양의 유한 실수여야 합니다.')
    before = fm_score(x, w, V, bias)
    error = float(y-before['prediction'])
    grad = prediction_gradients(x, V)
    new_w = w + learning_rate*error*grad['w']
    new_V = V + learning_rate*error*grad['V']
    after = fm_score(x, new_w, new_V, bias)
    return {'w': new_w, 'V': new_V, 'bias': bias,
            'prediction': before['prediction'], 'prediction_new': after['prediction'],
            'error': error, 'error_new': float(y-after['prediction']),
            'loss': .5*error**2, 'loss_new': .5*float(y-after['prediction'])**2,
            'prediction_gradient_w': grad['w'], 'prediction_gradient_V': grad['V'],
            'loss_gradient_w': -error*grad['w'], 'loss_gradient_V': -error*grad['V']}
