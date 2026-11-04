"""평점 벡터의 압축·복원과 관측 오차를 연결하는 W10B 공통 API.

작은 고정 모수 예제와 실제 데이터 학습을 구별한다. TensorFlow는 모델을
만들 때만 import한다. 원본 데이터 다운로드는 datasets.prepare_movielens가
담당하며 이 모듈의 데이터 준비 함수는 입력 표를 변경하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd


def toy_ratings() -> np.ndarray:
    """독자 설명용4명×4편 평점표의 새 사본. NaN은 미관측이다.

    관측12개의 평균은3이다. 파일/다운로드/학습이 없는 수계산 자료이며
    실제 MovieLens 성능 표본이 아니다. toy_ratings()[0] -> [5,nan,3,1].
    """
    return np.array([[5., np.nan, 3., 1.], [4., 2., np.nan, 3.],
                     [np.nan, 5., 3., 1.], [1., 3., 5., np.nan]])


def observation_mask(mask: Any, shape: tuple[int, ...]) -> np.ndarray:
    """같은 shape의 Boolean/0·1 마스크를 검증해 새 bool 배열로 반환.

    방송과0.5 같은 가중치를 관측 여부로 오인하지 않는다. 데이터 값이
    0인지 검사하는 함수가 아니다. observation_mask([1,0],(2,)).sum() -> 1.
    """
    value = np.asarray(mask)
    if value.shape != shape or not np.isin(value, [0, 1]).all():
        raise ValueError('관측 마스크는 대상과 같은 shape의 Boolean 또는0/1이어야 합니다.')
    return value.astype(bool, copy=True)


def masked_metrics(target: Any, prediction: Any, observed: Any) -> dict:
    """관측된 항의 전역 SSE/n과RMSE. target/prediction은 같은 shape.

    관측 밖의NaN은 허용하지만 관측된 값은 유한해야 한다. 마스크가 빈
    경우 오류로 전달한다. 배치RMSE를 평균하지 않는다. 파일/학습 부작용 없음.
    masked_metrics([2,0],[1,9],[1,0]) -> mse=1,rmse=1,n=1.
    """
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape:
        raise ValueError('정답과 예측의 shape가 같아야 합니다. 암묵적 방송은 허용하지 않습니다.')
    mask = observation_mask(observed, y.shape)
    n = int(mask.sum())
    if not n or not np.isfinite(y[mask]).all() or not np.isfinite(p[mask]).all():
        raise ValueError('유한한 관측과 예측이 적어도 하나 필요합니다.')
    sse = float(np.sum(np.square(y[mask] - p[mask]), dtype=np.float64))
    return {'sse': sse, 'n': n, 'mse': sse / n, 'rmse': float(np.sqrt(sse / n))}


def fixed_ae_forward(raw: Any = None, *, mean: float = 3.) -> dict:
    """주어진 고정 모수4→2(tanh)→4(linear)의 NumPy 순전파.

    raw: 길이4의1–5 평점/NaN, 기본toy사용자A. mean은 주어진 훈련 평균이며
    이 함수에서 재적합하지 않는다. 관측 여부를 먼저 보존하고 빈칸만0으로
    채운다. 반환 배열은 새 사본이다. 학습/파일 쓰기 없음.
    fixed_ae_forward()['pre'] -> [1,-1].
    """
    ratings = toy_ratings()[0] if raw is None else np.asarray(raw, dtype=float).copy()
    observed = np.isfinite(ratings)
    if (ratings.shape != (4,) or np.isinf(ratings).any() or not np.isfinite(mean)
            or ((ratings[observed] < 1) | (ratings[observed] > 5)).any()):
        raise ValueError('길이4의1–5 평점/NaN과 유한한 평균이 필요합니다.')
    x = np.where(observed, ratings - mean, 0.)
    encoder = np.array([[.5, 0.], [0., .5], [.5, .5], [0., .5]])
    decoder = np.array([[1., .5, 0., -1.], [0., .5, 1., 0.]])
    pre = x @ encoder  # 두 bias는 설명용으로0이다.
    latent = np.tanh(pre)
    centered = latent @ decoder
    return {'raw': ratings, 'observed': observed, 'mean': float(mean),
            'input': x, 'encoder': encoder, 'encoder_bias': np.zeros(2),
            'pre': pre, 'latent': latent, 'decoder': decoder,
            'decoder_bias': np.zeros(4), 'centered_prediction': centered,
            'rating_prediction': centered + mean}


def loss_comparison(target: Any, prediction: Any, observed: Any) -> pd.DataFrame:
    """같은 유한한 중심화 배열에서 세 계산 대상의 분모와RMSE를 비교.

    실제 관측/전체 칸/0이 아닌 정답의 차이를 진단하는 일반 계산이다.
    0이 아닌 정답이 없는 경우 해당 행만n=0,RMSE=NaN으로 표시한다.
    전체 칸의 오차가 항상 작다고 가정하지 않는다. 부작용 없음.
    """
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('같은 shape의 유한한 중심화 배열이 필요합니다.')
    masks = [('실제 관측', observation_mask(observed, y.shape)),
             ('전체 칸', np.ones(y.shape, dtype=bool)), ('정답!=0', y != 0)]
    rows = []
    for name, mask in masks:
        result = masked_metrics(y, p, mask) if mask.any() else {'n': 0, 'sse': 0., 'mse': np.nan, 'rmse': np.nan}
        rows.append({'계산 대상': name, **result})
    return pd.DataFrame(rows)


@dataclass
class AutoencoderData:
    """동일 관측 분할의NxM 행렬. N=사용자,M=아이템.

    values[train]만 모델 입력으로 쓴다. validation/test는 손실 비교용
    정답이며 입력이 아니다. masks는 중심화 전에 만든 관측 여부다.
    ratings는 원척도 평가용 전체 표이며 학습 함수에 입력하지 않는다.
    내부 배열/표는 가변이므로 변경 실험에는copy를 사용한다.
    """
    values: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    frames: dict[str, pd.DataFrame]
    ratings: np.ndarray
    mean: float
    user_map: dict
    item_map: dict
    item_titles: dict
    mode: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.ratings.shape

    @property
    def warm(self) -> np.ndarray:
        train = self.masks['train']
        return train.any(axis=1)[:, None] & train.any(axis=0)[None, :]


def prepare_autoencoder_data(prepared: Any, *, seed: int = 20261105,
                             max_rows: int | None = None) -> AutoencoderData:
    """PreparedMovieLens를 관측 단위70/15/15로 고정 분할해 행렬화.

    평균은train값만으로 계산한다. 축 사전은 전체 카탈로그의 정렬된ID로
    정하며 카탈로그 존재와 훈련 관측 존재를 구별한다. 중복(u,i)은 거부한다.
    max_rows는 분할 전 부분표본이며 실제100K 전체와 구별해야 한다.
    입력 변경/다운로드/학습/파일 쓰기 없음. synthetic모드도 mode에 남긴다.
    prepare_autoencoder_data(prepare_movielens(mode='synthetic')).shape -> (80,18).
    """
    ratings = prepared.data.ratings[['user_id', 'movie_id', 'rating']].copy()
    if (len(ratings) < 20 or ratings.duplicated(['user_id', 'movie_id']).any()
            or ratings.isna().any().any()):
        raise ValueError('누락/중복 없는 평점 관측20개 이상이 필요합니다.')
    scores = ratings['rating'].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 1) | (scores > 5)).any():
        raise ValueError('관측 평점은 유한한1–5 값이어야 합니다.')
    user_map = {v: j for j, v in enumerate(sorted(prepared.data.users['user_id'].unique()))}
    item_map = {v: j for j, v in enumerate(sorted(prepared.data.movies['movie_id'].unique()))}
    if not ratings['user_id'].isin(user_map).all() or not ratings['movie_id'].isin(item_map).all():
        raise ValueError('모든 평점ID가 카탈로그에 있어야 합니다.')
    rng = np.random.default_rng(seed)
    ratings['row_id'] = np.arange(len(ratings))
    if max_rows is not None:
        if type(max_rows) is not int or max_rows < 20:
            raise ValueError('max_rows는20 이상의 정수여야 합니다.')
        if max_rows < len(ratings):
            ratings = ratings.iloc[np.sort(rng.choice(len(ratings), max_rows, replace=False))].copy()
    order = rng.permutation(len(ratings))
    c1, c2 = int(len(ratings) * .7), int(len(ratings) * .85)
    names = ('train', 'validation', 'test')
    frames = {name: ratings.iloc[take].copy().reset_index(drop=True)
              for name, take in zip(names, (order[:c1], order[c1:c2], order[c2:]), strict=True)}
    mean = float(frames['train']['rating'].mean())
    shape = len(user_map), len(item_map)
    values = {name: np.zeros(shape, dtype=np.float32) for name in names}
    masks = {name: np.zeros(shape, dtype=bool) for name in names}
    raw = np.zeros(shape, dtype=np.float32)
    for name, frame in frames.items():
        u = frame['user_id'].map(user_map).to_numpy(dtype=int)
        i = frame['movie_id'].map(item_map).to_numpy(dtype=int)
        r = frame['rating'].to_numpy(dtype=np.float32)
        masks[name][u, i] = True
        values[name][u, i] = r - mean
        raw[u, i] = r
    titles = dict(zip(prepared.data.movies['movie_id'], prepared.data.movies['title'], strict=True))
    return AutoencoderData(values, masks, frames, raw, mean, user_map, item_map, titles, prepared.mode)


def oriented_arrays(data: AutoencoderData, direction: str = 'user') -> tuple[dict, dict]:
    """user는NxM,item은MxN으로 같은 값·마스크를 함께 전치한 view 반환.

    분할을 다시 추첨하지 않는다. 모델 입력은 반환values['train']뿐이다.
    배열 변경 시copy해야 한다. oriented_arrays(data,'item')[0]['train'].shape.
    """
    if direction not in {'user', 'item'}:
        raise ValueError('direction은user 또는item입니다.')
    convert = (lambda x: x) if direction == 'user' else (lambda x: x.T)
    return ({k: convert(v) for k, v in data.values.items()},
            {k: convert(v) for k, v in data.masks.items()})


def _keras(seed: int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return tf, tf.keras


def build_autoencoder(n_features: int, *, hidden: tuple[int, int] = (32, 8),
                      reg: float = .0003, seed: int = 20261105):
    """M→H1(tanh)→H2(linear,L1)→H1(tanh)→M(linear)의Keras 모형.

    입력/출력(B,M),학습 목표는 중심화 평점이다. 사용자/아이템 기준은
    n_features로 결정된다. TensorFlow import·seed·모수 초기화가 발생하며
    학습/다운로드는 없다. build_autoencoder(4,hidden=(3,2)).count_params()->48.
    """
    if n_features < 1 or len(hidden) != 2 or min(hidden) < 1 or reg < 0:
        raise ValueError('양의 입력폭/두 은닉폭과0 이상의L1 계수가 필요합니다.')
    _, keras = _keras(seed)
    inputs = keras.Input((n_features,), name='training_ratings')
    first = keras.layers.Dense(hidden[0], activation='tanh', name='encoder_dense')(inputs)
    latent = keras.layers.Dense(hidden[1], activation='linear',
                                kernel_regularizer=keras.regularizers.L1(reg), name='latent')(first)
    decoded = keras.layers.Dense(hidden[0], activation='tanh', name='decoder_dense')(latent)
    output = keras.layers.Dense(n_features, name='centered_ratings')(decoded)
    return keras.Model(inputs, output, name='rating_autoencoder')


def build_fixed_autoencoder():
    """설명용4→2(tanh)→4를 같은 고정값으로Keras화. 학습 모수22개.

    fixed_ae_forward와 같은 가중치/0인bias. 초기화와TensorFlow import만 수행.
    model(fixed_ae_forward()['input'][None,:])로 중심화 출력을 비교한다.
    """
    _, keras = _keras(20261105)
    trace = fixed_ae_forward()
    model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(2, activation='tanh'),
                              keras.layers.Dense(4)])
    model.set_weights([trace['encoder'].astype('float32'), np.zeros(2, dtype='float32'),
                       trace['decoder'].astype('float32'), np.zeros(4, dtype='float32')])
    return model


@dataclass
class AutoencoderFit:
    model: Any
    direction: str
    history: pd.DataFrame
    best_epoch: int
    train_seconds: float
    updates: int
    config: dict


def _predict_centered(model: Any, inputs: np.ndarray, batch_size: int = 256) -> np.ndarray:
    return np.concatenate([np.asarray(model(inputs[j:j + batch_size], training=False), dtype=float)
                           for j in range(0, len(inputs), batch_size)])


def fit_autoencoder(data: AutoencoderData, *, direction: str = 'user', epochs: int = 8,
                    hidden: tuple[int, int] = (32, 8), reg: float = .0003,
                    learning_rate: float = .001, batch_size: int = 128,
                    patience: int = 3, loss_kind: str = 'mse', seed: int = 20261105) -> AutoencoderFit:
    """관측 마스크를 명시해Adamax로 실제 학습, 검증RMSE 최상의 가중치 복원.

    train 입력만 쓰고validation으로 선택한다. test를 참조하지 않는다.
    기본은 관측MSE+L1,rmse는sqrt(MSE+1e-8)+L1인 안정화 경로다. 보고RMSE는
    전체 관측의 정확한SSE/n에서 구한다. 반환은 모형·이력·최상epoch·시간·갱신 수.
    데이터/디스크를 변경하지 않으며TensorFlow seed와 모형 가중치를 변경한다.
    예: fit_autoencoder(data,direction='user',epochs=3).
    """
    if (type(epochs) is not int or epochs < 1 or batch_size < 1 or patience < 1
            or learning_rate <= 0 or loss_kind not in {'mse', 'rmse'}):
        raise ValueError('양의epoch/batch/patience/학습률과mse/rmse가 필요합니다.')
    values, masks = oriented_arrays(data, direction)
    x, train_mask = values['train'], masks['train']
    if not train_mask.any() or not masks['validation'].any():
        raise ValueError('훈련과 검증 관측이 필요합니다.')
    tf, keras = _keras(seed)
    model = build_autoencoder(x.shape[1], hidden=hidden, reg=reg, seed=seed)
    model(x[:1], training=False)  # timing excludes imports and initial graph allocation.
    optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
    rng = np.random.default_rng(seed)
    warm = train_mask.any(axis=1)[:, None] & train_mask.any(axis=0)[None, :]
    history, best, best_weights, best_epoch, stale, updates = [], np.inf, None, 0, 0, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for j in range(0, len(x), batch_size):
            take = order[j:j + batch_size]
            if not train_mask[take].any():
                continue
            target = tf.convert_to_tensor(x[take])
            observed = tf.convert_to_tensor(train_mask[take], dtype=tf.float32)
            with tf.GradientTape() as tape:
                pred = model(target, training=True)
                mse = tf.reduce_sum(observed * tf.square(target - pred)) / tf.reduce_sum(observed)
                reconstruction = mse if loss_kind == 'mse' else tf.sqrt(mse + 1e-8)
                objective = reconstruction + tf.add_n(model.losses)
            gradients = tape.gradient(objective, model.trainable_variables)
            if not bool(tf.math.is_finite(objective)) or any(
                    g is None or not bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients):
                raise FloatingPointError('학습 목적함수 또는 기울기가 유한하지 않습니다.')
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            updates += 1
        pred = _predict_centered(model, x)
        # Cold axes use the training mean (centered0), in either orientation.
        pred[~warm] = 0.
        train = masked_metrics(values['train'], pred, masks['train'])
        val = masked_metrics(values['validation'], pred, masks['validation'])
        penalty = float(tf.add_n(model.losses))
        train_loss = train['mse'] if loss_kind == 'mse' else float(np.sqrt(train['mse'] + 1e-8))
        history.append({'epoch': epoch, 'train_mse': train['mse'], 'train_rmse': train['rmse'],
                        'validation_rmse': val['rmse'], 'l1_penalty': penalty,
                        'objective': train_loss + penalty})
        if val['rmse'] < best:
            best, best_epoch, stale = val['rmse'], epoch, 0
            best_weights = [a.copy() for a in model.get_weights()]
        else:
            stale += 1
        if stale >= patience:
            break
    model.set_weights(best_weights)
    config = {'hidden': list(hidden), 'l1': reg, 'learning_rate': learning_rate,
              'batch_size': batch_size, 'loss_kind': loss_kind, 'seed': seed,
              'parameter_count': model.count_params(), 'data_mode': data.mode}
    return AutoencoderFit(model, direction, pd.DataFrame(history), best_epoch,
                          time.perf_counter() - start, updates, config)


def predict_autoencoder(fitted: AutoencoderFit, data: AutoencoderData) -> np.ndarray:
    """훈련 입력만으로 원척도NxM 평점 예측. 평균을 한 번 복원, 미학습 축은 평균.

    가중치/데이터/파일을 변경하지 않으며1–5 clipping은 수행하지 않는다.
    item 방향의 출력은 원래NxM으로 돌린다. predict_autoencoder(fitted,data).shape.
    """
    values, _ = oriented_arrays(data, fitted.direction)
    centered = _predict_centered(fitted.model, values['train'])
    if fitted.direction == 'item':
        centered = centered.T
    ratings = centered + data.mean
    ratings[~data.warm] = data.mean
    return ratings


def evaluate_autoencoder(data: AutoencoderData, prediction: Any, split: str = 'test') -> dict:
    """원척도 예측과 지정split의 실제 관측만 평가. 미학습 축의 관측 수도 보고.

    split은train/validation/test. 전체RMSE에는cold 축의 평균 예측도 포함한다.
    원자료 행의 출력·파일 저장·모형 학습은 수행하지 않는다.
    """
    if split not in data.masks:
        raise ValueError('split은train/validation/test입니다.')
    result = masked_metrics(data.ratings, prediction, data.masks[split])
    result['cold_n'] = int((data.masks[split] & ~data.warm).sum())
    return result


def recommend_autoencoder(data: AutoencoderData, prediction: Any, user_id: Any, *, k: int = 5) -> pd.DataFrame:
    """알려진 전체 관측 이력을 제외한 후보 상위k개. 원척도 예측으로 안정 정렬.

    시험 라벨의 값을 모델 입력으로 쓰지 않는다. 결과는 표시용 후보 표다.
    movie_id/title/prediction을 반환하며 파일 저장/학습을 수행하지 않는다.
    """
    p = np.asarray(prediction, dtype=float)
    if user_id not in data.user_map or p.shape != data.shape or k < 1 or not np.isfinite(p).all():
        raise ValueError('등록ID·같은NxM의 유한한 예측·양의k가 필요합니다.')
    u = data.user_map[user_id]
    observed = np.logical_or.reduce(list(data.masks.values()))[u]
    rows = [{'movie_id': item, 'title': data.item_titles[item], 'prediction': float(p[u, j])}
            for item, j in data.item_map.items() if not observed[j]]
    return pd.DataFrame(rows, columns=['movie_id', 'title', 'prediction']).sort_values(
        ['prediction', 'movie_id'], ascending=[False, True], kind='stable').head(k).reset_index(drop=True)
