"""W10A: MF, 평점 MLP, 추가 특성의 재사용 가능한 교육용 계산.

TensorFlow는 함수 호출 시에만 import한다. 단순 import/NumPy 순전파에는
네트워크·학습·파일 쓰기가 없다. 실습은 평균을 뺀 평점을 학습하고 예측에
훈련 평균을 복원한다. 반환 점수는 1–5로 clipping하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def fixed_forward(user: str = 'A', item: str = 'P', *, activation: str = 'relu',
                  occupation: str | None = None) -> dict:
    """고정 합성 모수의 MF/MLP 중간값. 실제 학습 성능이 아니다.

    user A/B, item P/Q. activation relu/linear는 첫 Dense 뒤의 함수다.
    occupation None/student/engineer/artist는 추가 3차원 경로를 선택한다.
    반환: p/q (2,), z (6 또는9,), kernel (입력폭,2), pre/hidden (2,),
    centered_mlp/mlp_rating/mf_rating 실수. 평균은 주어진 상수1이다.
    매번 새 배열을 반환한다. fixed_forward()['mlp_rating'] -> 1.6.
    """
    users = {'A': ([1., 1.], .5), 'B': ([.3, -.2], .2)}
    items = {'P': ([1., 0.], -.5), 'Q': ([-.1, .4], -.1)}
    jobs = {'student': [0., 0., 0.], 'engineer': [1., 0., 1.], 'artist': [0., 1., 0.]}
    if user not in users or item not in items or activation not in {'relu', 'linear'}:
        raise ValueError('user=A/B, item=P/Q, activation=relu/linear로 선택하세요.')
    if occupation is not None and occupation not in jobs:
        raise ValueError('정의된 합성 직업만 선택하세요.')
    p, bu = users[user]
    q, bi = items[item]
    p, q = np.array(p), np.array(q)
    z = np.r_[p, q, bu, bi]
    kernel = np.array([[1., 0.], [-1., 1.], [.5, 1.], [0., 1.], [1., 0.], [0., 0.]])
    if occupation is not None:
        z = np.r_[z, jobs[occupation]]
        kernel = np.vstack([kernel, [[0., 1.], [1., 0.], [.5, .5]]])
    pre = z @ kernel + np.array([-2., 0.])
    hidden = np.maximum(pre, 0) if activation == 'relu' else pre.copy()
    # 교재 두 번째 Dense+linear의 축소 예: kernel=I, bias=0.
    second = hidden @ np.eye(2)
    centered = float(second @ np.array([.5, .25]) + .1)
    return {'p': p, 'q': q, 'user_bias': bu, 'item_bias': bi, 'mean': 1.,
            'z': z, 'kernel': kernel, 'pre': pre, 'hidden': hidden,
            'second': second, 'centered_mlp': centered, 'mlp_rating': 1. + centered,
            'mf_rating': float(1. + bu + bi + p @ q)}


@dataclass
class NeuralData:
    """ID0은 미학습/OOV 표지. 평균·스케일·사전은 훈련 행에서만 적합.

    frames의 train/validation/test에는 row_id와 원래 평점·특성이 있으며
    순서를 유지한다. user/movie/occupation 사전의 값은1부터 시작한다.
    내부 DataFrame은 가변이므로 호출자는 수정 시 copy해야 한다.
    """
    frames: dict[str, pd.DataFrame]
    user_map: dict
    movie_map: dict
    occupation_map: dict
    mean: float
    age_mean: float
    age_std: float
    mode: str

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """예약 ID0을 포함한 세 임베딩 표의 행 수(M,N,J)."""
        return len(self.user_map) + 1, len(self.movie_map) + 1, len(self.occupation_map) + 1

    def inputs(self, frame: pd.DataFrame, kind: str = 'mf') -> tuple[dict, np.ndarray]:
        """같은 행 순서의 입력 사전과 학습된 ID 여부 (B,)를 반환.

        기본 입력은 int32 (B,1), age_scaled만 float32 (B,1).
        kind mf/mlp/occupation/age. age는 직업도 포함한다.
        미학습 ID0은 known=False이며 predict_ratings에서 평균으로 대체한다.
        """
        if kind not in {'mf', 'mlp', 'occupation', 'age'}:
            raise ValueError('kind는 mf/mlp/occupation/age 중 하나여야 합니다.')
        u = frame['user_id'].map(self.user_map).fillna(0).to_numpy(dtype=np.int32)
        i = frame['movie_id'].map(self.movie_map).fillna(0).to_numpy(dtype=np.int32)
        values = {'user_id': u[:, None], 'movie_id': i[:, None]}
        known = (u > 0) & (i > 0)
        if kind in {'occupation', 'age'}:
            o = frame['occupation'].map(self.occupation_map).fillna(0).to_numpy(dtype=np.int32)
            values['occupation_id'] = o[:, None]
            known &= o > 0
        if kind == 'age':
            age = frame['age'].to_numpy(dtype=np.float32)
            values['age_scaled'] = ((age - self.age_mean) / self.age_std)[:, None]
        return values, known


def prepare_neural_data(prepared: Any, *, seed: int = 20261103,
                        max_rows: int | None = None) -> NeuralData:
    """PreparedMovieLens를 70/15/15 관측 단위로 고정 분할한다.

    prepared.data.ratings: user_id,movie_id,rating; users: user_id,occupation,age.
    many_to_one merge 후 원래 row_id 정렬을 확인한다. max_rows가 주어지면
    분할 전에 무작위 부분집합을 선택한다(실제 전체 자료 결과와 구별).
    사전/평균/나이 표준화는 훈련 행만 사용. 파일·네트워크 부작용 없음.
    예: prepare_neural_data(prepare_movielens(mode='synthetic')).dimensions.
    """
    ratings = prepared.data.ratings[['user_id', 'movie_id', 'rating']].copy()
    users = prepared.data.users[['user_id', 'occupation', 'age']].copy()
    if len(ratings) < 20 or users['user_id'].duplicated().any():
        raise ValueError('평점20행 이상과 사용자별 고유한 메타데이터가 필요합니다.')
    ratings['row_id'] = np.arange(len(ratings))
    frame = ratings.merge(users, on='user_id', how='left', validate='many_to_one', sort=False)
    if not np.array_equal(frame['row_id'], ratings['row_id']):
        raise ValueError('메타데이터 결합이 평점 행 순서를 바꾸었습니다.')
    if frame[['occupation', 'age', 'rating']].isna().any().any():
        raise ValueError('나이·직업·평점에 누락값이 있습니다.')
    if not np.isfinite(frame[['age', 'rating']].to_numpy(dtype=float)).all():
        raise ValueError('나이와 평점은 유한한 수여야 합니다.')
    rng = np.random.default_rng(seed)
    if max_rows is not None:
        if not isinstance(max_rows, int) or max_rows < 20:
            raise ValueError('max_rows는20 이상의 정수여야 합니다.')
        if max_rows < len(frame):
            frame = frame.iloc[np.sort(rng.choice(len(frame), max_rows, replace=False))].copy()
    order = rng.permutation(len(frame))
    cut1, cut2 = int(len(frame) * .7), int(len(frame) * .85)
    frames = {name: frame.iloc[idx].copy().reset_index(drop=True)
              for name, idx in zip(('train', 'validation', 'test'),
                                   (order[:cut1], order[cut1:cut2], order[cut2:]), strict=True)}
    train = frames['train']
    maps = [{v: n + 1 for n, v in enumerate(sorted(train[col].unique()))}
            for col in ('user_id', 'movie_id', 'occupation')]
    std = float(train['age'].std(ddof=0)) or 1.
    return NeuralData(frames, *maps, float(train['rating'].mean()),
                      float(train['age'].mean()), std, prepared.mode)


def _keras(seed: int):
    """TensorFlow를 읽고 Python/NumPy/TF의 전역 난수 상태를 초기화한다."""
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return tf, tf.keras


def build_mf(n_users: int, n_items: int, *, k: int = 8, reg: float = .0002,
             seed: int = 20261103):
    """교재6.1–6.2의 Dot→Add→Flatten. 출력은 중심화 평점 (B,1).

    n_users/n_items는 ID0을 포함한 임베딩 표 행 수; k는 잠재 차원,
    reg는 L2 계수(음수 불가). 평균은 모델 외부에서 복원한다.
    TensorFlow import·전역 난수 seed 설정·모수 초기화가 발생한다. 학습/다운로드는 없다.
    예: build_mf(81,19,k=8).count_params() -> 900.
    """
    if min(n_users, n_items, k) < 1 or reg < 0:
        raise ValueError('표 행 수와 k는 양수, reg는0 이상이어야 합니다.')
    _, keras = _keras(seed)
    regularizer = keras.regularizers.l2(reg)
    u, i = keras.Input((1,), dtype='int32', name='user_id'), keras.Input((1,), dtype='int32', name='movie_id')
    p = keras.layers.Embedding(n_users, k, embeddings_regularizer=regularizer, name='user_embedding')(u)
    q = keras.layers.Embedding(n_items, k, embeddings_regularizer=regularizer, name='item_embedding')(i)
    bu = keras.layers.Embedding(n_users, 1, embeddings_regularizer=regularizer, name='user_bias')(u)
    bi = keras.layers.Embedding(n_items, 1, embeddings_regularizer=regularizer, name='item_bias')(i)
    dot = keras.layers.Dot(axes=2, name='mf_dot')([p, q])
    added = keras.layers.Add(name='mf_add')([dot, bu, bi])
    output = keras.layers.Flatten(name='centered_rating')(added)
    return keras.Model([u, i], output, name='keras_mf')


def build_mlp(n_users: int, n_items: int, *, k: int = 8, hidden: tuple[int, int] = (16, 8),
              n_occupations: int | None = None, age: bool = False,
              reg: float = .0002, seed: int = 20261103):
    """6.3–6.4의 연결→Dense→ReLU→Dense→linear→Dense1.

    hidden은 두 Dense 폭. 직업은3차원 임베딩, age=True면 표준화 나이의
    Dense3+ReLU 경로(장말 문제에 대한 수업용 선택)를 추가한다.
    n_occupations는 예약0 포함 행 수. 나이 사용 시 직업도 필요하다.
    두 bias 임베딩은 연결 입력이며 가산 편향이라고 보장되지 않는다.
    반환 Keras Model: 중심화 평점(B,1). 전역 난수 seed를 설정하며 파일/다운로드는 없다.
    """
    if min(n_users, n_items, k, *hidden) < 1 or len(hidden) != 2 or reg < 0:
        raise ValueError('양의 표 크기/k/두 은닉폭과0 이상의 reg가 필요합니다.')
    if age and n_occupations is None:
        raise ValueError('age 모형에는 직업 임베딩 경로도 필요합니다.')
    if n_occupations is not None and n_occupations < 1:
        raise ValueError('직업 표의 행 수는 양수여야 합니다.')
    _, keras = _keras(seed)
    regularizer = keras.regularizers.l2(reg)
    u, i = keras.Input((1,), dtype='int32', name='user_id'), keras.Input((1,), dtype='int32', name='movie_id')
    inputs, parts = [u, i], []
    for value, rows, width, name in ((u,n_users,k,'user_embedding'), (i,n_items,k,'item_embedding'),
                                    (u,n_users,1,'user_bias'), (i,n_items,1,'item_bias')):
        layer = keras.layers.Embedding(rows, width, embeddings_regularizer=regularizer, name=name)(value)
        parts.append(keras.layers.Flatten(name=name+'_flat')(layer))
    if n_occupations is not None:
        job = keras.Input((1,), dtype='int32', name='occupation_id')
        inputs.append(job)
        embedding = keras.layers.Embedding(n_occupations, 3, name='occupation_embedding')(job)
        parts.append(keras.layers.Flatten(name='occupation_flat')(embedding))
    if age:
        years = keras.Input((1,), dtype='float32', name='age_scaled')
        inputs.append(years)
        parts.append(keras.layers.Dense(3, activation='relu', name='age_dense')(years))
    joined = keras.layers.Concatenate(name='joined_features')(parts)
    first = keras.layers.Dense(hidden[0], name='dense_1')(joined)
    active = keras.layers.Activation('relu', name='relu_1')(first)
    second = keras.layers.Dense(hidden[1], name='dense_2')(active)
    linear = keras.layers.Activation('linear', name='linear_2')(second)
    output = keras.layers.Dense(1, name='centered_rating')(linear)
    return keras.Model(inputs, output, name='rating_mlp')


def rating_metrics(y, prediction) -> dict:
    """길이가 같은 유한한1차원 평점의 MSE/RMSE. 암묵적 방송을 거부한다."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or not y.size:
        raise ValueError('정답과 예측은 길이가 같은 비어 있지 않은1차원 배열이어야 합니다.')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('정답과 예측은 유한해야 합니다.')
    mse = float(np.mean((y - prediction) ** 2))
    return {'mse': mse, 'rmse': float(np.sqrt(mse)), 'n': int(y.size)}


def predict_ratings(model, data: NeuralData, frame: pd.DataFrame, kind: str = 'mf') -> tuple[np.ndarray, np.ndarray]:
    """행 순서 그대로 평점(B,)과 warm-ID mask 반환. 미학습 ID는 훈련 평균.

    직접 model(inputs,training=False)를 배치 호출하여 예측. 학습이나 파일
    쓰기는 없다. 평균 복원은 여기서 한 번만 수행한다. clipping은 하지 않는다.
    """
    values, known = data.inputs(frame, kind)
    prediction = np.full(len(frame), data.mean, dtype=float)
    positions = np.flatnonzero(known)
    for start in range(0, len(positions), 2048):
        take = positions[start:start+2048]
        batch = {name: value[take] for name, value in values.items()}
        centered = np.asarray(model(batch, training=False)).reshape(-1)
        prediction[take] = centered + data.mean
    return prediction, known


def fit_neural(data: NeuralData, kind: str = 'mf', *, k: int = 8,
               hidden: tuple[int, int] = (16, 8), epochs: int = 8,
               batch_size: int = 128, learning_rate: float = .05,
               reg: float = .0002, seed: int = 20261103) -> dict:
    """작은 CPU 학습. 검증 RMSE로 epoch 선택 후 시험은 최종1회 평가.

    data는 준비된 관측 단위70/15/15분할. kind mf/mlp/occupation/age.
    SGD(momentum=.7); 교재와 폭/epoch/학습률 차이는 실험 설정에 표시한다.
    반환 dict: model,history(DataFrame),best_epoch,metrics(split별),
    baseline(split별),predictions(split별),counts와setting. 모델과 반환값을
    변경하며 반환 외의 전역 학습 상태/파일/네트워크를 사용하지 않는다.
    이 결과를 다른 데이터 모드의 성능으로 보고하지 않는다.
    """
    if kind not in {'mf', 'mlp', 'occupation', 'age'} or epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('유효한 kind와 양의 epoch/batch/학습률이 필요합니다.')
    tf, keras = _keras(seed)
    m, n, j = data.dimensions
    if kind == 'mf':
        model = build_mf(m,n,k=k,reg=reg,seed=seed)
    else:
        model = build_mlp(m,n,k=k,hidden=hidden,reg=reg,seed=seed,
                          n_occupations=j if kind in {'occupation','age'} else None, age=kind=='age')
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate,momentum=.7),
                  loss='mse', metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    def dataset(split, shuffle=False):
        frame = data.frames[split]
        values, known = data.inputs(frame,kind)
        if not known.any():
            raise ValueError('학습된 ID를 가진 검증/훈련 관측이 없습니다.')
        values = {name: value[known] for name,value in values.items()}
        y = (frame['rating'].to_numpy(dtype=np.float32)[known] - data.mean)[:,None]
        ds = tf.data.Dataset.from_tensor_slices((values,y))
        if shuffle:
            ds = ds.shuffle(len(y),seed=seed,reshuffle_each_iteration=True)
        return ds.batch(batch_size).with_options(options)
    stop = keras.callbacks.EarlyStopping(monitor='val_rmse',patience=3,restore_best_weights=True)
    history = model.fit(dataset('train',True),validation_data=dataset('validation'),
                        epochs=epochs,verbose=0,shuffle=False,callbacks=[stop])
    history = pd.DataFrame(history.history)
    history.insert(0,'epoch',np.arange(1,len(history)+1))
    metrics, baseline, predictions, counts = {}, {}, {}, {}
    for split, frame in data.frames.items():
        pred, warm = predict_ratings(model,data,frame,kind)
        truth = frame['rating'].to_numpy(dtype=float)
        metrics[split] = rating_metrics(truth,pred)
        baseline[split] = rating_metrics(truth,np.full(len(frame),data.mean))
        metrics[split]['warm_rmse'] = rating_metrics(truth[warm],pred[warm])['rmse'] if warm.any() else None
        predictions[split] = pred
        counts[split] = {'all':len(frame),'warm':int(warm.sum()),'cold':int((~warm).sum())}
    return {'model':model,'history':history,'best_epoch':int(history.loc[history.val_rmse.idxmin(),'epoch']),
            'metrics':metrics,'baseline':baseline,'predictions':predictions,'counts':counts,
            'setting':{'kind':kind,'mode':data.mode,'k':k,'hidden':list(hidden),'seed':seed,
                       'epochs_requested':epochs,'epochs_run':len(history),'batch_size':batch_size,
                       'learning_rate':learning_rate,'reg':reg,'parameters':model.count_params()}}
