"""W08A: 기존 평가지표·MF·SGD·SVD API를 연결하는 작은 복습 실험.

새 학습 알고리즘을 구현하지 않는다. 합성 설명 예제와 화면 연결만 제공하며
문제은행의 정답은 포함하지 않는다. train/validation 평점 예측과 완전 행렬의
SVD 재구성은 서로 다른 실험이며 성능 숫자를 직접 비교하지 않는다.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .collaborative import toy_ratings
from .mf_evaluation import fit_validated, predict_rows, rating_metrics
from .mf_sgd import sgd_step
from .svd_foundations import svd_steps


def metric_review_demo(predicted=None) -> dict:
    """설명용 실제값 [5,3,2,4]와 예측 4개 → 오차표와 n/MAE/RMSE.

    predicted=None이면 [4,3,3,3]. 입력·출력 반올림이나 clipping 없음.
    반환 table은 actual/predicted/error/absolute_error/squared_error 열,
    metrics는 기존 rating_metrics의 dict. 파일·네트워크 접근 없음.
    """
    actual = np.array([5., 3., 2., 4.])
    prediction = np.array([4., 3., 3., 3.] if predicted is None else predicted,
                          dtype=float, copy=True)
    metrics = rating_metrics(actual, prediction)
    error = actual - prediction
    return {"table": pd.DataFrame({"actual": actual, "predicted": prediction,
                                   "error": error, "absolute_error": abs(error),
                                   "squared_error": error ** 2}),
            "metrics": metrics}


def sgd_review_demo(alpha: float = .1, beta: float = .1) -> dict:
    """K=1 설명 예제의 한 관측 갱신. 기존 sgd_step만 호출한다.

    μ=3,b=.1,c=-.1,p=.5,q=.4,r=4. alpha>0,beta>=0.
    반환 step은 기존 sgd_step dict, table은 p/q/b/c 갱신 전후,
    settings는 입력 조건. 매 호출이 같은 갱신 전 상태에서 시작한다.
    두 요인 모두 old 값으로 갱신하며 정규화는 관측별 규약이다.
    """
    p, q = np.array([.5]), np.array([.4])
    step = sgd_step(p, q, 4., mean=3., user_bias=.1, item_bias=-.1,
                    learning_rate=alpha, regularization=beta)
    before = np.array([.5, .4, .1, -.1])
    after = np.array([step['p_new'][0], step['q_new'][0],
                      step['user_bias_new'], step['item_bias_new']])
    return {"step": step,
            "table": pd.DataFrame({"parameter": ['p', 'q', 'b', 'c'],
                                   "old": before, "new": after,
                                   "change": after - before}),
            "settings": {"alpha": alpha, "beta": beta, "mean": 3.,
                         "rating": 4., "K": 1}}


def mf_review_data() -> dict:
    """새 합성 train15/validation3 표와 별도 완전 2×2 행렬을 반환.

    기존 5사용자×5영화·18관측 toy_ratings에서 (2,2),(3,1),(4,4)를
    validation으로 보류한다. 원래 미관측 칸과 보류한 관측을 구별한다.
    complete_matrix=[[4,2],[2,4]]는 SVD 설명용이며 평점표의 0 채움이
    아니다. 모든 반환 객체는 호출별 사본이다. test는 탐색에 노출하지 않는다.
    """
    ratings = toy_ratings()
    heldout = {(2, 2), (3, 1), (4, 4)}
    mask = [(u, i) in heldout for u, i in
            ratings[['user_id', 'movie_id']].itertuples(index=False, name=None)]
    mask = np.array(mask)
    return {'train': ratings.loc[~mask].reset_index(drop=True).copy(),
            'validation': ratings.loc[mask].reset_index(drop=True).copy(),
            'complete_matrix': np.array([[4., 2.], [2., 4.]])}


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or float(value) != int(value)
            or not minimum <= int(value) <= maximum):
        raise ValueError(f'{name}은 {minimum}..{maximum} 범위의 정수여야 합니다.')
    return int(value)


def mf_review_experiment(K: int = 2, beta: float = .1, epochs: int = 12,
                         rank: int = 1, *, train: pd.DataFrame | None = None,
                         validation: pd.DataFrame | None = None) -> dict:
    """설정 → MF 검증 결과와 별도 완전 행렬의 SVD 재구성 결과.

    K=1..8,beta>=0,epochs=1..40,rank=0..2. 학습률 .02와 seed 20261020
    고정. train/validation을 바꾸려면 두 표를 함께 제공한다. test 인수 없음.
    fit_validated는 상한까지 학습한 뒤 최저 validation RMSE의 첫 상태를
    복원한다. epoch0도 후보이며 조기 종료가 아니다. 반환 model/history/
    best_epoch/validation_metrics/validation_predictions/train/validation/
    svd/settings. history와 입력 표는 별도 사본. 파일·네트워크 접근 없음.
    """
    K = _integer(K, 'K', 1, 8)
    epochs = _integer(epochs, 'epochs', 1, 40)
    rank = _integer(rank, 'rank', 0, 2)
    if not isinstance(beta, Real) or not np.isfinite(beta) or beta < 0:
        raise ValueError('beta는 유한한 0 이상의 숫자여야 합니다.')
    if (train is None) != (validation is None):
        raise ValueError('train과 validation을 함께 제공하세요.')
    data = mf_review_data()
    tr = data['train'] if train is None else train.copy(deep=True)
    va = data['validation'] if validation is None else validation.copy(deep=True)
    fitted = fit_validated(tr, va, n_factors=K, regularization=float(beta),
                           learning_rate=.02, epochs=epochs, seed=20261020)
    predicted = predict_rows(fitted.model, va)
    metrics = rating_metrics(va.rating.to_numpy(), predicted)
    table = va.copy(deep=True)
    table['prediction'] = predicted
    table['error'] = table.rating - table.prediction
    return {'model': fitted.model, 'history': fitted.history.copy(deep=True),
            'best_epoch': fitted.best_epoch, 'validation_metrics': metrics,
            'validation_predictions': table, 'train': tr.copy(deep=True),
            'validation': va.copy(deep=True),
            'svd': svd_steps(data['complete_matrix'], rank),
            'settings': {'K': K, 'beta': float(beta), 'epochs': epochs,
                         'rank': rank, 'learning_rate': .02, 'seed': 20261020}}


def metric_review_view(offset: float = 0.):
    """기준 예측 네 개에 같은 offset → 설명(str), 오차표(DataFrame).

    offset은 예측을 바꾸는 설명용 조작이며 학습이나 정답 보정이 아니다.
    """
    result = metric_review_demo(np.array([4., 3., 3., 3.]) + float(offset))
    m = result['metrics']
    return (f"동일한 {m['n']}개 관측: MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}. "
            '분모는 평점 합이 아니라 평가한 관측 수입니다.', result['table'])


def sgd_review_view(alpha: float = .1, beta: float = .1):
    """한 단계 학습률·정규화 → 설명(str), old/new/change 표(DataFrame)."""
    result = sgd_review_demo(float(alpha), float(beta))
    s = result['step']
    return (f"예측 {s['prediction']:.6f} → {s['prediction_new']:.6f}; "
            f"한 관측 손실 {s['loss']:.6f} → {s['loss_new']:.6f}. "
            '매 클릭은 같은 old 상태에서 한 번 계산합니다. 큰 학습률의 손실 감소는 보장하지 않습니다.',
            result['table'])


def mf_review_view(K=2, beta=.1, epochs=12, rank=1,
                   report_fn: Callable[[dict], str] | None = None):
    """앱 callback: 설명/history/검증예측표/SVD근사/Figure, 총 5개 반환.

    report_fn(result)->str는 학생이 수정하는 문장 함수이다. 함수에는 결과의
    깊은 사본을 전달하므로 학생 설명 코드가 표시할 학습 상태를 바꾸지 않는다.
    학습곡선과 SVD 특이값은 서로 다른 축에 표시하며 우열 비교하지 않는다.
    """
    r = mf_review_experiment(K, beta, epochs, rank)
    h, s = r['history'], r['svd']
    fig = Figure(figsize=(10, 3.4))
    left, right = fig.subplots(1, 2)
    left.plot(h.epoch, h.train_rmse, label='Train RMSE')
    left.plot(h.epoch, h.validation_rmse, label='Validation RMSE')
    left.axvline(r['best_epoch'], color='#8b5e83', linestyle='--', label='Restored epoch')
    left.set(xlabel='Epoch (0 = initial)', ylabel='Rating RMSE', title='MF: held-out ratings')
    left.legend(fontsize=8)
    right.bar(np.arange(1, len(s['s']) + 1), s['s'], color='#7087ab')
    right.set(xlabel='Singular-value index', ylabel='Singular value',
              title='SVD: separate complete matrix', xticks=[1, 2])
    fig.tight_layout()
    text = (f"MF: train {len(r['train'])}건, validation {len(r['validation'])}건. "
            f"{int(epochs)} epoch까지 계산 후 epoch {r['best_epoch']}의 모든 모수를 복원했습니다. "
            f"복원 상태 validation RMSE={r['validation_metrics']['rmse']:.6f}.\n\n"
            f"별도 SVD 행렬 [[4,2],[2,4]], rank={int(rank)}: "
            f"재구성오차={s['error']:.6f}, 보존 제곱합 비율={s['energy_fraction']:.1%}. "
            '이 비율은 숨긴 평점의 추천 정확도가 아닙니다. 두 실험의 오차를 직접 비교하지 않습니다.')
    if report_fn is not None:
        text += '\n\n' + str(report_fn(deepcopy(r)))
    return text, h, r['validation_predictions'], pd.DataFrame(s['reconstruction']), fig


def build_mf_review_app(report_fn: Callable[[dict], str] | None = None):
    """W08A 복습 Blocks 생성. 서버 시작/파일 저장/다운로드는 하지 않는다.

    새 지표·SGD·MF/SVD 복습 탭과 기존 build_svd_app의 누적 기능을 연결한다.
    report_fn은 세 번째 탭의 문장에 연결된다. 빈 문장 starter도 계산을 막지
    않으며 문법/실행 오류는 숨기지 않는다. .launch는 마지막 notebook 셀에서.
    """
    import gradio as gr
    from .svd_lab import build_svd_app

    previous = build_svd_app()
    with gr.Blocks(title='W08A · MF·SGD·SVD 복습 실험실', analytics_enabled=False) as app:
        gr.Markdown('# MF·SGD·SVD 복습 실험실\n설명 예제입니다. 연습문제 정답표가 아닙니다.')
        with gr.Tab('1 · 관측별 오차와 지표'):
            gr.Markdown('실제값 [5,3,2,4], 기준 예측 [4,3,3,3]. Q06의 큰 오차 해석과 Q10의 분모 원리를 이 별도 예제에 적용합니다.')
            offset = gr.Slider(-1, 1, value=0, step=.25, label='기준 예측에 더할 값')
            metric_button = gr.Button('같은 네 관측의 지표 계산')
            metric_text, metric_table = metric_review_view()
            metric_note = gr.Textbox(metric_text, label='지표와 의미')
            metric_out = gr.Dataframe(metric_table, interactive=False, label='관측별 계산')
            metric_button.click(metric_review_view, offset, [metric_note, metric_out], api_name='review_metrics')
        with gr.Tab('2 · SGD의 old → new'):
            gr.Markdown('Q26/Q31 연결: μ=3,b=.1,c=−.1,p=.5,q=.4,r=4. 두 요인 모두 갱신 전 값을 씁니다.')
            alpha = gr.Slider(.01, .5, value=.1, step=.01, label='학습률 α')
            beta_step = gr.Slider(0, 1, value=.1, step=.05, label='관측별 정규화 β')
            step_button = gr.Button('동일한 old 상태에서 한 번 갱신')
            step_text, step_table = sgd_review_view()
            step_note = gr.Textbox(step_text, label='한 관측의 변화')
            step_out = gr.Dataframe(step_table, interactive=False, label='동시 갱신 확인')
            step_button.click(sgd_review_view, [alpha, beta_step], [step_note, step_out], api_name='review_sgd')
        with gr.Tab('3 · 검증 선택과 별도 SVD 재구성'):
            gr.Markdown('Q39/Q48/Q51/Q58 연결: 먼저 조건을 고정하고 한 설정만 바꿉니다. my_report를 수정한 뒤 앱 생성 셀과 실행 셀을 다시 실행합니다.')
            with gr.Row():
                factors = gr.Slider(1, 8, value=2, step=1, label='MF 잠재요인 수 K')
                beta = gr.Slider(0, .5, value=.1, step=.05, label='MF 관측별 정규화 β')
                epochs = gr.Slider(1, 40, value=12, step=1, label='MF 계산 epoch 상한')
                rank = gr.Slider(0, 2, value=1, step=1, label='별도 SVD rank r')
            button = gr.Button('새 모델로 계산하고 근거 읽기', variant='primary')
            text = gr.Textbox(label='결과와 나의 설명', lines=7)
            history = gr.Dataframe(interactive=False, label='모든 epoch: train/validation')
            predictions = gr.Dataframe(interactive=False, label='복원 상태의 validation 예측')
            reconstruction = gr.Dataframe(interactive=False, label='별도 완전 행렬의 SVD 근사')
            plot = gr.Plot(label='서로 다른 두 실험')
            def callback(k, b, e, r):
                return mf_review_view(k, b, e, r, report_fn)
            controls = [factors, beta, epochs, rank]
            outputs = [text, history, predictions, reconstruction, plot]
            button.click(callback, controls, outputs, api_name='review_compare')
            app.load(callback, controls, outputs)
        with gr.Tab('4 · 앞선 MF·SVD 앱 이어보기'):
            gr.Markdown('기존 W06B 앱을 그대로 연결합니다. main=[[5,1],[1,5]] 등 과거 설명 예제는 위 W08A 예제와 다릅니다. 선택 확장 탭은 새 시험 핵심 범위를 추가하지 않습니다.')
            previous.render()
        gr.Markdown('Q58 기록: 고정 조건 → 변경 조건 → 관찰 수치 → 가능한 해석. 공유 주소는 실행 중인 런타임에 의존합니다.')
    return app
