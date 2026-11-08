"""기존FM/MF/MLP를 보존하고 AE의 입력·마스크·복원을 보여 주는 누적 앱."""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .autoencoder_recommenders import (
    fixed_ae_forward, loss_comparison, prepare_autoencoder_data,
    fit_autoencoder, predict_autoencoder, evaluate_autoencoder,
)


def ae_view(r1=5., r2=3., r3=3., r4=1., o1=True, o2=False, o3=True, o4=True, *, forward_fn: Callable = fixed_ae_forward):
    """8개 위젯(평점4,관측 여부4)→요약/평점표/잠재표/오차표/Figure.

    훈련 평균3과 모수는 고정이다. 관측False인 위젯값은 입력에서 제외한다.
    한 번 호출은 학습이 아니다. 관측0개이면 RMSE를 정의하지 않았다고
    표시한다. 파일/다운로드 없음. ae_view()[1]에서 계산 표를 확인한다.
    """
    ratings = np.array([r1, r2, r3, r4], dtype=float)
    observed = np.array([o1, o2, o3, o4], dtype=bool)
    if not np.isfinite(ratings[observed]).all():
        raise ValueError('관측으로 선택한 칸에 평점을 입력하세요.')
    raw = np.where(observed, ratings, np.nan)
    trace = forward_fn(raw)
    prediction = trace['rating_prediction']
    table = pd.DataFrame({'movie': ['P', 'Q', 'R', 'S'], 'rating': raw,
                          'observed': observed, 'centered_input': trace['input'],
                          'reconstructed': prediction,
                          'observed_squared_error': np.where(observed, (prediction-ratings)**2, np.nan)})
    latent = pd.DataFrame({'node': ['z1', 'z2'], 'before_tanh': trace['pre'],
                           'after_tanh': trace['latent']})
    errors = loss_comparison(trace['input'], trace['centered_prediction'], observed)
    n = int(observed.sum())
    value = errors.iloc[0]['rmse']
    score = f'관측 RMSE={value:.6f}' if n else '관측0칸: 관측 RMSE를 계산하지 않습니다.'
    message = (f'고정 합성 모수 · 4→2(tanh)→4 · 훈련 평균3 · 관측{n}칸\n'
               f'{score}\n현재 계산의 모수는 고정입니다. 별도 학습 탭에서는 새 소형 모형을 학습합니다.')
    fig = Figure(figsize=(8, 3.4))
    axes = fig.subplots(1, 2)
    axes[0].bar(['z1', 'z2'], trace['latent'], color=['#2563eb', '#7c3aed'])
    axes[0].axhline(0, color='#64748b', lw=.8)
    axes[0].set(title='Encoded vector (fixed weights)', ylim=(-1.1, 1.1))
    positions = np.arange(4)
    axes[1].bar(positions-.18, raw, width=.36, label='observed', color='#2563eb')
    axes[1].bar(positions+.18, prediction, width=.36, label='reconstructed', color='#e69b31')
    axes[1].set(xticks=positions, xticklabels=['P', 'Q', 'R', 'S'], title='Restore the training mean')
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return message, table, latent, errors, fig


def export_ae_csv(r1=5., r2=3., r3=3., r4=1., o1=True, o2=False, o3=True, o4=True,
                  *, view_fn: Callable = ae_view) -> str:
    """view_fn이 반환한 현재 평점표를UTF-8 CSV 임시파일로 저장해 경로 반환.

    위젯 순서는ae_view와 같다. 학생이 추가한 표의 열도 보존한다.
    합성 입력만 사용하며 학습/네트워크 요청은 없다. export_ae_csv()로 사용.
    """
    current = view_fn(r1, r2, r3, r4, o1, o2, o3, o4)[1]
    with tempfile.NamedTemporaryFile(prefix='w10b-ae-', suffix='.csv', delete=False) as handle:
        destination = Path(handle.name)
    current.to_csv(destination, index=False, encoding='utf-8-sig')
    return str(destination)


def train_synthetic_comparison(epochs: int = 5):
    """독자 합성80명/18편 자료로U/I 실제CPU 학습과 비교. 실데이터 성능 아님.

    epochs1–20. 다운로드나 파일 저장 없이 기존 synthetic_movielens를 쓰고
    같은 분할에서 평균/U/I를 평가한다. TF import·seed·모형 학습이 발생한다.
    반환은 표와Figure이며 runtime-check에서 callback 자체도 검사한다.
    """
    from .datasets import prepare_movielens
    epochs = int(epochs)
    if not 1 <= epochs <= 20:
        raise ValueError('앱의 합성 학습은1–20epoch로 설정하세요.')
    data = prepare_autoencoder_data(prepare_movielens(mode='synthetic'))
    baseline = evaluate_autoencoder(data, np.full(data.shape, data.mean))
    rows = [{'model': 'mean (synthetic)', 'test_rmse': baseline['rmse'],
             'n_test': baseline['n'], 'parameters': 0, 'updates': 0, 'seconds': 0.}]
    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    for direction, color in [('user', '#2563eb'), ('item', '#d27a19')]:
        fitted = fit_autoencoder(data, direction=direction, epochs=epochs, hidden=(8, 2))
        metric = evaluate_autoencoder(data, predict_autoencoder(fitted, data))
        rows.append({'model': direction+' AE (synthetic)', 'test_rmse': metric['rmse'],
                     'n_test': metric['n'], 'parameters': fitted.model.count_params(),
                     'updates': fitted.updates, 'seconds': fitted.train_seconds})
        ax.plot(fitted.history.epoch, fitted.history.validation_rmse, 'o-',
                label=direction+' validation', color=color)
    ax.set(xlabel='Epoch', ylabel='Observed RMSE', title='Synthetic data: validation only for selection')
    ax.legend()
    fig.tight_layout()
    return pd.DataFrame(rows), fig


def build_autoencoder_explainer(view_fn: Callable = ae_view):
    """기존 앱과AE 탭을 합친Gradio Blocks. launch는 호출자가 수행한다.

    view_fn은ae_view와 같은8개 인수/5개 반환값을 가진다. 학생의 표시 함수를
    연결할 수 있고CSV에도 같은 함수가 반환한 표를 쓴다. CSV는 임시 저장한다.
    학습 버튼은 합성 자료로 학습하며 실제MovieLens 행을 전송하지 않는다.
    """
    import gradio as gr
    from .neural_explainer import build_neural_explainer

    initial = view_fn()
    with gr.Blocks(analytics_enabled=False) as ae_app:
        gr.Markdown('# 오토인코더의 계산을 따라가기\n'
                    '입력→중심화→인코딩→복원→관측 오차를 순서대로 확인합니다. 훈련 평균은3,모수는 고정입니다.')
        with gr.Tab('고정 모수 계산'):
            with gr.Row():
                ratings = [gr.Number(value=value, label='평점 '+name, minimum=1, maximum=5)
                           for name, value in zip(['P', 'Q', 'R', 'S'], [5, 3, 3, 1], strict=True)]
            with gr.Row():
                observed = [gr.Checkbox(value=value, label=name+' 관측함')
                            for name, value in zip(['P', 'Q', 'R', 'S'], [True, False, True, True], strict=True)]
            controls = ratings+observed
            button = gr.Button('예상과 계산 비교', variant='primary')
            message = gr.Textbox(initial[0], label='계산의 의미', lines=3)
            table = gr.Dataframe(initial[1], label='입력과 복원', interactive=False)
            latent = gr.Dataframe(initial[2], label='tanh 전후', interactive=False)
            errors = gr.Dataframe(initial[3], label='오차에 포함할 항과 분모', interactive=False)
            plot = gr.Plot(initial[4], label='잠재값과 평점')
            button.click(view_fn, controls, [message, table, latent, errors, plot])

            def export_current(r1, r2, r3, r4, o1, o2, o3, o4):
                return export_ae_csv(r1, r2, r3, r4, o1, o2, o3, o4, view_fn=view_fn)

            download = gr.Button('현재 계산표 CSV 만들기')
            file = gr.File(label='계산표 다운로드', interactive=False)
            download.click(export_current, controls, file)
        with gr.Tab('작은 모델 실제 학습'):
            gr.Markdown('독자 합성 자료80명·18편입니다. 같은 관측 분할에서 두 방향을 학습합니다. '
                        '실제MovieLens 자료의 실험은 실습B에서 수행합니다.')
            epochs = gr.Slider(1, 20, value=5, step=1, label='합성 학습epoch')
            train = gr.Button('합성 자료로 실제 학습·비교')
            results = gr.Dataframe(label='합성 실험 결과', interactive=False)
            learning = gr.Plot(label='검증RMSE 이력')
            train.click(train_synthetic_comparison, epochs, [results, learning], concurrency_limit=1)
        gr.Markdown('수정 활동: 입력 위젯→view_fn→반환 표→CSV를 따라가며 열 하나를 추가합니다. '
                    '예상·변경 후 값·이유를 기록하세요. 공유 주소는 런타임이 켜져 있는 동안 이용할 수 있습니다.')
    return gr.TabbedInterface([build_neural_explainer(), ae_app],
                              ['FM·MF·MLP 복습', '오토인코더 계산·학습'],
                              title='W10B · 오토인코더 추천', analytics_enabled=False)
