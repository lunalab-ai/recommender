"""W06A의 검증·튜닝·SVD 화면. 입력 데이터에 test 집합을 받지 않는다."""
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .mf_evaluation import fit_validated, search_mf, svd_reconstruct


def history_figure(history: pd.DataFrame, best_epoch: int):
    """epoch/train_rmse/validation_rmse 표 → Matplotlib Figure. 파일 저장 없음."""
    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    ax.plot(history.epoch, history.train_rmse, label='Train', color='#2563eb')
    ax.plot(history.epoch, history.validation_rmse, label='Validation', color='#d97706')
    ax.axvline(best_epoch, color='#0f766e', linestyle='--', label=f'Selected epoch {best_epoch}')
    ax.set(xlabel='Epoch (0 = initialization)', ylabel='RMSE')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    return fig


def validation_view(train: pd.DataFrame, validation: pd.DataFrame, k=2, lr=.02,
                    reg=.02, epochs=15, report_fn=None):
    """설정→학습→요약/전체 history/곡선. report_fn(history,best_epoch)의 문장을 추가.

    입력 k/epochs는 양의 정수. report_fn은 학생이 수정하는 출력 연결점이다.
    반환 (str,DataFrame,Figure). test 채점이나 파일/네트워크 변경 없음.
    """
    result = fit_validated(train, validation, n_factors=k, learning_rate=lr,
                           regularization=reg, epochs=epochs)
    summary = (f'선택 epoch={result.best_epoch}, validation RMSE={result.best_validation_rmse:.4f}. '
               f'학습 {len(train):,}개, 검증 {len(validation):,}개. '
               '선택 상태를 복원했습니다. 테스트 성능은 이 화면에서 계산하지 않습니다.')
    if report_fn is not None:
        summary += '\n'+str(report_fn(result.history.copy(), result.best_epoch))
    return summary, result.history, history_figure(result.history, result.best_epoch)


def comparison_view(train: pd.DataFrame, validation: pd.DataFrame, epochs=10):
    """4개 후보(K=2/8, β=.02/.2)의 검증 결과 표. 최종 test 평점 사용 없음."""
    winner, table = search_mf(train, validation, epochs=epochs)
    return (f'현재 4개 후보에서 최소 validation RMSE={winner.best_validation_rmse:.4f}. '
            '이 설정이 다른 분할이나 전체 데이터에서도 최적이라는 뜻은 아닙니다.', table)


def svd_view(rank=1, zero_fill=False):
    """독자 완전 2×2 예제의 rank/0 대입 반례. 반환 설명/재구성 표/Figure.

    zero_fill은 설명을 위해 정답 1을 감춘 위치에 실제 숫자 0을 넣는다.
    실제 MovieLens 성능 실험이 아니며 누락값을 올바르게 처리하는 권고가 아니다.
    """
    original = np.array([[5., 1.], [1., 5.]])
    supplied = original.copy()
    if zero_fill:
        supplied[0, 1] = 0.
    result = svd_reconstruct(supplied, rank)
    fig = Figure(figsize=(9, 3))
    axes = fig.subplots(1, 3)
    for ax, values, title in zip(axes, [original,supplied,result['reconstruction']],
                                 ['Teaching truth','Numeric input','Rank-k reconstruction']):
        ax.imshow(values, vmin=0, vmax=5, cmap='Blues')
        for (i,j),value in np.ndenumerate(values):
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if value>3 else '#172333')
        ax.set_title(title); ax.set_xticks([0,1],['A','B']); ax.set_yticks([0,1],['U1','U2'])
    fig.tight_layout()
    summary = (f"입력 행렬에 대한 Frobenius 오차={result['frobenius_error']:.6f}; "
               f"특이값={np.round(result['s'],4).tolist()}. ")
    summary += ('0은 미관측 표시가 아니라 실제 재구성 대상 숫자로 들어갔습니다.' if zero_fill
                else '완전한 행렬의 재구성입니다. 보지 않은 평점에 대한 성능이 아닙니다.')
    return summary, pd.DataFrame(result['reconstruction'],index=['U1','U2'],columns=['A','B']), fig


def build_evaluation_app(train: pd.DataFrame, validation: pd.DataFrame, *, report_fn=None):
    """Gradio Blocks 생성만 수행. launch는 호출자가 실행한다.

    train/validation은 고정 관측 사본. 출력 수정은 report_fn(history,best_epoch).
    API: train_validate, compare, svd. Colab share 링크는 런타임 종료 시 만료.
    """
    import gradio as gr
    tr, va = train.copy(), validation.copy()
    with gr.Blocks(title='W06a · MF 평가 실험실') as app:
        gr.Markdown('# MF 평가 실험실\n예상 → 설정 변경 → 실행 → 수치 관찰 → 이유 설명 순서로 실험하세요.')
        gr.Markdown(f'현재 고정 데이터: 학습 **{len(tr):,}개**, 검증 **{len(va):,}개**. '
                    '같은 데이터를 사용해 설정을 비교합니다. 테스트 데이터는 이 앱에 전달하지 않습니다.')
        with gr.Tab('학습과 검증'):
            with gr.Row():
                k = gr.Slider(1,16,value=2,step=1,label='잠재요인 수 K',min_width=240)
                lr = gr.Slider(.001,.08,value=.02,step=.001,label='학습률 α',min_width=240)
            with gr.Row():
                reg = gr.Slider(0,.5,value=.02,step=.01,label='정규화 β',min_width=240)
                epochs = gr.Slider(1,40,value=15,step=1,label='최대 epoch',min_width=240)
            run = gr.Button('학습·검증 곡선 계산',variant='primary')
            summary = gr.Textbox(label='선택 결과와 나의 해석',lines=4)
            curve = gr.Plot(label='학습 곡선')
            history = gr.Dataframe(label='전체 epoch 기록')
            def train_callback(k,lr,reg,epochs):
                return validation_view(tr,va,k,lr,reg,epochs,report_fn)
            run.click(train_callback,[k,lr,reg,epochs],[summary,history,curve],api_name='train_validate')
        with gr.Tab('4개 설정 비교'):
            gr.Markdown('K=2/8 × β=0.02/0.2, α=0.02, 동일 seed. 선택 근거는 검증 RMSE입니다.')
            compare_epochs = gr.Slider(1,30,value=10,step=1,label='후보별 최대 epoch')
            compare = gr.Button('같은 분할로 4개 후보 비교')
            compare_summary = gr.Textbox(label='비교 결과',lines=3)
            table = gr.Dataframe(label='후보별 검증 결과')
            compare.click(lambda e:comparison_view(tr,va,e),compare_epochs,
                          [compare_summary,table],api_name='compare')
        with gr.Tab('SVD와 미관측'):
            gr.Markdown('작은 완전 행렬 [[5,1],[1,5]]의 실험입니다. 0 대입은 결측 처리의 한계를 보여주는 반례입니다.')
            rank = gr.Slider(1,2,value=1,step=1,label='재구성 rank')
            zero = gr.Checkbox(False,label='U1–B를 감추고 0으로 채우기 (반례)')
            svd_run = gr.Button('SVD 재구성')
            svd_summary = gr.Textbox(label='재구성과 예측의 구분',lines=3)
            svd_table = gr.Dataframe(label='재구성 값')
            svd_plot = gr.Plot(label='원본·입력·재구성')
            svd_run.click(svd_view,[rank,zero],[svd_summary,svd_table,svd_plot],api_name='svd')
        gr.Markdown('정리: 무엇을 고정했고 무엇을 바꿨나요? 어떤 수치가 달라졌나요? '
                    '이 결과로 새 사용자나 미래의 추천 만족도까지 말할 수 있나요?')
    return app
