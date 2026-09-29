"""MF 수업용 화면과 수치 실험: 알고리즘은 mf_sgd 모듈을 재사용한다."""
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .mf_sgd import MFSGD, sgd_step
from .model_comparison import comparison_toy


def step_view(rating: float=4., learning_rate: float=.05, regularization: float=.02):
    """고정 K=2 예제에서 실제 평점/학습률/정규화를 바꾼다.

    반환 (한국어 요약 문자열, 6행 갱신표, Figure). 표의 값은 반올림 전
    float, 그림은 새 객체다. 입력 배열/학습 모델을 변경하지 않는다.
    """
    p=np.array([.2,.4]);q=np.array([.5,-.1])
    s=sgd_step(p,q,float(rating),mean=3.,user_bias=.1,item_bias=-.2,
               learning_rate=float(learning_rate),regularization=float(regularization))
    before=np.r_[p,q,.1,-.2];after=np.r_[s['p_new'],s['q_new'],s['user_bias_new'],s['item_bias_new']]
    table=pd.DataFrame({'parameter':['p1','p2','q1','q2','user_bias','item_bias'],
                        'before':before,'after':after,'change':after-before})
    fig=Figure(figsize=(7,3));ax=fig.subplots();x=np.arange(2)
    ax.bar(x-.18,s['factor_contributions'],.36,label='Before');ax.bar(x+.18,s['contributions_new'],.36,label='After')
    ax.axhline(0,color='#64748b',lw=.7);ax.set(xticks=x,xticklabels=['Factor 1','Factor 2'],ylabel='Contribution to rating');ax.legend();fig.tight_layout()
    message=f"예측 {s['prediction']:.6f} → {s['prediction_new']:.6f} | 잔차 {s['error']:.6f} → {s['error_new']:.6f}\n한 관측 목적함수 {s['loss']:.6f} → {s['loss_new']:.6f}. 큰 학습률에서는 감소하지 않을 수 있습니다."
    return message,table,fig


def training_view(ratings: pd.DataFrame, n_factors: int=2, learning_rate: float=.02,
                  regularization: float=.02, epochs: int=30, user_id=1):
    """매 호출 같은 seed/init_scale에서 새 모델을 학습한다.

    ratings는 관측 평점 표. 반환 (요약, epoch별 표, 곡선 Figure, 미관측 추천 표).
    모든 오차는 train RMSE이며 일반화 점수가 아니다. 최대 실행량은 화면에서
    제한하며 Python 직접 호출에서는 호출자가 정한다. 다운로드 부작용 없음.
    """
    model=MFSGD(int(n_factors),float(learning_rate),float(regularization),int(epochs)).fit(ratings)
    h=model.history_.copy();fig=Figure(figsize=(7,3));ax=fig.subplots()
    ax.plot(h.epoch,h.train_rmse,color='#0f766e',marker='.',label='Train RMSE')
    ax.set(xlabel='Epoch (0 = initialization)',ylabel='Rating RMSE');ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    summary=f"관측 {len(ratings):,}개 | K={int(n_factors)} | train RMSE {h.train_rmse.iloc[0]:.4f} → {h.train_rmse.iloc[-1]:.4f}\n학습에 쓴 관측의 오차입니다. 새로운 데이터의 추천 성능을 측정한 결과가 아닙니다."
    return summary,h,fig,model.recommend(user_id,5)


def build_mf_app(ratings: pd.DataFrame | None=None):
    """두 탭의 Gradio Blocks를 만들어 반환한다. 이 함수는 서버를 열지 않는다.

    ratings=None이면 독자 5×5/18관측 예제. 별도 관측 표를 주면 최대
    3,000행을 seed 고정 표본으로 사용하고 화면에 명시한다. launch는 호출자.
    한 단계 탭은 고정 손계산 예제, 반복 학습 탭은 전달한 데이터이다.
    """
    import gradio as gr
    toy=ratings is None
    data=comparison_toy().ratings if toy else ratings.copy()
    if len(data)>3000: data=data.sample(3000,random_state=20261001).reset_index(drop=True)
    user=data.user_id.iloc[0]
    with gr.Blocks(title='W05B · MF와 SGD 실험실',analytics_enabled=False) as app:
        gr.Markdown('# MF와 SGD 실험실\n예상하기 → 한 설정 바꾸기 → 결과 비교하기 → 이유 설명하기')
        with gr.Tab('1 · 한 평점의 갱신'):
            gr.Markdown('p=(0.2, 0.4), q=(0.5, -0.1), 평균=3, 사용자 편향=0.1, 영화 편향=-0.2. 매번 이 상태에서 한 번 갱신합니다.')
            with gr.Row():
                r=gr.Slider(1,5,value=4,step=1,label='실제 평점')
                alpha=gr.Slider(.001,.5,value=.05,step=.001,label='한 단계 학습률')
                beta=gr.Slider(0,1,value=.02,step=.01,label='한 단계 정규화')
            button=gr.Button('한 단계 계산',variant='primary')
            msg=gr.Textbox(label='예측과 잔차',lines=3);table=gr.Dataframe(label='갱신 전후');plot=gr.Plot(label='요인별 내적 기여도')
            button.click(step_view,[r,alpha,beta],[msg,table,plot],api_name='step')
        with gr.Tab('2 · 관측 전체를 반복 학습'):
            gr.Markdown(f"데이터: {'독자 합성 5×5 예제' if toy else '전달한 실제 평점 표의 고정 표본'} / 관측 {len(data):,}개. 초기화 표준편차 0.1, seed 고정. 추천 대상 ID={user}.")
            with gr.Row():
                k=gr.Slider(1,16,value=2,step=1,label='잠재요인 수 K')
                lr=gr.Slider(.001,.1,value=.02,step=.001,label='반복 학습률')
                reg=gr.Slider(0,.5,value=.02,step=.01,label='반복 정규화')
                epochs=gr.Slider(1,80,value=30,step=1,label='Epoch')
            train=gr.Button('새 모델로 학습',variant='primary')
            summary=gr.Textbox(label='학습 결과와 해석 범위',lines=3)
            history=gr.Dataframe(label='Epoch별 train RMSE와 목적함수');curve=gr.Plot(label='학습 곡선');ranking=gr.Dataframe(label='미관측 영화의 예측 순위')
            def callback(k,lr,reg,epochs):
                return training_view(data,k,lr,reg,epochs,user)
            train.click(callback,[k,lr,reg,epochs],[summary,history,curve,ranking],api_name='train')
        gr.Markdown('기록: 고정한 조건 / 바꾼 설정 / 예측한 변화 / 관측한 수치 / 가능한 이유 / 이 실험으로 알 수 없는 것. Colab 공유 주소는 실행 중인 런타임에 의존합니다.')
    return app
