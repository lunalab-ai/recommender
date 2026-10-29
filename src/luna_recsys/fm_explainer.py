"""고정 합성 FM의 예측·표현·한 단계 갱신을 보여 주는 W09B 개념 계산기."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .fm_core import (toy_parameters, encode_observation, fm_score,
                      pair_contributions, to_sparse, fm_sgd_step)


def concept_view(user='A', item='P', context=1., y=4., learning_rate=.1, factors=2):
    """매 호출 새 합성 모수에서 계산한다. 반복 학습/실제 성능 평가가 아니다.

    user A/B, item P/Q, context/y 유한 실수, learning_rate>0, factors1/2.
    반환 (요약 str, 특성표5행, 활성쌍표, 희소표, 갱신표, Figure).
    다른 사용자의 상태를 공유하지 않는다. 파일/네트워크 부작용 없음.
    """
    if factors not in (1, 2):
        raise ValueError('이 계산기는 합성 V의 1개/2개 열만 선택합니다.')
    toy = toy_parameters()
    V = toy['V'][:, :int(factors)].copy()
    x = encode_observation(user, item, context)
    score = fm_score(x, toy['w'], V, toy['bias'])
    step = fm_sgd_step(x, float(y), toy['w'], V, toy['bias'], float(learning_rate))
    idx, values = to_sparse(x)
    features = pd.DataFrame({'feature': toy['names'], 'x': x, 'w': toy['w'],
                             'linear': score['linear_terms']})
    pairs = pd.DataFrame(pair_contributions(x, V), columns=['i','j','dot','x_product','contribution'])
    sparse = pd.DataFrame({'index_python0': idx, 'index_book1': idx+1, 'value': values})
    rows = []
    for i, name in enumerate(toy['names']):
        rows.append({'parameter': 'w:'+name, 'before': toy['w'][i], 'after': step['w'][i],
                     'prediction_derivative': step['prediction_gradient_w'][i]})
        for f in range(V.shape[1]):
            rows.append({'parameter': f'V:{name}:f{f+1}', 'before': V[i,f], 'after': step['V'][i,f],
                         'prediction_derivative': step['prediction_gradient_V'][i,f]})
    updates = pd.DataFrame(rows)
    fig = Figure(figsize=(8,3.2))
    ax = fig.subplots()
    ax.bar(['Bias','Linear','Interactions','Total'],
           [score['bias'],score['linear'],score['interaction'],score['prediction']],
           color=['#64748b','#2563eb','#0f766e','#7c3aed'])
    ax.axhline(0,color='#334155',lw=.8)
    ax.set_ylabel('Raw FM score')
    fig.tight_layout()
    message = (f"고정 합성 모수 · 예측 {score['prediction']:.6f} = 절편 {score['bias']:.6f} + "
               f"선형 {score['linear']:.6f} + 상호작용 {score['interaction']:.6f}\n"
               f"활성 특성 {len(idx)}개 · 활성 쌍 {len(pairs)}개 · 선택한 요인 열 {factors}개\n"
               f"한 단계: 오차 {step['error']:.6f}, 예측 {step['prediction_new']:.6f}, "
               f"손실 {step['loss']:.9f} → {step['loss_new']:.9f}. 매번 같은 초기 상태에서 계산합니다.")
    return message, features, pairs, sparse, updates, fig


def build_fm_explainer():
    """Gradio Blocks를 만들고 반환한다. launch/외부 공유는 호출자가 결정한다.

    입력 설정→공통 계산 함수→기여도/표의 연결을 확인하는 이론 보조 화면이다.
    합성 모수를 고정하고 V 열 선택 시 재학습하지 않는다. 정답/개인자료 없음.
    """
    import gradio as gr
    initial = concept_view()
    with gr.Blocks(title='W09B · FM 개념 계산기', analytics_enabled=False) as app:
        gr.Markdown('# FM 개념 계산기\n한 입력의 효과를 예상하고 표·합과 비교하세요. '
                    '주어진 합성 모수이며 학습된 추천 성능이 아닙니다. 맥락은 수업용 무단위 수치입니다.')
        with gr.Row():
            user = gr.Dropdown(['A','B'],value='A',label='사용자')
            item = gr.Dropdown(['P','Q'],value='P',label='아이템')
            context = gr.Number(value=1,label='맥락 특성 x5',minimum=-2,maximum=2)
            factors = gr.Radio([1,2],value=2,label='고정 V에서 사용할 요인 열')
        with gr.Row():
            y = gr.Number(value=4,label='관측 목표 y')
            alpha = gr.Number(value=.1,label='한 단계 학습률',minimum=.001,maximum=1)
        button = gr.Button('예상과 계산 비교',variant='primary')
        message = gr.Textbox(value=initial[0],label='계산 결과',lines=4)
        with gr.Tab('특성과 기여도'):
            features = gr.Dataframe(value=initial[1],label='선형항의 특성별 기여',interactive=False)
            pairs = gr.Dataframe(value=initial[2],label='활성 쌍 i<j · 0인 내적도 표시',interactive=False)
            plot = gr.Plot(value=initial[5],label='기여도와 예측')
        with gr.Tab('희소 표현'):
            sparse = gr.Dataframe(value=initial[3],label='동일 입력의 비영 index/value',interactive=False)
        with gr.Tab('한 단계 갱신'):
            updates = gr.Dataframe(value=initial[4],label='같은 갱신 전 상태로 계산',interactive=False)
        button.click(concept_view,[user,item,context,y,alpha,factors],
                     [message,features,pairs,sparse,updates,plot])
        gr.Markdown('숫자를 바꾼 뒤 버튼을 누르세요. 전체 학습·데이터 전처리·학생 앱 조립은 다음 차시입니다. '
                    '공유 주소는 실행 중인 런타임에 의존합니다.')
    return app
