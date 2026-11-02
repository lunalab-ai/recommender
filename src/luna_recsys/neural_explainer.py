"""FM 복습을 보존하고 MF/MLP 중간값을 연결하는 W10A 개념 앱."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from .neural_recommenders import fixed_forward


def forward_view(user='A', item='P', model='MLP', activation='relu', occupation='student'):
    """한 합성 관측의 중간값을 반환한다. 학습/실제 추천 성능이 아니다.

    model MF/MLP/MLP+occupation. activation relu/linear. 나머지 값은
    fixed_forward와 같다. 반환은 요약str, 입력표, 층별표, Figure다.
    입력 변경 전 결과를 예상하고 버튼을 눌러 비교한다. 다운로드/파일 쓰기 없음.
    """
    if model not in {'MF','MLP','MLP+occupation'}:
        raise ValueError('화면에 정의된 세 모형 중에서 선택하세요.')
    trace = fixed_forward(user,item,activation=activation,
                          occupation=occupation if model=='MLP+occupation' else None)
    names = ['p1','p2','q1','q2','b_u feature','b_i feature']
    if len(trace['z']) == 9:
        names += ['occupation1','occupation2','occupation3']
    table = pd.DataFrame({'input':names,'value':trace['z']})
    stages = pd.DataFrame({'unit':[1,2], 'Dense1 before activation':trace['pre'],
                           'after activation':trace['hidden'],'Dense2 linear':trace['second']})
    score = trace['mf_rating'] if model=='MF' else trace['mlp_rating']
    summary = (f'고정 합성 모수 · {model} · 예측 {score:.6f}\n'
               f'MF: 1 + {trace["user_bias"]:.3f} + {trace["item_bias"]:.3f} '
               f'+ 내적 {trace["p"] @ trace["q"]:.3f} = {trace["mf_rating"]:.6f}\n'
               f'MLP: 연결 폭 {len(trace["z"])} → Dense2 → {activation} → '
               f'Dense2(linear) → 출력1. 중심화 점수 {trace["centered_mlp"]:.6f} + 평균1.\n'
               '평균1은 주어진 합성 상수입니다. 데이터 학습 평균/모형 품질 실험과 구별하세요.')
    fig = Figure(figsize=(7,3.4))
    axes = fig.subplots(1,2)
    axes[0].bar(['z1','z2'],trace['pre'],color='#64748b',label='before')
    axes[0].bar(['z1','z2'],trace['hidden'],alpha=.7,color='#0f766e',label='after')
    axes[0].axhline(0,color='#334155',lw=.8)
    axes[0].legend()
    axes[0].set_title('Dense1 activation')
    axes[1].bar(['MF','MLP'],[trace['mf_rating'],trace['mlp_rating']],color=['#2563eb','#7c3aed'])
    axes[1].set_title('Fixed forward scores')
    axes[1].set_ylim(min(0,score-.5),max(3,score+.5))
    fig.tight_layout()
    return summary,table,stages,fig


def export_forward(user='A',item='P',model='MLP',activation='relu',occupation='student') -> str:
    """현재 합성 입력/층별값 CSV 경로. 시스템 임시폴더에 새 파일을 쓴다.

    민감 정보/학습 자료를 사용하지 않는다. 반환 파일은 앱 다운로드용이며
    현재 실행 환경의 임시 보관이다. 경로 인자를 받지 않는다.
    """
    _,inputs,stages,_ = forward_view(user,item,model,activation,occupation)
    with tempfile.NamedTemporaryFile(prefix='w10a-forward-',suffix='.csv',delete=False) as handle:
        path = Path(handle.name)
    combined = pd.concat([inputs.rename(columns={'input':'name'}),
                          stages.assign(name=lambda f:'unit'+f['unit'].astype(str))],ignore_index=True)
    combined.to_csv(path,index=False,encoding='utf-8-sig')
    return str(path)


def build_neural_explainer():
    """누적 Gradio Blocks. 기존 FM 계산 기능과 새 MF/MLP 탭을 조립한다.

    launch/share는 호출자가 정한다. 계산은 fixed_forward이며 학습 버튼이
    아니다. Gradio가 그림 캐시의 임시파일을 쓸 수 있다. CSV 버튼은 현재
    표를 별도 임시파일에 기록한다. 원자료 읽기/학습은 수행하지 않는다.
    반환 Blocks를 Colab에서 launch(share=True,inline=False)로 실행한다.
    """
    import gradio as gr
    from .fm_explainer import concept_view
    initial = forward_view()
    with gr.Blocks(title='W10A · FM에서 신경망으로',analytics_enabled=False) as app:
        gr.Markdown('# FM → MF → MLP\n변경 전 예측을 적고 중간값을 확인하세요. '
                    '이 화면은 고정 합성 모수의 순전파이며 실제 모형 학습 결과는 Colab의 평가 표에서 봅니다.')
        with gr.Tab('FM 복습'):
            gr.Markdown('기존 FM의 공통 계산 함수를 재사용합니다. 입력을 바꾸기 전에 예측하세요. '
                        '숫자를 입력한 뒤 Enter로 확정하고 비교 버튼을 누르세요.')
            fm_initial=concept_view()
            with gr.Row():
                fm_user=gr.Dropdown(['A','B'],value='A',label='FM 사용자')
                fm_item=gr.Dropdown(['P','Q'],value='P',label='FM 아이템')
                fm_context=gr.Number(value=1,label='FM 맥락',minimum=-2,maximum=2)
                fm_factors=gr.Radio([1,2],value=2,label='FM 요인 열')
            with gr.Row():
                fm_y=gr.Number(value=4,label='FM 관측 목표')
                fm_alpha=gr.Number(value=.1,label='FM 한 단계 학습률',minimum=.001,maximum=1)
            fm_button=gr.Button('FM 예상과 계산 비교')
            fm_message=gr.Textbox(fm_initial[0],label='FM 계산 결과',lines=4)
            with gr.Tab('특성과 기여도'):
                fm_features=gr.Dataframe(fm_initial[1],label='FM 선형 기여',interactive=False)
                fm_pairs=gr.Dataframe(fm_initial[2],label='FM 활성 쌍',interactive=False)
                fm_plot=gr.Plot(fm_initial[5],label='FM 기여도')
            with gr.Tab('희소 표현'):
                fm_sparse=gr.Dataframe(fm_initial[3],label='FM 비영 index/value',interactive=False)
            with gr.Tab('한 단계 갱신'):
                fm_updates=gr.Dataframe(fm_initial[4],label='FM 갱신 전후',interactive=False)
            fm_button.click(concept_view,[fm_user,fm_item,fm_context,fm_y,fm_alpha,fm_factors],
                [fm_message,fm_features,fm_pairs,fm_sparse,fm_updates,fm_plot])
        with gr.Tab('MF·MLP 계산 경로'):
            with gr.Row():
                user = gr.Dropdown(['A','B'],value='A',label='사용자')
                item = gr.Dropdown(['P','Q'],value='P',label='영화')
                model = gr.Radio(['MF','MLP','MLP+occupation'],value='MLP',label='계산 구조')
            with gr.Row():
                activation = gr.Radio(['relu','linear'],value='relu',label='첫 Dense 뒤 활성함수')
                occupation = gr.Dropdown(['student','engineer','artist'],value='student',label='합성 직업')
            button = gr.Button('예상과 중간값 비교',variant='primary')
            message = gr.Textbox(initial[0],label='계산과 의미',lines=5)
            inputs = gr.Dataframe(initial[1],label='연결 입력',interactive=False)
            stages = gr.Dataframe(initial[2],label='활성함수 전후',interactive=False)
            plot = gr.Plot(initial[3],label='중간값과 고정 점수')
            values = [user,item,model,activation,occupation]
            button.click(forward_view,values,[message,inputs,stages,plot])
            download = gr.Button('현재 중간값 CSV 만들기')
            file = gr.File(label='합성 계산 CSV',interactive=False)
            download.click(export_forward,values,file)
        gr.Markdown('수정 활동: 입력 위젯 → 버튼.click의 인자 순서 → forward_view의 인자 '
                    '→ 반환 표를 따라가세요. 결과 요약에 중심화 점수/평균을 추가하는 자신의 함수를 연결합니다. '
                    'Colab 공유 주소는 런타임이 켜져 있는 동안만 유지됩니다.')
    return app
