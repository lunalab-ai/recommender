"""W06B: 화살표, 단계별 SVD, 재구성 및 기존 MF 기능을 연결한 Gradio 앱."""
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .svd_foundations import teaching_matrix, svd_steps, transform_stages


def reconstruction_view(name: str = "main", rank: int = 1, report_fn=None):
    """예제 이름/rank → 설명(str), 근사(DataFrame), 비교(Figure).

    report_fn(report_dict)는 학생이 바꾸는 해석 문장 함수. 학습/다운로드 없음.
    입력 'main',1의 오차4, 보존 에너지36/52. 실제 평점 성능이 아니다.
    """
    if isinstance(rank, bool) or not float(rank).is_integer():
        raise ValueError("rank는 정수여야 합니다.")
    d = svd_steps(teaching_matrix(name), int(rank))
    fig = Figure(figsize=(10, 3.3)); axes = fig.subplots(1, 3)
    matrices = [d['A'], d['reconstruction'], d['A'] - d['reconstruction']]
    bound = max(1, float(np.max(np.abs(d['A']))))
    for ax, values, title in zip(axes, matrices, ['Original A', f'Rank {rank} approximation', 'A - approximation']):
        ax.imshow(values, cmap='RdBu_r', vmin=-bound, vmax=bound)
        for (i,j),v in np.ndenumerate(values):
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if abs(v)>.65*bound else '#172333')
        ax.set_title(title); ax.set_xticks(range(values.shape[1])); ax.set_yticks(range(values.shape[0]))
    fig.tight_layout()
    ratio = '정의하지 않음(원본이 영행렬)' if d['energy_fraction'] is None else f"{100*d['energy_fraction']:.2f}%"
    text = (f"입력 크기={d['A'].shape}, 수치적 rank={d['numerical_rank']}, 유지 k={rank}. "
            f"특이값={np.round(d['s'],6).tolist()}; Frobenius 오차={d['error']:.6f}; "
            f"제곱합 보존={ratio}. 보지 않은 평점의 정확도가 아닙니다.")
    if report_fn is not None:
        text += '\n' + str(report_fn(d))
    return text, pd.DataFrame(d['reconstruction']), fig


def direction_view(name: str = "main", degrees: float = 45):
    """2×2 예제와 각도(도) → x/Ax 표, 방향 잔차 설명, 화살표 그림.

    λ 후보=xᵀAx/(xᵀx). ||Ax−λx||가 작으면 현재 x가 고유방향이다.
    출력 순서 (str,DataFrame,Figure). Ax=0일 때 방향 없음도 명시한다.
    """
    a = teaching_matrix(name)
    if a.shape != (2,2) or not np.isfinite(degrees):
        raise ValueError('유한한 각도와 2×2 예제가 필요합니다.')
    angle = np.deg2rad(degrees); x = np.array([np.cos(angle), np.sin(angle)])
    y = a @ x; scale = float(x @ y); residual = float(np.linalg.norm(y - scale*x))
    description = '고유방향입니다.' if residual < 1e-8 else '같은 직선에 남지 않아 고유방향이 아닙니다.'
    if np.linalg.norm(y) < 1e-8:
        description = 'Ax=0: 출력 방향은 없으며, 이 x는 고유값0의 고유벡터입니다.'
    fig = Figure(figsize=(5,4)); ax = fig.subplots()
    for v, color, label in [(x,'#2563eb','x'),(y,'#e87831','Ax')]:
        ax.quiver(0,0,v[0],v[1],angles='xy',scale_units='xy',scale=1,color=color,label=label)
    bound=max(1.5,float(np.max(np.abs(y)))+1)
    ax.set(xlim=(-bound,bound),ylim=(-bound,bound),aspect='equal');ax.axhline(0,color='#ccd5df');ax.axvline(0,color='#ccd5df');ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    return (f'각도={degrees:g}°, 배율 후보={scale:.6f}, 방향 잔차={residual:.6f}. {description}',
            pd.DataFrame({'x':x,'Ax':y,'lambda*x':scale*x}), fig)


def stages_view(name: str = "main"):
    """단위원의 점을 Vᵀ→Σ→U 순서로 변환한 Figure와 설명 반환. 2×2만 사용."""
    a=teaching_matrix(name)
    t=np.linspace(0,2*np.pi,161);points=np.vstack([np.cos(t),np.sin(t)])
    stages=transform_stages(a,points)
    fig=Figure(figsize=(12,3.2));axes=fig.subplots(1,4)
    colors=np.linspace(0,1,points.shape[1])
    for ax,values,title in zip(axes,stages,['Unit circle','After Vt','After Sigma','After U = A @ x']):
        ax.scatter(*values,c=colors,cmap='viridis',s=9)
        limit=max(1.25,float(np.max(np.abs(values)))*1.18)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',title=title)
        ax.axhline(0,color='#ccd5df');ax.axvline(0,color='#ccd5df')
    fig.tight_layout()
    return '같은 색 점을 따라가세요. Vᵀ와 U는 길이를 보존하고 Σ가 방향별 길이를 바꿉니다. 패널별 축 범위를 확인하세요.', fig


def build_svd_app(report_fn=None):
    """새 SVD3탭과 W06A MF3탭을 포함하는 Blocks를 생성한다. launch는 별도.

    MF 복습에는 명시적 5×5 합성18관측을 사용하며 MovieLens 결과가 아니다.
    report_fn(dict)->str는 재구성 화면에 연결된다. 예: build_svd_app(lambda d: '오차를 해석하세요').
    """
    import gradio as gr
    from .model_comparison import comparison_toy
    from .mf_evaluation import split_three_way
    from .mf_evaluation_lab import build_evaluation_app
    split=split_three_way(comparison_toy().ratings)
    previous=build_evaluation_app(split.train,split.validation)
    with gr.Blocks(title='W06B · 눈으로 계산하는 SVD') as app:
        gr.Markdown('# 눈으로 계산하는 SVD\n**예상 → 한 설정 변경 → 실행 → 수치·그림 확인 → 이유 설명**')
        gr.Markdown('수업용 합성 행렬입니다. main=[[5,1],[1,5]], negative=diag(3,-2), deficient=[[3,0],[4,0]], rectangular=3×2, zero=영행렬.')
        with gr.Tab('1 · 고유방향'):
            name=gr.Dropdown(['main','negative','deficient','shear'],value='main',label='화살표 예제')
            angle=gr.Slider(0,180,value=45,step=15,label='벡터 각도(도)')
            run=gr.Button('화살표 계산',variant='primary')
            text,table,fig=direction_view()
            out=gr.Textbox(value=text,label='방향 해석',lines=3); frame=gr.Dataframe(value=table,label='x와 Ax');plot=gr.Plot(value=fig)
            run.click(direction_view,[name,angle],[out,frame,plot],api_name='direction')
        with gr.Tab('2 · SVD 세 단계'):
            name2=gr.Dropdown(['main','negative','deficient','shear'],value='main',label='변환 예제')
            run2=gr.Button('Vᵀ → Σ → U 보기')
            t2,f2=stages_view();out2=gr.Textbox(value=t2,label='그림 읽기');plot2=gr.Plot(value=f2)
            run2.click(stages_view,name2,[out2,plot2],api_name='stages')
        with gr.Tab('3 · rank와 재구성'):
            name3=gr.Dropdown(['main','rectangular','deficient','negative','zero'],value='main',label='재구성 예제')
            rank=gr.Slider(0,2,value=1,step=1,label='남길 항의 수 k')
            run3=gr.Button('재구성 계산',variant='primary')
            t3,df3,f3=reconstruction_view(report_fn=report_fn)
            out3=gr.Textbox(value=t3,label='재구성 해석과 나의 문장',lines=4);table3=gr.Dataframe(value=df3,label='근사 행렬');plot3=gr.Plot(value=f3)
            run3.click(lambda n,k:reconstruction_view(n,k,report_fn),[name3,rank],[out3,table3,plot3],api_name='reconstruct')
        with gr.Tab('4 · 지난 MF 실험 이어가기'):
            gr.Markdown('**작은 합성 자료 복습 모드**입니다. 기존 학습/검증·설정 비교·미관측 반례 기능을 보존했습니다. 표본이 매우 작아 모형 우열의 근거로 쓰지 않습니다.')
            previous.render()
        gr.Markdown('학습 기록: 무엇을 바꾸었나 / 예상은 무엇이었나 / 실제 수치와 그림 / 다른 결과가 나온 이유. 공유 링크는 실행 중인 런타임에 의존합니다.')
    return app
