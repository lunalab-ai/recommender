"""W07A interactive comparison; callbacks are independently executable."""
from __future__ import annotations
from typing import Callable
from .review_foundations import review_experiment


def review_view(method: str, min_count: int, top_k: int, group: str,
                report_fn: Callable[[dict],str] | None=None):
    """Return explanation, ranking rows, held-out rows, metrics and plot.

    Inputs match review_experiment. report_fn(result)->str lets learners add
    their own explanation; it never changes model results. Calls no network.
    """
    import matplotlib.pyplot as plt
    result=review_experiment(method,int(min_count),int(top_k),group)
    rows=[[r['item'],round(r['score'],4),r['count'],r['basis']] for r in result['ranked']]
    metric_rows=[[name,'정의되지 않음' if value is None else round(value,4)] for name,value in result['metrics'].items()]
    fig,ax=plt.subplots(figsize=(6,2.8))
    if rows:
        ax.bar([r[0] for r in rows],[r[1] for r in rows],color='#2563eb')
    else:ax.text(.5,.5,'No eligible candidates',ha='center',transform=ax.transAxes)
    ax.set_title('Synthetic data — eligible unseen items');ax.set_ylabel('Ranking score');fig.tight_layout()
    explanation=(f"합성 자료 · 방법 `{method}` · 최소 관측 {int(min_count)} · 요청 K={int(top_k)} · 반환 {len(rows)}편. "
                 "A/C는 이미 평가해서 제외합니다. 같은 점수이면 관측 수, 다음 ID 순으로 정렬합니다. "
                 "관련 영화는 test의 B/E이며, test 점수는 추천 통계에 사용하지 않습니다. "
                 "짧은 목록에서도 Precision 분모는 요청 K, Recall 분모는 관련 영화2개입니다. "
                 "커버리지는 전체 카탈로그5편 중 이번 한 목록에 등장한 서로 다른 영화 비율입니다.")
    if method in {'count','content'}:explanation+=' 평점 수/코사인 점수는 별점 예측이 아니므로 MAE/RMSE를 계산하지 않습니다.'
    if method=='group':explanation+=' 집단 목록이 모두 비면 영화평균 목록으로 대체합니다. 아래 개별 평점의 대체 경로와 구별하세요.'
    if report_fn is not None:explanation+='\n\n'+str(report_fn(result))
    return explanation,rows,result['predictions'],metric_rows,fig


def build_review_app(report_fn: Callable[[dict],str] | None=None):
    """Build (do not launch) Gradio Blocks with reviewed baseline and SVD tabs.

    CPU only. Learner report_fn adds a sentence to the comparison result.
    Call .launch(share=True, prevent_thread_lock=True) in Colab. Temporary
    share URLs expire; the fixed-version notebook is the durable entrypoint.
    """
    import gradio as gr
    from .svd_lab import reconstruction_view
    with gr.Blocks(title='W07A · 추천 기준을 바꾸며 이해하기') as app:
        gr.Markdown('# W07A · 추천 기준을 바꾸며 이해하기\n먼저 결과를 예상하고 → 설정을 바꾸고 → 표에서 근거를 찾아 설명하세요. 데이터는 수업용 합성 자료입니다.')
        with gr.Tab('기본 추천 비교'):
            with gr.Row():
                method=gr.Dropdown(['mean','count','group','content'],value='mean',label='추천 기준')
                group=gr.Dropdown(['G1','G2'],value='G1',label='집단: group에서만 사용')
            with gr.Row():
                n=gr.Slider(1,4,value=1,step=1,label='최소 평점 수')
                k=gr.Slider(1,3,value=3,step=1,label='Top-K')
            go=gr.Button('추천과 평가 다시 계산',variant='primary')
            explanation=gr.Markdown()
            rankings=gr.Dataframe(headers=['영화','순위 점수','관측 수','근거'],interactive=False,label='추천 목록')
            predictions=gr.Dataframe(headers=['영화','test 실제','예측','예측 근거','실제−예측'],interactive=False,label='평점 예측 (평균 방법만)')
            metrics=gr.Dataframe(headers=['지표','값'],interactive=False,label='같은 평가 조건의 결과')
            plot=gr.Plot()
            def callback(method,n,k,group):return review_view(method,n,k,group,report_fn)
            go.click(callback,[method,n,k,group],[explanation,rankings,predictions,metrics,plot])
            app.load(callback,[method,n,k,group],[explanation,rankings,predictions,metrics,plot])
            with gr.Accordion('작은 데이터 전체와 읽는 순서',open=False):
                from .review_foundations import review_data
                data=review_data()
                gr.Dataframe(value=[list(r) for r in data['train']],headers=['user','item','rating'],interactive=False,label='train: 통계 계산')
                gr.Dataframe(value=[list(r) for r in data['test']],headers=['user','item','rating'],interactive=False,label='test: 평가 전용')
                gr.Markdown('mean=영화 평균 / count=관측 수 / group=집단 평균 / content=좋아한 영화의 장르 프로필 코사인. u1/u2는 G1, u3/u4는 G2. 특징 순서는 액션·코미디이며 A/E=[1,0], B=[1,1], C/D=[0,1].')
        with gr.Tab('SVD 마무리: k와 재구성'):
            gr.Markdown('W06B의 동일한 SVD 구현을 사용합니다. 오차가 줄어드는 것은 주어진 완전 행렬에 대한 재구성 결과입니다.')
            rank=gr.Slider(0,2,value=1,step=1,label='남길 특이값 항 수 k')
            run=gr.Button('SVD 재구성')
            note=gr.Markdown();table=gr.Dataframe(interactive=False);figure=gr.Plot()
            run.click(lambda k:reconstruction_view('main',int(k)),rank,[note,table,figure])
    return app
