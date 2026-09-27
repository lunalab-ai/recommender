"""W05A 누적 웹 앱: 같은 평가 조건의 두 설정과 추천 이유를 비교한다."""
from __future__ import annotations
from dataclasses import asdict
from html import escape
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
from .model_comparison import ComparisonSuite,ModelSpec,baseline_specs,cohort


LABELS={'전체 평균':'global','영화 평균':'movie','집단 평균':'group','평가 수 인기':'count',
        '장르 내용 기반':'content','사용자 CF':'user','아이템 CF':'item','Pearson CF':'pearson'}


class ComparisonLab:
    """train/validation/메타데이터→공통 모델과 검증 지표 캐시.

    test는 받지 않는다. rank_limit=100, seed=20260929로 같은 사용자를 고정.
    suite는 선택적으로 이미 학습된 ComparisonSuite를 재사용한다.
    처음 설정 계산은 시간이 걸리며 같은 설정과 N은 캐시에서 읽는다.
    예: lab=ComparisonLab(train,validation,users,movies).
    """
    def __init__(self,train: pd.DataFrame,validation: pd.DataFrame,users: pd.DataFrame,
                 movies: pd.DataFrame,*,rank_limit: int = 100,seed: int = 20260929,
                 suite: ComparisonSuite | None = None,data_label: str = 'MovieLens100K'):
        self.suite=suite if suite is not None else ComparisonSuite(train,users,movies)
        if not self.suite.train.equals(train):raise ValueError('Supplied suite uses different training data')
        self.validation=validation.copy();self.ranking_users=cohort(validation,rank_limit,seed)
        self.cache={};self.data_label=data_label

    def metrics(self,spec: ModelSpec,top_n: int = 10) -> tuple[dict,bool]:
        """요약 지표 사본과 캐시 적중 여부. 처음만 실제 validation 평가."""
        key=(spec,top_n);hit=key in self.cache
        if not hit:self.cache[key]=self.suite.evaluate(spec,self.validation,top_n=top_n,ranking_users=self.ranking_users)[0]
        return self.cache[key].copy(),hit


def comparison_view(lab: ComparisonLab,left: ModelSpec,right: ModelSpec,user_id: int,
                    top_n: int = 10,history: list | None = None) -> tuple:
    """두 설정·사용자·N→요약HTML/좌추천/우추천/지표표/기록표/새기록/그림.

    평점과순위지표를별도로표시하고추천목록에제목·장르·근거·학습평가수를포함.
    사용자 변경은추천목록만변경하며집계지표의고정평가집단은유지된다.
    history 사본은gr.State의개인세션기록;원본리스트/모델설정은변경하지않는다.
    """
    import matplotlib.pyplot as plt
    metrics=[];tops=[];cached=[]
    for spec in (left,right):
        result,hit=lab.metrics(spec,top_n);metrics.append(result);cached.append(hit)
        tops.append(lab.suite.recommend(spec,user_id,top_n))
    records=list(history or [])
    for side,spec,m in zip(('기준','변경'),(left,right),metrics):
        records.append(dict(side=side,user_id=user_id,**asdict(spec),top_n=top_n,
            **{k:m[k] for k in ('rmse','mae','precision','recall','ndcg','cf_support','catalog_coverage')}))
    table=pd.DataFrame(metrics)[['name','mae','rmse','precision','recall','f1','ndcg','cf_support','rating_fallback','catalog_coverage','content_fallback','ranking_users','excluded_no_positive']]
    summary=(f'<h3>동일 조건 · 사용자 {int(user_id)} · Top-{top_n}</h3>'
        f'<p>{escape(lab.data_label)} · 학습 이력 {len(lab.suite.seen.get(user_id,set()))}편을 후보에서 제외합니다.<br>'
        f'평점 평가 {metrics[0]["rating_count"]:,}건 전체 / 순위 표본 {metrics[0]["ranking_cohort"]}명 중 '
        f'{metrics[0]["ranking_users"]}명, 관련 영화 없음 {metrics[0]["excluded_no_positive"]}명 제외.<br>'
        f'캐시 재사용: 기준 {cached[0]}, 변경 {cached[1]}. 표의 빈 RMSE/MAE는 해당없음입니다.<br>'
        '사용자 선택은 아래 추천 목록을 바꿉니다. 위 지표는 고정 평가집단의 평균입니다.</p>')
    fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    for ax,key,title in zip(axes,('rmse','ndcg','catalog_coverage'),('RMSE (lower)','NDCG@N (higher)','Catalog coverage')):
        values=[m[key] for m in metrics]
        ax.bar(['Before','After'],values,color=['#64748b','#0e8a82'])
        for i,v in enumerate(values):
            if np.isfinite(v):ax.text(i,v,f'{v:.4f}',ha='center',va='bottom',fontsize=9)
            else:ax.text(i,.05,'N/A',ha='center')
        ax.set_title(title);ax.spines[['top','right']].set_visible(False);ax.margins(y=.25)
    fig.tight_layout();plt.close(fig)
    return summary,tops[0],tops[1],table,pd.DataFrame(records),records,fig


def export_comparison(history: list) -> str:
    """개인 세션의 집계 비교 기록→UTF-8 BOM CSV 임시파일 경로.

    원본평점/사용자메타데이터는내보내지않는다. 새임시파일생성이부작용.
    반환파일은Gradio File로다운로드. 빈기록은ValueError.
    """
    if not history:raise ValueError('먼저 비교 버튼을 실행하세요.')
    with tempfile.NamedTemporaryFile(prefix='recommender-comparison-',suffix='.csv',delete=False) as f:path=Path(f.name)
    pd.DataFrame(history).to_csv(path,index=False,encoding='utf-8-sig')
    return str(path)


def build_comparison_app(lab: ComparisonLab,*,legacy=None):
    """실험 객체와 이전 Gradio Blocks→실행 전 누적 Blocks.

    .click 입력 순서가 모델설정·사용자·N·개인기록과 연결된다. 서버 시작은
    호출자의 launch에서만 수행. CPU/초기계산지연/공유주소수명을화면에안내.
    """
    import gradio as gr
    base={s.name:s for s in baseline_specs()}
    users=sorted(lab.suite.seen)
    def run(baseline,kind,user,k,common,minimum,centered,clip,group,group_min,liked,n,history):
        mode=LABELS[kind]
        settings=dict(k=0 if mode=='pearson' else int(k),min_common=max(2,int(common)) if mode=='pearson' else int(common),
            min_neighbors=int(minimum),centered=bool(centered) if mode=='user' else False,clip=bool(clip),
            group_col=group,min_group_ratings=int(group_min),like_threshold=float(liked))
        name=f'{mode}:k{settings["k"]}:c{settings["min_common"]}:m{settings["min_neighbors"]}:center{settings["centered"]}:clip{settings["clip"]}:group{group}{int(group_min)}:like{liked}'
        right=ModelSpec(name,mode,mode,**settings)
        return comparison_view(lab,base[baseline],right,int(user),int(n),history)
    with gr.Blocks(title='추천 모델 비교 실험실',fill_width=True) as app:
        gr.Markdown('# 추천 모델 비교 실험실\n가설을 세우고 설정 하나만 바꾼 뒤, 지표와 추천 이유를 함께 읽으세요.')
        with gr.Tab('W05A · 같은 조건에서 비교'):
            with gr.Row():
                left=gr.Dropdown(list(base),value='count',label='기준 모델',min_width=210)
                kind=gr.Dropdown(list(LABELS),value='사용자 CF',label='변경할 모델',min_width=210)
                user=gr.Dropdown(users,value=users[0],label='추천 목록을 볼 사용자',min_width=210)
            n=gr.Slider(5,20,value=10,step=5,label='추천 목록 길이 N · 양쪽 공통')
            with gr.Accordion('모델 설정 · 관련된 설정만 적용됩니다',open=True):
                with gr.Row():
                    k=gr.Slider(0,50,value=30,step=1,label='CF 이웃 수 k · 0 전체',min_width=220)
                    common=gr.Slider(1,10,value=3,step=1,label='CF 최소 공통수',min_width=220)
                    minimum=gr.Slider(1,5,value=2,step=1,label='CF 최소 이웃',min_width=220)
                with gr.Row():
                    centered=gr.Checkbox(True,label='사용자 CF 평균 보정')
                    clip=gr.Checkbox(True,label='User/Item CF 1–5점 제한')
                with gr.Row():
                    group=gr.Dropdown(['sex','occupation'],value='sex',label='집단 평균 속성',min_width=220)
                    group_min=gr.Slider(1,20,value=1,step=1,label='집단×영화 최소평가수',min_width=220)
                    liked=gr.Slider(3,5,value=4,step=1,label='내용 프로필의 학습 선호하한',min_width=220)
                gr.Markdown('평가 관련성은 항상 4점 이상입니다. 인기·전체/영화 평균은 위 모델 설정을 사용하지 않습니다. '
                    'Pearson은 전체 이웃·공통수 최소2를 사용하며 k·평균 보정은 적용하지 않습니다.')
            button=gr.Button('두 모델 비교하고 기록하기',variant='primary')
            summary=gr.HTML()
            with gr.Row():
                before=gr.Dataframe(label='기준 추천 · 제목/장르/점수/근거/학습 평가 수',interactive=False,wrap=True,min_width=0)
                after=gr.Dataframe(label='변경 추천 · 같은 후보와 사용자',interactive=False,wrap=True,min_width=0)
            metrics=gr.Dataframe(label='동일 validation 성과 · 빈 평점 지표는 해당없음',interactive=False,wrap=True,min_width=0)
            plot=gr.Plot()
            table=gr.Dataframe(label='개인 비교 기록',interactive=False,wrap=True,min_width=0)
            state=gr.State([])
            button.click(run,[left,kind,user,k,common,minimum,centered,clip,group,group_min,liked,n,state],
                [summary,before,after,metrics,table,state,plot],api_name='compare_models')
            download=gr.Button('비교 기록 CSV 만들기')
            file=gr.File(label='내 비교 기록 다운로드')
            download.click(export_comparison,[state],[file],api_name='export_comparison')
            gr.Markdown('처음 계산은 잠시 기다려 주세요. 같은 설정은 캐시로 재사용합니다. '
                '이 앱은 validation만 사용합니다. test를 보고 설정을 바꾸지 않습니다. '
                '평점 관측의 CF 근거 비율과 카탈로그 노출 비율은 분모가 다릅니다. '
                'MovieLens의 미관측 영화는 실제 비선호로 확인된 영화가 아닙니다.')
        if legacy is not None:
            with gr.Tab('이전 수업 누적 앱'):legacy.render()
        gr.Markdown('Colab 공유 링크는 런타임이 실행 중일 때만 유효합니다. 재접속할 때 노트북을 다시 실행하고 새 링크를 사용하세요.')
    return app
