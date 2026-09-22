"""W04B 누적 CF 실험실. 계산·평가와 화면 이벤트를 연결한다."""
from __future__ import annotations
from html import escape
import numpy as np
import pandas as pd
from .cf_synthesis import EvidenceCF, evaluate_cf

MODES = ['사용자 CF', '사용자 CF · 평균 보정', '아이템 CF']


class CFLab:
    """train/validation/카탈로그로 두 축을 fit하고 검증 지표를 캐시한다.

    train, validation: 분리된 관측 DataFrame. movies: ID/title 카탈로그.
    rank_limit=40은 빠른 순위평가 표본 상한, seed=20260923은 표본 재현용.
    평점 지표는 validation 전체, 순위 지표는 같은 고정 표본으로 계산한다.
    test는 받지 않는다. 앱의 조절로 test를 엿볼 수 없다. 파일/서버 부작용 없음.
    예: lab=CFLab(train, validation, movies); lab.model(MODES[1],30,3,2,True).
    """

    def __init__(self, train: pd.DataFrame, validation: pd.DataFrame, movies: pd.DataFrame,
                 *, rank_limit: int = 40, seed: int = 20260923, data_label: str = '사용자가 제공한 관측 자료'):
        if type(rank_limit) is not int or rank_limit < 1:
            raise ValueError('rank_limit must be a positive integer')
        self.models = {axis:EvidenceCF(axis=axis).fit(train) for axis in ('user','item')}
        self.validation, self.movies = validation.copy(), movies.copy()
        self.data_label = str(data_label)
        ids = np.sort(validation.user_id.unique())
        self.ranking_users = sorted(np.random.default_rng(seed).choice(ids,min(len(ids),rank_limit),replace=False).tolist())
        self.cache = {}

    def model(self, mode: str, k: int, min_common: int, min_neighbors: int, clip: bool) -> EvidenceCF:
        """화면 설정→공유 fit상태를 보존하는 모델 복사본. 입력 의미는 EvidenceCF와 같다."""
        if mode not in MODES:
            raise ValueError('Unknown CF mode')
        return self.models['item' if mode==MODES[2] else 'user'].configured(
            k=k,min_common=min_common,min_neighbors=min_neighbors,centered=mode==MODES[1],clip=clip)

    def metrics(self, mode: str, k: int, min_common: int, min_neighbors: int, clip: bool, top_n: int) -> dict:
        """설정별 검증 지표 사본 반환. 같은 설정은 재계산 없이 캐시 재사용한다.

        첫 호출은 validation 전체 예측과 고정 사용자 표본의 전체 후보 순위를 계산.
        실제 callback 지연은 장비/설정에 따라 달라진다. 성능값을 미리 만들지 않는다.
        """
        key=(mode,k,min_common,min_neighbors,clip,top_n)
        if key not in self.cache:
            self.cache[key]=evaluate_cf(self.model(*key[:5]),self.validation,self.movies,
                top_n=top_n,ranking_users=self.ranking_users)
        return self.cache[key].copy()


def synthesis_view(lab: CFLab, mode: str, user_id: int, k: int, min_common: int,
                   min_neighbors: int, clip: bool, top_n: int, history: list | None = None) -> tuple:
    """화면 입력→요약HTML/추천표/전체근거표/비교기록표/새개인기록/막대그림.

    ID와 개수는 정수, clip은bool. history는 세션별 기록 리스트(기본None).
    이전 리스트·공유모델은 변경하지 않는다. 숫자는 계산 시 반올림하지 않는다.
    그림은 최대 20개 이웃과 나머지 기여도 합을 표시한다. 전체 이웃은 근거표에 남긴다.
    """
    import matplotlib.pyplot as plt
    model=lab.model(mode,k,min_common,min_neighbors,clip)
    top=model.recommend(user_id,lab.movies,top_n)
    evidence=model.explain(user_id,int(top.iloc[0].movie_id)) if len(top) else pd.DataFrame()
    metrics=lab.metrics(mode,k,min_common,min_neighbors,clip,top_n)
    record=dict(방식=mode,k=k,공통수=min_common,최소이웃=min_neighbors,clip=clip,N=top_n,**metrics)
    new_history=list(history or [])+[record]
    summary=(f'<h3>{escape(mode)} · 설정 비교</h3><p>데이터: {escape(lab.data_label)}<br>평점 평가 {metrics["rating_count"]:,}개 · '
        f'순위 평가 고정표본 {metrics["ranking_cohort"]}명 중 {metrics["ranking_users"]}명 '
        f'(관련 영화 없음 {metrics["excluded_no_positive"]}명 제외)<br>'
        f'RMSE <b>{metrics["rmse"]:.4f}</b> · MAE {metrics["mae"]:.4f} · '
        f'Precision@{top_n} {metrics["precision"]:.4f} · Recall {metrics["recall"]:.4f} · F1 {metrics["f1"]:.4f}<br>'
        f'NDCG {metrics["ndcg"]:.4f} · CF 근거 확보 {metrics["cf_coverage"]:.1%} · '
        f'CF 후보가 있는 사용자 {metrics["cf_user_coverage"]:.1%} · 카탈로그 노출 {metrics["catalog_coverage"]:.1%}</p>')
    if len(top):
        row=top.iloc[0]
        summary+=(f'<p>첫 추천 <b>{escape(str(row.title))}</b>: {row.raw_prediction:.4f} → {row.prediction:.4f}점 · '
            f'확보 이웃 {row.eligible_neighbors} / 실제 사용 {row.n_contributors} · {escape(row.basis)}</p>')
    else:
        summary+='<p>이 사용자는 모든 카탈로그 영화를 이미 평가했습니다.</p>'
    fig,ax=plt.subplots(figsize=(8,max(2.4,.25*min(len(evidence),21))))
    if len(evidence):
        labels=evidence.neighbor_id.astype(str).tolist()[:20]
        values=evidence.contribution.tolist()[:20]
        if len(evidence)>20:
            labels.append(f'Other {len(evidence)-20} (sum)')
            values.append(float(evidence.contribution.iloc[20:].sum()))
        ax.barh(labels,values,color='#157f87')
        ax.invert_yaxis();ax.axvline(0,color='#64748b',lw=.8)
        ax.set(xlabel='Contribution to raw prediction',ylabel='User ID' if model.axis=='user' else 'Movie ID')
        base=float(model.user_means_.loc[user_id]) if model.centered else 0
        ax.set_title(f'Base {base:.3f} + contributions {evidence.contribution.sum():.3f}')
    else:
        ax.text(.5,.5,'No CF evidence: training-mean fallback',ha='center',va='center',transform=ax.transAxes)
        ax.axis('off')
    fig.tight_layout();plt.close(fig)
    columns=['방식','k','공통수','최소이웃','clip','N','rmse','mae','precision','recall','f1','ndcg','cf_coverage','catalog_coverage']
    return summary,top.round(6),evidence.round(6),pd.DataFrame(new_history)[columns].round(6),new_history,fig


def build_synthesis_app(lab: CFLab, *, legacy=None):
    """CFLab와 선택적 이전 Gradio Blocks→실행 전 Blocks. launch는 호출자가 한다.

    조절→버튼.click→synthesis_view→표/그래프가 연결된다. 사용자별 history를
    gr.State에 두어 비교 기록을 분리한다. 서버 시작/다운로드는 하지 않는다.
    """
    import gradio as gr
    users=sorted(lab.models['user'].seen_)
    def show(mode,user,k,common,minimum,clip,n,history):
        return synthesis_view(lab,mode,int(user),int(k),int(common),int(minimum),bool(clip),int(n),history)
    with gr.Blocks(title='CF 종합 실험실',fill_width=True) as app:
        gr.Markdown('# CF 종합 실험실\n설정을 하나씩 바꾸고, 추천 근거와 검증 지표를 함께 비교하세요.')
        with gr.Tab('신뢰도 · User/Item · 성과'):
            with gr.Row():
                mode=gr.Dropdown(MODES,value=MODES[1],label='CF 방식',min_width=220)
                user=gr.Dropdown(users,value=users[0],label='추천 대상 사용자',min_width=220)
            with gr.Row():
                k=gr.Slider(0,60,value=30,step=1,label='이웃 수 k · 0은 전체',min_width=220)
                common=gr.Slider(0,30,value=3,step=1,label='최소 공통 평가 수',min_width=220)
                minimum=gr.Slider(1,10,value=2,step=1,label='최소 유효 이웃 수',min_width=220)
            with gr.Row():
                clip=gr.Checkbox(value=True,label='예측을 1~5점으로 제한')
                n=gr.Slider(1,20,value=5,step=1,label='추천 수 N',min_width=220)
            button=gr.Button('계산하고 비교 기록에 추가',variant='primary')
            summary=gr.HTML()
            top=gr.Dataframe(label='추천 목록 · 확보 이웃과 실제 사용 이웃',interactive=False,wrap=True,min_width=0)
            plot=gr.Plot(label='첫 추천의 계산 근거')
            with gr.Accordion('첫 추천의 전체 기여 이웃',open=False):
                evidence=gr.Dataframe(interactive=False,wrap=True,min_width=0)
            history_table=gr.Dataframe(label='설정 변경 전후 비교 · 동일 검증 자료',interactive=False,wrap=True,min_width=0)
            history=gr.State([])
            button.click(show,[mode,user,k,common,minimum,clip,n,history],
                [summary,top,evidence,history_table,history,plot],api_name='synthesis_compare')
            gr.Markdown('평점 지표는 검증 관측 전체, 순위 지표는 고정 사용자 표본입니다. '
                '추천은 전체 카탈로그에서 학습 이력만 제외합니다. CF 근거 비율과 카탈로그 노출 비율의 분모는 다릅니다. '
                '미관측 영화는 실제 비선호가 아닙니다. 이 지표는 온라인 만족도 측정이 아닙니다.')
        if legacy is not None:
            with gr.Tab('이전 수업 누적 앱'):
                legacy.render()
        gr.Markdown('Colab 공유 주소는 런타임 실행 중에만 유지됩니다. 종료 후 마지막 실행 단계를 다시 수행하세요.')
    return app
