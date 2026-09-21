"""W04A 이웃/보정 조절 탭. W03A CF 및 이전 네 방법 비교 탭을 보존한다."""
from copy import copy
from html import escape
import pandas as pd
from .neighborhood import NeighborCF


def neighbor_view(model: NeighborCF, movies: pd.DataFrame, user_id: int,
                  k: int = 30, centered: bool = False, threshold: float = 0,
                  top_n: int = 5) -> tuple:
    """학습 모델과 화면 설정→(요약HTML,추천표,첫 추천 전체 근거표).

    model은 fit 완료, movies는 movie_id/title, 나머지는 NeighborCF 설정과 N.
    재학습/다운로드/서버 시작 없음. 얕은 복사본의 설정만 변경하여 동시 사용자가
    원본 설정을 바꾸지 않는다. 근거 표는 전체 이웃이며 수평 스크롤 가능하다.
    사용자 ID/k/N은 정수, centered는 bool. 오류는 해당 모델 API와 같다.
    """
    current = copy(model)
    current.k, current.centered, current.threshold = k, centered, threshold
    current._check_settings()
    rows = current.recommend(user_id,movies,top_n)
    if rows.empty:
        return '<p>미평가 영화가 없습니다.</p>', rows, pd.DataFrame()
    first = rows.iloc[0]
    evidence = current.explain(user_id,int(first.movie_id))
    base = float(current.user_means_.get(user_id,0)) if centered and not evidence.empty else 0
    cards = ''.join(f'<li>{escape(str(r.title))}: {r.prediction:.3f}점 ({r.n_contributors}명)</li>' for r in rows.head(3).itertuples())
    message = (f'<div style="overflow-wrap:anywhere"><h3>이웃으로 계산한 추천</h3><ol>{cards}</ol>'
        f'<p>첫 추천: {escape(str(first.title))} · {escape(first.basis)} · 실제 이웃 {first.n_contributors}명<br>'
        f'clip 전 {first.raw_prediction:.6f} → 최종 {first.prediction:.6f}점</p>')
    if len(evidence):
        message += f'<p>기준 {base:.6f} + 전체 기여도 합 {evidence.contribution.sum():.6f} = {first.raw_prediction:.6f}</p>'
    else:
        message += '<p>선택 가능한 이웃이 없어 학습 평균으로 대체했습니다.</p>'
    return message+'</div>', rows.round(6), evidence.round(6)


def build_neighbor_app(model: NeighborCF, movies: pd.DataFrame, *, legacy=None):
    """학습 모델/카탈로그와 선택적 기존 Gradio Blocks→미실행 Blocks.

    사용자,이웃수,보정,임계값,N을 callback에 연결한다. legacy는 build_cf_app
    반환값. 호출자가 Colab에서 launch(share=True)한다. fit/네트워크 없음.
    """
    import gradio as gr
    model._require_fit()
    def show(user,k,centered,threshold,n):
        return neighbor_view(model,movies,int(user),int(k),bool(centered),float(threshold),int(n))
    with gr.Blocks(title='내 취향 이웃 실험실', fill_width=True) as app:
        gr.Markdown('# 내 취향 이웃 실험실\n이웃 수와 평점 보정을 바꾸며 추천 근거를 비교하세요.')
        with gr.Tab('이웃 수 · 평가 경향'):
            with gr.Row():
                user=gr.Dropdown(sorted(model.seen_),value=min(model.seen_),label='사용자 ID',min_width=220)
                k=gr.Slider(0,60,value=30,step=1,label='이웃 수 k · 0은 전체',min_width=220)
            with gr.Row():
                centered=gr.Checkbox(value=False,label='사용자 평균 보정')
                threshold=gr.Slider(0,1,value=0,step=.05,label='유사도 임계값 · 이 값보다 커야 함',min_width=220)
                n=gr.Slider(1,20,value=5,step=1,label='추천 수 N',min_width=220)
            button=gr.Button('이웃 설정으로 추천',variant='primary')
            summary=gr.HTML()
            rows=gr.Dataframe(label='미평가 영화와 예측 근거',interactive=False,wrap=True,min_width=0)
            with gr.Accordion('첫 추천의 전체 이웃 근거',open=False):
                evidence=gr.Dataframe(label='평점 · 사용자 평균 · 편차 · 기여도',interactive=False,wrap=True,min_width=0)
            button.click(show,[user,k,centered,threshold,n],[summary,rows,evidence],api_name='neighbor_recommend')
            app.load(show,[user,k,centered,threshold,n],[summary,rows,evidence])
            gr.Markdown('k는 최대 이웃 수입니다. 목표 영화별 실제 이웃 수는 다릅니다. N은 화면에 보여줄 추천 수입니다. 보정 시 이웃 평균을 빼고 대상 사용자 평균을 더합니다.')
        if legacy is not None:
            with gr.Tab('이전 수업 앱'):
                legacy.render()
        gr.Markdown('공유 주소는 Colab 런타임 실행 중에만 유지됩니다. 이 예측은 영화 만족도 보장이 아닙니다.')
    return app
