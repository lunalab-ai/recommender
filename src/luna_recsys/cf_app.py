"""기존 네 방법 비교 앱에 기본 CF와 근거 표시를 추가한다."""
from __future__ import annotations

from html import escape
import pandas as pd

from .collaborative import UserCF
from .ranking import FourMethodRecommender


def cf_view(model: UserCF, movies: pd.DataFrame, user_id: int, top_n: int = 10) -> tuple:
    """CF 화면의 요약 HTML, 추천 표, 첫 추천의 근거 표를 반환한다.

    model은 학습된 UserCF, movies는 movie_id/title 카탈로그, user_id는 정수,
    top_n은 양의 정수(기본10). model.recommend와 explain을 호출하며 재학습하지 않는다.
    추천 표는 최대top_n행, 근거 표는 전체 기여자 중 유사도 상위20행이다.
    요약에는 전체/표시 기여자 수와 전체 기여도의 합을 명시한다. HTML의
    영화명은 escape 처리한다. 이 함수는 서버/다운로드 없이도 직접 검사할 수 있다.
    예: summary, rows, evidence = cf_view(model, movies, 1, 5).
    """
    rows = model.recommend(user_id, movies, top_n)
    if rows.empty:
        return '<p>조건에 맞는 미평가 영화가 없습니다.</p>', rows, pd.DataFrame()
    first = rows.iloc[0]
    evidence = model.explain(user_id, int(first.movie_id))
    cards = ''.join('<li><strong>'+escape(str(r.title))+'</strong><br>'
                    f'예측 {r.prediction:.3f}점 · 근거 {r.n_contributors}명 · '+escape(r.basis)+'</li>'
                    for r in rows.head(3).itertuples())
    summary = '<div style="overflow-wrap:anywhere;min-width:0"><h3>추천 상위 3편</h3><ol>'+cards+'</ol>'
    if evidence.empty:
        summary += '<p>첫 추천은 CF 근거가 부족하여 학습 평균으로 대체했습니다.</p>'
    else:
        summary += (f'<p>첫 추천 근거: 전체 {len(evidence)}명 중 상위 {min(20,len(evidence))}명 표시. '
                    f'전체 기여도 합 = {evidence.contribution.sum():.6f}점. '
                    '아래 표시된 일부 행만의 합과 구별하세요.</p>')
    return summary+'</div>', rows.round(6), evidence.head(20).round(6)


def build_cf_app(model: UserCF, movies: pd.DataFrame,
                 comparison: FourMethodRecommender | None = None):
    """학습된 모델을 입력받아 실행 전 Gradio Blocks 객체를 반환한다.

    model: 학습된 UserCF. movies: movie_id/title 카탈로그. comparison: 기본None;
    기존 학습된 FourMethodRecommender를 넘기면 이전 네 방법 탭도 유지한다.
    사용자 선택과 추천 수(1–20)를 cf_view에 연결한다. 반환 앱은 호출자가
    launch(share=True)로 Colab에서 실행한다. 이 함수는 서버를 띄우거나
    데이터를 내려받지 않는다. Gradio [apps] 의존성이 필요하다.
    """
    import gradio as gr
    model._require_fit()
    legacy = None
    if comparison is not None:
        from .comparison_app import build_comparison_app
        legacy = build_comparison_app(comparison)
    choices = sorted(model.seen_)

    def show(user_id, top_n):
        return cf_view(model, movies, int(user_id), int(top_n))

    with gr.Blocks(title='취향 이웃으로 추천하기', fill_width=True) as app:
        gr.Markdown('# 취향 이웃으로 추천하기\n평점 패턴 → 유사도 → 가중평균 → 추천 근거를 확인하세요.')
        with gr.Tab('기본 협업 필터링'):
            with gr.Row():
                uid = gr.Dropdown(choices, value=choices[0], label='사용자 ID', min_width=240)
                n = gr.Slider(1, 20, value=5, step=1, label='추천 수 N (이웃 수 아님)', min_width=240)
            button = gr.Button('CF 추천 계산', variant='primary')
            summary = gr.HTML()
            rows = gr.Dataframe(label='미평가 영화 추천 목록', interactive=False, wrap=True, min_width=0)
            with gr.Accordion('첫 추천의 기여 평가자', open=False):
                evidence = gr.Dataframe(label='상위 20명 · 전체 합은 위 요약 참고', interactive=False, wrap=True, min_width=0)
            button.click(show, [uid, n], [summary, rows, evidence], api_name='cf_recommend')
            app.load(show, [uid, n], [summary, rows, evidence])
            gr.Markdown('cf는 양의 유사도 가중평균, movie-mean/global-mean은 근거 부족 시 학습 평균 대체입니다. '
                        '추천 수 N을 바꿔도 예측에 사용하는 전체 평가자 집합은 줄지 않습니다.')
        if legacy is not None:
            with gr.Tab('이전 네 방법 비교'):
                legacy.render()
        gr.Markdown('Colab 공유 주소는 실행 중인 런타임에 의존합니다. 런타임 종료 후에는 다시 실행해야 합니다.')
    return app
