"""학습된 네 추천 기준을 재사용하는 W02B 웹 앱과 표 callback."""
from __future__ import annotations
import pandas as pd
from html import escape
from .ranking import FourMethodRecommender, METHODS


def comparison_table(model: FourMethodRecommender, user_id: int, k: int = 10) -> pd.DataFrame:
    """동일 user_id의 네 상위 목록을 합쳐 화면용 DataFrame을 반환한다.

    model: fit이 끝난 FourMethodRecommender. user_id: 카탈로그에 추천할
    사용자 ID(정수). k: 방법별 최대 추천 수, 기본10인 양의 정수다.
    반환 열은 method/rank/movie_id/title/genres/score/basis이다.
    rank는 방법마다1부터 시작한다. 최대4k행이며 score는 원래 단위를 유지.
    입력 모델을 재학습/변경하지 않는다. 잘못된 k나 미학습 모델의 오류는
    숨기지 않고 전달한다. Gradio import 없이도 callback을 검사할 수 있다.
    """
    results = []
    for method in METHODS:
        table = model.recommend(user_id, method, k)
        table.insert(0, "rank", range(1, len(table)+1))
        table.insert(0, "method", method)
        results.append(table)
    return pd.concat(results, ignore_index=True)[
        ["method", "rank", "movie_id", "title", "genres", "score", "basis"]]


def comparison_cards(table: pd.DataFrame) -> str:
    """comparison_table 결과에서 방법별 상위 3개를 반응형 HTML로 표시한다.

    table: method/rank/title/genres/score/basis 열이 있는 DataFrame.
    반환값: gradio.HTML에 넣을 문자열. 원본 table과 점수는 변경하지 않는다.
    제목과 근거는 HTML 이스케이프하여 텍스트로 표시한다. 작은 화면에서는
    카드가 한 열로 배치되며 전체 순위는 별도 표에서 확인한다.
    """
    labels = {"count": "평점 수", "mean": "평점 평균", "group": "사용자 집단", "content": "내용 기반"}
    sections = []
    for method in METHODS:
        rows = table.loc[table.method == method].head(3)
        items = []
        for row in rows.itertuples():
            items.append(f'<li><strong>{escape(str(row.title))}</strong><br>'
                         f'{escape(str(row.genres))}<br>점수 {row.score:.4g} · '
                         f'{escape(str(row.basis))}</li>')
        sections.append('<section style="padding:16px;border:1px solid #9ca3af;border-radius:12px;min-width:0;overflow-wrap:anywhere">'
                        f'<h3>{labels[method]}</h3><ol style="padding-left:24px">'
                        + ''.join(items) + '</ol></section>')
    return '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px">' + ''.join(sections) + '</div>'


def build_comparison_app(model: FourMethodRecommender):
    """학습된 model을 사용하는 gradio.Blocks 객체를 만들고 반환한다.

    model: fit이 끝난 FourMethodRecommender. 사용자 선택은 학습 사용자 ID,
    목록 길이는1–20이다. 반환 앱의 launch(share=True)는 호출자가 Colab에서
    실행한다. 이 함수 자체는 네트워크 서버를 띄우거나 데이터를 다운로드하지
    않는다. [apps] 선택 의존성인 Gradio가 없으면 ImportError가 발생한다.
    화면 입력은 callback에서 정수로 변환하여 comparison_table에 전달한다.
    """
    import gradio as gr
    if not hasattr(model, "seen_"):
        raise ValueError("Call fit before building the app")
    choices = sorted(model.seen_)

    def compare(user_id, k):
        """UI 숫자를 정수로 변환하고 같은 학습 객체에 조회를 위임한다."""
        table = comparison_table(model, int(user_id), int(k))
        return comparison_cards(table), table

    with gr.Blocks(title="네 가지 영화 추천 비교", fill_width=True) as app:
        gr.Markdown("# 네 가지 추천을 같은 사용자에게 적용하기\n"
                    "사용자와 목록 길이를 바꾸고 제목·장르·근거를 비교하세요. "
                    "평점 수, 예측 평점, 코사인은 단위가 다릅니다.")
        with gr.Row():
            uid = gr.Dropdown(choices, value=choices[0], label="익명 사용자 ID", min_width=180)
            k = gr.Slider(1, 20, value=10, step=1, label="방법별 추천 수", min_width=180)
        button = gr.Button("네 방법 비교", variant="primary")
        cards = gr.HTML(label="방법별 상위 3개")
        with gr.Accordion("전체 추천 목록과 상세 점수", open=False):
            output = gr.Dataframe(label="방법별 추천 목록", interactive=False, wrap=True, min_width=0)
        button.click(compare, [uid, k], [cards, output], api_name="compare")
        app.load(compare, [uid, k], [cards, output])
        gr.Markdown("학습에서 이미 평가한 영화는 제외됩니다. "
                    "basis는 집단 평균의 대체나 빈 내용 프로필의 대체 여부를 표시합니다. "
                    "이 앱은 W01B/W02A의 데이터→추천 함수→callback 흐름에 내용 기반 비교를 추가합니다.")
    return app
