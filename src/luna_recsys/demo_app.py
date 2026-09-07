"""Gradio prototype for the first MovieLens recommendation application."""

from __future__ import annotations

import pandas as pd

from .baselines import mean_rating_recommendations
from .data import MOVIELENS_100K_GENRES, MovieLens100K


def recommend_for_app(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    genre_label: str,
    min_ratings: int,
    top_n: int,
) -> pd.DataFrame:
    """Adapt UI values to the baseline and return student-friendly column labels."""
    genre = None if genre_label == "전체" else genre_label
    result = mean_rating_recommendations(
        ratings,
        movies,
        genre=genre,
        min_ratings=int(min_ratings),
        top_n=int(top_n),
    )
    return result.rename(
        columns={
            "movie_id": "영화 ID",
            "title": "영화 제목",
            "mean_rating": "평균 평점",
            "rating_count": "평점 수",
        }
    )


def build_movie_recommender_app(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame | None = None,
    *,
    data_mode: str = "데이터 모드 미지정",
):
    """Build the W01B Gradio Blocks app without launching a server."""
    if users is not None:
        return build_baseline_lab(MovieLens100K(users, movies, ratings), data_mode=data_mode)
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError('Install the app extra with: pip install ".[apps]"') from exc

    def callback(genre_label: str, min_ratings: int, top_n: int) -> pd.DataFrame:
        return recommend_for_app(ratings, movies, genre_label, min_ratings, top_n)

    with gr.Blocks(title="LUNA 영화 추천 실험실", analytics_enabled=False) as demo:
        gr.Markdown(
            "# 🎬 LUNA 영화 추천 실험실\n"
            "장르와 조건을 선택해 MovieLens 100K의 평균 평점 기반 추천을 관찰하세요."
        )
        with gr.Row():
            genre = gr.Dropdown(
                choices=["전체", *MOVIELENS_100K_GENRES[1:]],
                value="전체",
                label="장르",
            )
            min_ratings = gr.Slider(
                minimum=1,
                maximum=200,
                value=50,
                step=1,
                label="최소 평점 수",
            )
            top_n = gr.Slider(
                minimum=1,
                maximum=20,
                value=10,
                step=1,
                label="추천 개수",
            )
        run_button = gr.Button("추천 영화 보기", variant="primary")
        output = gr.Dataframe(
            headers=["영화 ID", "영화 제목", "평균 평점", "평점 수"],
            datatype=["number", "str", "number", "number"],
            interactive=False,
            label="추천 결과",
        )
        gr.Markdown(
            "이 결과는 모든 사용자에게 같은 집계 기준을 적용하는 첫 baseline입니다. "
            "개인 취향을 반영하지 않습니다."
        )
        run_button.click(callback, inputs=[genre, min_ratings, top_n], outputs=output)
    return demo


def build_baseline_lab(dataset: MovieLens100K, *, data_mode: str = "데이터 모드 미지정"):
    """Build W02A's cumulative responsive app; launch remains the caller's choice.

    Example: ``build_baseline_lab(prepared.data, data_mode=prepared.description).launch()``.
    W01B's two-argument builder and callback retain their behavior.
    """
    import html

    import gradio as gr

    from .baselines import baseline_recommendations
    from .evaluation import evaluate_means, split_ratings

    split = split_ratings(dataset.ratings)
    methods = [("평점 수 기반 인기", "count"), ("평균 평점", "mean"), ("집단별 평균 평점", "group")]
    group_choices = []
    for column, label in [("sex", "교재의 성별 그룹"), ("occupation", "직업 그룹")]:
        for value in sorted(dataset.users[column].astype(str).unique()):
            group_choices.append((f"{label}: {value}", f"{column}:{value}"))

    def recommend(method, group, genre, minimum, top_n):
        column, value = group.split(":", 1)
        result = baseline_recommendations(
            dataset.ratings,
            dataset.movies,
            users=dataset.users,
            method=method,
            group_col=column,
            group_value=value,
            genre=None if genre == "전체" else genre,
            min_ratings=int(minimum),
            top_n=int(top_n),
        )
        # Escape dataset titles: HTML is presentation, not trusted dataset content.
        cards = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:12px">'
        for rank, row in enumerate(result.itertuples(), start=1):
            cards += (
                f'<article style="border:1px solid #b7c7d6;border-radius:14px;padding:14px;overflow-wrap:anywhere">'
                f"<strong>{rank}. {html.escape(str(row.title))}</strong><p>평균 {row.mean_rating:.3f} · 평점 {row.rating_count}개</p>"
                f"<small>{html.escape(row.basis)}</small></article>"
            )
        cards += "</div>"
        if result.empty:
            cards = (
                "<p>조건을 만족하는 영화가 없습니다. 최소 평점 수를 낮추거나 장르를 바꾸세요.</p>"
            )
        table = result.rename(
            columns={
                "movie_id": "영화 ID",
                "title": "영화 제목",
                "mean_rating": "평균 평점",
                "rating_count": "평점 수",
                "basis": "사용한 기준",
            }
        )
        return cards, table

    def evaluate(column, minimum):
        result = evaluate_means(
            split, dataset.users, group_col=column, min_group_ratings=int(minimum)
        )
        summary = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:12px">'
        for row in result.itertuples():
            summary += (
                f'<article style="border:1px solid #b7c7d6;border-radius:14px;padding:14px">'
                f"<strong>{row.method}</strong><p>RMSE {row.rmse:.4f}</p>"
                f"<small>평가 {row.test_count:,}행 · 집단 평균 사용 {row.group_used:,}건</small></article>"
            )
        summary += "</div>"
        return summary, result.rename(
            columns={
                "method": "평점 예측 방법",
                "rmse": "RMSE",
                "test_count": "평가 행 수",
                "group_used": "집단 평균 사용",
                "movie_used": "영화 평균 사용",
                "global_used": "전체 평균 사용",
            }
        ).round(4)

    with gr.Blocks(
        title="LUNA 영화 추천 실험실 · W02A", analytics_enabled=False, fill_width=True
    ) as demo:
        gr.Markdown(
            "# 🎬 LUNA 영화 추천 실험실\n## 기본적인 추천 방법 (1)\n조건을 바꾸기 전에 결과를 예측하고, 추천 근거와 표본 수를 비교해 보세요."
        )
        gr.Markdown(f"**현재 데이터:** {data_mode}")
        with gr.Tab("추천"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    algorithm = gr.Dropdown(
                        methods, value="mean", label="추천 알고리즘", min_width=0
                    )
                    group = gr.Dropdown(
                        group_choices,
                        value=group_choices[0][1],
                        label="사용자 집단 (집단별 추천에서 사용)",
                        min_width=0,
                    )
                    genre = gr.Dropdown(
                        ["전체", *MOVIELENS_100K_GENRES[1:]],
                        value="전체",
                        label="장르",
                        min_width=0,
                    )
                    minimum = gr.Slider(1, 200, value=5, step=1, label="최소 평점 수", min_width=0)
                    top_n = gr.Slider(1, 20, value=5, step=1, label="추천 개수", min_width=0)
                    button = gr.Button("추천 영화 보기", variant="primary", min_width=0)
                with gr.Column(scale=2, min_width=260):
                    cards = gr.HTML(label="모바일 추천 카드")
                    table = gr.Dataframe(
                        label="추천 결과 · 좁은 화면에서는 표 내부를 가로로 이동",
                        interactive=False,
                        wrap=True,
                        min_width=0,
                    )
            gr.Markdown(
                "평점 수는 노출·참여량의 대리값이며 좋아한다는 뜻은 아닙니다. 집단 평균도 개인 취향을 보장하지 않습니다.\n집단에서 조건을 만족하는 영화가 없으면 전체 사용자 평균 목록으로 전환하며 카드에 표시합니다."
            )
            button.click(
                recommend,
                [algorithm, group, genre, minimum, top_n],
                [cards, table],
                api_name="recommend",
            )
            demo.load(recommend, [algorithm, group, genre, minimum, top_n], [cards, table])
        with gr.Tab("평가"):
            gr.Markdown(
                f"### 같은 평가 자료로 비교\n학습 {len(split.train):,}행 / 평가 {len(split.test):,}행 · seed 42 · {split.method}\n평가할 평점은 평균 계산에 사용하지 않습니다. **평점 수 순위는 RMSE 비교 대상이 아닙니다.**"
            )
            with gr.Row():
                eval_group = gr.Dropdown(
                    [("성별", "sex"), ("직업", "occupation")],
                    value="sex",
                    label="평가할 집단 속성",
                    min_width=0,
                )
                eval_min = gr.Slider(
                    1, 50, value=1, step=1, label="집단×영화 평균의 최소 평점 수", min_width=0
                )
            evaluate_button = gr.Button("동일한 분할로 평가", variant="primary", min_width=0)
            evaluation_cards = gr.HTML(label="평가 요약 카드")
            scores = gr.Dataframe(
                label="평점 예측 오차와 실제 사용 기준", interactive=False, min_width=0, wrap=True
            )
            evaluate_button.click(
                evaluate, [eval_group, eval_min], [evaluation_cards, scores], api_name="evaluate"
            )
            gr.Markdown(
                "집단×영화 평균 → 학습 영화 평균 → 학습 전체 평균 순서로 대체합니다.\nRMSE는 작을수록 좋지만, 이 숫자만으로 추천 목록의 만족도·다양성·순위 품질을 판단할 수 없습니다."
            )
        with gr.Tab("기록"):
            gr.Markdown(
                "### 지금 사용할 수 있는 기능\n- W01B: 데이터 로딩, 장르·최소 평점 수·Top-N, 평균 평점 추천\n- W02A: 평점 수 비교, 집단 선택, 표본 부족 안내, 공통 holdout 평가, 모바일 카드\n### 이후 수업 계획\n내용 기반, 협업 필터링, 행렬 분해 등을 순서대로 추가할 예정입니다. 아직 구현된 기능은 아닙니다.\n\n이 앱의 공유 링크는 실행 중인 Colab/Gradio 런타임에 연결됩니다."
            )
    return demo
