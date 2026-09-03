"""Gradio prototype for the first MovieLens recommendation application."""

from __future__ import annotations

import pandas as pd

from .baselines import mean_rating_recommendations
from .data import MOVIELENS_100K_GENRES


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


def build_movie_recommender_app(ratings: pd.DataFrame, movies: pd.DataFrame):
    """Build the W01B Gradio Blocks app without launching a server."""
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
