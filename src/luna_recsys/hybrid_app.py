"""W12A classroom app. Validation guides the slider and only the frozen weight sees test."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np

from .hybrid_lab import blend_scores, rank_catalog


@dataclass
class HybridDashboard:
    candidates: object
    curve: object
    selected: float
    blend_fn: object = blend_scores

    def current(self, weight):
        rows = rank_catalog(self.candidates, weight, blend_fn=self.blend_fn).head(3)
        return rows[["rank", "movie_id", "title", "score_cf", "score_mf", "cf_part", "mf_part", "combined"]]

    def validation_plot(self, weight):
        fig, ax = plt.subplots(figsize=(7, 2.9), layout="constrained")
        ax.plot(self.curve.weight_mf, self.curve.validation_rmse, color="#00568e", linewidth=2)
        ax.axvline(self.selected, color="#008076", linestyle=":", label="Frozen validation choice")
        ax.axvline(weight, color="#a75000", linestyle="--", label="Current preview")
        ax.set(xlabel="MF weight", ylabel="Validation RMSE", xlim=(0, 1))
        ax.legend(loc="best", fontsize=8)
        return fig

    def transitions_plot(self, weight):
        fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
        grid = np.linspace(0, 1, 51)
        leaders = self.candidates[self.candidates.movie_id.isin(
            set(self.current(0).movie_id) | set(self.current(1).movie_id)
        )]
        scores = np.column_stack([self.blend_fn(leaders, w) for w in grid])
        for (_, row), values in zip(leaders.iterrows(), scores):
            ax.plot(grid, values, label=str(row.title), linewidth=1.5)
        ax.axvline(weight, color="#322031", linestyle="--")
        ax.set(xlabel="MF weight", ylabel="Predicted rating", xlim=(0, 1))
        ax.legend(loc="best", fontsize=7)
        return fig

    def export(self, weight):
        with NamedTemporaryFile(prefix="w12a-top3-", suffix=".csv", delete=False) as file:
            path = Path(file.name)
        self.current(weight).to_csv(path, index=False, encoding="utf-8-sig")
        return str(path)


def build_dashboard(dashboard):
    import gradio as gr

    def refresh(weight):
        return dashboard.current(weight), dashboard.validation_plot(weight), dashboard.transitions_plot(weight)

    with gr.Blocks(title="W12A 추천 실험실", analytics_enabled=False) as demo:
        gr.Markdown("## 후보 영화의 점수와 순위\n"
                    "비중을 옮겨 두 점수의 기여와 순위 교차를 확인하세요. "
                    "검증 최적값은 점선으로 남고 시험 평점은 화면에 나오지 않습니다.")
        weight = gr.Slider(0, 1, value=dashboard.selected, step=.01, label="MF 비중")
        with gr.Row():
            curve = gr.Plot(label="검증 RMSE")
            lines = gr.Plot(label="후보 점수선")
        top = gr.Dataframe(label="현재 Top-3", interactive=False)
        export = gr.Button("현재 Top-3 CSV")
        file = gr.File(label="CSV")
        demo.load(refresh, weight, [top, curve, lines], api_name="initial")
        weight.release(refresh, weight, [top, curve, lines], api_name="preview")
        export.click(dashboard.export, weight, file, api_name="export")
    return demo
