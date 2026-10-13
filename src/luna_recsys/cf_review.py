"""W07B: CF의 관측·이웃·기여도·평가를 연결하는 작은 CPU 실험."""
from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd

from .collaborative import toy_ratings
from .cf_synthesis import EvidenceCF, evaluate_cf


def cf_review_data() -> dict:
    """새 합성 표를 반환한다. train15행 / validation3행, 카탈로그5편.

    W03A toy_ratings의 (2,2),(3,1),(4,4)를 검증용으로 숨긴다. 그 정답은
    fit에 전달하지 않는다. 반복 설정 탐색 화면이므로 test 성능이라 부르지
    않는다. 마지막 시험 평가는 별도 test가 필요하다. 파일/네트워크 없음.
    """
    ratings = toy_ratings()
    hidden = {(2, 2), (3, 1), (4, 4)}
    mask = np.array([(u, i) in hidden for u, i in
                     ratings[['user_id', 'movie_id']].itertuples(index=False, name=None)])
    return dict(train=ratings.loc[~mask].reset_index(drop=True),
                validation=ratings.loc[mask].reset_index(drop=True),
                movies=pd.DataFrame({'movie_id': range(1, 6), 'title': list('ABCDE')}))


def cf_review_experiment(axis: str = 'user', k: int = 2, min_common: int = 1,
                         min_neighbors: int = 1, centered: bool = False,
                         clip: bool = True, *, user_id: int = 1,
                         movie_id: int = 4) -> dict:
    """설정 → 설명 가능한 한 예측과 동일 검증3건의 지표를 반환.

    axis='item',centered=True는 오류로 거부한다. 편의를 위해 몰래 설정을
    바꾸지 않는다. 반환 model/details/contributions/candidates/validation/
    metrics/train/movies는 표 또는 dict다. k=0은 전체 유효 이웃이다.
    학생 문항의 정답 데이터와는 다른 합성 시연 자료다.
    """
    data = cf_review_data()
    model = EvidenceCF(axis=axis, k=k, min_common=min_common,
                       min_neighbors=min_neighbors, centered=centered, clip=clip).fit(data['train'])
    if user_id not in model.seen_ or movie_id not in set(data['movies'].movie_id):
        raise ValueError('Choose a known demo user and movie')
    if movie_id in model.seen_[user_id]:
        raise ValueError('The demonstration target must be unobserved in train')
    details = model.predict_details(pd.DataFrame({'user_id': [user_id], 'movie_id': [movie_id]}))
    contributions = model.explain(user_id, movie_id)
    candidates = data['movies'].loc[~data['movies'].movie_id.isin(model.seen_[user_id])].reset_index(drop=True)
    scores = model.predict_details(candidates.assign(user_id=user_id))
    candidates = pd.concat([candidates, scores], axis=1).sort_values(
        ['prediction', 'movie_id'], ascending=[False, True]).reset_index(drop=True)
    validation = data['validation'].copy()
    predictions = model.predict_details(validation[['user_id', 'movie_id']])
    validation = pd.concat([validation, predictions], axis=1)
    validation['error'] = validation.rating - validation.prediction
    metrics = evaluate_cf(model, data['validation'], data['movies'], top_n=2)
    return dict(**data, model=model, details=details, contributions=contributions,
                candidates=candidates, validation_predictions=validation, metrics=metrics,
                settings=dict(axis=axis, k=k, min_common=min_common,
                              min_neighbors=min_neighbors, centered=centered, clip=clip),
                user_id=user_id, movie_id=movie_id)


def cf_review_view(axis='user', k=2, min_common=1, min_neighbors=1,
                   centered=False, clip=True, report_fn: Callable[[dict], str] | None = None):
    """Gradio callback. 설명/이웃표/후보표/검증예측표/지표표/그림을 반환.

    부적합 설정은 명시 오류. 사용자가 수정한 report_fn(result)의 문장은
    화면에 추가된다. matplotlib figure는 호출자가 닫을 수 있다.
    """
    import matplotlib.pyplot as plt
    result = cf_review_experiment(axis, int(k), int(min_common), int(min_neighbors), bool(centered), bool(clip))
    row = result['details'].iloc[0]
    baseline = float(result['model'].user_means_.loc[1]) if centered else 0.0
    explanation = (
        f"### u1의 미관측 영화 D\n"
        f"축 **{axis}**, k={int(k)} (0=전체), 공통수≥{int(min_common)}, 최소 이웃={int(min_neighbors)}.\n\n"
        f"확보 이웃 **{int(row.eligible_neighbors)}**, 실제 CF 사용 **{int(row.n_contributors)}**, "
        f"근거 **{row.basis}**. 원예측 **{row.raw_prediction:.6f}** → 최종 **{row.prediction:.6f}**.\n\n"
        "표의 weight 합은 CF가 사용되면1입니다. contribution 합에 대상 기준점 "
        f"**{baseline:.4f}**을 더하면 원예측이 됩니다. 대체 분기에서는 이웃표가 비어 있습니다.\n\n"
        "train15건으로만 평균과 유사도를 만들었습니다. 아래 오차는 반복 탐색용 **validation3건**입니다. "
        "순위 지표는 관련 평점≥4인 검증 사용자1명(u4), K=2이며 관련 항목이 없는2명은 제외합니다. "
        "카탈로그 분모는5편입니다. 작은 시연이므로 모델의 일반적인 우열을 결론 내리지 마세요."
    )
    if report_fn is not None:
        explanation += '\n\n' + str(report_fn(result))
    table = result['contributions'].round(6)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if not table.empty:
        values = table.contribution.to_numpy()
        ax.bar(table.neighbor_id.astype(str), values,
               color=['#2563eb' if v >= 0 else '#dc2626' for v in values])
        ax.axhline(0, color='#64748b', linewidth=.7)
        ax.set(xlabel='Neighbor ID (user or item)', ylabel='Contribution')
    else:
        ax.text(.5, .5, f'Fallback: {row.basis}', ha='center', transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_title(f'CF contributions; baseline={baseline:.2f}')
    fig.tight_layout()
    metrics = pd.DataFrame([(key, float(value)) for key, value in result['metrics'].items()], columns=['metric', 'value'])
    return (explanation, table, result['candidates'].round(6),
            result['validation_predictions'].round(6), metrics.round(6), fig)


def build_cf_review_app(report_fn: Callable[[dict], str] | None = None):
    """CPU Gradio 앱을 생성만 한다. .launch는 별도 호출.

    report_fn은 학생이 고치는 설명 함수다. 공개 정답이나 시험풀이를 포함하지
    않는다. Colab 임시 share 링크보다 고정 버전 노트북을 진입점으로 사용한다.
    """
    import gradio as gr
    data = cf_review_data()
    with gr.Blocks(title='W07B · CF 이해 실험실') as app:
        gr.Markdown('# CF 이해 실험실\n한 조건만 바꾸기 전에 예상하고, 이웃표에서 이유를 찾아 한 문장으로 설명하세요.')
        with gr.Row():
            axis = gr.Dropdown(['user', 'item'], value='user', label='비교 축')
            k = gr.Slider(0, 4, value=2, step=1, label='최대 이웃 k (0=전체)')
        with gr.Row():
            common = gr.Slider(1, 4, value=1, step=1, label='최소 공통 평가 수')
            minimum = gr.Slider(1, 4, value=1, step=1, label='최소 이웃 수')
        with gr.Row():
            centered = gr.Checkbox(False, label='사용자 평균보정 (user에서만 선택)')
            clip = gr.Checkbox(True, label='최종 평점 1–5 제한')
        gr.Markdown('item으로 바꿀 때 평균보정을 해제하세요. 두 조건의 조합이 맞지 않으면 이유를 표시합니다.')
        go = gr.Button('예측과 근거 다시 계산', variant='primary')
        note = gr.Markdown()
        terms = gr.Dataframe(interactive=False, label='선택 이웃과 기여도')
        candidates = gr.Dataframe(interactive=False, label='u1 미관람 영화 순위')
        validation = gr.Dataframe(interactive=False, label='검증3건: 실제·예측·오차')
        metrics = gr.Dataframe(interactive=False, label='같은 검증 조건의 지표')
        plot = gr.Plot()
        def callback(axis, k, common, minimum, centered, clip):
            try:
                return cf_review_view(axis, k, common, minimum, centered, clip, report_fn)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
        inputs = [axis, k, common, minimum, centered, clip]
        outputs = [note, terms, candidates, validation, metrics, plot]
        go.click(callback, inputs, outputs)
        app.load(callback, inputs, outputs)
        with gr.Accordion('관측 표와 검증 분할을 펼쳐 보기', open=False):
            gr.Dataframe(data['train'], interactive=False, label='train15건')
            gr.Dataframe(data['validation'], interactive=False, label='validation3건: fit에 넣지 않음')
            gr.Markdown('합성 데이터입니다. A–E는 영화ID1–5입니다. 관측이 없는 칸은0점이 아닙니다. 이 앱에서 test를 탐색하지 않습니다.')
    return app
