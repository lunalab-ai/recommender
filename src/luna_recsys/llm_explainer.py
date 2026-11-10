"""W11B: real local LLM requests and trained embedding FM, with explicit callbacks."""
from __future__ import annotations
from pathlib import Path
import tempfile
import pandas as pd
from .llm_recommendation import run_prompt, build_prompt, validate_recommendations, ranking_overlap
from .embedding_fm import predict_embedding_fm


def export_displayed_table(table: pd.DataFrame) -> str:
    """Export the last displayed result, without another request or prediction."""
    if table is None:
        raise ValueError('먼저 결과표를 계산하세요.')
    with tempfile.NamedTemporaryFile(prefix='w11b-result-', suffix='.csv', delete=False) as handle:
        path = Path(handle.name)
    table.copy(deep=True).to_csv(path, index=False, encoding='utf-8-sig')
    return str(path)


def prompt_view(case, truth, seen, client, method='CF', shots=3, review_chars=140,
                *, prompt_fn=build_prompt, validator_fn=validate_recommendations,
                metric_fn=ranking_overlap):
    result = run_prompt(case, truth, seen, client, method, int(shots), int(review_chars),
                        prompt_fn=prompt_fn, validator_fn=validator_fn, metric_fn=metric_fn)
    message = (f"실제 {method} · 응답검사={result['validation']['status']} · "
               f"유효 {len(result['validation']['valid'])}/{case['k']}개 · "
               f"P@{case['k']}={result['metrics']['precision']:.3f} · "
               f"R={result['metrics']['recall']:.3f} · 입력 {result['prompt_chars']}문자 · "
               '이 점수는 보류제품 포함률이며 만족도나 확률이 아닙니다.')
    # Never return raw review text/messages to a shared interface.
    return message, result['table'].copy(), result['table'].copy()


def fm_view(model, pairs, user):
    chosen = pairs.loc[pairs.user_id == user, ['user_id', 'item_id']].drop_duplicates()
    if chosen.empty:
        raise ValueError('이 사용자의 평가 쌍이 없습니다.')
    table = predict_embedding_fm(model, chosen)
    table = table.sort_values('prediction', ascending=False, kind='stable').reset_index(drop=True)
    table.insert(0, 'rank', range(1, len(table)+1))
    message = (f"FM 실제 학습 · epoch={model['state']['epoch']} · "
               f"갱신={model['state']['updates']}회 · 임베딩={model['use_embeddings']} · "
               f"미관측제품 {int(table.unknown_item.sum())}건. "
               '현재 검증 쌍의 점수 순서이며 전체 후보에 대한 추천 성능은 아닙니다.')
    return message, table.copy(), table.copy()


def build_llm_explainer(*, client=None, case=None, truth=None, seen=None,
                        models=None, pairs=None, prompt_fn=build_prompt,
                        validator_fn=validate_recommendations, metric_fn=ranking_overlap,
                        include_legacy=True):
    """Capture the actual notebook functions/models. No training or LLM call at build."""
    import gradio as gr
    apps, names = [], []
    if include_legacy:
        from .transformer_explainer import build_transformer_explainer
        apps.append(build_transformer_explainer()); names.append('기존 FM·MF·MLP·AE·attention')
    if client is not None and case is not None:
        with gr.Blocks(analytics_enabled=False) as prompting:
            gr.Markdown('# 실제 LLM 추천\nCF는 다른 사용자 이력, CB는 관측 리뷰를 읽습니다. 원문 리뷰와 실사용자 정보는 공유하지 않습니다.')
            method = gr.Radio(['CF', 'CB'], value='CF', label='입력 방법')
            shots = gr.Slider(0, 4, value=3, step=1, label='CF 예시 사용자 수')
            chars = gr.Slider(40, 240, value=140, step=20, label='CB 제품별 리뷰 문자 수')
            message = gr.Textbox(label='실제 호출·검사·평가', lines=3)
            table = gr.Dataframe(label='가장 최근 계산한 추천표', interactive=False)
            state = gr.State(None)
            def calculate(m, s, c):
                return prompt_view(case, truth, seen, client, m, s, c,
                    prompt_fn=prompt_fn, validator_fn=validator_fn, metric_fn=metric_fn)
            gr.Button('실제 모델에 요청', variant='primary').click(calculate, [method, shots, chars], [message, table, state])
            download = gr.File(label='표와 같은 CSV', interactive=False)
            gr.Button('마지막 표 CSV 저장').click(export_displayed_table, state, download)
            gr.Markdown('설정만 바꾸면 표는 갱신되지 않습니다. 요청 버튼으로 다시 계산하세요. CSV는 마지막 표를 저장하며 모델을 다시 호출하지 않습니다.')
        apps.append(prompting); names.append('9.2 CF·CB 프롬프팅')
    if models and pairs is not None:
        with gr.Blocks(analytics_enabled=False) as embedding:
            gr.Markdown('# 훈련 리뷰 임베딩 + FM\nLLM 가중치는 고정, FM의 w와 V만 학습했습니다. 0–1 확률로 읽지 마세요.')
            mode = gr.Radio(list(models), value=next(iter(models)), label='고정 분할 비교')
            users = sorted(pairs.user_id.unique())
            user = gr.Dropdown(users, value=users[0], label='가명 사용자')
            message = gr.Textbox(label='학습 상태와 지원 범위', lines=3)
            table = gr.Dataframe(label='현재 검증 쌍의 예측 분해', interactive=False)
            state = gr.State(None)
            def calculate_fm(m, u):
                return fm_view(models[m], pairs, u)
            gr.Button('학습된 FM으로 계산', variant='primary').click(calculate_fm, [mode, user], [message, table, state])
            download = gr.File(label='표와 같은 CSV', interactive=False)
            gr.Button('마지막 표 CSV 저장').click(export_displayed_table, state, download)
        apps.append(embedding); names.append('9.3 임베딩·FM')
    if not apps:
        raise ValueError('Provide a real prompt case or fitted FM model.')
    return gr.TabbedInterface(apps, names, title='W11B · LLM 추천의 원리', analytics_enabled=False)
