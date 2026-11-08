"""W11A cumulative app: editable attention and real fitted next-item inference."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Any
import tempfile
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .sequential_recommenders import scaled_attention, rollout, select_candidates, ranking_metrics


def attention_view(query: int = 0, causal: bool = False, *, attention_fn: Callable = scaled_attention) -> tuple:
    """Chosen query0..2 / causal mask -> message, weight table, output table, heatmap.

    Fixed given Q/K/V, not learned weights or a MovieLens result. A supplied
    notebook attention_fn determines the actual table, heatmap and CSV.
    """
    query = int(query)
    if query not in range(3):
        raise ValueError('query must be 0,1,2.')
    q = np.array([[1., 0.], [0., 1.], [1., 1.]])
    v = np.array([[2., 0.], [0., 2.], [2., 2.]])
    mask = np.tril(np.ones((3, 3), bool)) if causal else np.ones((3, 3), bool)
    result = attention_fn(q, q, v, allowed=mask)
    weights = pd.DataFrame({'key': ['A', 'B', 'C'], 'scaled_score': result['scores'][query],
                            'allowed': mask[query], 'weight': result['weights'][query]})
    output = pd.DataFrame({'dimension': ['v1', 'v2'], 'weighted_output': result['output'][query]})
    fig = Figure(figsize=(5.8, 3.6)); ax = fig.subplots()
    ax.imshow(result['weights'], vmin=0, vmax=1, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{result['weights'][i,j]:.3f}", ha='center', va='center')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=['A','B','C'], yticklabels=['A','B','C'],
           xlabel='Key item', ylabel='Query item', title='Given Q/K/V: attention weights')
    fig.tight_layout()
    message = ('고정 Q/K/V 설명 계산입니다. 학습 결과가 아닙니다. '
               f'선택 query={query}, 허용 가중치 합={result["weights"][query].sum():.6f}. '
               '관측된 과거 전체로 다음 한 개를 분류할 때는 과거 내 양방향 attention을 쓸 수 있습니다.')
    return message, weights, output, fig


def export_attention(query: int = 0, causal: bool = False, *, attention_fn: Callable = scaled_attention) -> str:
    """Write precisely the currently displayed attention weight table to a temporary CSV."""
    with tempfile.NamedTemporaryFile(prefix='w11a-attention-', suffix='.csv', delete=False) as f:
        path = Path(f.name)
    attention_view(query, causal, attention_fn=attention_fn)[1].to_csv(path, index=False, encoding='utf-8-sig')
    return str(path)


def sequence_view(data: Any, fitted: Any, user: int, *, candidate_fn: Callable = select_candidates) -> tuple:
    """Actual model + chosen test history -> decoded display table and held-out set metrics.

    Only display decodes IDs. No fit, mutation or future-label input to rollout.
    Metrics assess seven generated items against the user's last ten recorded items.
    """
    user = int(user)
    if user not in data.histories:
        raise ValueError('Choose a registered test user.')
    generated = rollout(lambda x: fitted.model(x, training=False).numpy(), data.histories[user],
                        maxlen=data.maxlen, n=7, candidate_fn=candidate_fn)
    truth = set(data.targets[user])
    table = pd.DataFrame([{'rank': j+1, 'internal_id': i, 'movie_id': data.decode[i],
                            'title': data.titles.get(data.decode[i], ''), 'heldout_hit': i in truth}
                          for j, i in enumerate(generated)])
    metric = ranking_metrics(generated, truth)
    message = (f'자료 모드={data.mode}; 사용자={user}; 학습 모수={fitted.model.count_params()}; '
               f'실제 갱신={fitted.updates}. P@7={metric["precision"]:.4f}, '
               f'R@10={metric["recall"]:.4f}, F1={metric["f1"]:.4f}. '
               '순서 일치율이 아닌 보류 집합 포함률이며, 평점 기록 시각의 순서를 사용합니다.')
    return message, table


def export_sequence(data: Any, fitted: Any, user: int, *, candidate_fn: Callable = select_candidates) -> str:
    """Save the same generated recommendation table, including notebook callback changes."""
    table = sequence_view(data, fitted, user, candidate_fn=candidate_fn)[1]
    with tempfile.NamedTemporaryFile(prefix='w11a-recommendation-', suffix='.csv', delete=False) as f:
        path = Path(f.name)
    table.to_csv(path, index=False, encoding='utf-8-sig')
    return str(path)


def build_transformer_explainer(*, attention_fn: Callable = scaled_attention,
                                data: Any = None, fitted: Any = None,
                                candidate_fn: Callable = select_candidates):
    """Return cumulative Gradio app; launch remains the caller's explicit action.

    Notebook functions are captured explicitly by UI and download callbacks.
    Prior FM/MF/MLP/AE tabs remain available. No extra model is trained here.
    """
    import gradio as gr
    from .autoencoder_explainer import build_autoencoder_explainer
    with gr.Blocks(analytics_enabled=False) as attention_app:
        gr.Markdown('# Q·K·V의 계산을 바꿔 보기\n주어진 작은 수치입니다. 가중치가 어떻게 출력에 반영되는지 확인하세요.')
        query = gr.Slider(0, 2, value=0, step=1, label='어느 query 행을 읽을까요?')
        causal = gr.Checkbox(False, label='각 query보다 뒤인 위치 가리기')
        initial = attention_view(attention_fn=attention_fn)
        message = gr.Textbox(initial[0], label='계산 의미', lines=3)
        weights = gr.Dataframe(initial[1], label='점수와 가중치', interactive=False)
        output = gr.Dataframe(initial[2], label='V 가중합', interactive=False)
        figure = gr.Plot(initial[3])
        def calculate(q, c):
            return attention_view(q, c, attention_fn=attention_fn)
        gr.Button('계산하고 예상과 비교', variant='primary').click(calculate, [query, causal], [message, weights, output, figure])
        file = gr.File(label='현재 가중치표 CSV', interactive=False)
        def csv(q, c):
            return export_attention(q, c, attention_fn=attention_fn)
        gr.Button('현재 표 CSV 만들기').click(csv, [query, causal], file)
    apps, names = [build_autoencoder_explainer(), attention_app], ['FM·MF·MLP·AE 복습', 'attention 직접 계산']
    if data is not None and fitted is not None:
        with gr.Blocks(analytics_enabled=False) as sequence_app:
            gr.Markdown('# 실제 학습 모형으로 다음 아이템 추천\n정답을 입력하지 않고 모형이 생성한 내부 ID를 반복해서 추가합니다.')
            user = gr.Dropdown(data.users['test'], value=data.users['test'][0], label='시험 사용자')
            initial = sequence_view(data, fitted, data.users['test'][0], candidate_fn=candidate_fn)
            message = gr.Textbox(initial[0], label='자료·학습·평가', lines=3)
            table = gr.Dataframe(initial[1], label='자기 예측 7회와 보류 정답', interactive=False)
            def recommend(u):
                return sequence_view(data, fitted, u, candidate_fn=candidate_fn)
            gr.Button('추천 계산', variant='primary').click(recommend, user, [message, table])
            csv_file = gr.File(label='현재 추천표 CSV', interactive=False)
            def download(u):
                return export_sequence(data, fitted, u, candidate_fn=candidate_fn)
            gr.Button('현재 추천 CSV 만들기').click(download, user, csv_file)
            gr.Markdown('노트북 candidate_fn을 수정한 뒤 앱을 다시 조립하면 추천과 CSV에 같은 규칙을 적용합니다. 공유 주소는 런타임 종료 시 만료됩니다.')
        apps.append(sequence_app); names.append('실제 순차 추천')
    return gr.TabbedInterface(apps, names, title='W11A · Transformer 추천', analytics_enabled=False)
