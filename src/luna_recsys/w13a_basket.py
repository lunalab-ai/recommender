"""Order-based item CF for W13A, with a shared browser and assignment contract."""
from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

RUNS = ("popularity", "cf-1", "cf-3", "cf-5")
SETTINGS = {"popularity": None, "cf-1": 1, "cf-3": 3, "cf-5": 5}
CSV_HEADER = ("run_id", "model", "min_coorders", "n_orders", "hits_at_5", "hit_at_5", "ndcg_at_5")
DEMO_SEEDS = ("22423",)


def read_retail(path):
    """Load the reviewed public training and validation snapshot."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != "uci-retail-w13a-v1":
        raise ValueError("수업 데이터 버전이 다릅니다.")
    return data


def order_matrix(baskets, size):
    matrix = np.zeros((len(baskets), size), dtype=np.int32)
    for i, basket in enumerate(baskets):
        matrix[i, basket] = 1
    return matrix


def fit(data, item_cosine):
    """Call the student's scalar formula on observed pairs only."""
    x = order_matrix(data["train"], len(data["catalog"]))
    shared = x.T @ x
    counts = np.diag(shared).copy()
    scores = np.zeros(shared.shape, dtype=np.float64)
    for i, j in zip(*np.triu(shared > 0, k=1).nonzero()):
        value = float(item_cosine(int(shared[i, j]), int(counts[i]), int(counts[j])))
        if not math.isfinite(value) or not 0 <= value <= 1.000001:
            raise ValueError("유사도는 0~1 사이의 유한한 숫자여야 합니다.")
        scores[i, j] = scores[j, i] = value
    # The same quantization is used by Python and the exported browser.
    return counts, shared, np.rint(scores * 1_000_000).astype(np.int64)


def rank(sku, counts, shared, similarities, basket, cutoff, is_candidate, limit=5):
    basket = tuple(dict.fromkeys(basket))
    scores = np.zeros(len(sku), dtype=np.int64)
    if cutoff is not None and basket:
        rows = list(basket)
        scores = np.where(shared[rows] >= cutoff, similarities[rows], 0).sum(axis=0)
    order = sorted(range(len(sku)), key=lambda j: (-int(scores[j]), -int(counts[j]), sku[j]))
    return [j for j in order if is_candidate(j, basket)][:limit]


def evaluate(data, model, is_candidate):
    sku = [item["sku"] for item in data["catalog"]]
    total = len(data["validation"])
    results = []
    for run, minimum in SETTINGS.items():
        hits, discounted = 0, 0.0
        for visible, hidden in data["validation"]:
            found = rank(sku, *model, visible, minimum, is_candidate)
            if hidden in found:
                hits += 1
                discounted += 1 / math.log2(found.index(hidden) + 2)
        results.append(dict(
            run_id=run, model="popularity" if minimum is None else "item_cf",
            min_coorders="" if minimum is None else minimum, n_orders=total,
            hits_at_5=hits, hit_at_5=f"{hits / total:.9f}", ndcg_at_5=f"{discounted / total:.9f}",
        ))
    return results


def preferred_runs(results):
    # Comparisons use the published nine-decimal metric values and the exact hit count.
    keys = [(row["hits_at_5"], float(row["ndcg_at_5"])) for row in results]
    best = max(keys)
    return [row["run_id"] for row, key in zip(results, keys) if key == best]


def browser_payload(data, model, metrics, chosen, student_code="", media=None):
    if chosen not in RUNS:
        raise ValueError("실험표에 있는 모델을 선택하세요.")
    counts, shared, scores = model
    neighbors = []
    for i in range(len(counts)):
        neighbors.append([[int(j), int(shared[i, j]), int(scores[i, j])] for j in np.flatnonzero(scores[i])])
    sku = [item["sku"] for item in data["catalog"]]
    return dict(
        catalog=data["catalog"], counts=counts.tolist(), neighbors=neighbors,
        version=data["version"], snapshot=data["source_sha256"], metrics=metrics,
        selected=chosen, student_code=student_code, media=media or {},
        featured=[sku.index(code) for code in DEMO_SEEDS if code in sku],
    )


def _safe_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", r"\u003c")


def render_shop(template, payload):
    if template.count("<!-- W13A_DATA -->") != 1:
        raise ValueError("기본 화면 템플릿 버전이 다릅니다.")
    return template.replace("<!-- W13A_DATA -->", _safe_json(payload))


def submission_manifest(payload):
    return dict(schema_version=1, assignment="2026-2-recommender-w13a-a1",
                data_version=payload["version"], dataset_sha256=payload["snapshot"],
                runs=list(RUNS), selected_run_id=payload["selected"])


def metrics_csv(rows):
    return ",".join(CSV_HEADER) + "\r\n" + "".join(
        ",".join(str(row[key]) for key in CSV_HEADER) + "\r\n" for row in rows
    )


def export_zip(template, payload):
    if not payload["student_code"]:
        raise ValueError("먼저 두 함수를 완성하고 실행하세요.")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("submission.json", json.dumps(submission_manifest(payload), ensure_ascii=False))
        zipped.writestr("student_model.py", payload["student_code"])
        zipped.writestr("metrics.csv", metrics_csv(payload["metrics"]))
        zipped.writestr("shop.html", render_shop(template, payload))
    return stream.getvalue()
