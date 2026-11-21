"""Order-based W13A experiment protocol and app export helpers."""
from __future__ import annotations

import csv
import io
import itertools
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCALE = 1_000_000
TOP_K = 5
RANKS = (8, 16, 32)
SUPPORTS = (1, 3, 5)
CRITERIA = ("hit_at_5", "ndcg_at_5")
CSV_FIELDS = (
    "run_id", "model", "support", "rank", "latent_weight", "orders",
    "hits", "hit_at_5", "ndcg_at_5", "one_item_hit_at_5", "multi_item_hit_at_5",
)
ASSIGNMENT = "2026-2-recommender-w13a-a1"
DESIGN_MARKER = "<!-- W13A_DESIGN_DATA -->"


@dataclass(frozen=True)
class Trial:
    kind: str
    support: int = 0
    rank: int = 0
    latent_weight: int = 0

    def __post_init__(self):
        if any(type(value) is not int for value in (self.support, self.rank, self.latent_weight)):
            raise ValueError("설정값은 정수여야 합니다.")
        if self.kind == "popularity":
            valid = (self.support, self.rank, self.latent_weight) == (0, 0, 0)
        elif self.kind == "cf":
            valid = self.support in SUPPORTS and self.rank == self.latent_weight == 0
        elif self.kind == "svd":
            valid = self.support == 0 and self.rank in RANKS and self.latent_weight == 100
        elif self.kind == "hybrid":
            valid = self.support in SUPPORTS and self.rank in RANKS and 0 < self.latent_weight < 100
        else:
            valid = False
        if not valid:
            raise ValueError("모델 종류와 설정값을 확인하세요.")

    @property
    def id(self):
        return f"{self.kind}-s{self.support}-k{self.rank}-w{self.latent_weight}"


POPULAR = Trial("popularity")
CF_REFERENCE = Trial("cf", support=3)


def order_matrix(orders, size):
    matrix = np.zeros((len(orders), size), dtype=np.int32)
    for i, order in enumerate(orders):
        matrix[i, order] = 1
    return matrix


def load_data(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != "uci-retail-w13a-v1":
        raise ValueError("수업용 주문 스냅샷의 버전이 다릅니다.")
    return data


def coorders(data):
    orders = order_matrix(data["train"], len(data["catalog"]))
    shared = orders.T @ orders
    return shared, shared.diagonal().copy()


def quantized_cosine(value, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("코사인 행렬의 모양 또는 숫자가 올바르지 않습니다.")
    if np.min(result) < -1e-12 or np.max(result) > 1.000001:
        raise ValueError("코사인 점수는 0~1로 계산합니다.")
    result = np.floor(np.clip(result, 0, 1) * SCALE + .5).astype(np.int64)
    np.fill_diagonal(result, 0)
    return result


def latent_directions(shared):
    """Top right singular directions for order matrix X from X.T @ X."""
    _, directions = np.linalg.eigh(shared.astype(np.float64))
    return directions[:, ::-1]


def quantized_kernel(value, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)) or np.max(np.abs(result)) > 1.000001:
        raise ValueError("장바구니 투영에서 나온 상품×상품 행렬을 확인하세요.")
    return np.floor(np.clip(result, -1, 1) * SCALE + .5).astype(np.int64)


def scale_rows(masked_scores):
    """Map the maximum positive candidate score in each row to SCALE."""
    values = np.maximum(np.asarray(masked_scores, dtype=np.int64), 0)
    scale = np.maximum(values.max(axis=1, keepdims=True), 1)
    return (values * SCALE + scale // 2) // scale


def raw_components(visible, trial, shared, similarity, kernels):
    if trial.support:
        cf = visible @ np.where(shared >= trial.support, similarity, 0)
    else:
        cf = np.zeros_like(visible, dtype=np.int64)
    if trial.rank:
        latent = visible @ kernels[trial.rank]
    else:
        latent = np.zeros_like(cf)
    return cf, latent


def top_scores(scores, visible, counts, sku):
    masked = np.where(visible, -1, scores)
    return np.lexsort(
        (np.broadcast_to(sku, masked.shape), np.broadcast_to(-counts, masked.shape), -masked),
        axis=1,
    )[:, :TOP_K]


def ordered_parts(visible, trial, shared, counts, similarity, kernels, blender, sku):
    cf, latent = raw_components(visible, trial, shared, similarity, kernels)
    if trial == POPULAR:
        left, right = np.zeros_like(cf), np.zeros_like(latent)
        score = np.broadcast_to(counts, left.shape)
    else:
        left, right = (np.asarray(part, dtype=np.int64) for part in
                       blender(cf, latent, visible.astype(bool), trial.latent_weight))
        if left.shape != visible.shape or right.shape != visible.shape:
            raise ValueError("결합 결과는 장바구니 수×상품 수 배열 두 개여야 합니다.")
        if np.any(left < 0) or np.any(right < 0) or np.any(left + right > 100 * SCALE):
            raise ValueError("결합 점수 범위를 확인하세요.")
        score = left + right
    return top_scores(score, visible, counts, sku), left, right


def evaluate(pairs, trial, shared, counts, similarity, kernels, blender, sku):
    visible = order_matrix([p[0] for p in pairs], len(sku))
    targets = np.array([p[1] for p in pairs])
    top, _, _ = ordered_parts(visible, trial, shared, counts, similarity, kernels, blender, sku)
    matched = top == targets[:, None]
    hit = matched.any(axis=1)
    one = visible.sum(axis=1) == 1

    def rate(mask):
        return round(float(hit[mask].mean()), 9) if mask.any() else None

    return dict(
        run_id=trial.id, model=trial.kind, support=trial.support, rank=trial.rank,
        latent_weight=trial.latent_weight, orders=len(pairs), hits=int(hit.sum()),
        hit_at_5=round(float(hit.mean()), 9),
        ndcg_at_5=round(float((matched / np.log2(np.arange(2, TOP_K + 2))).sum(axis=1).mean()), 9),
        one_item_hit_at_5=rate(one), multi_item_hit_at_5=rate(~one),
    )


def trial_coverage(trials):
    """Confirm controlled comparisons without requiring any particular winner."""
    grouped = {kind: [trial for trial in trials if trial.kind == kind]
               for kind in ("popularity", "cf", "svd", "hybrid")}
    controlled_hybrid = any(
        a.support == b.support and a.rank == b.rank and a.latent_weight != b.latent_weight
        for a, b in itertools.combinations(grouped["hybrid"], 2)
    )
    return {
        "popularity": bool(grouped["popularity"]),
        "cf": len({t.support for t in grouped["cf"]}) >= 2,
        "svd": len({t.rank for t in grouped["svd"]}) >= 2,
        "hybrid": controlled_hybrid,
    }


def preferred(metrics, criterion):
    if criterion not in CRITERIA:
        raise ValueError("출시 전에 Hit@5 또는 NDCG@5를 선택하세요.")
    best = max(row[criterion] for row in metrics)
    return {row["run_id"] for row in metrics if row[criterion] == best}


def compare_basket(basket, first, second, shared, counts, similarity, kernels, blender, sku):
    ids = {item: i for i, item in enumerate(sku)}
    visible = order_matrix([[ids[item] for item in basket]], len(sku))
    r_a, _, _ = ordered_parts(visible, first, shared, counts, similarity, kernels, blender, sku)
    r_b, _, _ = ordered_parts(visible, second, shared, counts, similarity, kernels, blender, sku)
    return [[sku[i] for i in row[0]] for row in (r_a, r_b)]


def exercise_vectors(cosine, kernel, parts):
    """Small public fixtures. Return outputs to be checked against the reference."""
    joint = np.array([[3, 2, 0], [2, 4, 0], [0, 0, 0]], dtype=np.int32)
    cos = quantized_cosine(cosine(joint, joint.diagonal()), joint.shape)
    basis = np.array([[.8,.6,0], [-.6,.8,0], [0,0,1]], dtype=float)
    factor = quantized_kernel(kernel(basis, 1), joint.shape)
    left, right = parts(np.array([[10, 100, 2]]), np.array([[40, 900, -4]]),
                        np.array([[False, True, False]]), 25)
    weights = [parts(np.array([[1, 2]]), np.array([[2, 1]]),
                     np.array([[False, False]]), w) for w in (0, 100)]
    return {
        "cosine": cos.tolist(), "kernel": factor.tolist(),
        "mask_and_mix": [np.asarray(piece, dtype=np.int64).tolist() for piece in (left, right)],
        "endpoints": [[np.asarray(piece, dtype=np.int64).tolist() for piece in pair]
                      for pair in weights],
    }


def make_manifest(data, trials, criterion, selected, evidence, checks):
    return dict(schema_version=2, assignment=ASSIGNMENT, data_version=data["version"],
                dataset_sha256=data["source_sha256"], trials=[asdict(trial) for trial in trials],
                criterion=criterion, selected_run_id=selected, evidence=evidence, checks=checks)


def browser_payload(data, trials, metrics, shared, counts, similarity, kernels, manifest, media):
    cf_raw = {str(support): np.where(shared >= support, similarity, 0).tolist()
              for support in sorted({t.support for t in trials} | {CF_REFERENCE.support}) if support}
    return dict(catalog=data["catalog"], media=media, counts=counts.tolist(), similarities=cf_raw,
                kernels={str(k): kernels[k].tolist() for k in sorted({t.rank for t in trials}) if k},
                scale=SCALE, metrics=metrics, selected=manifest["selected_run_id"],
                specs={t.id: asdict(t) for t in (*trials, CF_REFERENCE)}, reference=CF_REFERENCE.id,
                featured=["22423", "22776"], screenshot_case=manifest["evidence"]["basket"])


def render(template, payload):
    if template.count(DESIGN_MARKER) != 1:
        raise ValueError("설계 실습 화면 템플릿을 확인하세요.")
    return template.replace(DESIGN_MARKER, json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("<", r"\u003c"))


def csv_text(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def package(manifest, student_code, metrics, html):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("submission.json", json.dumps(manifest, ensure_ascii=False, allow_nan=False))
        z.writestr("student_model.py", student_code)
        z.writestr("metrics.csv", csv_text(metrics))
        z.writestr("shop.html", html)
    return stream.getvalue()
