"""작은 실수 행렬의 SVD를 관찰하는 수업 함수. 파일/네트워크/학습 상태 없음."""
from __future__ import annotations
import numpy as np


def teaching_matrix(name: str = "main") -> np.ndarray:
    """명시적 합성 예제의 새 float 배열을 반환한다. 실제 평점 자료가 아니다.

    main (2,2), rectangular (3,2), deficient (2,2), negative (2,2),
    shear (2,2), zero (2,2). 예: teaching_matrix('main') → [[5,1],[1,5]].
    호출마다 사본을 반환하므로 변경해도 다음 호출에 영향을 주지 않는다.
    """
    examples = {
        "main": [[5, 1], [1, 5]], "rectangular": [[3, 0], [0, 2], [0, 0]],
        "deficient": [[3, 0], [4, 0]], "negative": [[3, 0], [0, -2]],
        "shear": [[2, 1], [0, 1]], "zero": [[0, 0], [0, 0]],
    }
    if name not in examples:
        raise ValueError(f"Unknown teaching matrix: {name}")
    return np.array(examples[name], dtype=float)


def unit_vector(vector) -> np.ndarray:
    """유한한 1D 실수 벡터 → 길이1인 새 배열. 0벡터는 방향이 없어 거부한다.

    예: unit_vector([3,4]) → [0.6,0.8]. 입력 배열은 변경하지 않는다.
    """
    raw = np.asarray(vector)
    if np.iscomplexobj(raw):
        raise ValueError("실수 벡터만 사용합니다.")
    v = np.asarray(raw, dtype=float)
    if v.ndim != 1 or not v.size or not np.isfinite(v).all():
        raise ValueError("비어 있지 않은 유한한 1차원 벡터가 필요합니다.")
    size = float(np.linalg.norm(v))
    if not np.isfinite(size) or size == 0:
        raise ValueError("길이가 0이거나 수치 범위를 벗어난 벡터는 정규화할 수 없습니다.")
    return v / size


def _matrix(value) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("이 실습은 실수 행렬을 사용합니다.")
    a = np.array(raw, dtype=float, copy=True)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("빈 칸/NaN/무한대가 없는 2차원 행렬이 필요합니다.")
    return a


def symmetric_eigen(matrix) -> dict:
    """대칭 실수 (n,n) 행렬 → 내림차순 고유값 values와 정규 고유벡터 columns.

    반환 keys: values (n,), vectors (n,n), residual (float).
    np.linalg.eigh는 오름차순이므로 값과 벡터의 열을 함께 뒤집는다.
    열의 최대 절댓값 좌표가 양수가 되도록 표시 부호를 통일한다.
    부호/중복 고유값의 기저는 유일하지 않다. 예: symmetric_eigen([[2,1],[1,2]]).
    """
    a = _matrix(matrix)
    if a.shape[0] != a.shape[1] or not np.allclose(a, a.T, rtol=1e-12, atol=1e-12):
        raise ValueError("eigh 실습에는 대칭 정방행렬이 필요합니다.")
    values, vectors = np.linalg.eigh(a)
    values, vectors = values[::-1].copy(), vectors[:, ::-1].copy()
    for j in range(len(values)):
        if vectors[np.argmax(np.abs(vectors[:, j])), j] < 0:
            vectors[:, j] *= -1
    return {"values": values, "vectors": vectors,
            "residual": float(np.linalg.norm(a @ vectors - vectors * values))}


def svd_steps(matrix, rank: int | None = None) -> dict:
    """완전한 (m,n) 실수 행렬의 축소 SVD와 rank-k 근사를 관찰한다.

    rank=None은 min(m,n), 정수0..min(m,n) 허용. 0은 영행렬 근사.
    반환 U(m,q), s(q,), Vt(q,n), gram(n,n), reconstruction(m,n),
    error(float), discarded_energy(float), energy_fraction(float|None),
    numerical_rank(int), positive_left(m,r), positive_right(n,r).
    q=min(m,n), r은 수치적 rank. 특이값0에 대해서는 나누지 않는다.
    Gram 행렬은 손계산 설명용이며 실제 분해는 안정적인 np.linalg.svd 사용.
    에너지 비율은 Frobenius 제곱합 기준이며 추천 정확도가 아니다.
    예: svd_steps([[5,1],[1,5]],1)의 error는4, reconstruction은 모든 칸3.
    """
    a = _matrix(matrix)
    q = min(a.shape)
    k = q if rank is None else rank
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k <= q:
        raise ValueError(f"rank는 0부터 {q}까지의 정수여야 합니다.")
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    for j in range(q):
        if vt[j, np.argmax(np.abs(vt[j]))] < 0:
            vt[j] *= -1
            u[:, j] *= -1
    reconstruction = (u[:, :k] * s[:k]) @ vt[:k, :]
    tol = max(a.shape) * np.finfo(float).eps * s[0]
    positive = s > tol
    right = vt[positive].T
    left = (a @ right) / s[positive] if positive.any() else np.empty((a.shape[0], 0))
    energy = float(s @ s)
    return {"A": a, "U": u, "s": s, "Vt": vt, "gram": a.T @ a,
            "reconstruction": reconstruction, "error": float(np.linalg.norm(a - reconstruction)),
            "discarded_energy": float(np.sum(s[k:] ** 2)),
            "energy_fraction": float(np.sum(s[:k] ** 2) / energy) if energy else None,
            "numerical_rank": int(positive.sum()), "positive_left": left,
            "positive_right": right, "rank": int(k)}


def transform_stages(matrix, points) -> tuple[np.ndarray, ...]:
    """2×2 행렬과 2×N 점 → 원본,Vᵀ 후,Σ 후,U 후의 4개 배열.

    한 열이 하나의 점. 마지막 배열은 matrix @ points와 같다. 입력은 변경하지 않는다.
    """
    a = _matrix(matrix)
    x = _matrix(points)
    if a.shape != (2, 2) or x.shape[0] != 2:
        raise ValueError("변환 표시에는 2×2 행렬과 2×N 점이 필요합니다.")
    d = svd_steps(a)
    first = d["Vt"] @ x
    second = d["s"][:, None] * first
    return x.copy(), first, second, d["U"] @ second
