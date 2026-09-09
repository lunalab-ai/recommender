"""같은 카탈로그에서 비교하는 네 가지 교육용 추천 기준.

학습 평점으로 통계/선호 프로필을 만들고, 사용자의 학습 이력만 후보에서
제외한다. test 평점은 이 모듈에 전달하지 않는다. 점수의 단위는 방법마다
다르므로 순위를 평가하며 count/cosine을 예측 평점으로 해석하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MOVIELENS_100K_GENRES
from .rating_models import MeanRatingPredictor, validate_ratings

METHODS = ("count", "mean", "group", "content")


class FourMethodRecommender:
    """평점 수·영화 평균·집단 평균·장르 프로필을 같은 후보에서 비교한다.

    Parameters
    ----------
    group_col : str, default 'sex'
        users의 집단 열 이름. 'occupation'도 가능하다. 개인 취향과 집단을
        동일시하지 않는다. 이 데이터에 주어진 집단 속성만 사용한다.
    min_group_ratings : int, default 1
        집단×영화 평균을 사용할 최소 학습 평점 수. 부족하면 영화 평균,
        영화 학습 이력도 없으면 전체 평균으로 대체한다.
    like_threshold : float, default 4
        내용 프로필에 넣을 학습 평점의 하한(1–5). 평가의 관련성 기준과
        구별되는 모델 설정이며 test 결과를 보고 조정하지 않는다.

    Notes
    -----
    생성자는 설정만 저장한다. fit(train, users, movies)이 학습을 수행한다.
    movies의 전체 장르는 추천 시 이미 알려진 카탈로그 메타데이터로 가정한다.
    genre_vectors_는 unknown을 뺀 18개 장르의 L2 정규화 행렬(M×18),
    profiles_는 사용자 ID→정규화 선호 벡터 사전이다. 원자료 변경이 학습
    상태를 바꾸지 않도록 복사한다. 이력 없는 사용자는 내용 방식에서 영화
    평균 목록으로 대체한다. 장르 없는 영화는 정상 프로필에서 유사도 0이다.
    """

    def __init__(self, *, group_col: str = "sex", min_group_ratings: int = 1,
                 like_threshold: float = 4):
        """group_col/min_group_ratings/like_threshold 설정만 보관한다.

        반환값은 없으며 아직 추천할 수 없다. 유효하지 않은 평점 하한이나
        최소 집단 표본 수에는 ValueError가 발생한다.
        """
        if not np.isfinite(like_threshold) or not 1 <= like_threshold <= 5:
            raise ValueError("like_threshold must be between 1 and 5")
        # 같은 기준 검사를 재사용하되 이 시점에는 fit을 호출하지 않는다.
        MeanRatingPredictor("group", group_col=group_col,
                            min_group_ratings=min_group_ratings)
        self.group_col = group_col
        self.min_group_ratings = min_group_ratings
        self.like_threshold = like_threshold

    def fit(self, train: pd.DataFrame, users: pd.DataFrame,
            movies: pd.DataFrame) -> FourMethodRecommender:
        """train 레이블로 통계와 프로필을 학습하고 self를 반환한다.

        train: N행 DataFrame, user_id/movie_id/rating(1–5) 필수, 쌍은 유일.
        users: user_id와 group_col 필수, 한 사용자 한 행.
        movies: movie_id/title과 MovieLens 장르 18열 필수, 한 영화 한 행.
        train의 모든 ID는 users/movies에 있어야 한다. 추가 열은 허용한다.

        counts_는 영화별 학습 관측 수, mean_model_/group_model_은 학습된
        평균 예측기, seen_은 사용자별 학습 영화 집합이다. profiles_는
        4점 이상(기본값) 학습 영화의 단위 장르 벡터 평균을 다시 단위벡터로
        정규화한다. 빈/영벡터 프로필은 저장하지 않아 대체 규칙이 적용된다.
        입력 표를 수정하지 않는다. 잘못된 스키마·중복·범위는 ValueError.
        """
        validate_ratings(train)
        genres = [g for g in MOVIELENS_100K_GENRES if g != "unknown"]
        needed = {"movie_id", "title", *genres}
        if not needed.issubset(movies.columns) or movies.empty:
            raise ValueError("movies needs movie_id, title and all 18 known genre columns")
        if movies[list(needed)].isna().any().any() or movies.movie_id.duplicated().any():
            raise ValueError("Movie metadata must have unique nonmissing identifiers and values")
        if train.duplicated(["user_id", "movie_id"]).any():
            raise ValueError("Repeated user/movie observations must be resolved before splitting")
        if not train.movie_id.isin(movies.movie_id).all():
            raise ValueError("Training movie missing from catalog")
        if not movies[genres].isin([0, 1]).all().all():
            raise ValueError("Genre features must be binary multi-hot values")
        catalog = movies.sort_values("movie_id").reset_index(drop=True).copy()
        group = MeanRatingPredictor("group", group_col=self.group_col,
                                    min_group_ratings=self.min_group_ratings).fit(train, users)
        mean = MeanRatingPredictor("movie").fit(train)
        features = catalog[genres].to_numpy(dtype=float)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        vectors = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
        positions = pd.Series(np.arange(len(catalog)), index=catalog.movie_id)
        profiles = {}
        liked = train.loc[train.rating.ge(self.like_threshold)]
        for uid, rows in liked.groupby("user_id"):
            profile = vectors[positions.loc[rows.movie_id].to_numpy()].mean(axis=0)
            norm = np.linalg.norm(profile)
            if norm > 0:
                profiles[uid] = profile / norm
        # 검증/계산이 성공한 뒤 학습 상태를 한 번에 교체한다.
        self.catalog_, self.genres_, self.genre_vectors_ = catalog, genres, vectors
        self.mean_model_, self.group_model_ = mean, group
        self.counts_ = train.groupby("movie_id").size().reindex(catalog.movie_id, fill_value=0)
        self.seen_ = {u: set(rows.movie_id) for u, rows in train.groupby("user_id")}
        self.profiles_ = profiles
        return self

    def score_catalog(self, user_id: int, method: str = "content") -> pd.DataFrame:
        """한 user_id의 전체 카탈로그 점수 표를 movie_id 순서로 반환한다.

        method는 count/mean/group/content 중 하나. 반환 M행의 열은
        movie_id, score(float), basis(str)이며 아직 본 영화도 포함한다.
        count는 학습 평점 개수(없으면0), mean/group은 예측 평점,
        content는 코사인 유사도(0–1)다. 내용 프로필이 없으면 이 사용자
        목록 전체를 영화 평균으로 대체하고 basis에 'empty-profile/…'를
        표시한다. 정상 프로필에서 장르 없는 영화는 'zero-genres'다.
        fit 전 또는 알 수 없는 method에는 ValueError. 상태 변경은 없다.
        """
        if not hasattr(self, "catalog_"):
            raise ValueError("Call fit before scoring")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        pairs = pd.DataFrame({"user_id": user_id, "movie_id": self.catalog_.movie_id})
        if method == "count":
            values = self.counts_.to_numpy(dtype=float)
            basis = np.full(len(pairs), "count", dtype=object)
        elif method in {"mean", "group"}:
            model = self.mean_model_ if method == "mean" else self.group_model_
            details = model.predict_details(pairs)
            values, basis = details.prediction.to_numpy(), details.level.to_numpy()
        elif user_id not in self.profiles_:
            details = self.mean_model_.predict_details(pairs)
            values = details.prediction.to_numpy()
            basis = ("empty-profile/" + details.level).to_numpy()
        else:
            # 양쪽이 단위벡터이므로 내적이 코사인이다. 모든 영화 쌍 행렬은 불필요.
            values = np.clip(self.genre_vectors_ @ self.profiles_[user_id], 0, 1)
            basis = np.where(self.genre_vectors_.any(axis=1), "genre-cosine", "zero-genres")
        return pd.DataFrame({"movie_id": self.catalog_.movie_id, "score": values, "basis": basis})

    def recommend(self, user_id: int, method: str = "content", k: int = 10) -> pd.DataFrame:
        """user_id의 학습 영화만 제외하고 상위 k개를 반환한다.

        method: 네 기준 중 하나, k: 양의 정수(기본10). score 내림차순,
        동점은 movie_id 오름차순으로 정한다. 후보가 k개 미만이면 전부 반환.
        출력 열: movie_id/score/basis/title/genres(장르 이름을 |로 연결).
        score를 반올림해서 정렬하지 않는다. test에서 본 영화를 제외하면
        정답 후보를 숨기는 오류가 되므로 test 이력을 받지 않는다.
        """
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        scores = self.score_catalog(user_id, method)
        candidates = scores.loc[~scores.movie_id.isin(self.seen_.get(user_id, set()))]
        top = candidates.sort_values(["score", "movie_id"], ascending=[False, True]).head(k)
        top = top.merge(self.catalog_, on="movie_id", how="left", validate="one_to_one")
        top["genres"] = pd.Series(dtype=str) if top.empty else top[self.genres_].apply(
            lambda row: "|".join(g for g in self.genres_ if row[g] == 1), axis=1)
        return top[["movie_id", "score", "basis", "title", "genres"]].reset_index(drop=True)


def ranking_metrics(recommended: list[int], relevant: set[int], *, k: int = 10) -> dict[str, float]:
    """사용자 한 명의 이진 관련성 순위 지표를 계산한다.

    recommended: 점수순 영화 ID 목록, 중복 불허. 앞의 k개만 평가한다.
    relevant: 후보에 포함되는 test 평점>=4 영화 ID 집합(비어 있으면 오류).
    k: 양의 정수. Precision=적중/k, Recall=적중/관련 영화 수.
    DCG=sum(hit_at_rank/log2(rank+1)), NDCG=DCG/IDCG.
    IDCG는 min(k, 관련 영화 수)개를 맨 위에 놓은 이상적 이진 순위다.
    후보 부족으로 추천이 k개보다 짧으면 빈 자리도 Precision 분모 k에
    포함한다. 반환 키 precision/recall/ndcg/hits, float 값. 평균은 호출자가
    사용자별 결과를 모아서 계산한다. 입력을 수정하지 않는다.
    """
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not relevant or len(recommended) != len(set(recommended)):
        raise ValueError("Need nonempty relevance and unique recommendations")
    hits = np.array([item in relevant for item in recommended[:k]], dtype=float)
    discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = (1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum()
    return {"precision": float(hits.sum() / k), "recall": float(hits.sum() / len(relevant)),
            "ndcg": float(hits @ discounts / ideal), "hits": float(hits.sum())}


def evaluate_rankings(model: FourMethodRecommender, test: pd.DataFrame, *, k: int = 10,
                      relevance_threshold: float = 4) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """학습 완료 model을 고정하고 동일 test/후보/사용자에서 네 방법을 평가한다.

    test: user_id/movie_id/rating의 관측 표. 중복 쌍과 train 쌍 겹침을 거부.
    k: 공통 목록 길이(10), relevance_threshold: 평가 관련성 하한(4).
    관련 test 영화가 하나도 없는 사용자는 네 방법 모두에서 제외하며
    metadata에 수를 기록한다. test는 모델 재학습이나 후보 제거에 사용하지
    않는다. 평균은 평가 사용자별 단순평균(macro), coverage는 추천된 고유
    영화 수/전체 카탈로그 수다. 대체율은 반환된 상위 목록 항목 기준이다.

    Returns
    -------
    summary : DataFrame
        method, precision, recall, ndcg, catalog_coverage, fallback_fraction,
        zero_genre_fraction, users. 점수는 반올림하지 않는다.
    per_user : DataFrame
        user_id/method와 precision/recall/ndcg/hits/relevant_count.
        사용자 식별자와 평가 결과를 담으므로 저작 검증용으로만 저장한다.
    metadata : dict
        test_users/evaluated_users/excluded_no_positive/catalog_size/k 및
        relevance_threshold. 관측되지 않은 영화는 비선호가 아니라 관련성
        미확인이다. 이 holdout 결과는 온라인 클릭/만족도의 보증이 아니다.
    """
    validate_ratings(test)
    if not 1 <= relevance_threshold <= 5:
        raise ValueError("relevance_threshold must be between 1 and 5")
    if test.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Repeated test user/movie pair")
    catalog = set(model.catalog_.movie_id)
    if not test.movie_id.isin(catalog).all():
        raise ValueError("Test movie missing from catalog")
    for uid, rows in test.groupby("user_id"):
        if model.seen_.get(uid, set()).intersection(rows.movie_id):
            raise ValueError("Train and test user/movie pairs overlap")
    positives = test.loc[test.rating.ge(relevance_threshold)]
    relevant = {u: set(rows.movie_id) for u, rows in positives.groupby("user_id")}
    if not relevant:
        raise ValueError("No evaluable users with positive test ratings")
    rows, summaries = [], []
    for method in METHODS:
        union, used, fallback, zero = set(), 0, 0, 0
        method_rows = []
        for uid, truth in sorted(relevant.items()):
            top = model.recommend(uid, method, k)
            metrics = ranking_metrics(top.movie_id.tolist(), truth, k=k)
            row = dict(user_id=uid, method=method, relevant_count=len(truth), **metrics)
            rows.append(row)
            method_rows.append(row)
            union.update(top.movie_id)
            used += len(top)
            if method == "group":
                fallback += int(top.basis.ne("group").sum())
            elif method == "mean":
                fallback += int(top.basis.eq("global").sum())
            elif method == "content":
                fallback += int(top.basis.str.startswith("empty-profile/").sum())
                zero += int(top.basis.eq("zero-genres").sum())
        average = pd.DataFrame(method_rows)[["precision", "recall", "ndcg"]].mean().to_dict()
        summaries.append(dict(method=method, **average, catalog_coverage=len(union)/len(catalog),
                              fallback_fraction=fallback/used if used else 0,
                              zero_genre_fraction=zero/used if used else 0, users=len(relevant)))
    metadata = dict(test_users=int(test.user_id.nunique()), evaluated_users=len(relevant),
                    excluded_no_positive=int(test.user_id.nunique())-len(relevant),
                    catalog_size=len(catalog), k=k, relevance_threshold=relevance_threshold)
    return pd.DataFrame(summaries), pd.DataFrame(rows), metadata
