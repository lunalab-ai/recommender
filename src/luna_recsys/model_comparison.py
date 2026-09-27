"""W05A: 같은 관측·후보·사용자로 비교하는 추천 실험 도구."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from time import perf_counter
from pathlib import Path
import hashlib
import inspect
import json
import re
import numpy as np
import pandas as pd
from .rating_models import MeanRatingPredictor, validate_ratings
from .ranking import FourMethodRecommender, ranking_metrics
from .collaborative import UserCF, toy_ratings
from .cf_synthesis import EvidenceCF
from .evaluation import split_ratings
from .data import MovieLens100K, MOVIELENS_100K_GENRES


@dataclass(frozen=True)
class ModelSpec:
    """모델 설정. kind=global/movie/group/count/content/user/item/pearson.

    name은 실험의 고유 이름, family는 같은 목적에서 튜닝할 모델군이다.
    k는 CF 이웃 수(0은 전체); 추천 목록 길이 top_n과 다르다.
    min_common/min_neighbors는 EvidenceCF 조건, centered는 user만 허용.
    group_col/min_group_ratings는 group, like_threshold는 content에 적용.
    생성만으로 학습하지 않는다. 예: ModelSpec('u30','user-k','user',k=30).
    """
    name: str
    family: str
    kind: str
    k: int = 30
    min_common: int = 1
    min_neighbors: int = 1
    centered: bool = False
    clip: bool = True
    group_col: str = 'sex'
    min_group_ratings: int = 1
    like_threshold: float = 4


def baseline_specs() -> list[ModelSpec]:
    """기존 수업의 12개 비교 설정을 새 리스트로 반환. 학습/파일 접근 없음."""
    return [ModelSpec('global','global','global'), ModelSpec('movie','movie','movie'),
        ModelSpec('group-sex-1','group','group'), ModelSpec('count','count','count'),
        ModelSpec('content-4','content','content'),
        ModelSpec('user-all','user-all','user',k=0),
        ModelSpec('user-k30','user-k','user'),
        ModelSpec('user-centered-k30','user-centered','user',centered=True),
        ModelSpec('user-evidence-c3-m2','user-evidence','user',centered=True,min_common=3,min_neighbors=2),
        ModelSpec('item-k30','item-k','item'),
        ModelSpec('item-evidence-c3-m2','item-evidence','item',min_common=3,min_neighbors=2),
        ModelSpec('pearson-all-c3','pearson','pearson',k=0,min_common=3)]


def tuning_specs() -> list[ModelSpec]:
    """기준 설정과 1요소 변경 후보. 전체 Cartesian grid를 만들지 않는다.

    집단2속성×최소수4개, 내용 선호하한3개, 두 축 k5/10/30/50,
    중심화 User의 k, 신뢰도 두 축의 공통수1/3/5·최소이웃1/2/3.
    같은 name은 한 번만 반환. clip 효과는 별도 학생 실험으로 둔다.
    """
    specs=baseline_specs()
    for col in ('sex','occupation'):
        for minimum in (1,5,10,20):
            specs.append(ModelSpec(f'group-{col}-{minimum}','group','group',group_col=col,min_group_ratings=minimum))
    for threshold in (3,5):
        specs.append(ModelSpec(f'content-{threshold}','content','content',like_threshold=threshold))
    for family,kind,centered in [('user-k','user',False),('user-centered','user',True),('item-k','item',False)]:
        for k in (5,10,50):
            specs.append(ModelSpec(f'{family}{k}',family,kind,k=k,centered=centered))
    for kind,centered in [('user',True),('item',False)]:
        base=ModelSpec(f'{kind}-evidence-c3-m2',f'{kind}-evidence',kind,centered=centered,min_common=3,min_neighbors=2)
        for common in (1,5):specs.append(replace(base,name=f'{kind}-evidence-c{common}-m2',min_common=common))
        for minimum in (1,3):specs.append(replace(base,name=f'{kind}-evidence-c3-m{minimum}',min_neighbors=minimum))
    return list({s.name:s for s in specs}.values())


def split_three(ratings: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """중복 없는 관측→train60%/validation20%/test20% 사본.

    기존 W04B와 동일: test .2 seed20260922, 남은 자료 .25 seed20260923.
    시간순/새 사용자 평가가 아닌 무작위 관측 분할. 원본 인덱스 보존.
    """
    if ratings.duplicated(['user_id','movie_id']).any():raise ValueError('Duplicate pairs')
    outer=split_ratings(ratings,test_size=.2,seed=20260922)
    inner=split_ratings(outer.train,test_size=.25,seed=20260923)
    return inner.train,inner.test,outer.test


def cohort(heldout: pd.DataFrame, limit: int | None = 100, seed: int = 20260929) -> list[int]:
    """관측된 사용자 ID만으로 고정 표본 생성. 평점 레이블은 읽지 않는다.

    limit=None은 전체, 양의 정수는 표본 상한. 관련영화 없는 사용자는
    evaluate에서 제외 수와 함께 보고한다. 모든 설정에 같은 리스트 사용.
    """
    ids=np.sort(heldout.user_id.unique())
    if limit is None:return ids.tolist()
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<1:raise ValueError('positive limit required')
    return sorted(np.random.default_rng(seed).choice(ids,min(limit,len(ids)),replace=False).tolist())


def pair_digest(frame: pd.DataFrame) -> str:
    """ID쌍을 정렬한 SHA256. 레이블을 포함하지 않는 분할 식별자."""
    raw=frame[['user_id','movie_id']].sort_values(['user_id','movie_id']).to_csv(index=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class ComparisonSuite:
    """학습 표와 메타데이터를 보관하고 기존 모델을 공통 평가에 연결한다.

    train: user_id/movie_id/rating, 중복 없는1–5점 표.
    users: user_id/sex/occupation, movies: movie_id/title/18장르 열.
    생성은 학습 표 복사·스키마 검사·이력 구성. model(spec)이 필요한
    축의 fit을 최초 한 번 수행하며 이후 k/신뢰도 변경은 fit 상태를 공유한다.
    test를 생성자에 받지 않는다. SGD나 신경망 학습을 추가하지 않는다.
    예: suite=ComparisonSuite(train,users,movies); suite.recommend(spec,1,10).
    """
    def __init__(self, train: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame):
        validate_ratings(train)
        if train.duplicated(['user_id','movie_id']).any():raise ValueError('Duplicate train pairs')
        self.train=train.copy();self.users=users.copy();self.movies=movies.sort_values('movie_id').reset_index(drop=True).copy()
        self.base=FourMethodRecommender().fit(train,users,self.movies)
        self.seen=self.base.seen_;self._models={};self._axes={}
        self.genres=self.base.genres_
        self.genre_text=self.movies[self.genres].apply(lambda r:'|'.join(g for g in self.genres if r[g]),axis=1)
        self._counts=self.train.groupby('movie_id').size()

    def model(self, spec: ModelSpec):
        """ModelSpec→학습된 기존 모델. 설정별 캐시를 재사용한다.

        meanはMeanRatingPredictor、count/contentはFourMethodRecommender、
        user/itemはEvidenceCF、pearsonは従来UserCF。未知kindはValueError。
        """
        if spec not in self._models:
            if spec.kind in {'global','movie','group'}:
                m=MeanRatingPredictor(spec.kind,group_col=spec.group_col,min_group_ratings=spec.min_group_ratings).fit(self.train,self.users)
            elif spec.kind in {'count','content'}:
                m=self.base if spec.like_threshold==4 else FourMethodRecommender(like_threshold=spec.like_threshold).fit(self.train,self.users,self.movies)
            elif spec.kind in {'user','item'}:
                if spec.kind not in self._axes:self._axes[spec.kind]=EvidenceCF(axis=spec.kind).fit(self.train)
                m=self._axes[spec.kind].configured(k=spec.k,min_common=spec.min_common,min_neighbors=spec.min_neighbors,centered=spec.centered,clip=spec.clip)
            elif spec.kind=='pearson':m=UserCF(metric='pearson',min_common=spec.min_common).fit(self.train)
            else:raise ValueError('Unknown model kind')
            self._models[spec]=m
        return self._models[spec]

    def recommend(self, spec: ModelSpec, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """같은 카탈로그에서 train 이력 제외→점수 내림차순/ID 오름차순.

        최대 top_n행 DataFrame: movie_id/title/genres/score/basis/train_count.
        count의 score는 관측 개수, content는 코사인(빈 프로필은 평균 대체),
        나머지는 예측 평점. 모든 단위를 같은 평점이라고 해석하지 않는다.
        평가 레이블을 받지 않으며 처음 model 호출 외 학습/파일 변경 없음.
        """
        if type(top_n) is not int or top_n<1:raise ValueError('positive top_n required')
        m=self.model(spec)
        if spec.kind in {'count','content'}:scores=m.score_catalog(user_id,spec.kind)
        else:
            details=m.predict_details(pd.DataFrame({'user_id':user_id,'movie_id':self.movies.movie_id}))
            scores=pd.DataFrame({'movie_id':self.movies.movie_id,'score':details.prediction.to_numpy(),
                'basis':details['basis' if 'basis' in details else 'level'].to_numpy()})
        scores=scores.loc[~scores.movie_id.isin(self.seen.get(user_id,set()))]
        top=scores.sort_values(['score','movie_id'],ascending=[False,True]).head(top_n)
        meta=self.movies[['movie_id','title']].assign(genres=self.genre_text)
        top=top.merge(meta,on='movie_id',validate='one_to_one')
        top['train_count']=top.movie_id.map(self._counts).fillna(0).astype(int)
        return top[['movie_id','title','genres','score','basis','train_count']].reset_index(drop=True)

    def evaluate(self, spec: ModelSpec, heldout: pd.DataFrame, *, top_n: int = 10,
                 ranking_users: list[int] | None = None) -> tuple[dict,pd.DataFrame]:
        """관측 전체 평점 오차와 공통 사용자 순위 지표→요약dict/사용자별표.

        heldout의 train 중복/카탈로그 밖/중복 관측은 거부. 관련성>=4 고정.
        ranking_users=None은전체. P=hits/top_n, R=hits/relevant_count,
        NDCG는이진 gain; 사용자 macro평균. 관련성0 사용자는 제외 수 보고.
        count/content의 mae/rmse는NaN(해당없음). CF 대체도평점분모에 포함.
        cf_support는전체평가관측중CF근거비율; catalog_coverage는추천된고유영화/
        전체카탈로그. 서로다른분모다. heldout은여기서만채점하며fit에전달안함.
        """
        validate_ratings(heldout)
        if heldout.duplicated(['user_id','movie_id']).any():raise ValueError('Duplicate evaluation pairs')
        if not heldout.movie_id.isin(self.movies.movie_id).all():raise ValueError('Unknown evaluation catalog items')
        for u,rows in heldout.groupby('user_id'):
            if self.seen.get(u,set()).intersection(rows.movie_id):raise ValueError('Train and evaluation overlap')
        ids=cohort(heldout,None) if ranking_users is None else list(ranking_users)
        if len(set(ids))!=len(ids) or not set(ids).issubset(set(heldout.user_id)):raise ValueError('Invalid cohort')
        start=perf_counter();m=self.model(spec)
        result=dict(name=spec.name,family=spec.family,kind=spec.kind,mae=np.nan,rmse=np.nan,
            cf_support=np.nan,rating_fallback=np.nan,rating_count=len(heldout),top_n=top_n)
        if spec.kind not in {'count','content'}:
            detail=m.predict_details(heldout[['user_id','movie_id']])
            err=heldout.rating.to_numpy()-detail.prediction.to_numpy()
            result.update(mae=float(np.abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))))
            basis=detail['basis' if 'basis' in detail else 'level']
            primary=spec.kind if spec.kind in {'global','movie','group'} else 'cf'
            result['rating_fallback']=float(basis.ne(primary).mean())
            if primary=='cf':result['cf_support']=float(basis.eq('cf').mean())
        positive={u:set(r.movie_id) for u,r in heldout.loc[heldout.rating.ge(4)].groupby('user_id')}
        rows=[];union=set();fallback_items=0;returned=0
        for uid in ids:
            truth=positive.get(uid,set())
            if not truth:continue
            top=self.recommend(spec,uid,top_n)
            metrics=ranking_metrics(top.movie_id.tolist(),truth,k=top_n)
            p,r=metrics['precision'],metrics['recall']
            rows.append(dict(user_id=uid,**metrics,f1=2*p*r/(p+r) if p+r else 0,
                relevant_count=len(truth),candidate_count=len(self.movies)-len(self.seen.get(uid,set())),
                returned=len(top),train_activity=len(self.seen.get(uid,set()))))
            union.update(top.movie_id);returned+=len(top)
            if spec.kind=='content':fallback_items+=int(top.basis.str.startswith('empty-profile/').sum())
        if not rows:raise ValueError('No users with relevant heldout items')
        per_user=pd.DataFrame(rows)
        result.update(per_user[['precision','recall','f1','ndcg']].mean().to_dict())
        result.update(ranking_cohort=len(ids),ranking_users=len(rows),excluded_no_positive=len(ids)-len(rows),
            catalog_coverage=len(union)/len(self.movies),catalog_size=len(self.movies),
            content_fallback=fallback_items/returned if spec.kind=='content' and returned else np.nan,
            candidate_min=int(per_user.candidate_count.min()),candidate_max=int(per_user.candidate_count.max()),
            cohort_sha256=hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
            heldout_pairs_sha256=pair_digest(heldout),seconds=perf_counter()-start)
        return result,per_user


def compare(suite: ComparisonSuite, specs: list[ModelSpec], heldout: pd.DataFrame, *,
            top_n: int = 10, ranking_users: list[int] | None = None) -> tuple[pd.DataFrame,pd.DataFrame]:
    """고정 설정 목록을 같은 관측·사용자 집합에서 평가하고 두 표를 반환.

    specs는 모델 설정, top_n은 추천 수. 중복 이름은 거부, 입력은 불변.
    예: table,users=compare(suite,specs,validation). 개인표에는 name이 붙는다.
    """
    if not specs or len({s.name for s in specs})!=len(specs):raise ValueError('Need unique nonempty specs')
    summary=[];users=[]
    for spec in specs:
        row,per_user=suite.evaluate(spec,heldout,top_n=top_n,ranking_users=ranking_users)
        summary.append(row);users.append(per_user.assign(name=spec.name,family=spec.family))
    return pd.DataFrame(summary),pd.concat(users,ignore_index=True)


def select_best(validation_results: pd.DataFrame, metric: str = 'ndcg') -> pd.DataFrame:
    """검증 표에서 family별 최고 설정. metric=ndcg(높음)/rmse(낮음)만 허용.

    동점 name 오름차순. RMSE 해당없는모델제외. test표나개별레이블인자 없음.
    호출자가 validation 표를 전달해야 한다. 함수는 자료 출처를 추측하지 않는다.
    후보/분모가 다른 표를 섞으면 오류. 반환은 선택된행 사본, 입력변경없음.
    """
    if metric not in {'ndcg','rmse'}:raise ValueError('Choose ndcg or rmse')
    for c in ['top_n','cohort_sha256','heldout_pairs_sha256']:
        if validation_results[c].nunique()!=1:raise ValueError('Different evaluation protocols')
    rows=validation_results.dropna(subset=[metric])
    return rows.sort_values([metric,'name'],ascending=[metric=='rmse',True]).drop_duplicates('family').copy()


def definition_link(obj, ref: str = '2026-fall-w05a') -> str:
    """실제 설치된 luna_recsys 객체의 정의 시작 줄로 GitHub 바로가기 생성.

    obj는 공개 함수·클래스·메소드; ref는노트북설치태그와같아야한다.
    inspect로읽기만하고 파일/네트워크를변경하지않는다. 비공개객체거부.
    예: definition_link(ComparisonSuite.evaluate). 반환 https URL 문자열.
    """
    if not re.fullmatch(r'[A-Za-z0-9._-]+',ref):raise ValueError('Use a fixed safe ref')
    if not getattr(obj,'__module__','').startswith('luna_recsys'):raise ValueError('Course public API only')
    filename=Path(inspect.getsourcefile(obj)).name
    _,line=inspect.getsourcelines(obj)
    return f'https://github.com/lunalab-ai/recommender/blob/{ref}/src/luna_recsys/{filename}#L{line}'


def comparison_toy() -> MovieLens100K:
    """기존5×5·18관측에 독자 집단/장르를 붙인 합성예. 원본자료가 아니다.

    users5행(id/sex/occupation), movies5행(id/title/19장르),ratings18행.
    나중에 별도평가를설계할때관측쌍중복에유의. 파일/다운로드없음.
    """
    users=pd.DataFrame({'user_id':range(1,6),'sex':['F','F','M','M','F'],
        'occupation':['student','student','artist','artist','student']})
    movies=pd.DataFrame({'movie_id':range(1,6),'title':['A','B','C','D','E']})
    for g in MOVIELENS_100K_GENRES:movies[g]=0
    movies['Action']=[1,1,0,1,0];movies['Comedy']=[0,1,1,0,1]
    return MovieLens100K(users,movies,toy_ratings())
