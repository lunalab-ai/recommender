"""관측 평점만 사용하는 교육용 biased MF와 갱신 추적 도구.

한 관측의 목적함수는 0.5*e**2 + 0.5*reg*(||p||²+||q||²+bu²+bi²).
각 관측 방문 때 이 함수를 갱신하므로 전체 정규화 합은 관측 빈도로
가중된다. 전체 모수에 정규화를 단 한 번 더하는 다른 규약과 구별한다.
예측은 clipping하지 않는다. RMSE는 입력받은 관측 행에 대해 계산한다.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


def sgd_step(p: np.ndarray, q: np.ndarray, rating: float, *, mean: float = 0.,
             user_bias: float = 0., item_bias: float = 0., learning_rate: float = .05,
             regularization: float = .02, use_bias: bool = True) -> dict:
    """한 관측의 갱신 전/후를 반환하며 입력 벡터를 변경하지 않는다.

    p,q: 같은 길이의 유한한 1D 실수 배열. rating/mean/bias는 실수.
    learning_rate>0, regularization>=0. use_bias=False이면 mean과 두
    편향을 예측에서 제외하고 두 편향 출력은 0이다. p_new/q_new는 모두
    갱신 전 p/q 및 같은 error를 사용한다. 반환 dict는 예측/오차/손실,
    새 벡터/편향 및 factor_contributions를 포함한다. 파일/네트워크 없음.
    예: sgd_step(np.array([.2,.4]),np.array([.5,-.1]),4,mean=3).
    """
    p = np.asarray(p, dtype=float).copy()
    q = np.asarray(q, dtype=float).copy()
    if p.ndim != 1 or p.shape != q.shape or not p.size:
        raise ValueError('p와 q는 길이가 같은 비어 있지 않은 1차원 벡터여야 합니다.')
    scalars = [rating, mean, user_bias, item_bias, learning_rate, regularization]
    if not np.isfinite(scalars).all() or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('유한한 숫자만 입력하세요.')
    if learning_rate <= 0 or regularization < 0:
        raise ValueError('학습률은 양수, 정규화는 0 이상이어야 합니다.')
    b, c, mu = (user_bias, item_bias, mean) if use_bias else (0., 0., 0.)
    prediction = float(mu+b+c+p@q)
    error = float(rating-prediction)
    p_new = p + learning_rate*(error*q-regularization*p)
    q_new = q + learning_rate*(error*p-regularization*q)
    b_new = b + learning_rate*(error-regularization*b) if use_bias else 0.
    c_new = c + learning_rate*(error-regularization*c) if use_bias else 0.
    prediction_new = float(mu+b_new+c_new+p_new@q_new)
    penalty = regularization*.5*(p@p+q@q+b*b+c*c)
    penalty_new = regularization*.5*(p_new@p_new+q_new@q_new+b_new*b_new+c_new*c_new)
    return dict(prediction=prediction,error=error,p_new=p_new,q_new=q_new,
                user_bias_new=float(b_new),item_bias_new=float(c_new),
                prediction_new=prediction_new,error_new=float(rating-prediction_new),
                loss=float(.5*error**2+penalty),
                loss_new=float(.5*(rating-prediction_new)**2+penalty_new),
                factor_contributions=p*q,contributions_new=p_new*q_new)


@dataclass
class MFSGD:
    """희소한 관측 평점 표를 NumPy SGD로 학습하는 교육용 모델.

    n_factors=8: 잠재벡터 길이; learning_rate=.02: 한 갱신의 보폭;
    regularization=.02: 관측별 L2 강도; epochs=10: 전체 관측 방문 횟수;
    seed=20261001: 난수; use_bias=True: 평균과 편향 사용;
    init_scale=.1: K와 독립인 초기 정규분포 표준편차, 0은 반례 실험용.
    생성자는 설정만 보관한다. fit 이후에만 P_,Q_,bias,history_를 읽는다.
    예: MFSGD(n_factors=2,epochs=50).fit(ratings).predict(1,4).
    """
    n_factors: int = 8
    learning_rate: float = .02
    regularization: float = .02
    epochs: int = 10
    seed: int = 20261001
    use_bias: bool = True
    init_scale: float = .1

    def fit(self, ratings: pd.DataFrame) -> MFSGD:
        """user_id/movie_id/rating 열의 관측 표를 복사하여 새로 학습한다.

        빈 표·결측/무한 값·중복 관측쌍은 거부한다. 0점도 관측으로 인정한다.
        매 호출은 초기 상태로 재설정한다. 행 순서는 그대로 보관하고 epoch
        내부 방문만 shuffle한다. history_는 epoch 0부터 train_rmse/objective.
        mean_은 이 표의 관측 평균이며 use_bias=False일 때 0. 외부 파일 없음.
        """
        if isinstance(self.n_factors,bool) or int(self.n_factors)!=self.n_factors or self.n_factors<1:
            raise ValueError('n_factors는 양의 정수여야 합니다.')
        if isinstance(self.epochs,bool) or int(self.epochs)!=self.epochs or self.epochs<0:
            raise ValueError('epochs는 0 이상의 정수여야 합니다.')
        if not np.isfinite([self.learning_rate,self.regularization,self.init_scale]).all() or self.learning_rate<=0 or self.regularization<0 or self.init_scale<0:
            raise ValueError('학습률/정규화/초기화 설정을 확인하세요.')
        required=['user_id','movie_id','rating']
        if not set(required).issubset(ratings.columns) or ratings.empty:
            raise ValueError('비어 있지 않은 user_id/movie_id/rating 표가 필요합니다.')
        rows=ratings[required].copy()
        if rows.isna().any().any() or not np.isfinite(rows.rating.to_numpy(dtype=float)).all():
            raise ValueError('미관측 값은 행에서 제외하고 유한한 관측 평점만 전달하세요.')
        if rows.duplicated(['user_id','movie_id']).any():
            raise ValueError('사용자-영화 관측쌍 중복: 집계 규칙을 먼저 정하세요.')
        self.train_=rows
        self.user_ids_=pd.unique(rows.user_id).tolist()
        self.item_ids_=pd.unique(rows.movie_id).tolist()
        self.user_index_={v:i for i,v in enumerate(self.user_ids_)}
        self.item_index_={v:i for i,v in enumerate(self.item_ids_)}
        u=rows.user_id.map(self.user_index_).to_numpy(dtype=int)
        i=rows.movie_id.map(self.item_index_).to_numpy(dtype=int)
        y=rows.rating.to_numpy(dtype=float)
        rng=np.random.default_rng(self.seed)
        self.P_=rng.normal(0,self.init_scale,(len(self.user_ids_),int(self.n_factors)))
        self.Q_=rng.normal(0,self.init_scale,(len(self.item_ids_),int(self.n_factors)))
        self.user_bias_=np.zeros(len(self.user_ids_)); self.item_bias_=np.zeros(len(self.item_ids_))
        self.mean_=float(y.mean()) if self.use_bias else 0.
        history=[]
        def record(epoch):
            pred=self.mean_+self.user_bias_[u]+self.item_bias_[i]+np.sum(self.P_[u]*self.Q_[i],axis=1)
            mse=float(np.mean((y-pred)**2))
            norms=np.sum(self.P_[u]**2,axis=1)+np.sum(self.Q_[i]**2,axis=1)+self.user_bias_[u]**2+self.item_bias_[i]**2
            objective=float(.5*mse+.5*self.regularization*np.mean(norms))
            if not np.isfinite(objective):
                raise FloatingPointError('학습이 발산했습니다. 학습률을 낮추고 초기 설정을 확인하세요.')
            history.append(dict(epoch=epoch,train_rmse=mse**.5,objective=objective))
        record(0)
        for epoch in range(1,int(self.epochs)+1):
            for row in rng.permutation(len(rows)):
                a,b=u[row],i[row]
                # copy: 두 벡터는 동일한 갱신 전 상태의 기울기를 사용한다.
                p_old=self.P_[a].copy(); q_old=self.Q_[b].copy()
                e=y[row]-(self.mean_+self.user_bias_[a]+self.item_bias_[b]+p_old@q_old)
                self.P_[a] += self.learning_rate*(e*q_old-self.regularization*p_old)
                self.Q_[b] += self.learning_rate*(e*p_old-self.regularization*q_old)
                if self.use_bias:
                    self.user_bias_[a] += self.learning_rate*(e-self.regularization*self.user_bias_[a])
                    self.item_bias_[b] += self.learning_rate*(e-self.regularization*self.item_bias_[b])
            record(epoch)
        self.history_=pd.DataFrame(history)
        return self

    def explain(self, user_id, movie_id) -> dict:
        """ID 두 개의 예측을 평균/두 편향/내적 기여도로 분해한다.

        알려지지 않은 ID의 벡터/편향은 0. 알려진 쪽 편향은 유지한다.
        known_user/known_item으로 cold start를 표시한다. clipping 없음.
        fit 전에는 ValueError. 입력/학습 상태를 수정하지 않는다.
        """
        if not hasattr(self,'P_'): raise ValueError('먼저 fit(ratings)를 실행하세요.')
        u=self.user_index_.get(user_id); i=self.item_index_.get(movie_id)
        bu=float(self.user_bias_[u]) if u is not None else 0.
        bi=float(self.item_bias_[i]) if i is not None else 0.
        contributions=self.P_[u]*self.Q_[i] if u is not None and i is not None else np.zeros(int(self.n_factors))
        interaction=float(contributions.sum())
        return dict(mean=self.mean_,user_bias=bu,item_bias=bi,interaction=interaction,
                    prediction=self.mean_+bu+bi+interaction,known_user=u is not None,
                    known_item=i is not None,contributions=contributions.copy())

    def predict(self, user_id, movie_id) -> float:
        """평점 단위 실수 1개. 알려지지 않은 ID 처리와 무클리핑은 explain 참조."""
        return float(self.explain(user_id,movie_id)['prediction'])

    def rmse(self, ratings: pd.DataFrame) -> float:
        """주어진 관측 행의 RMSE. train/heldout 여부는 호출자가 구분한다.

        필수 열 user_id/movie_id/rating. 행 순서대로 예측하며 빈 표나 유한하지
        않은 정답을 거부한다. 알려지지 않은 ID도 대체 규칙으로 평가에 포함.
        """
        if ratings.empty or not np.isfinite(ratings.rating.to_numpy(dtype=float)).all():
            raise ValueError('유한한 평점의 비어 있지 않은 평가 표가 필요합니다.')
        predicted=np.array([self.predict(u,i) for u,i in ratings[['user_id','movie_id']].itertuples(index=False,name=None)])
        return float(np.sqrt(np.mean((ratings.rating.to_numpy()-predicted)**2)))

    def recommend(self, user_id, top_n: int = 5) -> pd.DataFrame:
        """학습에서 본 영화 제외, 예측 내림차순/학습 ID 등장순 동점 처리.

        top_n>=1. 반환 movie_id/prediction/known_user/known_item 표. 카탈로그는
        학습에서 한 번 이상 관측한 영화로 한정한다. 제목은 외부 메타 표와 merge.
        """
        if not hasattr(self,'train_'): raise ValueError('먼저 fit을 실행하세요.')
        if isinstance(top_n,bool) or int(top_n)!=top_n or top_n<1: raise ValueError('top_n은 양의 정수입니다.')
        seen=set(self.train_.loc[self.train_.user_id==user_id,'movie_id'])
        rows=[dict(movie_id=i,**{k:v for k,v in self.explain(user_id,i).items() if k in ['prediction','known_user','known_item']}) for i in self.item_ids_ if i not in seen]
        result=pd.DataFrame(rows,columns=['movie_id','prediction','known_user','known_item'])
        return result.sort_values('prediction',ascending=False,kind='stable').head(int(top_n)).reset_index(drop=True)
