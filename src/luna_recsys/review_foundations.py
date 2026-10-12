"""W07A: small, transparent recommendation experiments (not exam answers).

All fixtures are original synthetic teaching data. No download, network, or
hidden training happens in these functions. Inputs are ordinary Python lists.
"""
from __future__ import annotations
from math import isfinite, sqrt, log2
from typing import Callable

Rating = tuple[str, str, float]


def review_data() -> dict:
    """Return fresh synthetic train/test lists, groups and genre vectors.

    Train has 9 distinct (user,item) observations; test has all three unseen
    items for u1. Ratings are 1–5. Features order is [action, comedy].
    >>> len(review_data()['train'])
    9
    """
    return {
        'train': [('u1','A',5),('u1','C',2),('u2','A',3),('u2','B',4),
                  ('u2','D',5),('u3','A',2),('u3','B',1),('u3','E',4),('u4','E',4)],
        'test': [('u1','B',4),('u1','D',1),('u1','E',5)],
        'groups': {'u1':'G1','u2':'G1','u3':'G2','u4':'G2'},
        'features': {'A':[1.,0.],'B':[1.,1.],'C':[0.,1.],'D':[0.,1.],'E':[1.,0.]},
    }


def rating_stats(rows: list[Rating], groups: dict[str,str] | None = None) -> dict:
    """Aggregate observed training rows by item or (group,item).

    Return {key: {'sum':float,'count':int,'mean':float}}. Reject duplicate
    user-item observations, unknown groups, missing/out-of-range ratings.
    Empty input returns {}. No sorting or filtering is performed.
    >>> rating_stats([('u','A',4)])['A']['mean']
    4.0
    """
    out,seen={},set()
    for user,item,rating in rows:
        if (user,item) in seen:raise ValueError('Duplicate user-item observation')
        seen.add((user,item))
        if not isinstance(rating,(int,float)) or isinstance(rating,bool) or not isfinite(rating) or not 1<=rating<=5:
            raise ValueError('Observed ratings must be finite numbers from 1 to 5')
        if groups is not None and user not in groups:raise ValueError('Unknown user group')
        key=(groups[user],item) if groups is not None else item
        entry=out.setdefault(key,{'sum':0.,'count':0,'mean':0.})
        entry['sum']+=float(rating);entry['count']+=1
        entry['mean']=entry['sum']/entry['count']
    return out


def mean_predict(rows: list[Rating], item: str, *, groups: dict[str,str] | None = None,
                 group: str | None = None, min_count: int = 2) -> tuple[float,str]:
    """Predict a rating: sufficient group×item → item → global training mean.

    min_count is the minimum group sample, integer >=1. Returns (rating,basis).
    This per-pair rule differs from whole-list fallback in rank_candidates.
    No test values are accepted. Empty training data raises ValueError.
    >>> mean_predict([('u','A',4)], 'Z')
    (4.0, 'global mean')
    """
    if isinstance(min_count,bool) or not isinstance(min_count,int) or min_count<1:
        raise ValueError('min_count must be a positive integer')
    stats=rating_stats(rows)
    if not rows:raise ValueError('No training observations')
    if group is not None:
        if groups is None:raise ValueError('groups required')
        gs=rating_stats(rows,groups).get((group,item))
        if gs and gs['count']>=min_count:return gs['mean'],'group mean'
    if item in stats:return stats[item]['mean'],'item mean'
    return sum(r for _,_,r in rows)/len(rows),'global mean'


def cosine_score(x: list[float], y: list[float]) -> float | None:
    """Cosine of equal nonempty vectors; None for a zero vector.

    Features must have the same order. Score is not a probability or rating.
    Reject nonfinite coordinates; no implicit imputation/normalization state.
    >>> cosine_score([1,0], [0,1])
    0.0
    """
    if not x or len(x)!=len(y):raise ValueError('Same nonempty dimensions required')
    if not all(isfinite(v) for v in x+y):raise ValueError('Finite coordinates required')
    nx=sqrt(sum(v*v for v in x));ny=sqrt(sum(v*v for v in y))
    return None if nx==0 or ny==0 else sum(a*b for a,b in zip(x,y))/(nx*ny)


def rank_candidates(data: dict, *, user: str='u1', method: str='mean',
                    group: str='G1', min_count: int=1, top_k: int=3) -> list[dict]:
    """Rank eligible unseen items using mean/count/group/content.

    All statistics use data['train']; test is never read here. Filter before
    sorting. Ties: score descending, observed count descending, ID ascending.
    For group lists, if NO eligible unseen group item exists, switch the whole
    list to global item means and label fallback. Content uses the average
    feature vector of this user's training likes (rating >=4). A zero/absent
    profile yields no candidates. min_count applies to item support, or group
    support in group mode. top_k is a maximum; a short list is not padded.
    Return dictionaries item/score/count/basis; no side effects.
    """
    if method not in {'mean','count','group','content'}:raise ValueError('Unknown method')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (min_count,top_k)):
        raise ValueError('min_count and top_k must be positive integers')
    rows=data['train'];stats=rating_stats(rows);features=data['features']
    seen={i for u,i,_ in rows if u==user}
    gs=rating_stats(rows,data['groups']) if method=='group' else {}
    eligible=[i for i in features if i not in seen]
    group_ok=[i for i in eligible if gs.get((group,i),{}).get('count',0)>=min_count]
    fallback=method=='group' and not group_ok
    profile=None
    if method=='content':
        liked=[features[i] for u,i,r in rows if u==user and r>=4 and i in features]
        if not liked:return []
        width=len(liked[0]);profile=[sum(v[j] for v in liked)/len(liked) for j in range(width)]
    out=[]
    for item in eligible:
        st=gs.get((group,item)) if method=='group' and not fallback else stats.get(item)
        if st is None or st['count']<min_count:continue
        score=st['count'] if method=='count' else st['mean']
        basis='item mean fallback' if fallback else 'group mean' if method=='group' else method
        if method=='content':
            score=cosine_score(profile,features[item])
            if score is None:continue
        out.append(dict(item=item,score=float(score),count=st['count'],basis=basis))
    return sorted(out,key=lambda x:(-x['score'],-x['count'],x['item']))[:top_k]


def rating_metrics(actual: list[float], predicted: list[float]) -> dict[str,float]:
    """MAE/RMSE over aligned nonempty rating lists, each observation equal weight.

    Caller must supply rating predictions, not count/cosine ranking scores.
    Empty, unequal, or nonfinite inputs raise ValueError; output unit is rating.
    """
    if not actual or len(actual)!=len(predicted):raise ValueError('Aligned nonempty inputs required')
    if not all(isfinite(v) for v in actual+predicted):raise ValueError('Finite inputs required')
    e=[y-p for y,p in zip(actual,predicted)]
    return {'MAE':sum(abs(v) for v in e)/len(e),'RMSE':sqrt(sum(v*v for v in e)/len(e))}


def ranking_metrics(recommended: list[str], relevant: set[str], catalog: set[str],
                    *, k: int) -> dict[str,float | None]:
    """Binary precision@K, recall@K, NDCG@K, single-list catalog coverage.

    A short list still uses requested K as the precision denominator. No
    relevant items => recall/NDCG None; an empty catalog is invalid. Duplicate
    recommendations, unknown IDs, or more than K returned items are rejected.
    This synthetic experiment knows relevance for every unseen candidate.
    """
    if isinstance(k,bool) or not isinstance(k,int) or k<1 or not catalog:raise ValueError('Positive K and catalog required')
    if len(set(recommended))!=len(recommended) or len(recommended)>k:raise ValueError('Invalid recommendation list')
    if not set(recommended)<=catalog or not relevant<=catalog:raise ValueError('Unknown catalog item')
    hits=sum(i in relevant for i in recommended)
    dcg=sum((i in relevant)/log2(j+2) for j,i in enumerate(recommended))
    ideal=sum(1/log2(j+2) for j in range(min(k,len(relevant))))
    return {'Precision@K':hits/k,'Recall@K':hits/len(relevant) if relevant else None,
            'NDCG@K':dcg/ideal if ideal else None,'Catalog coverage':len(set(recommended))/len(catalog)}


def review_experiment(method: str='mean', min_count: int=1, top_k: int=3,
                      group: str='G1') -> dict:
    """Run the fixed synthetic experiment; return data, rank rows and metrics.

    u1 likes held-out B/E (ratings >=4); D is known nonrelevant. Other observed
    u1 items A/C are excluded. MAE/RMSE only for rating methods mean/group.
    All three held-out observations are evaluated even if not recommended.
    This is a demonstration, not evidence of MovieLens generalization.
    """
    data=review_data();ranked=rank_candidates(data,method=method,min_count=min_count,top_k=top_k,group=group)
    relevant={i for _,i,r in data['test'] if r>=4}
    metrics=ranking_metrics([r['item'] for r in ranked],relevant,set(data['features']),k=top_k)
    predictions=[]
    if method in {'mean','group'}:
        for _,item,actual in data['test']:
            prediction,basis=mean_predict(data['train'],item,groups=data['groups'],group=group if method=='group' else None,min_count=min_count)
            predictions.append([item,actual,prediction,basis,actual-prediction])
        metrics.update(rating_metrics([r[1] for r in predictions],[r[2] for r in predictions]))
    return dict(data=data,ranked=ranked,metrics=metrics,predictions=predictions,method=method,
                min_count=min_count,top_k=top_k,group=group)
