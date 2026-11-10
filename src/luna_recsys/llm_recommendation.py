"""Chapter-9 review data, visible prompting, catalog validation and honest evaluation.

The official Software sample differs from the publisher's Amazon files. Review
timestamps are observations, not verified purchase times. Raw reviews stay local.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .llm_runtime import CHAT_MODEL, download_checked

AMAZON_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Software_5.json.gz'
AMAZON_SHA256 = '1495184a2d3a166516b1fbc5be57c5dbcc9fd44eb6be3df72a67aeb5546cd9ed'


def load_amazon_reviews(cache: str | Path | None = None) -> pd.DataFrame:
    """Prepare the official 12,805-row archive; return deduplicated pseudonymous rows.

    Earliest timestamp/source-row breaks duplicate user/item ties. Keep only the
    required text/rating/time fields, never reviewer names. Fail on corrupt data.
    """
    cache = Path(cache or Path.home() / '.cache/luna-recsys/w11b-data')
    path = download_checked(AMAZON_URL, cache / 'Software_5.json.gz', AMAZON_SHA256)
    records = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            records.append({'raw_user': str(row['reviewerID']), 'raw_item': str(row['asin']),
                'rating': float(row['overall']), 'timestamp': int(row['unixReviewTime']),
                'text': str(row.get('reviewText') or ''), 'row_id': position})
    frame = pd.DataFrame(records)
    if len(frame) != 12805 or not frame.rating.between(1, 5).all():
        raise ValueError('Official review schema/row count differs from the verified sample.')
    frame = frame.sort_values(['timestamp', 'row_id'], kind='stable').drop_duplicates(['raw_user', 'raw_item'])
    users = {uid: f'U{i+1:04d}' for i, uid in enumerate(sorted(frame.raw_user.unique()))}
    items = {iid: f'P{i+1:04d}' for i, iid in enumerate(sorted(frame.raw_item.unique()))}
    frame['user_id'] = frame.raw_user.map(users)
    frame['item_id'] = frame.raw_item.map(items)
    frame = frame[['user_id', 'item_id', 'rating', 'timestamp', 'text', 'row_id']].reset_index(drop=True)
    frame.attrs['provenance'] = {'mode': 'real', 'dataset': 'Amazon Reviews 2018 Software 5-core',
        'url': AMAZON_URL, 'sha256': AMAZON_SHA256, 'raw_rows': 12805,
        'deduplicated_rows': len(frame), 'users': len(users), 'items': len(items),
        'deduplication': 'earliest timestamp then source-row position'}
    return frame


def select_cohort(frame: pd.DataFrame, max_users: int = 32, max_history: int = 20,
                  seed: int = 20261111) -> pd.DataFrame:
    """Select most-observed users (seeded ties), then keep latest observations.

    This structural classroom subset uses observation counts, never rating values
    or model errors. It is not representative population/future-performance evidence.
    """
    needed = {'user_id', 'item_id', 'rating', 'timestamp', 'text', 'row_id'}
    if not needed.issubset(frame.columns) or max_users < 8 or max_history < 7:
        raise ValueError('Required columns, at least8 users and history length7 are needed.')
    if frame.row_id.duplicated().any() or frame.duplicated(['user_id', 'item_id']).any():
        raise ValueError('Resolve duplicated observations before selecting/splitting.')
    counts = frame.groupby('user_id').size()
    eligible = np.array(sorted(counts.loc[counts >= 7].index))
    if len(eligible) < 8:
        raise ValueError('There are fewer than8 eligible users.')
    rng = np.random.default_rng(seed)
    ranked = pd.DataFrame({'user_id': rng.permutation(eligible)})
    ranked['count'] = ranked.user_id.map(counts)
    ranked = ranked.sort_values('count', ascending=False, kind='stable')
    chosen = set(ranked.user_id.head(max_users))
    out = frame[frame.user_id.isin(chosen)].sort_values(['user_id', 'timestamp', 'row_id'], kind='stable')
    out = out.groupby('user_id', sort=True).tail(max_history).reset_index(drop=True).copy()
    out.attrs['provenance'] = dict(frame.attrs.get('provenance', {}), cohort_users=out.user_id.nunique(),
        cohort_items=out.item_id.nunique(), cohort_rows=len(out), max_history=max_history, seed=seed,
        cohort_rule='most observed users; seeded ties; latest history cap; no rating-value selection')
    return out


def split_fm_observations(frame: pd.DataFrame, seed: int = 20261111) -> dict:
    """Outer75/25 user-stratified observations, then20% of outer train for validation.

    This is a random-observation protocol, not future prediction/new-user testing.
    """
    positions = np.arange(len(frame))
    development, test = train_test_split(positions, test_size=.25, random_state=seed, stratify=frame.user_id)
    train, validation = train_test_split(development, test_size=.2, random_state=seed,
        stratify=frame.iloc[development].user_id)
    parts = {name: frame.iloc[ids].sort_values('row_id').copy()
             for name, ids in [('train', train), ('validation', validation), ('test', test)]}
    ids = {name: set(part.row_id) for name, part in parts.items()}
    if any(ids[a] & ids[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise AssertionError('Observation partitions overlap.')
    parts['protocol'] = {'kind': 'user-stratified-random-observations', 'seed': seed,
        'rows': {name: len(parts[name]) for name in ('train', 'validation', 'test')}}
    return parts


def aggregate_reviews(frame: pd.DataFrame, by: str = 'item_id', max_reviews: int = 3,
                      max_chars: int = 300, positive_only: bool = False) -> dict:
    """Group only the supplied rows; retain source IDs and explicit text truncation.

    Returns texts/source_rows/omitted_chars by ID. No encoding, writes or learning.
    """
    if by not in ('user_id', 'item_id') or min(max_reviews, max_chars) < 1:
        raise ValueError('Choose a valid grouping key and positive text limits.')
    groups = frame.sort_values(['timestamp', 'row_id'], kind='stable').groupby(by, sort=True)
    texts, source_rows, omitted = {}, {}, {}
    for key, group in groups:
        chosen = group[group.rating.ge(4)] if positive_only else group
        chosen = chosen[chosen.text.fillna('').str.strip().ne('')].tail(max_reviews)
        original = chosen.text.astype(str).tolist()
        clipped = [text[:max_chars] for text in original]
        texts[key] = '\n'.join(clipped)
        source_rows[key] = chosen.row_id.astype(int).tolist()
        omitted[key] = sum(len(a) - len(b) for a, b in zip(original, clipped))
    return {'texts': texts, 'source_rows': source_rows, 'omitted_chars': omitted}


def prepare_prompt_case(frame: pd.DataFrame, *, candidates: int = 8, k: int = 3,
                        seed: int = 20261111) -> dict:
    """Separate example users from target users; hide target's final3 observations.

    Candidate popularity uses only example users. Catalog membership is known,
    but held-out items are never inserted just to make the experiment look better.
    """
    if not 1 <= k <= candidates:
        raise ValueError('Need 1<=K<=candidate count.')
    users = np.array(sorted(frame.user_id.unique()))
    shuffled = np.random.default_rng(seed).permutation(users)
    cut = int(.75 * len(users))
    example_users = set(shuffled[:cut])
    target_users = sorted(shuffled[cut:])
    target_user = target_users[0]
    pool = frame[frame.user_id.isin(example_users)].copy()
    target = frame[frame.user_id.eq(target_user)].sort_values(['timestamp', 'row_id'], kind='stable')
    observed, heldout = target.iloc[:-3].copy(), target.iloc[-3:].copy()
    seen = set(observed.item_id)
    popularity = pool.groupby('item_id').size().rename('count').reset_index()
    popularity = popularity.sort_values(['count', 'item_id'], ascending=[False, True], kind='stable')
    candidate_ids = popularity.loc[~popularity.item_id.isin(seen), 'item_id'].head(candidates).tolist()
    if len(candidate_ids) < k:
        raise ValueError('Not enough unseen training-catalog products for Top-K.')
    examples = [{'customer': uid, 'history': group.sort_values(['timestamp', 'row_id']).item_id.tolist()[-6:]}
                for uid, group in pool.groupby('user_id', sort=True)]
    reviews = aggregate_reviews(pool)
    preferences = observed.sort_values(['rating', 'timestamp', 'row_id'], ascending=[False, True, True]).head(3)
    input_case = {'customer': target_user, 'history': observed.item_id.tolist()[-6:],
        'examples': examples, 'candidates': candidate_ids,
        'catalog_reviews': {iid: reviews['texts'].get(iid, '') for iid in candidate_ids},
        'preference_reviews': [{'rating': float(r.rating), 'text': str(r.text)[:240]}
                               for r in preferences.itertuples()], 'k': k}
    truth = heldout.item_id.tolist()
    return {'input': input_case, 'truth': truth, 'seen': seen,
        'observed_rows': observed.row_id.tolist(), 'heldout_rows': heldout.row_id.tolist(),
        'example_users': sorted(example_users), 'target_users': target_users,
        'catalog_source_rows': reviews['source_rows'],
        'candidate_truth_coverage': len(set(truth) & set(candidate_ids)) / len(set(truth))}


def build_prompt(case: dict, method: str = 'CF', shots: int = 3, review_chars: int = 140) -> list[dict]:
    """Visible input construction; receives past-only input, never the target answers.

    CF uses example histories; CB uses training-product/past-preference reviews.
    Changing these fields changes the actual subsequent model request.
    """
    if method not in ('CF', 'CB') or not 0 <= shots <= 4 or not 1 <= review_chars <= 240:
        raise ValueError('Choose CF/CB, 0–4 examples and 1–240 review characters.')
    payload = {'customer': case['customer'], 'K': case['k'], 'method': method,
               'observed_history': case['history']}
    if method == 'CF':
        payload['examples'] = case['examples'][:shots]
        payload['catalog'] = case['candidates']
    else:
        payload['preferences'] = case['preference_reviews']
        payload['catalog'] = [{'product': iid, 'reviews': case['catalog_reviews'][iid][:review_chars]}
                              for iid in case['candidates']]
    instruction = ('Recommend K distinct product codes from the provided catalog. '
        'Use the other customers histories for CF, or the review contents and ratings for CB. '
        'Do not output customer codes. Return JSON with a recommendations list of product codes.')
    return [{'role': 'system', 'content': instruction},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]


def recommendation_schema(candidates: list[str], k: int) -> dict:
    """Constrain JSON shape/catalog tokens; semantic validity still needs inspection."""
    return {'type': 'object', 'properties': {'recommendations': {'type': 'array',
        'items': {'type': 'string', 'enum': list(candidates)}, 'minItems': k, 'maxItems': k}},
        'required': ['recommendations'], 'additionalProperties': False}


def validate_recommendations(raw: str, candidates: list[str], seen: set[str], k: int = 3) -> dict:
    """Validate actual output, reporting errors instead of inventing replacement IDs."""
    if k < 1:
        raise ValueError('K must be positive.')
    try:
        obj = json.loads(raw)
        values = obj['recommendations']
        if not isinstance(values, list):
            raise ValueError('recommendations must be a list')
    except (ValueError, KeyError, TypeError) as exc:
        return {'status': 'invalid', 'valid': [], 'rejected': [{'reason': 'invalid-json-or-schema'}],
                'requested': k, 'returned': 0, 'detail': str(exc)}
    valid, rejected = [], []
    for value in values:
        reason = ('not-a-string' if not isinstance(value, str) else
                  'already-observed' if value in seen else
                  'outside-catalog' if value not in candidates else
                  'duplicate' if value in valid else
                  'beyond-requested-K' if len(valid) >= k else '')
        if reason:
            rejected.append({'value': value, 'reason': reason})
        else:
            valid.append(value)
    status = 'valid' if len(valid) == k and not rejected else 'partial' if valid else 'invalid'
    return {'status': status, 'valid': valid, 'rejected': rejected,
            'requested': k, 'returned': len(values)}


def ranking_overlap(recommendations: list[str], truth: list[str], k: int = 3) -> dict:
    """Observed-future overlap, not calibrated probability or proven satisfaction.

    Precision's denominator stays requested K even when output is short/invalid.
    """
    if k < 1 or not truth:
        raise ValueError('Positive K and a nonempty held-out set are required.')
    hits = len(set(recommendations[:k]) & set(truth))
    precision, recall = hits / k, hits / len(set(truth))
    return {'hits': hits, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'requested_k': k, 'valid_count': len(set(recommendations[:k]))}


def run_prompt(case: dict, truth: list[str], seen: set[str], client, method: str = 'CF',
               shots: int = 3, review_chars: int = 140, *, prompt_fn: Callable = build_prompt,
               validator_fn: Callable = validate_recommendations,
               metric_fn: Callable = ranking_overlap) -> dict:
    """Shared notebook/app path; editable functions determine actual requests/results."""
    messages = prompt_fn(case, method=method, shots=shots, review_chars=review_chars)
    raw, metadata = client.chat(messages, recommendation_schema(case['candidates'], case['k']))
    checked = validator_fn(raw, case['candidates'], seen, case['k'])
    metrics = metric_fn(checked['valid'], truth, case['k'])
    prompt_text = json.dumps(messages, ensure_ascii=False)
    table = pd.DataFrame([{'rank': j + 1, 'product': iid, 'heldout_match': iid in truth,
        'method': method, 'model': CHAT_MODEL, 'requested_k': case['k']}
        for j, iid in enumerate(checked['valid'])],
        columns=['rank', 'product', 'heldout_match', 'method', 'model', 'requested_k'])
    return {'table': table, 'metrics': metrics, 'validation': checked, 'raw': raw,
        'metadata': metadata, 'messages': messages, 'method': method, 'shots': shots,
        'prompt_chars': len(prompt_text),
        'prompt_sha256': hashlib.sha256(prompt_text.encode()).hexdigest()}
