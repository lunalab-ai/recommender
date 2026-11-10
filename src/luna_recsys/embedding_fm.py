"""Training-review embeddings as real-valued FM inputs, with editable callbacks.

LLM/encoder weights are frozen. Only the FM's w/V are fitted; its intercept is the
training mean. Validation selects a checkpoint; held-out test rows never enter fit.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import Callable

import numpy as np
import pandas as pd

from .fm_core import fm_score
from .llm_recommendation import aggregate_reviews
from .llm_runtime import EMBED_MODEL, MODEL_DIGESTS, file_sha256


def normalize_embeddings(vectors, dimension: int = 8) -> np.ndarray:
    """Select a prefix then L2-normalize rows. Zero prefixes stay zero, not unit vectors.

    Input (N,D), output (N,dimension), new float array. This is not PCA or learned
    feature selection. Coordinates do not have fixed human-readable genre names.
    """
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or not 1 <= dimension <= values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Need finite (N,D) values and 1<=dimension<=D.')
    prefix = values[:, :dimension].copy()
    norms = np.linalg.norm(prefix, axis=1, keepdims=True)
    return np.divide(prefix, norms, out=np.zeros_like(prefix), where=norms > 0)


def _function_sha(function: Callable) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        source = repr((function.__code__.co_code, function.__code__.co_consts))
    return hashlib.sha256(source.encode()).hexdigest()


def make_feature_recipe(train: pd.DataFrame, client, cache: str | Path, *, dimension: int = 8,
                        positive_only: bool = False, review_fn: Callable = aggregate_reviews,
                        normalize_fn: Callable = normalize_embeddings) -> dict:
    """Encode only supplied training reviews and cache actual model vectors.

    Text/model/split/dimension/aggregation/normalizer changes invalidate the key.
    Missing documents/unknown IDs use an explicit training-only mean, not a fake
    encoder result. API/shape/checksum failures propagate rather than fall back.
    """
    if train.empty or not 1 <= dimension <= 768:
        raise ValueError('Training rows and 1<=dimension<=768 are required.')
    users, items = sorted(train.user_id.unique()), sorted(train.item_id.unique())
    grouped = {kind: review_fn(train, by=kind + '_id', max_reviews=3, max_chars=300,
                               positive_only=positive_only) for kind in ('user', 'item')}
    documents = {kind + ':' + key: 'classification: ' + value
                 for kind, result in grouped.items() for key, value in result['texts'].items() if value.strip()}
    keys = sorted(documents)
    if not keys:
        raise ValueError('No nonempty training review documents exist.')
    description = {'model': EMBED_MODEL, 'digest': MODEL_DIGESTS[EMBED_MODEL],
        'train_rows': sorted(train.row_id.astype(int).tolist()), 'texts': documents,
        'dimension': dimension, 'normalization': _function_sha(normalize_fn),
        'aggregation': _function_sha(review_fn), 'positive_only': positive_only,
        'max_reviews': 3, 'max_chars': 300, 'prefix': 'classification: '}
    key = hashlib.sha256(json.dumps(description, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path, manifest = cache / (key + '.npz'), cache / (key + '.json')
    hit = path.exists() and manifest.exists()
    calls_before = len(client.calls)
    if hit:
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        if file_sha256(path) != meta['vectors_sha256']:
            raise ValueError('Embedding cache bytes changed; preserve and inspect that cache.')
        with np.load(path, allow_pickle=False) as archive:
            if archive['keys'].tolist() != keys:
                raise ValueError('Embedding cache ID order differs from its inputs.')
            raw = archive['vectors'].copy()
    else:
        batches = []
        for start in range(0, len(keys), 16):
            vectors, _ = client.embed([documents[k] for k in keys[start:start+16]])
            batches.extend(vectors)
        raw = np.asarray(batches, dtype=float)
        if raw.shape != (len(keys), 768) or not np.isfinite(raw).all():
            raise RuntimeError('Actual encoder dimensions/values differ from the verified model.')
        temporary = cache / (key + '.tmp.npz')
        np.savez_compressed(temporary, keys=np.array(keys), vectors=raw)
        temporary.replace(path)
        meta = {'model': EMBED_MODEL, 'model_digest': MODEL_DIGESTS[EMBED_MODEL],
            'raw_dimension': 768, 'dimension': dimension, 'documents': len(keys),
            'train_rows_sha256': hashlib.sha256(json.dumps(description['train_rows']).encode()).hexdigest(),
            'cache_key': key, 'vectors_sha256': file_sha256(path)}
        manifest.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if raw.shape != (len(keys), 768) or not np.isfinite(raw).all():
        raise RuntimeError('Invalid actual embedding matrix.')
    selected = normalize_fn(raw, dimension)
    if selected.shape != (len(keys), dimension) or not np.isfinite(selected).all():
        raise ValueError('The editable normalizer returned an invalid matrix.')
    vectors = dict(zip(keys, selected))
    by_kind = {kind: {key: vectors[kind + ':' + key].copy() for key in ids if kind + ':' + key in vectors}
               for kind, ids in [('user', users), ('item', items)]}
    defaults = {kind: np.mean(list(values.values()), axis=0) if values else np.zeros(dimension)
                for kind, values in by_kind.items()}
    return {'users': tuple(users), 'items': tuple(items), 'dimension': dimension, 'raw_dimension': 768,
        'user_index': {u: i for i, u in enumerate(users)},
        'item_index': {iid: i for i, iid in enumerate(items)},
        'user_vectors': by_kind['user'], 'item_vectors': by_kind['item'],
        'default_user': defaults['user'], 'default_item': defaults['item'],
        'metadata': dict(meta, cache_hit=hit, new_encoder_calls=len(client.calls)-calls_before,
            omitted_chars=sum(sum(g['omitted_chars'].values()) for g in grouped.values()),
            empty_documents=sum(not t.strip() for g in grouped.values() for t in g['texts'].values())),
        'source_rows': {kind: result['source_rows'] for kind, result in grouped.items()}}


def assemble_features(user: str, item: str, recipe: dict, use_embeddings: bool = True) -> np.ndarray:
    """One observation -> [user one-hot,item one-hot,user vector,item vector].

    n=U+I+2d with embeddings, U+I without. Unknown IDs have zero ID blocks and
    training-mean vectors; their absence is separately reported by prediction.
    """
    n_users, n_items, d = len(recipe['users']), len(recipe['items']), recipe['dimension']
    out = np.zeros(n_users + n_items + (2*d if use_embeddings else 0))
    if user in recipe['user_index']:
        out[recipe['user_index'][user]] = 1.
    if item in recipe['item_index']:
        out[n_users + recipe['item_index'][item]] = 1.
    if use_embeddings:
        offset = n_users + n_items
        out[offset:offset+d] = recipe['user_vectors'].get(user, recipe['default_user'])
        out[offset+d:offset+2*d] = recipe['item_vectors'].get(item, recipe['default_item'])
    return out


def embedding_fm_step(x, y: float, w, V, bias: float, learning_rate: float = .03,
                       predict_fn: Callable = fm_score) -> dict:
    """One simultaneous SGD update for L=.5*(y-yhat)^2, fixed bias and zero L2.

    x,w:(n,), V:(n,K). Gradient uses the same pre-update w/V for every coordinate.
    A changed prediction callback is explicitly used in the actual residual.
    """
    if not np.isfinite(y) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Finite target and positive learning rate are required.')
    x, w, V = np.asarray(x, float), np.asarray(w, float), np.asarray(V, float)
    prediction = predict_fn(x, w, V, bias)['prediction']
    error = float(y - prediction)
    weighted = V * x[:, None]
    gradient_w = x
    gradient_V = x[:, None] * (weighted.sum(axis=0) - weighted)
    new_w = w + learning_rate * error * gradient_w
    new_V = V + learning_rate * error * gradient_V
    if not np.isfinite(new_w).all() or not np.isfinite(new_V).all():
        raise ValueError('The update diverged; inspect features and the learning rate.')
    return {'w': new_w, 'V': new_V, 'bias': float(bias), 'prediction': prediction,
            'error': error, 'loss': .5*error**2}


def fm_epoch(X, y, state: dict, order, learning_rate: float = .03,
             step_fn: Callable = embedding_fm_step, predict_fn: Callable = fm_score) -> dict:
    """Repeat the visible one-observation update; return new state, never mutate input."""
    current = {'w': state['w'].copy(), 'V': state['V'].copy(), 'bias': state['bias'],
               'updates': state.get('updates', 0)}
    for position in order:
        update = step_fn(X[position], y[position], current['w'], current['V'],
                         current['bias'], learning_rate, predict_fn=predict_fn)
        current = {'w': update['w'], 'V': update['V'], 'bias': update['bias'],
                   'updates': current['updates'] + 1}
    return current


def feature_names(recipe: dict, use_embeddings: bool = True) -> list[str]:
    names = ['user:' + u for u in recipe['users']] + ['item:' + i for i in recipe['items']]
    if use_embeddings:
        names += [f'E_{kind}:{j}' for kind in ('user', 'item') for j in range(recipe['dimension'])]
    return names


def _initial_state(names: list[str], factors: int, bias: float, seed: int) -> dict:
    if factors < 1:
        raise ValueError('FM factor count must be positive.')
    # Same named input columns get identical initial parameters across ablations.
    rows = []
    for name in names:
        key = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')
        rows.append(np.random.default_rng(key).normal(0, .02, factors))
    return {'w': np.zeros(len(names)), 'V': np.asarray(rows), 'bias': bias, 'updates': 0}


def _scores(X, state: dict, predict_fn: Callable) -> np.ndarray:
    return np.array([predict_fn(row, state['w'], state['V'], state['bias'])['prediction'] for row in X])


def fit_embedding_fm(parts: dict, recipe: dict, *, use_embeddings: bool = True, epochs: int = 25,
                      factors: int = 4, learning_rate: float = .03, seed: int = 20261111,
                      feature_fn: Callable = assemble_features, epoch_fn: Callable = fm_epoch,
                      step_fn: Callable = embedding_fm_step, predict_fn: Callable = fm_score) -> dict:
    """Fit only train, choose/restore best validation weights, never read test labels.

    No regularization/clipping here. LLM vectors stay fixed. Held-out testing is a
    separate call; changing the feature/step/score callback changes actual fitting.
    """
    if epochs < 1 or parts['train'].empty or parts['validation'].empty:
        raise ValueError('Nonempty train/validation and positive epochs are required.')
    matrices = {key: np.stack([feature_fn(r.user_id, r.item_id, recipe, use_embeddings)
                for r in parts[key].itertuples()]) for key in ('train', 'validation')}
    targets = {key: parts[key].rating.to_numpy(float) for key in ('train', 'validation')}
    names = feature_names(recipe, use_embeddings)
    state = _initial_state(names, factors, float(targets['train'].mean()), seed)
    rng = np.random.default_rng(seed)
    history, best, best_value = [], None, np.inf
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(targets['train']))
        state = epoch_fn(matrices['train'], targets['train'], state, order, learning_rate,
                         step_fn=step_fn, predict_fn=predict_fn)
        errors = {key: float(np.sqrt(np.mean((_scores(matrices[key], state, predict_fn)-targets[key])**2)))
                  for key in ('train', 'validation')}
        if not all(np.isfinite(value) for value in errors.values()):
            raise RuntimeError('Training/evaluation produced a nonfinite value.')
        history.append({'epoch': epoch, 'train_rmse': errors['train'], 'validation_rmse': errors['validation']})
        if errors['validation'] < best_value:
            best_value = errors['validation']
            best = {'w': state['w'].copy(), 'V': state['V'].copy(), 'bias': state['bias'],
                    'updates': state['updates'], 'epoch': epoch}
    return {'state': best, 'history': pd.DataFrame(history), 'recipe': recipe, 'use_embeddings': use_embeddings,
        'feature_fn': feature_fn, 'predict_fn': predict_fn, 'feature_names': names,
        'config': {'epochs': epochs, 'factors': factors, 'learning_rate': learning_rate, 'seed': seed,
                   'dimension': recipe['dimension'], 'regularization': 0., 'clip': False},
        'train_rows': tuple(parts['train'].row_id.astype(int)),
        'validation_rows': tuple(parts['validation'].row_id.astype(int))}


def predict_embedding_fm(model: dict, pairs: pd.DataFrame, *, feature_fn: Callable | None = None,
                         predict_fn: Callable | None = None) -> pd.DataFrame:
    """Actual model predictions and cold-ID flags; observed rating is not an input."""
    features, score = feature_fn or model['feature_fn'], predict_fn or model['predict_fn']
    recipe, state = model['recipe'], model['state']
    rows = []
    for row in pairs.itertuples():
        x = features(row.user_id, row.item_id, recipe, model['use_embeddings'])
        terms = score(x, state['w'], state['V'], state['bias'])
        rows.append({'user_id': row.user_id, 'item_id': row.item_id,
            'prediction': float(terms['prediction']), 'bias': float(terms['bias']),
            'linear': float(terms['linear']), 'interaction': float(terms['interaction']),
            'unknown_user': row.user_id not in recipe['user_index'],
            'unknown_item': row.item_id not in recipe['item_index']})
    return pd.DataFrame(rows)


def evaluate_embedding_fm(model: dict, rows: pd.DataFrame, *, split_name: str = 'validation') -> dict:
    predictions = predict_embedding_fm(model, rows[['user_id', 'item_id']])
    errors = predictions.prediction.to_numpy() - rows.rating.to_numpy(float)
    return {'method': '임베딩+FM' if model['use_embeddings'] else 'ID만 FM',
        'split': split_name, 'rows': len(rows), 'rmse': float(np.sqrt(np.mean(errors**2))),
        'mae': float(np.mean(np.abs(errors))), 'best_epoch': model['state']['epoch'],
        'unknown_users': int(predictions.unknown_user.sum()), 'unknown_items': int(predictions.unknown_item.sum())}


def toy_embedding_fm() -> dict:
    """Provided synthetic coordinates for arithmetic; not actual LLM embeddings."""
    recipe = {'users': ('U01', 'U02'), 'items': ('P01', 'P02', 'P03'), 'dimension': 1,
        'user_index': {'U01': 0, 'U02': 1}, 'item_index': {'P01': 0, 'P02': 1, 'P03': 2},
        'user_vectors': {'U01': np.array([.6]), 'U02': np.array([-.2])},
        'item_vectors': {'P01': np.array([.8]), 'P02': np.array([-.5]), 'P03': np.array([.1])},
        'default_user': np.array([.2]), 'default_item': np.array([.4/3])}
    x = assemble_features('U01', 'P01', recipe)
    w, V = np.zeros(7), np.zeros((7, 2))
    active = [0, 2, 5, 6]
    w[active] = [.1, .2, .3, -.1]
    V[active] = [[1, 0], [0, 1], [1, 1], [-1, 1]]
    return {'recipe': recipe, 'x': x, 'w': w, 'V': V, 'bias': 3., 'y': 4., 'active': active}
