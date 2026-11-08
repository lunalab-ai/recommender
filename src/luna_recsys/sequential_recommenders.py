"""W11A: prefix classification, inspectable attention, and next-item rollout.

Independent teaching implementation. MovieLens time means rating-record time;
ties keep file order, which does not establish actual viewing order. No data is
downloaded by this module. TensorFlow is imported only when a model is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import time
import numpy as np
import pandas as pd


def scaled_attention(q: Any, k: Any, v: Any, allowed: Any = None) -> dict:
    """(T,D),(S,D),(S,V) -> scores/weights/output; True permits a key.

    A fully masked query has zero weights/output. Does not learn or mutate inputs.
    Example: scaled_attention(np.eye(2),np.eye(2),np.eye(2))['weights'].
    """
    q, k, v = [np.asarray(x, dtype=float) for x in (q, k, v)]
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[1] != k.shape[1] or len(k) != len(v) or not q.shape[1]:
        raise ValueError('Q/K/V must have compatible two-dimensional shapes.')
    if not all(np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError('Q/K/V must be finite.')
    scores = q @ k.T / np.sqrt(q.shape[1])
    mask = np.ones(scores.shape, bool) if allowed is None else np.broadcast_to(np.asarray(allowed, bool), scores.shape)
    masked = np.where(mask, scores, -1e9)
    unnormalized = np.exp(masked - masked.max(axis=-1, keepdims=True)) * mask
    denominator = unnormalized.sum(axis=-1, keepdims=True)
    weights = np.divide(unnormalized, denominator, out=np.zeros_like(unnormalized), where=denominator > 0)
    return {'scores': scores, 'weights': weights, 'output': weights @ v}


def pad_history(history: Any, maxlen: int) -> np.ndarray:
    """Right-pad/truncate the most recent positive internal IDs. New int32 array."""
    ids = np.asarray(history, dtype=np.int32)
    if ids.ndim != 1 or not len(ids) or np.any(ids <= 0) or maxlen < 1:
        raise ValueError('A nonempty positive-ID history and maxlen>=1 are required.')
    out = np.zeros(maxlen, dtype=np.int32)
    tail = ids[-maxlen:]
    out[:len(tail)] = tail
    return out


def prefix_pairs(history: Any, maxlen: int = 20, min_prefix: int = 20) -> tuple:
    """Observed prefix -> next label; never includes that label in its input.

    A history of length n yields max(0,n-min_prefix) rows. No shuffle or learning.
    """
    ids = list(history)
    if min_prefix < 1:
        raise ValueError('min_prefix must be positive.')
    xs = [pad_history(ids[:end], maxlen) for end in range(min_prefix, len(ids))]
    ys = ids[min_prefix:]
    return np.asarray(xs, dtype=np.int32).reshape(-1, maxlen), np.asarray(ys, dtype=np.int32)


@dataclass
class SequenceData:
    x: dict
    y: dict
    users: dict
    histories: dict
    targets: dict
    encode: dict
    decode: dict
    titles: dict
    popularity: np.ndarray
    maxlen: int
    mode: str
    protocol: dict


def prepare_sequence_data(prepared: Any, *, maxlen: int = 20, max_users: int = 120,
                          max_train_pairs: int = 1500, seed: int = 20261110,
                          min_prefix: int = 20, holdout: int = 10,
                          pair_fn: Callable = prefix_pairs) -> SequenceData:
    """Split users before prefix augmentation, then cap rows for a CPU lesson.

    75% train pool/25% test; 20% of train pool is validation (about60/15/25).
    Vocabulary uses the item catalog, not future interaction labels. Eligible
    users need > min_prefix+holdout records. Test targets never enter popularity
    or fit. User limit and sample cap change the book experiment population.
    """
    frame = prepared.data.ratings.copy()
    needed = {'user_id', 'movie_id', 'timestamp'}
    if not needed.issubset(frame.columns) or maxlen < 1 or max_train_pairs < 1:
        raise ValueError('Required rating schema and positive limits are missing.')
    if frame.duplicated(['user_id', 'movie_id']).any():
        raise ValueError('This lesson expects unique user-item rating records.')
    frame['_row'] = np.arange(len(frame))
    frame = frame.sort_values(['user_id', 'timestamp', '_row'], kind='stable')
    item_ids = sorted(int(i) for i in prepared.data.movies.movie_id.unique())
    encode = {item: j+1 for j, item in enumerate(item_ids)}
    decode = {j: item for item, j in encode.items()}
    all_histories = {int(u): [encode[int(i)] for i in g.movie_id]
                     for u, g in frame.groupby('user_id', sort=True)
                     if len(g) > min_prefix + holdout}
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(sorted(all_histories))[:max_users]
    if len(chosen) < 8:
        raise ValueError('At least eight eligible users are needed; no synthetic fallback.')
    n_pool = int(.75 * len(chosen))
    pool, test = chosen[:n_pool], chosen[n_pool:]
    n_val = max(1, int(.2 * len(pool)))
    users = {'train': pool[n_val:].tolist(), 'validation': pool[:n_val].tolist(), 'test': test.tolist()}
    xs, ys = {}, {}
    for split in ('train', 'validation'):
        pairs = [pair_fn(all_histories[u], maxlen, min_prefix) for u in users[split]]
        x, y = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
        limit = max_train_pairs if split == 'train' else min(500, max_train_pairs)
        take = rng.permutation(len(y))[:limit]
        xs[split], ys[split] = x[take], y[take]
    histories = {u: all_histories[u][:-holdout] for u in users['test']}
    targets = {u: all_histories[u][-holdout:] for u in users['test']}
    popularity = np.zeros(len(encode)+1)
    for u in users['train']:
        np.add.at(popularity, all_histories[u], 1)
    titles = dict(zip(prepared.data.movies.movie_id.astype(int), prepared.data.movies.title))
    tied = int(frame.duplicated(['user_id', 'timestamp']).sum())
    protocol = {'seed': seed, 'max_users': max_users, 'min_prefix': min_prefix,
                'holdout': holdout, 'train_pairs': len(ys['train']), 'validation_pairs': len(ys['validation']),
                'timestamp': 'rating-record time; tied timestamps keep original file row order',
                'same_user_tied_extra_rows': tied, 'split': 'disjoint users, before augmentation; not global temporal split',
                'candidates': 'exclude0, duplicates, observed history; never consult held-out labels'}
    return SequenceData(xs, ys, users, histories, targets, encode, decode, titles,
                        popularity, maxlen, prepared.mode, protocol)


def build_sequence_model(vocabulary: int, *, maxlen: int = 20, embed_dim: int = 16,
                         heads: int = 2, key_dim: int = 8, ff_dim: int = 32,
                         blocks: int = 1, dense_dim: int = 64, dropout: float = .1,
                         seed: int = 20261110) -> tuple:
    """Item+position -> encoder(s) -> Flatten/Dense -> next-item softmax.

    Return (classifier, first-block attention probe). Explicit key padding mask
    and post-block zeroing avoid relying on custom-layer mask propagation.
    Prefix-to-one-label task permits attention throughout the observed prefix.
    key_dim is per head; output width returns to embed_dim via projection.
    """
    import tensorflow as tf
    import keras
    if min(vocabulary, maxlen, embed_dim, heads, key_dim, ff_dim, blocks, dense_dim) < 1:
        raise ValueError('All model dimensions must be positive.')
    keras.utils.set_random_seed(seed)
    ids = keras.Input((maxlen,), dtype='int32', name='internal_item_ids')
    valid = keras.layers.Lambda(lambda z: tf.not_equal(z, 0), name='valid_items')(ids)
    positions = keras.layers.Lambda(lambda z: tf.broadcast_to(tf.range(maxlen)[None, :], tf.shape(z)), name='positions')(ids)
    x = keras.layers.Embedding(vocabulary, embed_dim, name='item_embedding')(ids)
    pos = keras.layers.Embedding(maxlen, embed_dim, name='position_embedding')(positions)
    x = keras.layers.Add()([x, pos])
    multiplier = keras.layers.Lambda(lambda z: tf.cast(z[..., None], tf.float32), name='pad_multiplier')(valid)
    x = keras.layers.Multiply()([x, multiplier])
    attention_mask = keras.layers.Lambda(lambda z: z[:, None, :], name='key_padding_mask')(valid)
    first_scores = None
    for j in range(blocks):
        attention = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, name=f'attention_{j}')
        attended, scores = attention(x, x, attention_mask=attention_mask, return_attention_scores=True)
        if first_scores is None:
            first_scores = scores
        attended = keras.layers.Dropout(dropout)(attended)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(keras.layers.Add()([x, attended]))
        f = keras.layers.Dense(ff_dim, activation='relu')(x)
        f = keras.layers.Dense(embed_dim)(f)
        f = keras.layers.Dropout(dropout)(f)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(keras.layers.Add()([x, f]))
        x = keras.layers.Multiply()([x, multiplier])
    flat = keras.layers.Flatten()(x)
    hidden = keras.layers.Dense(dense_dim, activation='relu')(flat)
    probs = keras.layers.Dense(vocabulary, activation='softmax', name='next_item_probabilities')(hidden)
    return keras.Model(ids, probs), keras.Model(ids, first_scores)


def sequence_train_step(model: Any, optimizer: Any, x: Any, y: Any, loss_fn: Callable = None) -> dict:
    """One real update; integer labels(B,), probs(B,V), CE mean over rows.

    loss_fn(y,probabilities) may be supplied from an editable notebook cell.
    Changes weights/optimizer state. Returns scalar loss and gradient norm.
    """
    import tensorflow as tf
    import keras
    loss_fn = loss_fn or keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        probs = model(x, training=True)
        loss = tf.reduce_mean(loss_fn(y, probs))
        if model.losses:
            loss = loss + tf.add_n(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    if not bool(tf.math.is_finite(loss)) or any(g is None or not bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(g)))) for g in grads):
        raise FloatingPointError('Nonfinite loss/gradient; no fallback.')
    norm = float(tf.linalg.global_norm(grads))
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return {'loss': float(loss), 'gradient_norm': norm}


@dataclass
class SequenceFit:
    model: Any
    probe: Any
    history: pd.DataFrame
    best_epoch: int
    updates: int
    seconds: float
    config: dict


def fit_sequence(data: SequenceData, *, epochs: int = 3, batch_size: int = 64,
                 learning_rate: float = .0003, seed: int = 20261110,
                 step_fn: Callable = sequence_train_step, loss_fn: Callable = None,
                 model_fn: Callable = build_sequence_model,
                 **model_options) -> SequenceFit:
    """CPU-sized real training; select weights by validation CE only.

    Explicit step_fn/loss_fn ensures notebook changes affect the actual fit.
    No test-label use. Book-size architecture is an optional model_options profile.
    """
    import keras
    import inspect
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('Positive training controls required.')
    model, probe = model_fn(len(data.encode)+1, maxlen=data.maxlen, seed=seed, **model_options)
    opt = keras.optimizers.Adam(learning_rate)
    rng = np.random.default_rng(seed)
    rows, best, weights, best_epoch, updates = [], np.inf, None, 0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(data.y['train']))
        total = 0.
        for j in range(0, len(order), batch_size):
            take = order[j:j+batch_size]
            result = step_fn(model, opt, data.x['train'][take], data.y['train'][take], loss_fn=loss_fn)
            total += result['loss'] * len(take)
            updates += 1
        p = np.asarray(model(data.x['validation'], training=False))
        val = float(np.mean(-np.log(np.maximum(p[np.arange(len(p)), data.y['validation']], 1e-12))))
        rows.append({'epoch': epoch, 'train_ce': total/len(order), 'validation_ce': val})
        if val < best:
            best, best_epoch = val, epoch
            weights = model.get_weights()
    model.set_weights(weights)
    names = {'embed_dim', 'heads', 'key_dim', 'ff_dim', 'blocks', 'dense_dim', 'dropout'}
    declared = {k: p.default for k, p in inspect.signature(model_fn).parameters.items()
                if k in names and p.default is not inspect.Parameter.empty}
    declared.update(model_options)
    return SequenceFit(model, probe, pd.DataFrame(rows), best_epoch, updates,
                       time.perf_counter()-started, {'seed': seed, 'maxlen': data.maxlen,
                       'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                       'parameters': model.count_params(), 'mode': data.mode, **declared})


def select_candidates(scores: Any, history: Any, k: int = 1, *, exclude_seen: bool = True) -> list:
    """Rank internal IDs; exclude0/nonfinite and optionally already observed IDs.

    Stable ID-order tie break. No target-label access; does not mutate scores.
    """
    scores = np.asarray(scores, dtype=float).copy()
    if scores.ndim != 1 or k < 1:
        raise ValueError('A score vector and k>=1 are needed.')
    permitted = np.isfinite(scores)
    permitted[0] = False
    if exclude_seen:
        seen = np.asarray(list(history), dtype=int)
        if len(seen) and (seen.min() < 1 or seen.max() >= len(scores)):
            raise ValueError('History must contain internal vocabulary IDs.')
        permitted[seen] = False
    ids = np.flatnonzero(permitted)
    return ids[np.argsort(-scores[ids], kind='stable')[:k]].tolist()


def rollout(score_fn: Callable, history: Any, *, maxlen: int, n: int = 7,
            candidate_fn: Callable = select_candidates) -> list:
    """Append own predicted INTERNAL IDs to a copied history, never future truth.

    score_fn receives one padded array(1,T) and returns(1,V) or(V,).
    candidate_fn(scores,history,k) controls actual later recommendation/app output.
    """
    current = list(history)
    if n < 1:
        raise ValueError('n must be positive.')
    chosen = []
    for _ in range(n):
        scores = np.asarray(score_fn(pad_history(current, maxlen)[None, :])).reshape(-1)
        ids = candidate_fn(scores, current, 1)
        if not ids:
            raise ValueError('No eligible candidates remain.')
        item = int(ids[0])
        if item <= 0 or item >= len(scores):
            raise ValueError('Candidate is not a valid internal ID.')
        chosen.append(item)
        current.append(item)
    return chosen


def ranking_metrics(recommended: Any, target: Any) -> dict:
    """Set hits / returned-list length and unique held-out targets; not positional accuracy."""
    rec, truth = list(recommended), set(target)
    if not rec or not truth or len(set(rec)) != len(rec) or 0 in rec:
        raise ValueError('A nonempty unique recommendation and nonempty target set are required.')
    hits = len(set(rec) & truth)
    precision, recall = hits/len(rec), hits/len(truth)
    return {'hits': hits, 'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if hits else 0.}


def evaluate_sequence(data: SequenceData, score_fn: Callable, *, n: int = 7,
                      candidate_fn: Callable = select_candidates) -> pd.DataFrame:
    """Each test user's own history/targets; return per-user metrics and equal-user macro row externally."""
    rows = []
    for user in data.users['test']:
        rec = rollout(score_fn, data.histories[user], maxlen=data.maxlen, n=n, candidate_fn=candidate_fn)
        rows.append({'user_id': user, **ranking_metrics(rec, data.targets[user])})
    return pd.DataFrame(rows)
