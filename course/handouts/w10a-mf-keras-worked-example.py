"""MF worked example: four trainable tables, exact gradients, one SGD step.

Teaching demonstration, not an assignment solution or a performance experiment.
The MF graph preserves the public 2026-fall-w10a build_mf structure.
Run: python w10a-mf-keras-worked-example.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

keras = tf.keras
NAMES = ("user_embedding", "item_embedding", "user_bias", "item_bias")


def build_mf(n_users, n_items, *, k=8, reg=0.0002, seed=20261103):
    """Same MF graph as the course API; seed helper is inlined here."""
    if min(n_users, n_items, k) < 1 or reg < 0:
        raise ValueError("Positive table sizes/k and nonnegative reg required")
    keras.utils.set_random_seed(seed)
    regularizer = keras.regularizers.l2(reg)
    u = keras.Input((1,), dtype="int32", name="user_id")
    i = keras.Input((1,), dtype="int32", name="movie_id")
    p = keras.layers.Embedding(n_users, k, embeddings_regularizer=regularizer,
                               name="user_embedding")(u)
    q = keras.layers.Embedding(n_items, k, embeddings_regularizer=regularizer,
                               name="item_embedding")(i)
    bu = keras.layers.Embedding(n_users, 1, embeddings_regularizer=regularizer,
                                name="user_bias")(u)
    bi = keras.layers.Embedding(n_items, 1, embeddings_regularizer=regularizer,
                                name="item_bias")(i)
    dot = keras.layers.Dot(axes=2, name="mf_dot")([p, q])
    added = keras.layers.Add(name="mf_add")([dot, bu, bi])
    output = keras.layers.Flatten(name="centered_rating")(added)
    return keras.Model([u, i], output, name="keras_mf")


def demo_model(reg=0.0):
    """Create a fresh model and set explicit teaching values in all four tables."""
    model = build_mf(3, 4, k=2, reg=reg)
    tables = {
        "user_embedding": [[0, 0], [1, 1], [0.4, 0.2]],
        "item_embedding": [[0, 0], [1, 0], [0, 1], [0.5, 0.5]],
        "user_bias": [[0], [0.5], [0.1]],
        "item_bias": [[0], [-0.5], [0.2], [0]],
    }
    for name, values in tables.items():
        model.get_layer(name).set_weights([np.array(values, dtype="float32")])
    return model


def tables(model):
    return {name: model.get_layer(name).get_weights()[0].copy() for name in NAMES}


def dense_gradients(model, inputs, target, include_regularization=False):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = tf.reduce_mean(tf.square(prediction - target))
        if include_regularization:
            loss += tf.add_n(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    result = {}
    for name in NAMES:
        weight = model.get_layer(name).trainable_weights[0]
        gradient = next(g for v, g in zip(model.trainable_variables, gradients)
                        if v is weight)
        result[name] = tf.convert_to_tensor(gradient).numpy()
    return result


def run_demo():
    # This is the full tiny training set; its mean is computed once, not per batch.
    train_ratings = np.array([5.0, 1.0], dtype="float32")
    mean = float(train_ratings.mean())
    inputs = {"user_id": np.array([[1]], dtype="int32"),
              "movie_id": np.array([[1]], dtype="int32")}
    target = np.array([[train_ratings[0] - mean]], dtype="float32")
    pairs = {"user_id": np.array([[1], [1], [2], [2]], dtype="int32"),
             "movie_id": np.array([[1], [2], [1], [2]], dtype="int32")}
    labels = ["A-X", "A-Y", "B-X", "B-Y"]

    model = demo_model()
    before = tables(model)
    before_predictions = np.asarray(model(pairs, training=False)).ravel() + mean
    gradients = dense_gradients(model, inputs, target)
    probe = keras.Model(model.inputs, [model.get_layer(name).output for name in
                        ("user_embedding", "item_embedding", "user_bias",
                         "item_bias", "mf_dot", "mf_add", "centered_rating")])
    shapes = {name: list(value.shape) for name, value in zip(
        ("p", "q", "bu", "bi", "dot", "added", "output"), probe(inputs))}

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.0),
                  loss="mse")
    loss_before_update = float(model.train_on_batch(inputs, target))
    after = tables(model)
    after_predictions = np.asarray(model(pairs, training=False)).ravel() + mean
    centered_after = float(np.asarray(model(inputs, training=False))[0, 0])
    snapshot = tables(model)
    model(pairs, training=False)
    inference_unchanged = all(np.array_equal(snapshot[n], tables(model)[n]) for n in NAMES)

    expected = {
        "user_embedding": [[0, 0], [1.2, 1.0], [0.4, 0.2]],
        "item_embedding": [[0, 0], [1.2, 0.2], [0, 1], [0.5, 0.5]],
        "user_bias": [[0], [0.7], [0.1]],
        "item_bias": [[0], [-0.3], [0.2], [0]],
    }
    for name, value in expected.items():
        np.testing.assert_allclose(after[name], value, rtol=0, atol=2e-6)
    np.testing.assert_allclose(gradients["user_embedding"][1], [-2, 0], atol=2e-6)
    np.testing.assert_allclose(gradients["item_embedding"][1], [-2, -2], atol=2e-6)
    np.testing.assert_allclose(before_predictions, [4, 4.7, 3, 3.5], atol=2e-6)
    np.testing.assert_allclose(after_predictions, [5.04, 4.9, 3.32, 3.5], atol=2e-6)
    assert model.count_params() == 21 and len(model.trainable_weights) == 4
    assert abs(loss_before_update - 1) < 2e-6 and inference_unchanged

    # Fresh, separate model: L2 affects unused rows too. Never train this in place
    # of the reg=0 model when comparing the hand calculation above.
    regularized = demo_model(reg=0.0002)
    regularized(inputs)
    penalty = float(tf.add_n(regularized.losses))
    reg_gradients = dense_gradients(regularized, inputs, target, True)
    np.testing.assert_allclose(penalty, 0.00105, atol=2e-8)
    np.testing.assert_allclose(reg_gradients["user_embedding"][2], [0.00016, 0.00008], atol=2e-8)

    result = {
        "versions": {"python": sys.version.split()[0], "tensorflow": tf.__version__,
                     "keras": keras.__version__, "numpy": np.__version__},
        "mean": mean, "target_centered": float(target[0, 0]),
        "trainable_tensors": len(model.trainable_weights), "parameters": model.count_params(),
        "shapes": shapes, "before": {n: v.tolist() for n, v in before.items()},
        "after": {n: v.tolist() for n, v in after.items()},
        "gradients": {n: v.tolist() for n, v in gradients.items()},
        "loss_before_update": loss_before_update,
        "centered_after": centered_after,
        "rating_after": centered_after + mean,
        "loss_after_update": (centered_after - float(target[0, 0])) ** 2,
        "predictions": [{"pair": label, "before": float(b), "after": float(a)}
                        for label, b, a in zip(labels, before_predictions, after_predictions)],
        "inference_unchanged": inference_unchanged,
        "l2_penalty": penalty,
        "unused_B_gradient_with_l2": reg_gradients["user_embedding"][2].tolist(),
        "default_count_81_19_k8": build_mf(81, 19, k=8).count_params(),
        "status": "MF_KERAS_WORKED_EXAMPLE_PASS",
    }
    return result


if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
