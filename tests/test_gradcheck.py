"""The test that matters: does backprop agree with finite differences?

If this passes, the whole backward path is consistent. If it fails, the bug
is in a layer, not in the training loop.
"""

import numpy as np

from src.scratch.gradcheck import check_model, relative_error
from src.scratch.network import MLP


def make_batch(n=8, d=12, c=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)), rng.integers(0, c, size=n)


def to_float64(model):
    """Finite differences need double precision to be meaningful."""
    for layer in model.layers:
        for key, value in layer.parameters().items():
            setattr(layer, key, value.astype(np.float64))
    return model


def test_relu_network_gradients():
    x, y = make_batch()
    model = to_float64(MLP([12, 16, 8, 4], activation="relu", seed=0))
    errors = check_model(model, x, y, n_samples=25)
    assert max(errors.values()) < 1e-6, errors


def test_tanh_network_gradients():
    x, y = make_batch(seed=1)
    model = to_float64(MLP([12, 10, 4], activation="tanh", seed=1))
    errors = check_model(model, x, y, n_samples=25)
    assert max(errors.values()) < 1e-6, errors


def test_sigmoid_network_gradients():
    x, y = make_batch(seed=2)
    model = to_float64(MLP([12, 10, 4], activation="sigmoid", seed=2))
    errors = check_model(model, x, y, n_samples=25)
    assert max(errors.values()) < 1e-6, errors


def test_deep_network_gradients():
    x, y = make_batch(seed=3)
    model = to_float64(MLP([12, 16, 16, 16, 4], activation="relu", seed=3))
    errors = check_model(model, x, y, n_samples=15)
    assert max(errors.values()) < 1e-6, errors


def test_relative_error_helper():
    a = np.array([1.0, 2.0])
    assert relative_error(a, a) == 0.0
    assert relative_error(a, -a) > 0.5
