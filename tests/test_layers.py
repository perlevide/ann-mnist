"""Linear and Dropout shapes and gradients."""

import numpy as np

from src.scratch.layers import Dropout, Linear


def test_linear_forward_shape_and_value():
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    out = layer.forward(x)
    assert out.shape == (5, 3)
    assert np.allclose(out, x @ layer.W + layer.b, atol=1e-5)


def test_linear_backward_shapes():
    rng = np.random.default_rng(1)
    layer = Linear(4, 3, rng)
    x = rng.standard_normal((5, 4)).astype(np.float32)
    layer.forward(x)
    grad_in = layer.backward(np.ones((5, 3), dtype=np.float32))
    assert grad_in.shape == x.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape


def test_linear_gradients_match_finite_difference():
    rng = np.random.default_rng(2)
    layer = Linear(3, 2, rng)
    layer.W = layer.W.astype(np.float64)
    layer.b = layer.b.astype(np.float64)
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 2))

    layer.forward(x)
    layer.backward(upstream)

    h = 1e-6
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            original = layer.W[i, j]
            layer.W[i, j] = original + h
            up = np.sum(layer.forward(x) * upstream)
            layer.W[i, j] = original - h
            down = np.sum(layer.forward(x) * upstream)
            layer.W[i, j] = original
            assert abs((up - down) / (2 * h) - layer.dW[i, j]) < 1e-5


def test_dropout_is_identity_in_eval_mode():
    rng = np.random.default_rng(3)
    layer = Dropout(0.5, rng)
    x = rng.standard_normal((100, 20))
    assert np.array_equal(layer.forward(x, training=False), x)


def test_inverted_dropout_preserves_expectation():
    rng = np.random.default_rng(4)
    layer = Dropout(0.5, rng)
    x = np.ones((5000, 20))
    out = layer.forward(x, training=True)
    assert abs(out.mean() - 1.0) < 0.05
    assert (out == 0).mean() > 0.4
