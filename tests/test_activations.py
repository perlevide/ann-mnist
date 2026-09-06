"""Check each activation against a finite difference of itself."""

import numpy as np
import pytest

from src.scratch import activations as act


@pytest.mark.parametrize("name", ["relu", "leaky_relu", "sigmoid", "tanh", "identity"])
def test_derivative_matches_finite_difference(name):
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4, 5)) * 2.0
    z[np.abs(z) < 1e-3] = 0.5  # avoid the ReLU kink, where the derivative is undefined

    layer = act.get(name)
    layer.forward(z)
    analytic = layer.backward(np.ones_like(z))

    h = 1e-6
    numeric = (act.get(name).forward(z + h) - act.get(name).forward(z - h)) / (2 * h)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(1)
    probs = act.softmax(rng.standard_normal((7, 10)))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs > 0).all()


def test_softmax_is_shift_invariant_and_stable():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    probs = act.softmax(z)
    assert np.isfinite(probs).all()
    assert np.allclose(probs, act.softmax(z - 1000.0))


def test_log_softmax_matches_log_of_softmax():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((5, 10))
    assert np.allclose(act.log_softmax(z), np.log(act.softmax(z)), atol=1e-6)


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        act.get("swish")
