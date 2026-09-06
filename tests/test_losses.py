"""Cross entropy: known values, and the gradient identity p - y."""

import numpy as np

from src.scratch.activations import softmax
from src.scratch.losses import MeanSquaredError, SoftmaxCrossEntropy, l2_penalty


def test_uniform_logits_give_log_of_n_classes():
    logits = np.zeros((3, 10))
    loss = SoftmaxCrossEntropy().forward(logits, np.array([0, 5, 9]))
    assert abs(loss - np.log(10)) < 1e-6


def test_confident_correct_prediction_has_small_loss():
    logits = np.full((1, 10), -10.0)
    logits[0, 3] = 10.0
    assert SoftmaxCrossEntropy().forward(logits, np.array([3])) < 1e-6


def test_gradient_equals_probabilities_minus_one_hot():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, 10))
    targets = rng.integers(0, 10, size=6)

    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(logits, targets)
    analytic = loss_fn.backward()

    expected = softmax(logits)
    expected[np.arange(6), targets] -= 1.0
    expected /= 6
    assert np.allclose(analytic, expected)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((4, 10))
    targets = rng.integers(0, 10, size=4)

    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(logits, targets)
    analytic = loss_fn.backward()

    h = 1e-6
    numeric = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            up, down = logits.copy(), logits.copy()
            up[i, j] += h
            down[i, j] -= h
            numeric[i, j] = (
                SoftmaxCrossEntropy().forward(up, targets)
                - SoftmaxCrossEntropy().forward(down, targets)
            ) / (2 * h)
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_mse_gradient():
    rng = np.random.default_rng(2)
    predictions = rng.standard_normal((5, 3))
    targets = rng.standard_normal((5, 3))
    loss_fn = MeanSquaredError()
    loss_fn.forward(predictions, targets)
    assert np.allclose(loss_fn.backward(), (predictions - targets) / 5)


def test_l2_penalty_is_zero_when_disabled():
    assert l2_penalty([np.ones((3, 3))], 0.0) == 0.0
    assert abs(l2_penalty([np.ones((3, 3))], 2.0) - 9.0) < 1e-9
