"""Optimizers should reduce a quadratic, and Adam's bias correction should show."""

import numpy as np

from src.scratch.optimizers import SGD, Adam, RMSProp, build


def descend(optimizer, steps=200):
    """Minimize f(w) = sum(w^2) / 2, whose gradient is w and whose minimum is 0."""
    w = np.array([3.0, -4.0])
    for _ in range(steps):
        optimizer.step([(w, w.copy(), True)])
    return np.linalg.norm(w)


def test_sgd_converges():
    assert descend(SGD(0.1)) < 1e-3


def test_momentum_is_faster_than_plain_sgd():
    plain = descend(SGD(0.05), steps=40)
    with_momentum = descend(SGD(0.05, momentum=0.9), steps=40)
    assert with_momentum < plain


def test_adam_converges():
    assert descend(Adam(0.1), steps=400) < 1e-2


def test_rmsprop_converges_to_a_floor_set_by_the_learning_rate():
    """RMSProp normalizes the step to roughly the learning rate in size.

    That is the point of the rule, and it means the iterate settles into a
    band of about that width around the minimum instead of landing on it.
    A smaller learning rate gives a tighter band.
    """
    coarse = descend(RMSProp(0.05), steps=400)
    fine = descend(RMSProp(0.005), steps=2000)
    assert coarse < 0.1
    assert fine < coarse


def test_adam_first_step_size_is_close_to_the_learning_rate():
    """After bias correction the first update is about lr, not lr * (1 - beta1)."""
    w = np.array([1.0])
    Adam(0.01).step([(w, np.array([1.0]), True)])
    assert abs(abs(1.0 - w[0]) - 0.01) < 1e-3


def test_weight_decay_only_touches_weights():
    weight = np.array([1.0])
    bias = np.array([1.0])
    optimizer = SGD(0.1, weight_decay=1.0)
    optimizer.step([(weight, np.zeros(1), True), (bias, np.zeros(1), False)])
    assert weight[0] < 1.0
    assert bias[0] == 1.0


def test_build_names():
    for name in ("sgd", "nesterov", "rmsprop", "adam"):
        assert build(name, 0.01) is not None
