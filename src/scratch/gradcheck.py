"""Numerical gradient checking.

Backpropagation is easy to get subtly wrong: a missing transpose, a sum over
the wrong axis, a factor of N. The check here compares the analytic gradient
against a finite difference estimate of the same quantity.

The central difference

    df/dw ~ (f(w + h) - f(w - h)) / (2h)

has error O(h^2), while the one sided version has error O(h). The two extra
evaluations are worth it. The two gradients are compared with the relative
error

    |a - n| / max(|a| + |n|, eps)

which stays meaningful whether the gradients are large or tiny. Below 1e-7
the implementation is almost certainly correct, and anything above 1e-4
deserves attention.
"""

from __future__ import annotations

import numpy as np


def relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), eps)))


def numerical_gradient(loss_fn, param: np.ndarray, h: float = 1e-5, indices=None) -> np.ndarray:
    """Finite difference gradient of `loss_fn()` with respect to `param`.

    `param` is modified in place and restored, so `loss_fn` must read the
    same array object. Checking every entry of a large weight matrix is slow,
    so `indices` can restrict the check to a random sample.
    """
    grad = np.zeros_like(param)
    flat = param.ravel()
    flat_grad = grad.ravel()
    positions = range(flat.size) if indices is None else indices

    for i in positions:
        original = flat[i]
        flat[i] = original + h
        loss_plus = loss_fn()
        flat[i] = original - h
        loss_minus = loss_fn()
        flat[i] = original
        flat_grad[i] = (loss_plus - loss_minus) / (2 * h)

    return grad


def check_model(model, x: np.ndarray, y: np.ndarray, n_samples: int = 20, h: float = 1e-5, seed: int = 0) -> dict:
    """Check every parameter array of an MLP on a small batch.

    Use float64 inputs. In float32 the rounding noise in f(w + h) is of the
    same order as the difference being measured and the check reports a
    false failure.
    """
    rng = np.random.default_rng(seed)
    results = {}

    logits = model.forward(x, training=False)
    model.loss_fn.forward(logits, y)
    model.backward(model.loss_fn.backward())

    analytic = {}
    for index, layer in enumerate(model.layers):
        for key, grad in layer.gradients().items():
            analytic[(index, key)] = grad.copy()

    for index, layer in enumerate(model.layers):
        for key, param in layer.parameters().items():
            def loss_fn():
                return model.loss_fn.forward(model.forward(x, training=False), y)

            picks = rng.choice(param.size, size=min(n_samples, param.size), replace=False)
            numeric = numerical_gradient(loss_fn, param, h, indices=picks)
            a = analytic[(index, key)].ravel()[picks]
            n = numeric.ravel()[picks]
            results[f"layer{index}.{key}"] = relative_error(a, n)

    return results
