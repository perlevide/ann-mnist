"""Activation functions and their derivatives.

Each class keeps whatever it needs from the forward pass so that `backward`
can compute the local derivative without recomputing the input. The
convention throughout the package is that `backward` receives dL/d(output)
and returns dL/d(input).
"""

from __future__ import annotations

from ..backend import xp as np


class Activation:
    """Elementwise nonlinearity. Holds no trainable parameters."""

    def forward(self, z: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> dict:
        return {}

    def gradients(self) -> dict:
        return {}

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    def __repr__(self):
        return f"{type(self).__name__}()"


class ReLU(Activation):
    """max(0, z).

    Derivative is 1 where z > 0 and 0 elsewhere. At z = 0 the function is
    not differentiable and the subgradient 0 is used, which costs nothing in
    practice because exact zeros are rare in floating point.
    """

    def forward(self, z, training=True):
        self.mask = z > 0
        return np.where(self.mask, z, 0.0)

    def backward(self, grad_out):
        return grad_out * self.mask


class LeakyReLU(Activation):
    """max(alpha * z, z), with a small alpha.

    Keeps a nonzero gradient on the negative side so that a unit pushed
    below zero can recover instead of staying dead for the rest of training.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, z, training=True):
        self.mask = z > 0
        return np.where(self.mask, z, self.alpha * z)

    def backward(self, grad_out):
        return grad_out * np.where(self.mask, 1.0, self.alpha)


class Sigmoid(Activation):
    """1 / (1 + exp(-z)), with derivative s(1 - s).

    The largest possible derivative is 0.25 at z = 0, so gradients shrink by
    at least a factor of four per sigmoid layer. Stack a few and the early
    layers stop learning. This is the vanishing gradient problem that ReLU
    was introduced to avoid.
    """

    def forward(self, z, training=True):
        self.out = np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
            np.exp(np.clip(z, -500, 500)) / (1.0 + np.exp(np.clip(z, -500, 500))),
        )
        return self.out

    def backward(self, grad_out):
        return grad_out * self.out * (1.0 - self.out)


class Tanh(Activation):
    """tanh(z), with derivative 1 - tanh(z)^2.

    Zero centered, unlike the sigmoid, which keeps the mean activation near
    zero and makes the next layer's job easier.
    """

    def forward(self, z, training=True):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad_out):
        return grad_out * (1.0 - self.out ** 2)


class Identity(Activation):
    """Pass through. Useful as the last layer when the loss handles softmax."""

    def forward(self, z, training=True):
        return z

    def backward(self, grad_out):
        return grad_out


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Row wise softmax, computed in a numerically safe way.

    Subtracting the row maximum before the exponential leaves the result
    unchanged, because exp(z - m) / sum(exp(z - m)) equals exp(z) /
    sum(exp(z)), but it keeps every exponent at or below zero so nothing
    overflows.
    """
    shifted = z - np.max(z, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(z)) without forming the softmax first."""
    shifted = z - np.max(z, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


REGISTRY = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "identity": Identity,
}


def get(name: str) -> Activation:
    """Look up an activation by name, as used in TrainConfig."""
    try:
        return REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}, choose from {sorted(REGISTRY)}") from None
