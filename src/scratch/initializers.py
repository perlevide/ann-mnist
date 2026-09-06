"""Weight initialization.

Initialization decides the scale of the signal entering each layer. Too
small and the activations collapse toward zero as they move forward, so the
gradients vanish on the way back. Too large and they blow up. The two
standard schemes both pick a variance that keeps the scale roughly constant
from layer to layer.
"""

from __future__ import annotations

import numpy as np


def zeros(shape: tuple, rng: np.random.Generator | None = None) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def normal(shape: tuple, rng: np.random.Generator, std: float = 0.01) -> np.ndarray:
    """Fixed small standard deviation. Included mostly to show why it fails
    for deep networks: the scale is independent of the layer width."""
    return (rng.standard_normal(shape) * std).astype(np.float32)


def xavier(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Glorot initialization, for tanh and sigmoid.

    Variance 2 / (fan_in + fan_out), which balances the forward variance
    against the backward variance. Drawn here from the uniform distribution
    on [-limit, limit] with limit = sqrt(6 / (fan_in + fan_out)).
    """
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def he(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Kaiming initialization, for ReLU.

    ReLU zeroes about half its inputs, so it halves the variance of what
    passes through. Compensating with variance 2 / fan_in keeps the forward
    signal at constant scale through a deep stack.
    """
    fan_in = shape[0]
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


REGISTRY = {"zeros": zeros, "normal": normal, "xavier": xavier, "he": he}


def get(name: str):
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown initializer {name!r}, choose from {sorted(REGISTRY)}") from None


def default_for(activation: str) -> str:
    """He for ReLU family, Xavier for the saturating functions."""
    return "he" if activation in ("relu", "leaky_relu") else "xavier"
