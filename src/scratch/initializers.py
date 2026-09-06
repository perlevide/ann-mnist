"""Weight initialization.

Scalar arithmetic here uses `math`, not the array module. `np.sqrt(2.0 /
fan_in)` on a Python float asks the array module to launch a kernel for a
single square root, which on CuPy means compiling one. Keep scalars on the
host and let the array module handle arrays.

Initialization decides the scale of the signal entering each layer. Too
small and the activations collapse toward zero as they move forward, so the
gradients vanish on the way back. Too large and they blow up. The two
standard schemes both pick a variance that keeps the scale roughly constant
from layer to layer.
"""

from __future__ import annotations

import math

from ..backend import xp as np


def zeros(shape: tuple, rng=None) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def normal(shape: tuple, rng, std: float = 0.01) -> np.ndarray:
    """Fixed small standard deviation. Included mostly to show why it fails
    for deep networks: the scale is independent of the layer width."""
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(std)


def xavier(shape: tuple, rng) -> np.ndarray:
    """Glorot initialization, for tanh and sigmoid.

    Variance 2 / (fan_in + fan_out), which balances the forward variance
    against the backward variance. Drawn here from the uniform distribution
    on [-limit, limit] with limit = sqrt(6 / (fan_in + fan_out)).
    """
    fan_in, fan_out = shape
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def he(shape: tuple, rng) -> np.ndarray:
    """Kaiming initialization, for ReLU.

    ReLU zeroes about half its inputs, so it halves the variance of what
    passes through. Compensating with variance 2 / fan_in keeps the forward
    signal at constant scale through a deep stack.
    """
    fan_in = shape[0]
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(math.sqrt(2.0 / fan_in))


REGISTRY = {"zeros": zeros, "normal": normal, "xavier": xavier, "he": he}


def get(name: str):
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown initializer {name!r}, choose from {sorted(REGISTRY)}") from None


def default_for(activation: str) -> str:
    """He for ReLU family, Xavier for the saturating functions."""
    return "he" if activation in ("relu", "leaky_relu") else "xavier"
