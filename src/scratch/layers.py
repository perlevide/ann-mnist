"""Layers with trainable parameters, plus dropout.

Shape convention: a batch is (N, D) with samples along the first axis. A
linear layer maps (N, in) to (N, out) with weights of shape (in, out), so
the forward pass is a plain matrix product and no transposes are needed.
"""

from __future__ import annotations

from ..backend import xp as np

from . import initializers


class Layer:
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> dict:
        return {}

    def gradients(self) -> dict:
        return {}


class Linear(Layer):
    """y = x W + b.

    Backward pass, with L the scalar loss and G = dL/dy of shape (N, out):

        dL/dW = x^T G          (in, out)
        dL/db = sum of G rows  (out,)
        dL/dx = G W^T          (N, in)

    The first two come from the chain rule applied to every entry of W and
    b; the sum over the batch appears because each parameter is reused by
    every sample. The third is what gets handed to the previous layer.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        rng: np.random.Generator,
        init: str = "he",
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.W = initializers.get(init)((n_in, n_out), rng)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def parameters(self):
        return {"W": self.W, "b": self.b}

    def gradients(self):
        return {"W": self.dW, "b": self.db}

    def __repr__(self):
        return f"Linear({self.n_in} -> {self.n_out})"


class Dropout(Layer):
    """Zero a random fraction p of the activations during training.

    Inverted dropout: the surviving activations are divided by (1 - p) so
    that the expected value of the output matches the evaluation pass. That
    way inference needs no rescaling and can simply return the input.
    """

    def __init__(self, p: float, rng: np.random.Generator):
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = rng
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.p == 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        self.mask = (self.rng.random(x.shape) < keep).astype(np.float32) / keep
        return x * self.mask

    def backward(self, grad_out):
        if self.mask is None:
            return grad_out
        return grad_out * self.mask

    def __repr__(self):
        return f"Dropout(p={self.p})"
