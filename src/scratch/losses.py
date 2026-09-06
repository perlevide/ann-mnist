"""Loss functions.

A loss turns the network output and the true label into one number, and
returns the gradient of that number with respect to the output. Everything
upstream is the chain rule applied to that starting gradient.
"""

from __future__ import annotations

from ..backend import xp as np

from .activations import log_softmax, softmax


class Loss:
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class SoftmaxCrossEntropy(Loss):
    """Softmax and cross entropy fused into one step.

    For one sample with logits z, true class k and p = softmax(z):

        L = -log p_k
        dL/dz_j = p_j - [j == k]

    That derivative is the reason the two are fused. Computed separately,
    the softmax Jacobian is a full C by C matrix and the cross entropy
    derivative has a 1 / p_k that overflows when the model is confidently
    wrong. Together they cancel into a subtraction that is both cheap and
    stable.

    The mean over the batch is used, so the gradient is divided by N and the
    learning rate does not have to be retuned when the batch size changes.
    """

    def forward(self, logits, targets):
        self.n = len(logits)
        self.targets = targets
        self.log_probs = log_softmax(logits)
        self.probs = np.exp(self.log_probs)
        return float(-self.log_probs[np.arange(self.n), targets].mean())

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(self.n), self.targets] -= 1.0
        return grad / self.n


class MeanSquaredError(Loss):
    """L = mean over the batch of ||y_hat - y||^2 / 2, with dL/dy_hat = (y_hat - y) / N.

    Fine for regression. A poor choice for classification: paired with a
    sigmoid or softmax output it multiplies the already small activation
    derivative into the gradient, so a badly wrong prediction produces an
    almost flat slope and learning stalls. Cross entropy has no such factor.
    """

    def forward(self, predictions, targets):
        self.n = len(predictions)
        self.diff = predictions - targets
        return float(0.5 * np.sum(self.diff ** 2) / self.n)

    def backward(self):
        return self.diff / self.n


def l2_penalty(parameters: list, weight_decay: float) -> float:
    """Value of the L2 term, (lambda / 2) * sum of squared weights.

    Only weight matrices are penalized. Shrinking biases moves the whole
    decision surface toward the origin without reducing model complexity.
    """
    if weight_decay == 0.0:
        return 0.0
    return 0.5 * weight_decay * sum(float(np.sum(p ** 2)) for p in parameters)


REGISTRY = {"cross_entropy": SoftmaxCrossEntropy, "mse": MeanSquaredError}


def get(name: str) -> Loss:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown loss {name!r}, choose from {sorted(REGISTRY)}") from None


__all__ = ["Loss", "SoftmaxCrossEntropy", "MeanSquaredError", "l2_penalty", "get", "softmax"]
