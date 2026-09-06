"""Optimizers.

An optimizer owns the update rule and whatever state that rule needs
between steps. It receives a list of (parameter, gradient) pairs and
modifies the parameters in place.
"""

from __future__ import annotations

from ..backend import xp as np


class Optimizer:
    def __init__(self, learning_rate: float, weight_decay: float = 0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_count = 0

    def step(self, params_and_grads: list) -> None:
        raise NotImplementedError

    def _decayed(self, param: np.ndarray, grad: np.ndarray, is_weight: bool) -> np.ndarray:
        """Add the L2 gradient, lambda * W, for weight matrices only."""
        if self.weight_decay and is_weight:
            return grad + self.weight_decay * param
        return grad


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum.

    Plain update:      w <- w - lr * g
    With momentum:     v <- mu * v + g,  w <- w - lr * v

    The velocity v is a running average of past gradients. It damps the
    oscillation across a narrow valley and accumulates speed along the
    valley floor, which is why momentum usually reaches a given loss in
    fewer epochs than the plain rule.

    Nesterov evaluates the gradient after the momentum step rather than
    before it, so the correction is slightly better aimed.
    """

    def __init__(self, learning_rate=0.05, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity: dict = {}

    def step(self, params_and_grads):
        self.step_count += 1
        for index, (param, grad, is_weight) in enumerate(params_and_grads):
            grad = self._decayed(param, grad, is_weight)
            if self.momentum == 0.0:
                param -= self.learning_rate * grad
                continue
            v = self.velocity.setdefault(index, np.zeros_like(param))
            v *= self.momentum
            v += grad
            if self.nesterov:
                param -= self.learning_rate * (grad + self.momentum * v)
            else:
                param -= self.learning_rate * v


class RMSProp(Optimizer):
    """Divide each step by the running root mean square of that coordinate.

        s <- rho * s + (1 - rho) * g^2
        w <- w - lr * g / (sqrt(s) + eps)

    Coordinates with consistently large gradients get small steps and
    coordinates with small gradients get relatively larger ones, so a single
    learning rate can serve parameters at very different scales.
    """

    def __init__(self, learning_rate=0.001, rho=0.9, eps=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.rho = rho
        self.eps = eps
        self.square_avg: dict = {}

    def step(self, params_and_grads):
        self.step_count += 1
        for index, (param, grad, is_weight) in enumerate(params_and_grads):
            grad = self._decayed(param, grad, is_weight)
            s = self.square_avg.setdefault(index, np.zeros_like(param))
            s *= self.rho
            s += (1.0 - self.rho) * grad ** 2
            param -= self.learning_rate * grad / (np.sqrt(s) + self.eps)


class Adam(Optimizer):
    """Momentum and RMSProp in one rule, with a bias correction.

        m <- b1 * m + (1 - b1) * g
        s <- b2 * s + (1 - b2) * g^2
        m_hat = m / (1 - b1^t),  s_hat = s / (1 - b2^t)
        w <- w - lr * m_hat / (sqrt(s_hat) + eps)

    Both averages start at zero, which biases them toward zero for the first
    few steps. Dividing by (1 - beta^t) removes exactly that bias, and the
    correction fades as t grows.

    Adam converges quickly with little tuning, which is why it is the usual
    first choice. Well tuned SGD with momentum still generalizes better on
    some problems, so it is worth trying both.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict = {}
        self.v: dict = {}

    def step(self, params_and_grads):
        self.step_count += 1
        t = self.step_count
        for index, (param, grad, is_weight) in enumerate(params_and_grads):
            grad = self._decayed(param, grad, is_weight)
            m = self.m.setdefault(index, np.zeros_like(param))
            v = self.v.setdefault(index, np.zeros_like(param))
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad ** 2
            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


def build(name: str, learning_rate: float, momentum: float = 0.9, weight_decay: float = 0.0):
    name = name.lower()
    if name == "sgd":
        return SGD(learning_rate, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return SGD(learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSProp(learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return Adam(learning_rate, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")
