"""The multilayer perceptron: a list of layers plus a training loop."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as _np

from .. import backend
from ..backend import xp as np
from ..data import iterate_minibatches
from ..metrics import accuracy
from . import activations as act
from . import initializers
from .layers import Dropout, Linear
from .losses import SoftmaxCrossEntropy, l2_penalty
from .optimizers import Optimizer, build as build_optimizer


class MLP:
    """Fully connected network of arbitrary depth.

    `layer_sizes` lists every width from input to output, so [784, 256, 10]
    is one hidden layer of 256 units. Hidden layers get the chosen
    activation and optional dropout; the output layer is left linear because
    SoftmaxCrossEntropy applies the softmax itself.
    """

    def __init__(
        self,
        layer_sizes: list,
        activation: str = "relu",
        dropout: float = 0.0,
        init: str | None = None,
        seed: int = 0,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("need at least an input and an output size")

        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.dropout_p = dropout
        self.init_name = init or initializers.default_for(activation)
        self.seed = seed

        rng = backend.default_rng(seed)
        self.layers: list = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], rng, self.init_name))
            is_output = i == len(layer_sizes) - 2
            if not is_output:
                self.layers.append(act.get(activation))
                if dropout > 0.0:
                    self.layers.append(Dropout(dropout, rng))

        self.loss_fn = SoftmaxCrossEntropy()
        self.history: dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Run the input through every layer and return the logits."""
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad: np.ndarray) -> None:
        """Walk the layers in reverse, handing each one dL/d(its output)."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params_and_grads(self) -> list:
        """Flat list of (parameter, gradient, is_weight) for the optimizer."""
        pairs = []
        for layer in self.layers:
            params, grads = layer.parameters(), layer.gradients()
            for key in params:
                pairs.append((params[key], grads[key], key == "W"))
        return pairs

    def weight_matrices(self) -> list:
        return [layer.W for layer in self.layers if isinstance(layer, Linear)]

    def predict_logits(self, x: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        outputs = [self.forward(x[i : i + batch_size], training=False) for i in range(0, len(x), batch_size)]
        return np.concatenate(outputs, axis=0)

    def predict(self, x: np.ndarray, batch_size: int = 1000) -> "_np.ndarray":
        """Predicted class per sample, always returned as a NumPy array.

        Metrics and plotting live on the host, so results come back from the
        device here rather than at every call site.
        """
        return backend.to_numpy(self.predict_logits(x, batch_size).argmax(axis=1))

    def predict_proba(self, x: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        return act.softmax(self.predict_logits(x, batch_size))

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 1000) -> tuple:
        logits = self.predict_logits(x, batch_size)
        loss = self.loss_fn.forward(logits, y)
        predicted = backend.to_numpy(logits.argmax(axis=1))
        return loss, accuracy(predicted, backend.to_numpy(y))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 20,
        batch_size: int = 128,
        optimizer: Optimizer | str = "sgd",
        learning_rate: float = 0.05,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        lr_decay: float = 1.0,
        seed: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """Train with mini batch gradient descent.

        One epoch is one pass over the training set. Within an epoch the data
        is reshuffled and split into batches; each batch produces one forward
        pass, one backward pass and one parameter update.
        """
        if isinstance(optimizer, str):
            optimizer = build_optimizer(optimizer, learning_rate, momentum, weight_decay)

        rng = backend.default_rng(self.seed if seed is None else seed)
        n_batches = int(_np.ceil(len(x_train) / batch_size))
        started = time.time()

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size, rng):
                logits = self.forward(x_batch, training=True)
                running_loss += self.loss_fn.forward(logits, y_batch)
                self.backward(self.loss_fn.backward())
                optimizer.step(self.params_and_grads())

            train_loss = running_loss / n_batches
            if weight_decay:
                train_loss += l2_penalty(self.weight_matrices(), weight_decay)
            _, train_acc = self.evaluate(x_train, y_train)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            message = f"epoch {epoch:>3}/{epochs}  loss {train_loss:.4f}  acc {train_acc:.4f}"
            if x_val is not None:
                val_loss, val_acc = self.evaluate(x_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                message += f"  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"

            if lr_decay != 1.0:
                optimizer.learning_rate *= lr_decay
                message += f"  lr {optimizer.learning_rate:.5f}"

            if verbose:
                print(message)

        backend.synchronize()
        self.history["seconds"] = time.time() - started
        return self.history

    def save(self, path: Path) -> None:
        """Store the weights in an .npz file and the settings alongside it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for index, layer in enumerate(self.layers):
            for key, value in layer.parameters().items():
                arrays[f"{index}.{key}"] = backend.to_numpy(value)
        _np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "layer_sizes": self.layer_sizes,
                    "activation": self.activation_name,
                    "dropout": self.dropout_p,
                    "init": self.init_name,
                    "seed": self.seed,
                    "history": {k: v for k, v in self.history.items() if k != "seconds"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "MLP":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        model = cls(
            meta["layer_sizes"],
            activation=meta["activation"],
            dropout=meta.get("dropout", 0.0),
            init=meta.get("init"),
            seed=meta.get("seed", 0),
        )
        arrays = _np.load(path)
        for index, layer in enumerate(model.layers):
            for key in layer.parameters():
                layer.parameters()[key][...] = backend.asarray(arrays[f"{index}.{key}"])
        model.history = meta.get("history", model.history)
        return model

    def summary(self) -> str:
        lines = [f"MLP {self.layer_sizes} activation={self.activation_name} init={self.init_name}"]
        total = 0
        for layer in self.layers:
            count = sum(p.size for p in layer.parameters().values())
            total += count
            lines.append(f"  {layer!r:<28} {count:>10,} parameters" if count else f"  {layer!r}")
        lines.append(f"  {'total':<28} {total:>10,} parameters")
        return "\n".join(lines)

    def __repr__(self):
        return f"MLP({self.layer_sizes}, activation={self.activation_name!r})"
