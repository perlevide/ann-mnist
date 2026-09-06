"""Model definition."""

from __future__ import annotations

import torch
from torch import nn

from ..config import N_CLASSES, N_PIXELS

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Same architecture as `src.scratch.network.MLP`.

    The last layer produces raw logits. `nn.CrossEntropyLoss` applies
    log softmax internally, for the same numerical reason the NumPy version
    fuses the two steps.
    """

    def __init__(
        self,
        layer_sizes=(N_PIXELS, 256, 128, N_CLASSES),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")

        layers: list = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(ACTIVATIONS[activation]())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.dropout = dropout
        self.reset_parameters(activation)

    def reset_parameters(self, activation: str) -> None:
        """Match the initialization used by the NumPy version.

        PyTorch defaults to a Kaiming uniform variant for nn.Linear. Setting
        it explicitly removes one difference between the two runs.
        """
        nonlinearity = "relu" if activation in ("relu", "leaky_relu") else "tanh"
        for module in self.net:
            if isinstance(module, nn.Linear):
                if nonlinearity == "relu":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config, device: str = "cpu") -> MLP:
    model = MLP(
        layer_sizes=config.layer_sizes(),
        activation=config.activation,
        dropout=config.dropout,
    )
    return model.to(device)
