"""The same network in PyTorch.

Nothing here is new mathematically. The forward pass is the same, the loss is
the same and the update rule is the same. What changes is that autograd
records the forward pass and produces the backward pass for you, so the
hand written derivatives in `src/scratch` disappear.

Read this package after the NumPy one. It is short on purpose.
"""

from .model import MLP, build_model

__all__ = ["MLP", "build_model"]
