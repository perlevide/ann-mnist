"""The from scratch implementation: NumPy only, no autograd.

Every derivative in this package is written out by hand. That is the point.
Once the shapes and the chain rule are familiar, `src/torchmlp` does the same
job in a tenth of the code.
"""

from .network import MLP
from .optimizers import SGD, Adam, RMSProp

__all__ = ["MLP", "SGD", "Adam", "RMSProp"]
