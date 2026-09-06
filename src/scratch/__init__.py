"""The from scratch implementation: NumPy only, no autograd.

Every derivative in this package is written out by hand. That is the point.
Once the shapes and the chain rule are familiar, `src/torchmlp` does the same
job in a tenth of the code.

The array module comes from `src.backend`, so the same code runs on NumPy or
on CuPy. Call `backend.select()` before importing this package; importing it
binds the choice.
"""

from .. import backend
from .network import MLP
from .optimizers import SGD, Adam, RMSProp

backend.lock()

__all__ = ["MLP", "SGD", "Adam", "RMSProp"]
