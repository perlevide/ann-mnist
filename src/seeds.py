"""Seeding helpers.

Training is stochastic: weight initialization, batch shuffling and dropout
all draw random numbers. Fixing the seed makes a run repeatable, which is
what lets you compare two settings and trust the difference.
"""

import os
import random

import numpy as np


def seed_everything(seed: int = 0) -> np.random.Generator:
    """Seed the standard library, NumPy and PyTorch if it is installed.

    Returns a NumPy Generator. Prefer passing this generator around
    explicitly rather than relying on the global NumPy state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return np.random.default_rng(seed)
