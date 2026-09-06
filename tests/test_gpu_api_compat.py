"""Catch CuPy incompatibilities without a GPU.

CuPy's Generator implements a subset of NumPy's. Code written against NumPy
can call something CuPy does not have, and the failure only shows up on a
machine with a card in it, halfway through the first epoch.

`CupyLikeGenerator` is a NumPy Generator with everything CuPy 14 lacks taken
away. Anything in the training path that works with it will work on the GPU,
at least as far as the Generator API is concerned.
"""

import numpy as np
import pytest

from src import backend
from src.data import iterate_minibatches
from src.scratch import initializers
from src.scratch.layers import Dropout, Linear


class CupyLikeGenerator:
    """A NumPy Generator restricted to the methods CuPy 14 provides."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        if item not in backend.CUPY_GENERATOR_METHODS:
            raise AttributeError(f"CuPy's Generator has no {item!r}")
        return getattr(self._rng, item)


def test_the_restriction_is_real():
    rng = CupyLikeGenerator()
    assert not hasattr(rng, "permutation")
    assert not hasattr(rng, "shuffle")
    assert not hasattr(rng, "choice")
    assert hasattr(rng, "random")
    assert hasattr(rng, "standard_normal")
    assert hasattr(rng, "uniform")


@pytest.mark.parametrize("name", ["zeros", "normal", "xavier", "he"])
def test_initializers_work_without_the_missing_methods(name):
    weights = initializers.get(name)((32, 16), CupyLikeGenerator(1))
    assert weights.shape == (32, 16)
    assert weights.dtype == np.float32


def test_linear_and_dropout_build_and_run():
    rng = CupyLikeGenerator(2)
    layer = Linear(8, 4, rng)
    out = layer.forward(np.ones((5, 8), dtype=np.float32))
    assert out.shape == (5, 4)

    dropout = Dropout(0.5, CupyLikeGenerator(3))
    assert dropout.forward(out, training=True).shape == (5, 4)


def test_permutation_helper_falls_back_to_sorting_random_keys():
    order = backend.permutation(CupyLikeGenerator(4), 50)
    assert sorted(order.tolist()) == list(range(50))


def test_permutation_helper_is_deterministic():
    a = backend.permutation(CupyLikeGenerator(5), 100)
    b = backend.permutation(CupyLikeGenerator(5), 100)
    assert np.array_equal(a, b)


def test_numpy_generator_keeps_its_own_permutation():
    """The NumPy path must not change, or every number in docs/ moves."""
    rng = np.random.default_rng(6)
    expected = np.random.default_rng(6).permutation(100)
    assert np.array_equal(backend.permutation(rng, 100), expected)


def test_minibatches_shuffle_without_a_permutation_method():
    x = np.arange(60).reshape(60, 1).astype(np.float32)
    y = np.arange(60)
    seen = np.concatenate(
        [batch_y for _, batch_y in iterate_minibatches(x, y, 16, rng=CupyLikeGenerator(7))]
    )
    assert sorted(seen.tolist()) == list(range(60))
