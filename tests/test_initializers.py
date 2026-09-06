"""Initializer scale, dtype, and one rule that keeps the GPU path working."""

import re
from pathlib import Path

import numpy as np
import pytest

from src.scratch import initializers


@pytest.mark.parametrize("name", ["zeros", "normal", "xavier", "he"])
def test_returns_float32(name):
    rng = np.random.default_rng(0)
    weights = initializers.get(name)((64, 32), rng)
    assert weights.dtype == np.float32
    assert weights.shape == (64, 32)


def test_he_variance_matches_two_over_fan_in():
    rng = np.random.default_rng(0)
    fan_in = 512
    weights = initializers.he((fan_in, 256), rng)
    assert abs(weights.std() - np.sqrt(2.0 / fan_in)) < 0.005


def test_xavier_stays_inside_its_limit():
    rng = np.random.default_rng(1)
    fan_in, fan_out = 256, 128
    weights = initializers.xavier((fan_in, fan_out), rng)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    assert np.abs(weights).max() <= limit


def test_zeros_are_identical_across_units():
    weights = initializers.zeros((16, 8))
    assert not weights.any()


def test_default_for_activation():
    assert initializers.default_for("relu") == "he"
    assert initializers.default_for("leaky_relu") == "he"
    assert initializers.default_for("tanh") == "xavier"
    assert initializers.default_for("sigmoid") == "xavier"


def test_scalar_math_does_not_go_through_the_array_module():
    """Guard for the CuPy backend.

    `np` in this package is whichever array module is active. Calling
    `np.sqrt` on a Python float asks that module to launch a kernel for one
    square root, which on CuPy means compiling one, and it fails outright
    when the CUDA headers are missing. Scalars belong to `math`.
    """
    source = Path(initializers.__file__).read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    body = code.split('"""')[-1]
    assert not re.search(r"\bnp\.sqrt\(", body), "use math.sqrt for scalars in initializers.py"
    assert "import math" in code


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        initializers.get("orthogonal")
