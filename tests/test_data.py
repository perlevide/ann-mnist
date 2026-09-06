"""Preprocessing and batching, without needing the dataset on disk."""

import numpy as np

from src.data import flatten_and_scale, iterate_minibatches, one_hot


def test_flatten_and_scale_unit_range():
    images = np.random.default_rng(0).integers(0, 256, size=(5, 28, 28), dtype=np.uint8)
    flat = flatten_and_scale(images, "unit")
    assert flat.shape == (5, 784)
    assert flat.dtype == np.float32
    assert 0.0 <= flat.min() and flat.max() <= 1.0


def test_standardized_data_is_roughly_centered():
    images = np.random.default_rng(1).integers(0, 256, size=(200, 28, 28), dtype=np.uint8)
    flat = flatten_and_scale(images, "standard")
    assert abs(flat.mean()) < 1.5


def test_one_hot():
    encoded = one_hot(np.array([0, 3, 9]), 10)
    assert encoded.shape == (3, 10)
    assert np.array_equal(encoded.argmax(axis=1), [0, 3, 9])
    assert encoded.sum() == 3


def test_minibatches_cover_every_sample_once():
    x = np.arange(100).reshape(100, 1).astype(np.float32)
    y = np.arange(100)
    seen = np.concatenate([batch_y for _, batch_y in iterate_minibatches(x, y, 32)])
    assert sorted(seen) == list(range(100))


def test_last_batch_may_be_smaller():
    x = np.zeros((10, 3), dtype=np.float32)
    y = np.zeros(10, dtype=np.int64)
    sizes = [len(b) for b, _ in iterate_minibatches(x, y, 4, shuffle=False)]
    assert sizes == [4, 4, 2]
