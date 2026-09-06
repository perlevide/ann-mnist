"""MNIST loading, IDX parsing and batching.

The IDX format is a header followed by raw bytes. For images the header is
four big endian 32 bit integers: magic number, count, rows, columns. For
labels it is two: magic number and count. Nothing else, no compression
beyond the gzip wrapper.
"""

from __future__ import annotations

import gzip
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .config import (
    DATA_DIR,
    MNIST_FILES,
    MNIST_MEAN,
    MNIST_MIRRORS,
    MNIST_STD,
    N_CLASSES,
)


def download(dest: Path = DATA_DIR, force: bool = False) -> None:
    """Fetch the four MNIST files, trying each mirror in turn."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES.values():
        target = dest / name
        if target.exists() and not force:
            print(f"have {name}")
            continue
        for base in MNIST_MIRRORS:
            url = base + name
            try:
                print(f"get  {url}")
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"     failed: {exc}")
                continue
            target.write_bytes(payload)
            print(f"     saved {len(payload) / 1e6:.1f} MB")
            break
        else:
            raise RuntimeError(f"could not download {name} from any mirror")


def read_idx(path: Path) -> np.ndarray:
    """Parse one IDX file into a NumPy array."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", handle.read(8))
            shape = (count, rows, cols)
        elif magic == 2049:
            shape = (count,)
        else:
            raise ValueError(f"{path.name}: unexpected magic number {magic}")
        buffer = handle.read()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(shape)


def load_raw(data_dir: Path = DATA_DIR) -> dict:
    """Return the four arrays as stored on disk, still uint8."""
    missing = [n for n in MNIST_FILES.values() if not (data_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {', '.join(missing)} in {data_dir}. Run: python download_data.py"
        )
    return {key: read_idx(data_dir / name) for key, name in MNIST_FILES.items()}


def flatten_and_scale(images: np.ndarray, normalize: str = "standard") -> np.ndarray:
    """Turn (N, 28, 28) uint8 into (N, 784) float32.

    "unit"     maps pixels to [0, 1].
    "standard" additionally subtracts the dataset mean and divides by the
               dataset standard deviation, which centers the inputs near
               zero. Centered inputs keep the first layer's pre-activations
               in the useful part of the activation function.
    """
    flat = images.reshape(len(images), -1).astype(np.float32) / 255.0
    if normalize == "unit":
        return flat
    if normalize == "standard":
        return (flat - MNIST_MEAN) / MNIST_STD
    raise ValueError(f"unknown normalize mode {normalize!r}")


def one_hot(labels: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """(N,) integer labels to (N, C) one hot rows."""
    encoded = np.zeros((len(labels), n_classes), dtype=np.float32)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def load_splits(
    data_dir: Path = DATA_DIR,
    val_fraction: float = 0.1,
    normalize: str = "standard",
    seed: int = 0,
) -> dict:
    """Load MNIST and carve a validation split out of the training set.

    The test set is never touched during training or tuning. Every decision
    about hyperparameters is made on the validation split, so the test
    number at the end is an estimate you can believe.
    """
    raw = load_raw(data_dir)

    x_train_full = flatten_and_scale(raw["train_images"], normalize)
    y_train_full = raw["train_labels"].astype(np.int64)
    x_test = flatten_and_scale(raw["test_images"], normalize)
    y_test = raw["test_labels"].astype(np.int64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train_full))
    n_val = int(round(val_fraction * len(x_train_full)))
    val_idx, train_idx = order[:n_val], order[n_val:]

    return {
        "x_train": x_train_full[train_idx],
        "y_train": y_train_full[train_idx],
        "x_val": x_train_full[val_idx],
        "y_val": y_train_full[val_idx],
        "x_test": x_test,
        "y_test": y_test,
    }


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
):
    """Yield (x_batch, y_batch) pairs covering the data once.

    Works on NumPy or CuPy arrays: the permutation is built with whichever
    module owns `x`, so a GPU run never round trips indices through the host.

    Shuffling matters. If the batches always arrive in the same order the
    gradient noise becomes periodic and the model can lock onto that order
    instead of the data.
    """
    from . import backend

    xp = backend.array_module(x)
    n = len(x)
    if shuffle:
        if rng is None:
            rng = xp.random.default_rng()
        order = rng.permutation(n)
    else:
        order = xp.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]
