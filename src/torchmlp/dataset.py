"""Datasets and loaders built from the same IDX parser as the NumPy path.

torchvision would download MNIST in one line. It is skipped here so that
both implementations read exactly the same bytes with exactly the same
preprocessing, which makes the comparison between them honest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import DATA_DIR
from ..data import load_splits


def to_tensors(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y))


def build_loaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    normalize: str = "standard",
    seed: int = 0,
    num_workers: int = 0,
) -> dict:
    """Return train, validation and test loaders.

    Only the training loader shuffles. Shuffling the evaluation sets would
    change nothing about the metrics and would make per sample debugging
    harder.
    """
    splits = load_splits(data_dir, val_fraction=val_fraction, normalize=normalize, seed=seed)

    loaders = {}
    for name, shuffle in (("train", True), ("val", False), ("test", False)):
        dataset = to_tensors(splits[f"x_{name}"], splits[f"y_{name}"])
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size if name == "train" else 1000,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )
    loaders["splits"] = splits
    return loaders
