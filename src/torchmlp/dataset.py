"""Datasets and loaders built from the same IDX parser as the NumPy path.

torchvision would download MNIST in one line. It is skipped here so that
both implementations read exactly the same bytes with exactly the same
preprocessing, which makes the comparison between them honest.

Two loaders live here. `DataLoader` is the standard PyTorch object and it is
what you will see in every tutorial. `DeviceBatches` keeps the whole split in
GPU memory and slices it there, which matters for a model this small: with a
DataLoader, copying 128 images from host to device and back through the
Python iterator costs more than the forward and backward passes put
together, and the GPU spends most of the run waiting.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import DATA_DIR
from ..data import load_splits


def to_tensors(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y))


class DeviceBatches:
    """Mini batches sliced from tensors that already live on the device.

    MNIST is small enough to hold entirely in GPU memory: 54000 by 784
    float32 is 169 MB. Uploading it once removes the per batch transfer, and
    shuffling becomes a `randperm` on the device instead of an index list in
    Python.

    The interface matches what the training loop needs from a DataLoader:
    iteration yields (x, y) pairs and `len` gives the number of batches.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, device, shuffle: bool):
        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).to(device)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def __iter__(self):
        n = len(self.x)
        order = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for start in range(0, n, self.batch_size):
            index = order[start : start + self.batch_size]
            yield self.x[index], self.y[index]

    def bytes_held(self) -> int:
        return self.x.numel() * self.x.element_size() + self.y.numel() * self.y.element_size()


def build_loaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    normalize: str = "standard",
    seed: int = 0,
    num_workers: int = 0,
    device=None,
    resident: bool | None = None,
) -> dict:
    """Return train, validation and test loaders.

    `resident` chooses between the two loaders. Left as None it follows the
    device: resident tensors on a GPU, DataLoader on the CPU. Pass it
    explicitly to compare the two.

    Only the training loader shuffles. Shuffling the evaluation sets would
    change nothing about the metrics and would make per sample debugging
    harder.
    """
    splits = load_splits(data_dir, val_fraction=val_fraction, normalize=normalize, seed=seed)

    device = torch.device("cpu") if device is None else torch.device(device)
    if resident is None:
        resident = device.type == "cuda"

    loaders: dict = {}
    for name, shuffle in (("train", True), ("val", False), ("test", False)):
        size = batch_size if name == "train" else 1000
        x, y = splits[f"x_{name}"], splits[f"y_{name}"]
        if resident:
            loaders[name] = DeviceBatches(x, y, size, device, shuffle)
        else:
            loaders[name] = DataLoader(
                to_tensors(x, y),
                batch_size=size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )

    loaders["splits"] = splits
    loaders["resident"] = resident
    return loaders
