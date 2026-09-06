"""Figures for the report: learning curves, confusion matrix, sample errors.

Matplotlib only, with the Agg backend so the scripts run without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import IMAGE_SIZE, OUT_DIR


def learning_curves(history: dict, path: Path = OUT_DIR / "fig_curves.png") -> Path:
    """Loss and accuracy against epoch, training against validation.

    The gap between the two curves is the part to read. Training loss falling
    while validation loss rises is overfitting, and the epoch where the
    validation curve turns is where early stopping would cut the run.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    if history.get("val_acc"):
        axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    return _save(figure, path)


def confusion_heatmap(matrix: np.ndarray, path: Path = OUT_DIR / "fig_confusion.png") -> Path:
    figure, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(len(matrix)))
    ax.set_yticks(range(len(matrix)))
    threshold = matrix.max() / 2
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] > threshold else "black",
            )
    figure.colorbar(image, ax=ax, shrink=0.85)
    figure.tight_layout()
    return _save(figure, path)


def sample_errors(
    images: np.ndarray,
    true: np.ndarray,
    predicted: np.ndarray,
    n: int = 24,
    path: Path = OUT_DIR / "fig_errors.png",
) -> Path:
    """Grid of misclassified digits, labelled true then predicted.

    Worth looking at before changing anything. Many MNIST errors are digits a
    person would also hesitate over, and no amount of tuning fixes those.
    """
    wrong = np.flatnonzero(true != predicted)[:n]
    if len(wrong) == 0:
        return path

    columns = 8
    rows = int(np.ceil(len(wrong) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.3, rows * 1.5))
    for ax, index in zip(np.ravel(axes), wrong):
        ax.imshow(images[index].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
        ax.set_title(f"{true[index]} -> {predicted[index]}", fontsize=8)
    for ax in np.ravel(axes):
        ax.axis("off")
    figure.tight_layout()
    return _save(figure, path)


def weight_grid(weights: np.ndarray, n: int = 32, path: Path = OUT_DIR / "fig_weights.png") -> Path:
    """First layer weights reshaped back to 28 by 28.

    Each column of W1 is a filter over the input image. After training some
    of them look like strokes and blobs positioned where digits differ.
    """
    n = min(n, weights.shape[1])
    columns = 8
    rows = int(np.ceil(n / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.2, rows * 1.2))
    for i, ax in enumerate(np.ravel(axes)):
        ax.axis("off")
        if i < n:
            ax.imshow(weights[:, i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="RdBu_r")
    figure.tight_layout()
    return _save(figure, path)


def _save(figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")
    return path
