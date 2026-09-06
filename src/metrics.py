"""Evaluation metrics.

Accuracy alone hides the interesting part. On MNIST a model at 98 percent
still gets about 200 test images wrong, and those 200 are not spread evenly:
4 and 9 account for a large share of them. The confusion matrix is what
shows that.
"""

from __future__ import annotations

import numpy as np

from .config import N_CLASSES


def accuracy(predicted: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(predicted == true))


def confusion_matrix(predicted: np.ndarray, true: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Row = true class, column = predicted class."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def per_class_report(matrix: np.ndarray) -> dict:
    """Precision, recall and F1 per class, from the confusion matrix.

    precision = TP / (TP + FP), how often a prediction of this class is right
    recall    = TP / (TP + FN), how much of this class the model finds
    F1        = harmonic mean of the two
    """
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_total = matrix.sum(axis=0).astype(np.float64)
    actual_total = matrix.sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted_total > 0, true_positive / predicted_total, 0.0)
        recall = np.where(actual_total > 0, true_positive / actual_total, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_total.astype(np.int64),
    }


def format_report(matrix: np.ndarray) -> str:
    """Readable table of the per class report plus the totals."""
    report = per_class_report(matrix)
    lines = [f"{'class':>6} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>8}"]
    for c in range(len(matrix)):
        lines.append(
            f"{c:>6} {report['precision'][c]:>10.4f} {report['recall'][c]:>8.4f} "
            f"{report['f1'][c]:>8.4f} {report['support'][c]:>8d}"
        )
    total = matrix.sum()
    correct = np.trace(matrix)
    lines.append("")
    lines.append(f"accuracy {correct}/{total} = {correct / total:.4f}")
    lines.append(f"macro f1 {report['f1'].mean():.4f}")
    return "\n".join(lines)


def top_confusions(matrix: np.ndarray, k: int = 5) -> list:
    """The k most frequent (true, predicted, count) mistakes."""
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    flat = np.argsort(off_diagonal, axis=None)[::-1][:k]
    rows, cols = np.unravel_index(flat, off_diagonal.shape)
    return [(int(r), int(c), int(off_diagonal[r, c])) for r, c in zip(rows, cols) if off_diagonal[r, c] > 0]
