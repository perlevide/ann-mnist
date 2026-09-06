"""Default paths and hyperparameters.

Every script reads its defaults from here so that a change in one place
affects the whole project. Command line flags override these values.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "out"

IMAGE_SIZE = 28
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE
N_CLASSES = 10

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass
class TrainConfig:
    """Hyperparameters shared by the NumPy and the PyTorch trainer."""

    hidden_sizes: tuple = (256, 128)
    activation: str = "relu"
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.05
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0
    dropout: float = 0.0
    lr_decay: float = 1.0
    val_fraction: float = 0.1
    seed: int = 0
    normalize: str = "standard"

    def layer_sizes(self) -> list:
        return [N_PIXELS, *self.hidden_sizes, N_CLASSES]

    def as_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}
