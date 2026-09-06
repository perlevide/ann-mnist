"""Training and evaluation loops.

Compare `train_one_epoch` with the loop in `src/scratch/network.py`. The
structure is identical. The only difference is that `loss.backward()`
replaces the hand written chain rule.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def backend_info() -> dict:
    """What PyTorch build this is and whether it can see a GPU.

    `torch.version.cuda` is None on a CPU-only wheel. That is the usual
    reason a machine with a working NVIDIA card still trains on the CPU:
    `pip install torch` on Windows takes the CPU-only build from PyPI.
    """
    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "gpu": None,
        "capability": None,
    }
    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(str(n) for n in torch.cuda.get_device_capability(0))
    return info


def explain_no_cuda(info: dict) -> str:
    """Why CUDA is unavailable, and what to do about it."""
    if info["cuda_build"] is None:
        return (
            f"PyTorch {info['torch']} is the CPU-only build, so it cannot use a GPU at all.\n"
            "On Windows, plain `pip install torch` gives this build. Reinstall from the CUDA index:\n"
            "  pip uninstall -y torch\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu130\n"
            "Check your driver version with `nvidia-smi` first. If it is older than the CUDA\n"
            "release above, use an earlier tag (cu128, cu126) or update the driver.\n"
            "pytorch.org/get-started/locally builds the exact command for your setup."
        )
    return (
        f"PyTorch {info['torch']} was built against CUDA {info['cuda_build']} but no GPU is\n"
        "visible to it. Run `nvidia-smi`. If that fails, the driver is missing or too old.\n"
        "If it works, the driver is likely older than the CUDA version this build needs."
    )


def pick_device(preference: str = "auto", verbose: bool = True) -> torch.device:
    """Resolve --device into a torch.device, and say why when it is not cuda.

    "auto" falls back to the CPU with a warning rather than silently.
    "cuda" is a request, not a hint: it fails loudly instead of falling back,
    because a silent fallback is how a run ends up 20 times slower than
    expected without anyone noticing.
    """
    preference = preference.lower()
    info = backend_info()

    if preference == "cpu":
        return torch.device("cpu")

    if preference.startswith("cuda"):
        if not info["cuda_available"]:
            raise SystemExit("--device cuda was requested but is unavailable.\n\n" + explain_no_cuda(info))
        return torch.device(preference)

    if preference != "auto":
        return torch.device(preference)

    if info["cuda_available"]:
        return torch.device("cuda")
    if verbose:
        print("no GPU available, training on the CPU")
        print(explain_no_cuda(info))
        print()
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    info = backend_info()
    if device.type != "cuda":
        return f"device cpu (torch {info['torch']})"
    return (
        f"device cuda: {info['gpu']}, compute capability {info['capability']}, "
        f"torch {info['torch']} built against CUDA {info['cuda_build']}"
    )


def enable_tf32() -> None:
    """Let matmuls use TensorFloat-32 on Ampere and newer.

    TF32 keeps the float32 exponent and truncates the mantissa to 10 bits.
    Accuracy on this model is unaffected and the matmuls run several times
    faster on the tensor cores. It does nothing on the CPU or on older GPUs.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    name = config.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if name == "nesterov":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            nesterov=True,
            weight_decay=config.weight_decay,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """One pass over the training data. Returns the mean batch loss.

    `model.train()` turns dropout on. `optimizer.zero_grad()` matters because
    PyTorch accumulates gradients into `.grad` instead of overwriting them,
    so a missing zero_grad silently sums the gradients of every batch seen so
    far.
    """
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    """Loss and accuracy with dropout off and no graph recorded."""
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device) -> tuple:
    """Return (predicted labels, true labels) as NumPy arrays."""
    model.eval()
    predictions, truths = [], []
    for x, y in loader:
        predictions.append(model(x.to(device, non_blocking=True)).argmax(dim=1).cpu().numpy())
        truths.append(y.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(truths)


def fit(model, loaders, config, device, verbose: bool = True) -> dict:
    """Full training run with a per epoch report and best checkpoint tracking."""
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = None
    if config.lr_decay != 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        _, train_acc = evaluate(model, loaders["train"], criterion, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best["val_acc"]:
            best.update(
                val_acc=val_acc,
                epoch=epoch,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(
                f"epoch {epoch:>3}/{config.epochs}  loss {train_loss:.4f}  acc {train_acc:.4f}"
                f"  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
            )

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    history["seconds"] = time.time() - started
    history["best_epoch"] = best["epoch"]
    history["best_val_acc"] = best["val_acc"]
    return history


def save_checkpoint(model, config, history, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "layer_sizes": model.layer_sizes,
            "activation": model.activation,
            "dropout": model.dropout,
            "config": config.as_dict(),
            "history": history,
        },
        path,
    )


def load_checkpoint(path: Path, device="cpu"):
    from .model import MLP

    blob = torch.load(Path(path), map_location=device, weights_only=False)
    model = MLP(blob["layer_sizes"], blob["activation"], blob["dropout"])
    model.load_state_dict(blob["state_dict"])
    return model.to(device), blob
