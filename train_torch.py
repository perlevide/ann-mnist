"""Train the PyTorch network.

    python train_torch.py
    python train_torch.py --optimizer adam --lr 0.001 --dropout 0.2 --epochs 30
    python train_torch.py --device cuda
"""

import argparse
import json

from src.config import MODEL_DIR, OUT_DIR, TrainConfig
from src.metrics import confusion_matrix, format_report, top_confusions
from src.plots import confusion_heatmap, learning_curves, sample_errors
from src.seeds import seed_everything

# The torch imports live inside main() so that --help works on a machine
# without PyTorch installed.


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="*", default=list(defaults.hidden_sizes))
    parser.add_argument("--activation", default=defaults.activation)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--optimizer", default=defaults.optimizer)
    parser.add_argument("--momentum", type=float, default=defaults.momentum)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--lr-decay", type=float, default=defaults.lr_decay)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N. 'cuda' fails loudly if no GPU is usable; "
        "'auto' falls back to the CPU and explains why",
    )
    parser.add_argument(
        "--resident",
        dest="resident",
        action="store_true",
        default=None,
        help="hold the whole split in device memory (default on a GPU)",
    )
    parser.add_argument(
        "--no-resident",
        dest="resident",
        action="store_false",
        help="use a standard DataLoader instead (default on the CPU)",
    )
    parser.add_argument("--no-tf32", action="store_true", help="disable TensorFloat-32 matmuls")
    parser.add_argument("--out", default=MODEL_DIR / "torch_mlp.pt")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from torch import nn
    except ImportError:
        raise SystemExit(
            "PyTorch is not installed. Run: pip install -r requirements-torch.txt\n"
            "The NumPy trainer needs nothing extra: python train_scratch.py"
        ) from None

    from src.torchmlp.dataset import build_loaders
    from src.torchmlp.engine import (
        describe_device,
        enable_tf32,
        evaluate,
        fit,
        pick_device,
        predict,
        save_checkpoint,
    )
    from src.torchmlp.model import build_model

    config = TrainConfig(
        hidden_sizes=tuple(args.hidden),
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        lr_decay=args.lr_decay,
        seed=args.seed,
    )
    seed_everything(config.seed)

    device = pick_device(args.device)
    if device.type == "cuda" and not args.no_tf32:
        enable_tf32()

    loaders = build_loaders(
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        normalize=config.normalize,
        seed=config.seed,
        device=device,
        resident=args.resident,
    )
    model = build_model(config, device)

    print(describe_device(device))
    held = loaders["train"].bytes_held() if loaders["resident"] else 0
    print(
        "batches: tensors resident on the device" + (f", {held / 1e6:.0f} MB for the training split" if held else "")
        if loaders["resident"]
        else "batches: DataLoader, copied to the device per batch"
    )
    print(model)
    print(f"{model.n_parameters():,} trainable parameters\n")

    history = fit(model, loaders, config, device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
    predicted, truth = predict(model, loaders["test"], device)
    matrix = confusion_matrix(predicted, truth)

    print()
    print(f"best epoch {history['best_epoch']} (val acc {history['best_val_acc']:.4f})")
    print(f"test loss {test_loss:.4f}  test accuracy {test_acc:.4f}")
    print(f"training time {history['seconds']:.1f} s")
    print()
    print(format_report(matrix))
    print()
    print("most frequent mistakes (true -> predicted, count):")
    for true_class, predicted_class, count in top_confusions(matrix):
        print(f"  {true_class} -> {predicted_class}: {count}")

    save_checkpoint(model, config, history, args.out)
    print(f"\nsaved {args.out}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "torch_summary.json").write_text(
        json.dumps(
            {"config": config.as_dict(), "test_loss": test_loss, "test_accuracy": test_acc},
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.no_plots:
        learning_curves(history, OUT_DIR / "fig_curves_torch.png")
        confusion_heatmap(matrix, OUT_DIR / "fig_confusion_torch.png")
        sample_errors(
            loaders["splits"]["x_test"], truth, predicted, path=OUT_DIR / "fig_errors_torch.png"
        )


if __name__ == "__main__":
    main()
