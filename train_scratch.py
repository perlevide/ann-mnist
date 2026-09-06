"""Train the from scratch network, on the CPU with NumPy or on a GPU with CuPy.

    python train_scratch.py
    python train_scratch.py --device cuda
    python train_scratch.py --hidden 512 256 --optimizer adam --lr 0.001 --epochs 30
    python train_scratch.py --activation sigmoid --init normal

The model code is the same either way. `src/backend.py` swaps the array
module, so `--device cuda` runs the hand written forward and backward passes
on the GPU. It needs CuPy: pip install cupy-cuda13x
"""

import argparse
import json

from src import backend
from src.config import MODEL_DIR, OUT_DIR, TrainConfig
from src.metrics import confusion_matrix, format_report, top_confusions
from src.plots import confusion_heatmap, learning_curves, sample_errors, weight_grid
from src.seeds import seed_everything


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="*", default=list(defaults.hidden_sizes))
    parser.add_argument("--activation", default=defaults.activation)
    parser.add_argument("--init", default=None, help="zeros, normal, xavier or he")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--optimizer", default=defaults.optimizer)
    parser.add_argument("--momentum", type=float, default=defaults.momentum)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--lr-decay", type=float, default=defaults.lr_decay)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--out", default=MODEL_DIR / "scratch_mlp.npz")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu (NumPy), cuda (CuPy), or auto. 'cuda' fails loudly if CuPy or a GPU is missing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    backend.select(args.device)
    from src.data import load_splits
    from src.scratch.network import MLP

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

    splits = load_splits(val_fraction=config.val_fraction, normalize=config.normalize, seed=config.seed)
    host = {k: v for k, v in splits.items()}
    if backend.name != "numpy":
        splits = {k: backend.asarray(v) for k, v in splits.items()}

    print(backend.describe())
    print(
        f"train {len(splits['x_train'])}  val {len(splits['x_val'])}  test {len(splits['x_test'])}"
    )

    model = MLP(
        config.layer_sizes(),
        activation=config.activation,
        dropout=config.dropout,
        init=args.init,
        seed=config.seed,
    )
    print(model.summary())
    print()

    history = model.fit(
        splits["x_train"],
        splits["y_train"],
        splits["x_val"],
        splits["y_val"],
        epochs=config.epochs,
        batch_size=config.batch_size,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        lr_decay=config.lr_decay,
        seed=config.seed,
    )

    test_loss, test_acc = model.evaluate(splits["x_test"], splits["y_test"])
    predicted = model.predict(splits["x_test"])
    matrix = confusion_matrix(predicted, host["y_test"])

    print()
    print(f"test loss {test_loss:.4f}  test accuracy {test_acc:.4f}")
    print(f"training time {history['seconds']:.1f} s")
    print()
    print(format_report(matrix))
    print()
    print("most frequent mistakes (true -> predicted, count):")
    for true_class, predicted_class, count in top_confusions(matrix):
        print(f"  {true_class} -> {predicted_class}: {count}")

    model.save(args.out)
    print(f"\nsaved {args.out}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scratch_summary.json").write_text(
        json.dumps(
            {
                "config": config.as_dict(),
                "backend": backend.name,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "seconds": history["seconds"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.no_plots:
        learning_curves(history, OUT_DIR / "fig_curves_scratch.png")
        confusion_heatmap(matrix, OUT_DIR / "fig_confusion_scratch.png")
        sample_errors(
            host["x_test"], host["y_test"], predicted, path=OUT_DIR / "fig_errors_scratch.png"
        )
        weight_grid(
            backend.to_numpy(model.weight_matrices()[0]), path=OUT_DIR / "fig_weights_scratch.png"
        )


if __name__ == "__main__":
    main()
