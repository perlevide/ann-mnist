"""Small ablations that turn the theory in docs/ into numbers.

    python experiments.py --study activation
    python experiments.py --study all --epochs 5
    python experiments.py --study all --seeds 5

Each setting is trained once per seed and the table reports the mean
validation accuracy with the spread across seeds. The spread is the point.
A single short run separates settings by a few tenths of a percent, and that
gap moves when anything perturbs the random stream, so a table of single
runs invites conclusions the data does not support. Read a difference as
real only when it clears the spread.
"""

import argparse
import csv
import time
from pathlib import Path
from statistics import mean

from src import backend
from src.config import OUT_DIR, TrainConfig
from src.seeds import seed_everything

STUDIES = {
    "activation": [
        {"activation": name, "label": name} for name in ("relu", "tanh", "sigmoid", "leaky_relu")
    ],
    "init": [{"init": name, "label": name} for name in ("zeros", "normal", "xavier", "he")],
    "optimizer": [
        {"optimizer": "sgd", "momentum": 0.0, "learning_rate": 0.05, "label": "sgd"},
        {"optimizer": "sgd", "momentum": 0.9, "learning_rate": 0.05, "label": "sgd+momentum"},
        {"optimizer": "nesterov", "momentum": 0.9, "learning_rate": 0.05, "label": "nesterov"},
        {"optimizer": "rmsprop", "learning_rate": 0.001, "label": "rmsprop"},
        {"optimizer": "adam", "learning_rate": 0.001, "label": "adam"},
    ],
    "depth": [
        {"hidden_sizes": (), "label": "0 hidden (linear)"},
        {"hidden_sizes": (128,), "label": "1 hidden"},
        {"hidden_sizes": (256, 128), "label": "2 hidden"},
        {"hidden_sizes": (256, 128, 64), "label": "3 hidden"},
    ],
    "regularization": [
        {"label": "none"},
        {"weight_decay": 1e-4, "label": "L2 1e-4"},
        {"dropout": 0.2, "label": "dropout 0.2"},
        {"weight_decay": 1e-4, "dropout": 0.2, "label": "both"},
    ],
}


def train_once(splits, base: TrainConfig, override: dict, init: str | None, seed: int) -> tuple:
    from src.scratch.network import MLP

    settings = {**base.__dict__, **override, "seed": seed}
    config = TrainConfig(**settings)
    seed_everything(config.seed)

    model = MLP(
        config.layer_sizes(),
        activation=config.activation,
        dropout=config.dropout,
        init=init,
        seed=config.seed,
    )
    model.fit(
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
        verbose=False,
    )
    _, train_acc = model.evaluate(splits["x_train"], splits["y_train"])
    val_loss, val_acc = model.evaluate(splits["x_val"], splits["y_val"])
    return train_acc, val_acc, val_loss


def run_setting(splits, base: TrainConfig, override: dict, init: str | None, seeds: int) -> dict:
    """Train one setting once per seed and summarize."""
    label = override.pop("label")
    started = time.time()
    results = [train_once(splits, base, override, init, base.seed + i) for i in range(seeds)]

    train_accs = [r[0] for r in results]
    val_accs = [r[1] for r in results]
    val_losses = [r[2] for r in results]

    return {
        "setting": label,
        "seeds": seeds,
        "train_acc": round(mean(train_accs), 4),
        "val_acc": round(mean(val_accs), 4),
        "val_acc_min": round(min(val_accs), 4),
        "val_acc_max": round(max(val_accs), 4),
        "val_acc_spread": round(max(val_accs) - min(val_accs), 4),
        "val_loss": round(mean(val_losses), 4),
        "seconds": round(time.time() - started, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="all", choices=[*STUDIES, "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset", type=int, default=20000, help="training samples, 0 for all")
    parser.add_argument("--device", default="cpu", help="cpu (NumPy) or cuda (CuPy)")
    parser.add_argument("--seeds", type=int, default=3, help="runs per setting")
    args = parser.parse_args()

    backend.select(args.device)
    from src.data import load_splits

    base = TrainConfig(epochs=args.epochs)
    splits = load_splits(val_fraction=base.val_fraction, normalize=base.normalize, seed=base.seed)
    if args.subset:
        splits["x_train"] = splits["x_train"][: args.subset]
        splits["y_train"] = splits["y_train"][: args.subset]
    if backend.name != "numpy":
        splits = {k: backend.asarray(v) for k, v in splits.items()}
    print(backend.describe())

    names = list(STUDIES) if args.study == "all" else [args.study]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{args.seeds} seeds per setting, {args.epochs} epochs, {len(splits['x_train'])} training samples")
    for name in names:
        print(f"\n=== {name} ===")
        rows = []
        for override in [dict(o) for o in STUDIES[name]]:
            init = override.pop("init", None)
            row = run_setting(splits, base, override, init, args.seeds)
            rows.append(row)
            print(
                f"{row['setting']:<20} train {row['train_acc']:.4f}  val {row['val_acc']:.4f}"
                f"  spread {row['val_acc_spread']:.4f}  val_loss {row['val_loss']:.4f}"
                f"  {row['seconds']:>5.1f} s"
            )
        path = Path(OUT_DIR) / f"study_{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
