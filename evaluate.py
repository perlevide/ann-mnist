"""Score a saved model on the test set and write the report to out/.

    python evaluate.py --model models/scratch_mlp.npz
    python evaluate.py --model models/torch_mlp.pt
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.config import MODEL_DIR, OUT_DIR, TrainConfig
from src.data import load_splits
from src.metrics import accuracy, confusion_matrix, format_report, top_confusions


def load_any(path: Path):
    """Dispatch on the file extension: .npz is the NumPy model, .pt the PyTorch one."""
    path = Path(path)
    if path.suffix == ".npz":
        from src.scratch.network import MLP

        model = MLP.load(path)
        return lambda x: model.predict(x), model
    if path.suffix == ".pt":
        import torch

        from src.torchmlp.engine import load_checkpoint

        model, _ = load_checkpoint(path)
        model.eval()

        def predict(x):
            with torch.no_grad():
                return model(torch.from_numpy(np.ascontiguousarray(x))).argmax(dim=1).numpy()

        return predict, model
    raise ValueError(f"do not know how to load {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DIR / "scratch_mlp.npz")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    defaults = TrainConfig()
    splits = load_splits(val_fraction=defaults.val_fraction, normalize=defaults.normalize, seed=defaults.seed)
    x, y = splits[f"x_{args.split}"], splits[f"y_{args.split}"]

    predict, _ = load_any(args.model)
    predicted = predict(x)
    matrix = confusion_matrix(predicted, y)

    print(f"model {args.model}  split {args.split}  n {len(y)}")
    print()
    print(format_report(matrix))
    print()
    for true_class, predicted_class, count in top_confusions(matrix):
        print(f"  {true_class} -> {predicted_class}: {count}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"eval_confusion_{args.split}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *range(matrix.shape[1])])
        for i, row in enumerate(matrix):
            writer.writerow([i, *row])
    print(f"\naccuracy {accuracy(predicted, y):.4f}")
    print(f"wrote {out_dir / f'eval_confusion_{args.split}.csv'}")


if __name__ == "__main__":
    main()
