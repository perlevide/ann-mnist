"""Run a saved model on your own digit images.

    python predict.py --model models/scratch_mlp.npz --images my_digits
    python predict.py --model models/torch_mlp.pt --images my_digits --out out/predictions.csv

The images are converted to MNIST's convention: 28 by 28 grayscale, white
stroke on a black background, the digit centered by its center of mass. A
photo of a digit written in black ink on white paper needs inverting, which
`--invert` does. Skipping this step is the usual reason a model with 98
percent test accuracy fails on your own handwriting.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.config import IMAGE_SIZE, MNIST_MEAN, MNIST_STD, MODEL_DIR, OUT_DIR


def load_image(path: Path, invert: bool) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if invert:
        array = 1.0 - array
    return array


def center_by_mass(image: np.ndarray) -> np.ndarray:
    """Shift the digit so its center of mass sits in the middle of the frame.

    MNIST was built this way. A model trained on centered digits has never
    seen one in the corner and will happily misread it.
    """
    total = image.sum()
    if total <= 0:
        return image
    rows, cols = np.indices(image.shape)
    shift_y = int(round(image.shape[0] / 2 - (rows * image).sum() / total))
    shift_x = int(round(image.shape[1] / 2 - (cols * image).sum() / total))
    return np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DIR / "scratch_mlp.npz")
    parser.add_argument("--images", required=True, help="folder of image files")
    parser.add_argument("--out", default=OUT_DIR / "predictions.csv")
    parser.add_argument("--invert", action="store_true", help="dark stroke on light paper")
    parser.add_argument("--no-center", action="store_true")
    args = parser.parse_args()

    paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not paths:
        raise SystemExit(f"no images found in {args.images}")

    batch = []
    for path in paths:
        image = load_image(path, args.invert)
        if not args.no_center:
            image = center_by_mass(image)
        batch.append(image.reshape(-1))
    x = (np.stack(batch) - MNIST_MEAN) / MNIST_STD

    from evaluate import load_any

    predict, _ = load_any(args.model)
    predicted = predict(x.astype(np.float32))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "prediction"])
        for path, label in zip(paths, predicted):
            print(f"{path.name:>30}  {label}")
            writer.writerow([path.name, int(label)])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
