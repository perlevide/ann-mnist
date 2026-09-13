"""Run a saved model on your own digit images.

    python predict.py --model models/scratch_mlp.npz --images my_digits --invert
    python predict.py --model models/torch_mlp.pt --images my_digits --out out/predictions.csv --invert
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import MNIST_MEAN, MNIST_STD, MODEL_DIR, OUT_DIR


def load_image(path: Path, invert: bool) -> np.ndarray:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0

    if invert:
        array = 1.0 - array

    array[array < 0.50] = 0.0

    mask = array > 0.0
    if mask.any():
        nz_y, nz_x = np.nonzero(mask)
        array = array[nz_y.min() : nz_y.max() + 1, nz_x.min() : nz_x.max() + 1]

    h, w = array.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    digit_pil = Image.fromarray((array * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    digit_array = np.asarray(digit_pil, dtype=np.float32) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = digit_array

    return canvas


def center_by_mass(image: np.ndarray) -> np.ndarray:
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

    processed_dir = Path("processed_digits")
    processed_dir.mkdir(parents=True, exist_ok=True)

    batch = []
    for path in paths:
        image = load_image(path, args.invert)
        if not args.no_center:
            image = center_by_mass(image)

        save_img = Image.fromarray((image * 255).astype(np.uint8))
        save_img.save(processed_dir / f"processed_{path.name}")

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
    print(f"saved processed images to {processed_dir.resolve()}")


if __name__ == "__main__":
    main()
    