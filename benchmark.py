"""Time both implementations on every device available, and write a table.

    python benchmark.py
    python benchmark.py --epochs 5

Writes out/benchmark.md and out/benchmark.json. Each run is a separate
process, because the array backend is chosen once at import time and cannot
be swapped inside a running interpreter.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import OUT_DIR, ROOT


def has_cupy() -> bool:
    try:
        import cupy
    except ImportError:
        return False
    return cupy.cuda.runtime.getDeviceCount() > 0


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def has_torch_cuda() -> bool:
    return has_torch() and __import__("torch").cuda.is_available()


def run(script: str, device: str, epochs: int, extra: list) -> dict:
    """Run one trainer and read back the summary it writes."""
    summary = OUT_DIR / ("scratch_summary.json" if "scratch" in script else "torch_summary.json")
    summary.unlink(missing_ok=True)

    command = [sys.executable, script, "--device", device, "--epochs", str(epochs), "--no-plots", *extra]
    print("  " + " ".join(command[1:]))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"{script} failed on {device}")

    return json.loads(summary.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    jobs = [("from scratch (NumPy)", "train_scratch.py", "cpu", [])]
    if has_cupy():
        jobs.append(("from scratch (CuPy)", "train_scratch.py", "cuda", []))
    if has_torch():
        jobs.append(("PyTorch, DataLoader", "train_torch.py", "cpu", []))
    if has_torch_cuda():
        jobs.append(("PyTorch, resident", "train_torch.py", "cuda", []))
        jobs.append(("PyTorch, DataLoader", "train_torch.py", "cuda", ["--no-resident"]))

    rows = []
    for label, script, device, extra in jobs:
        print(f"{label} on {device}")
        summary = run(script, device, args.epochs, extra)
        rows.append(
            {
                "run": label,
                "device": device,
                "seconds": round(summary["seconds"], 1),
                "test_accuracy": round(summary["test_accuracy"], 4),
                "test_loss": round(summary["test_loss"], 4),
            }
        )

    baseline = rows[0]["seconds"]
    lines = [
        f"Twenty epochs is the default; this run used {args.epochs}. "
        "Same architecture, same data, same seed.",
        "",
        "| run | device | time | speedup | test accuracy | test loss |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['device']} | {row['seconds']} s "
            f"| {baseline / row['seconds']:.2f}x | {row['test_accuracy']} | {row['test_loss']} |"
        )

    table = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    (OUT_DIR / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print()
    print(table)
    print()
    print(f"wrote {OUT_DIR / 'benchmark.md'}")


if __name__ == "__main__":
    main()
