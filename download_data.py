"""Download the four MNIST files into data/raw.

    python download_data.py
    python download_data.py --force
"""

import argparse

from src.config import DATA_DIR
from src.data import download, load_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=DATA_DIR, help="where to put the files")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    download(args.dest, force=args.force)
    raw = load_raw(args.dest)
    print()
    for key, array in raw.items():
        print(f"{key:>13}  shape {array.shape}  dtype {array.dtype}")


if __name__ == "__main__":
    main()
