# data

`download_data.py` writes the four MNIST IDX files into `raw/`:

```
raw/train-images-idx3-ubyte.gz   60000 training images, 28x28, uint8
raw/train-labels-idx1-ubyte.gz   60000 training labels, 0-9
raw/t10k-images-idx3-ubyte.gz    10000 test images
raw/t10k-labels-idx1-ubyte.gz    10000 test labels
```

The files are ignored by git. Run the download script once after cloning.

`src/data.py` parses the IDX format directly, so no dataset library is
needed. The parser is short enough to read in one sitting and it shows what
a raw dataset looks like before any framework wraps it.
