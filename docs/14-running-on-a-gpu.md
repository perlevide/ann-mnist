# 14. Running on a GPU

Code: `src/backend.py`, `src/torchmlp/dataset.py`, `check_gpu.py`, `benchmark.py`

Both trainers take `--device cuda`. They get there by different routes,
because they depend on different libraries.

## Two independent questions

`train_torch.py` needs a PyTorch build with CUDA in it.

`train_scratch.py` needs CuPy, which implements the NumPy API on CUDA.
`src/backend.py` swaps `numpy` for `cupy` before the model code is imported,
so the hand written forward and backward passes run on the GPU with no
change to the maths. Every derivative in chapter 06 is still doing the work.

Neither implies the other. You can have a working PyTorch GPU setup and a
CPU-only from scratch trainer, which is the default.

```
python check_gpu.py
```

reports both, plus a matmul timing on each backend it finds.

## Setup

```bash
nvidia-smi                                   # driver, and the CUDA version it supports

pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install cupy-cuda13x
```

Match the cu tag to what `nvidia-smi` reports. Use cu128 or cu126 and
`cupy-cuda12x` on an older driver.

One trap is worth naming, because it costs people hours. On Windows, plain
`pip install torch` gives the CPU-only wheel from PyPI, about 120 MB with no
CUDA inside it. The card is fine, the driver is fine, and
`torch.cuda.is_available()` is still False. `check_gpu.py` prints
`built with CUDA no (CPU-only build)` when this is what happened.

## Failing loudly

`--device cuda` on either trainer stops with an explanation rather than
falling back to the CPU. `--device auto` falls back but prints the reason.

A silent fallback is how a run ends up twenty times slower than expected and
nobody notices for a week. If you asked for the GPU, not getting it should
be an error.

## Keeping the data on the device

A GPU is thousands of small cores waiting for work large enough to fill
them. Getting work to them costs something, and for a small model that cost
is most of the run.

The standard PyTorch pattern is a DataLoader holding CPU tensors, with each
batch copied to the device inside the loop. For 128 images of 784 floats,
that copy plus the Python iterator overhead is larger than the forward and
backward passes together, so the GPU spends most of its time idle.

`DeviceBatches` in `src/torchmlp/dataset.py` uploads the whole split once and
slices it on the device. MNIST as float32 is 170 MB for the training split,
which fits on any card that can run CUDA at all. Shuffling becomes a
`randperm` on the device instead of an index list in Python.

This is the default on a GPU. `--no-resident` restores the DataLoader if you
want to compare. Even on the CPU the resident path is faster in this
project, 8.3 s against 11.8 s over eight epochs, purely from dropping the
DataLoader machinery.

The two loaders draw their shuffles from different random streams, NumPy for
the DataLoader and `torch.randperm` for the resident one, so switching
between them moves the result slightly at the same seed. Each is
reproducible on its own.

TF32 is also enabled on a GPU. It keeps the float32 exponent and truncates
the mantissa to 10 bits, which lets the tensor cores do the matmuls.
Accuracy on this model is unaffected. `--no-tf32` turns it off.

## Where CuPy is not NumPy

CuPy is a drop-in replacement until it is not, and the gaps only appear at
run time on a machine with a card in it. Two showed up building this.

Scalars must stay on the host. `np.sqrt(2.0 / fan_in)` looks harmless and is
correct under NumPy. Under CuPy it asks the GPU to compile and launch a
kernel to take one square root, and it fails outright when the CUDA headers
are missing. `math.sqrt` for scalars, the array module for arrays.

CuPy's `Generator` is a subset of NumPy's. As of CuPy 14 it has `random`,
`uniform`, `standard_normal` and `integers`, but no `permutation`, no
`shuffle` and no `choice`. Shuffling a training set therefore cannot use
`rng.permutation(n)`. `backend.permutation` sorts n random keys instead,
which gives the same result in O(n log n) rather than O(n), microseconds at
this size, and keeps the indices on the device.

`tests/test_gpu_api_compat.py` catches this class of bug without a GPU. It
wraps a NumPy Generator and hides every method CuPy lacks, so anything that
works against it will work on the device.

## Timing it honestly

Both CUDA libraries queue work rather than running it. A timing block
without a synchronize measures how fast Python can submit kernels, and
reports a speedup that is not real.

```python
started = time.perf_counter()
a @ b
torch.cuda.synchronize()          # cupy.cuda.Device().synchronize() for CuPy
elapsed = time.perf_counter() - started
```

`check_gpu.py` and `benchmark.py` both do this. `MLP.fit` calls
`backend.synchronize()` before stopping its clock for the same reason.

## Measure your own machine

```
python benchmark.py
```

runs every combination available as a separate process, because the array
backend is chosen once at import time and cannot be swapped inside a running
interpreter. It writes `out/benchmark.md` and `out/benchmark.json`.

CPU reference from this project, twenty epochs, both rows on the same
machine:

| run | time | test accuracy |
|---|---|---|
| from scratch (NumPy) | 34.4 s | 0.9828 |
| PyTorch, DataLoader | 39.6 s | 0.9832 |

Absolute times depend on the CPU, so treat the ratio as the finding and
generate your own table.

## What to expect

Modest gains, and possibly none.

235,146 parameters with a batch of 128 is three small matmuls per step. A
modern CPU finishes each one in microseconds, so the run is dominated by
launch overhead and Python, neither of which a GPU fixes. CuPy can come out
slower than NumPy here for exactly that reason: every small operation
becomes a kernel launch, and this network performs many small operations per
step.

What makes a GPU pull ahead is arithmetic per byte moved. Try:

```
python train_torch.py --device cuda --hidden 4096 4096 --batch-size 1024
python train_scratch.py --device cuda --hidden 4096 4096 --batch-size 1024
```

Now each layer is a large matmul, the batch is eight times bigger, and the
ordering is no longer in doubt.

This is worth internalizing before reaching for bigger hardware. A GPU is
not a speed setting. It pays off when the work per launch is large, which is
why convolutional networks and transformers live on them and a small MLP on
MNIST does not need one.

## Back to the guide

[Guide index](README.md)
