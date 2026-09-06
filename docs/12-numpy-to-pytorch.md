# 12. From NumPy to PyTorch

Code: `src/torchmlp/`

Same network, same maths, a tenth of the code. What disappears is the
backward pass, because autograd builds it.

## Side by side

The model:

```python
# src/scratch/network.py
self.layers = [
    Linear(784, 256, rng, "he"), ReLU(),
    Linear(256, 128, rng, "he"), ReLU(),
    Linear(128, 10, rng, "he"),
]
```

```python
# src/torchmlp/model.py
self.net = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
```

The training step:

```python
# NumPy
logits = self.forward(x_batch, training=True)
loss = self.loss_fn.forward(logits, y_batch)
self.backward(self.loss_fn.backward())
optimizer.step(self.params_and_grads())
```

```python
# PyTorch
optimizer.zero_grad(set_to_none=True)
loss = criterion(model(x), y)
loss.backward()
optimizer.step()
```

`loss.backward()` replaces every derivative in chapter 06. Autograd recorded
the operations of the forward pass in a graph and walks it in reverse, doing
exactly what `Linear.backward` and the activation classes did by hand.

## What autograd actually does

Each tensor with `requires_grad=True` remembers the operation that produced
it. The forward pass builds a graph of those operations. `backward()` walks
it from the loss to the leaves, applying the local derivative at each node
and accumulating into `.grad`.

The rules it applies are the ones already derived: for a matrix product,
$\partial J / \partial \mathbf{W} = \mathbf{A}^\top \boldsymbol{\delta}$; for
an elementwise function, a Hadamard product with its derivative. Nothing new
is happening, it is just automated. Reverse mode differentiation is the
technical name, and it predates neural networks by decades.

## Things that bite

Gradients accumulate. `.grad` is added to, not overwritten. Skip
`zero_grad()` and step $k$ uses the sum of the first $k$ gradients. This is
deliberate, because it lets you split a large batch across several forward
passes, and it is the most common PyTorch bug.

`model.train()` and `model.eval()` switch dropout and batch normalization
between their two behaviours. Evaluating without `eval()` gives noisy
numbers; training after forgetting to switch back turns the regularization
off.

`torch.no_grad()` skips graph construction. Evaluation without it still
works, but it builds a graph you will never use and spends the memory for
it.

`nn.CrossEntropyLoss` takes logits. Passing softmax output applies softmax
twice, which flattens the distribution and slows training without raising an
error.

`.item()` pulls a Python float out of a one element tensor. Accumulating the
tensor itself instead keeps the entire graph alive and leaks memory across an
epoch.

## Matching the two implementations

`src/torchmlp/` deliberately does not use `torchvision.datasets.MNIST`. Both
paths read the same IDX files through `src/data.py` with the same
normalization and the same validation split, so the comparison measures the
implementations rather than two different preprocessing pipelines.
`reset_parameters` also sets He or Xavier explicitly, because PyTorch's
default for `nn.Linear` is a Kaiming uniform variant that differs slightly.

## The numbers

Twenty epochs, `784 -> 256 -> 128 -> 10`, ReLU, SGD with momentum 0.9,
learning rate 0.05, batch 128, seed 0. Both columns measured on the same CPU
in one `benchmark.py` run:

| | NumPy | PyTorch |
|---|---|---|
| test accuracy | 0.9827 | 0.9832 |
| test loss | 0.0918 | 0.0908 |
| parameters | 235,146 | 235,146 |
| time | 26.5 s | 42.7 s |

The accuracies agree to within run to run noise, which is the point: the
hand written backward pass is correct.

PyTorch being slower here is not a mistake. This model is small enough that
its per operation overhead, the DataLoader, and the graph bookkeeping cost
more than they save, while the NumPy version is a handful of large BLAS calls
with nothing around them. Move to a GPU, or to a convolutional network, and
the ordering reverses by a wide margin.

## Running on a GPU

Both trainers take `--device cuda`. PyTorch needs a CUDA build; the from
scratch trainer needs CuPy. [Chapter 14](14-running-on-a-gpu.md) covers the
setup, the measurements, and why a model this small does not benefit much.

## What PyTorch gives you that NumPy does not

GPU execution by moving tensors with `.to(device)` and nothing else. Autograd
for any architecture you can write, including ones whose backward pass would
take a day to derive by hand. Layers that would be tedious to implement:
convolution, batch normalization, attention. A data pipeline with parallel
loading. And a saving format and ecosystem other people can read.

Which is why the from scratch version exists for one project and PyTorch for
everything after it.

## Next

[13. What comes after the MLP](13-what-comes-next.md).
