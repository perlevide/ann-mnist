# Guide

Fourteen short chapters. They go in order: the maths comes first, the code
that implements it comes second, and each chapter points at the file where
that piece lives.

If you only read three, read 03, 06 and 08. Those cover the forward pass,
backpropagation and the optimizers, which is most of what a neural network
is.

| # | Chapter | Code it explains |
|---|---------|------------------|
| 01 | [What a neural network is](01-what-is-a-neural-network.md) | |
| 02 | [The perceptron](02-the-perceptron.md) | |
| 03 | [Structure and the forward pass](03-structure-and-forward-pass.md) | `src/scratch/layers.py` |
| 04 | [Activation functions](04-activation-functions.md) | `src/scratch/activations.py` |
| 05 | [Loss functions](05-loss-functions.md) | `src/scratch/losses.py` |
| 06 | [Backpropagation](06-backpropagation.md) | `src/scratch/network.py` |
| 07 | [Initialization](07-initialization.md) | `src/scratch/initializers.py` |
| 08 | [Optimizers](08-optimizers.md) | `src/scratch/optimizers.py` |
| 09 | [Regularization](09-regularization.md) | `src/scratch/layers.py` |
| 10 | [The training loop](10-training-loop.md) | `src/scratch/network.py` |
| 11 | [Evaluation](11-evaluation.md) | `src/metrics.py` |
| 12 | [From NumPy to PyTorch](12-numpy-to-pytorch.md) | `src/torchmlp/` |
| 13 | [What comes after the MLP](13-what-comes-next.md) | |
| 14 | [Running on a GPU](14-running-on-a-gpu.md) | `src/backend.py` |

Also here: a [glossary](glossary.md) of the terms in both English and
Vietnamese, and [references](references.md) for the papers and books behind
the chapters.

## Notation

The same symbols are used everywhere in the guide and in the code.

| Symbol | Meaning | Shape |
|---|---|---|
| $N$ | samples in a batch | |
| $L$ | number of layers with weights | |
| $\mathbf{X}$ | input batch | $(N, 784)$ |
| $\mathbf{W}^{[l]}$ | weights of layer $l$ | $(n_{l-1}, n_l)$ |
| $\mathbf{b}^{[l]}$ | biases of layer $l$ | $(n_l,)$ |
| $\mathbf{Z}^{[l]}$ | pre-activations of layer $l$ | $(N, n_l)$ |
| $\mathbf{A}^{[l]}$ | activations of layer $l$ | $(N, n_l)$ |
| $g$ | activation function | |
| $J$ | loss averaged over the batch | scalar |
| $\eta$ | learning rate | scalar |

Row convention: one sample is a row, so a batch is $(N, D)$ and a layer is
$\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$. Many textbooks use the
column convention, $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, where
$\mathbf{W}$ has shape $(n_l, n_{l-1})$. Both are correct. Mixing them is
the fastest way to get lost in transposes, so this project stays with rows
throughout, which is also what NumPy and PyTorch use.
