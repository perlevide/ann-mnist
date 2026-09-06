# 03. Structure and the forward pass

Code: `src/scratch/layers.py`, `src/scratch/network.py`

## Layers

A fully connected layer takes every input and feeds it to every unit. With
$n_{l-1}$ inputs and $n_l$ units, layer $l$ holds

$$\mathbf{W}^{[l]} \in \mathbb{R}^{n_{l-1} \times n_l}, \qquad \mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$$

For one sample $\mathbf{a}^{[l-1]}$ written as a row:

$$\mathbf{z}^{[l]} = \mathbf{a}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}, \qquad \mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$$

The network used here is $784 \to 256 \to 128 \to 10$:

```
input 784
  Linear(784, 256)  ->  ReLU  ->  256
  Linear(256, 128)  ->  ReLU  ->  128
  Linear(128, 10)                  10 logits
```

The output layer has no activation. The softmax that turns logits into
probabilities lives inside the loss function, for reasons in chapter 05.

## Batching

Processing one image at a time wastes the machine. Stack $N$ images as rows
of $\mathbf{X} \in \mathbb{R}^{N \times 784}$ and the same formula covers
the whole batch:

$$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$$

$\mathbf{A}^{[l-1]}\mathbf{W}^{[l]}$ is $(N, n_{l-1}) \times (n_{l-1}, n_l)
= (N, n_l)$, and $\mathbf{b}^{[l]}$ of shape $(n_l,)$ is broadcast down the
rows. In NumPy that is one line:

```python
def forward(self, x, training=True):
    self.x = x
    return x @ self.W + self.b
```

Speed comes from the batch. A single $(128, 784) \times (784, 256)$ product
runs in one BLAS call on contiguous memory; 128 separate vector products do
the same arithmetic while spending most of the time on Python overhead and
cache misses.

## Shapes, in full

For a batch of 128:

| Quantity | Shape | Count |
|---|---|---|
| $\mathbf{X}$ | (128, 784) | |
| $\mathbf{W}^{[1]}$, $\mathbf{b}^{[1]}$ | (784, 256), (256,) | 200,960 |
| $\mathbf{A}^{[1]}$ | (128, 256) | |
| $\mathbf{W}^{[2]}$, $\mathbf{b}^{[2]}$ | (256, 128), (128,) | 32,896 |
| $\mathbf{A}^{[2]}$ | (128, 128) | |
| $\mathbf{W}^{[3]}$, $\mathbf{b}^{[3]}$ | (128, 10), (10,) | 1,290 |
| $\mathbf{Z}^{[3]}$ | (128, 10) | |

235,146 parameters, of which 85 percent sit in the first layer. That is what
"fully connected" costs: every one of the 784 inputs gets its own weight into
every one of the 256 hidden units. `model.summary()` prints this table.

## Input scaling

Raw pixels are 0 to 255. Feed those in directly and the first
pre-activations land in the hundreds, which saturates a sigmoid completely
and makes the first gradient step enormous for a ReLU. Two steps fix it:

$$x' = \frac{x}{255} \in [0, 1], \qquad x'' = \frac{x' - \mu}{\sigma}$$

with $\mu = 0.1307$ and $\sigma = 0.3081$, the mean and standard deviation
of the MNIST training pixels. The result is centered near zero with unit
scale, which is the range activation functions and initialization schemes
are designed around.

Use the training statistics for the test set too. Computing $\mu$ and
$\sigma$ on the test set leaks information about data the model is not
supposed to have seen, and on a deployed model there is no test set to
compute them from.

`flatten_and_scale` in `src/data.py` does both steps and offers `"unit"` if
you want to see what dropping the second one costs.

## The forward pass in code

```python
def forward(self, x, training=True):
    for layer in self.layers:
        x = layer.forward(x, training=training)
    return x
```

`self.layers` is a flat list, `[Linear, ReLU, Linear, ReLU, Linear]`, so the
loop is the whole forward pass. Activations are objects rather than
functions because each one has to remember something from the forward pass
for its backward pass, which is chapter 06.

The `training` flag matters for dropout, which drops units during training
and does nothing at evaluation time. Forgetting to switch it off is a common
bug: the model then reports noisy accuracy that changes on every call.

## Next

[04. Activation functions](04-activation-functions.md), the $g$ that has been
sitting in these formulas.
