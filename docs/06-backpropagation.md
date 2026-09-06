# 06. Backpropagation

Code: `src/scratch/network.py`, `src/scratch/layers.py`, `src/scratch/gradcheck.py`

Backpropagation is the chain rule, applied to a composed function, evaluated
in the order that avoids repeated work. It is not an approximation and it is
not a learning algorithm. It computes exact derivatives, and gradient
descent does the learning.

## What has to be computed

The network is a composition:

$$J = \text{Loss}\big(g(\dots g(\mathbf{X}\mathbf{W}^{[1]} + \mathbf{b}^{[1]})\dots \mathbf{W}^{[L]} + \mathbf{b}^{[L]}), \mathbf{y}\big)$$

Gradient descent needs $\partial J / \partial \mathbf{W}^{[l]}$ and $\partial
J / \partial \mathbf{b}^{[l]}$ for every $l$. For this network that is
235,146 partial derivatives.

Finite differences would need two forward passes per parameter, so about
470,000 forward passes for one gradient. Backpropagation gets all of them
from one forward pass and one backward pass. That ratio is why neural
networks are trainable at all.

## Why backwards

Consider $J = f_3(f_2(f_1(x)))$. The chain rule gives

$$\frac{\partial J}{\partial x} = \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}$$

Multiplication is associative, so you may group left to right or right to
left. The grouping decides the cost.

Left to right, starting at the input, means multiplying Jacobian by Jacobian:
matrix times matrix, which is expensive. This is forward mode.

Right to left, starting at the loss, means multiplying a row vector by a
Jacobian at each step: vector times matrix, which is cheap. This is reverse
mode, and because the loss is a single scalar, it is the right choice. One
sweep gives every partial derivative.

Reverse mode differentiation on a computational graph is exactly what
backpropagation is, and it is what PyTorch's autograd does.

## The key quantity

Define the error signal at layer $l$:

$$\boldsymbol{\delta}^{[l]} = \frac{\partial J}{\partial \mathbf{Z}^{[l]}}$$

the derivative of the loss with respect to that layer's pre-activations,
shape $(N, n_l)$. Once you have $\boldsymbol{\delta}^{[l]}$, both parameter
gradients follow immediately, and the previous layer's $\boldsymbol{\delta}$
follows too. So the algorithm is: get $\boldsymbol{\delta}$ at the output,
then push it back one layer at a time.

## The four equations

### 1. Output layer

From chapter 05, with softmax and cross entropy and the mean over the batch:

$$\boldsymbol{\delta}^{[L]} = \frac{1}{N}(\mathbf{P} - \mathbf{Y})$$

where $\mathbf{P}$ holds the predicted probabilities and $\mathbf{Y}$ the one
hot labels, both $(N, C)$.

### 2. Weight gradient

For a single sample, $z_j = \sum_i a_i W_{ij} + b_j$, so $\partial z_j /
\partial W_{ij} = a_i$. Only the $j$-th pre-activation depends on
$W_{ij}$, so the chain rule has a single term:

$$\frac{\partial J}{\partial W_{ij}} = \frac{\partial J}{\partial z_j}\frac{\partial z_j}{\partial W_{ij}} = \delta_j a_i$$

Over a batch every sample contributes, so the terms add:

$$\frac{\partial J}{\partial W_{ij}} = \sum_{n=1}^{N} A_{ni}\, \delta_{nj}$$

which is the $(i, j)$ entry of $\mathbf{A}^\top \boldsymbol{\delta}$:

$$\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \big(\mathbf{A}^{[l-1]}\big)^\top \boldsymbol{\delta}^{[l]}$$

Shapes check: $(n_{l-1}, N) \times (N, n_l) = (n_{l-1}, n_l)$, the same shape
as $\mathbf{W}^{[l]}$. A gradient always has the shape of the thing it
differentiates, and checking that catches most transpose errors.

### 3. Bias gradient

$\partial z_j / \partial b_j = 1$, so the batch sum is all that remains:

$$\frac{\partial J}{\partial \mathbf{b}^{[l]}} = \sum_{n=1}^{N} \boldsymbol{\delta}^{[l]}_{n}$$

`grad_out.sum(axis=0)`, shape $(n_l,)$.

### 4. Propagating back

The activation $a_i$ of layer $l-1$ feeds every unit $j$ of layer $l$, so
its derivative collects a term from each:

$$\frac{\partial J}{\partial a_i} = \sum_j \frac{\partial J}{\partial z_j}\frac{\partial z_j}{\partial a_i} = \sum_j \delta_j W_{ij}$$

In matrix form:

$$\frac{\partial J}{\partial \mathbf{A}^{[l-1]}} = \boldsymbol{\delta}^{[l]} \big(\mathbf{W}^{[l]}\big)^\top$$

Shapes: $(N, n_l) \times (n_l, n_{l-1}) = (N, n_{l-1})$.

Then through the activation, which is elementwise, so it is a Hadamard
product and not a matrix product:

$$\boldsymbol{\delta}^{[l-1]} = \frac{\partial J}{\partial \mathbf{A}^{[l-1]}} \odot g'\big(\mathbf{Z}^{[l-1]}\big)$$

Those four equations are the whole algorithm. Everything else is
bookkeeping.

## Worked example

Two layers, ReLU, one sample, no bias, so the arithmetic fits on a page.

$$\mathbf{x} = [1, 2], \quad \mathbf{W}^{[1]} = \begin{bmatrix} 0.5 & -0.5 \\ 1.0 & 0.5\end{bmatrix}, \quad \mathbf{W}^{[2]} = \begin{bmatrix} 1.0 & 0.0 \\ -1.0 & 2.0\end{bmatrix}$$

True class $k = 0$.

Forward:

$$\mathbf{z}^{[1]} = [1, 2]\mathbf{W}^{[1]} = [0.5 + 2.0,\; -0.5 + 1.0] = [2.5,\; 0.5]$$
$$\mathbf{a}^{[1]} = \text{ReLU}([2.5, 0.5]) = [2.5,\; 0.5]$$
$$\mathbf{z}^{[2]} = [2.5, 0.5]\mathbf{W}^{[2]} = [2.5 - 0.5,\; 0 + 1.0] = [2.0,\; 1.0]$$

Softmax:

$$\mathbf{p} = \frac{[e^{2}, e^{1}]}{e^{2} + e^{1}} = \frac{[7.389,\; 2.718]}{10.107} = [0.731,\; 0.269]$$

Loss $= -\log 0.731 = 0.313$.

Backward:

$$\boldsymbol{\delta}^{[2]} = \mathbf{p} - \mathbf{y} = [0.731 - 1,\; 0.269] = [-0.269,\; 0.269]$$

$$\frac{\partial J}{\partial \mathbf{W}^{[2]}} = (\mathbf{a}^{[1]})^\top \boldsymbol{\delta}^{[2]}
= \begin{bmatrix} 2.5 \\ 0.5 \end{bmatrix}[-0.269,\; 0.269]
= \begin{bmatrix} -0.673 & 0.673 \\ -0.135 & 0.135 \end{bmatrix}$$

$$\frac{\partial J}{\partial \mathbf{a}^{[1]}} = \boldsymbol{\delta}^{[2]}(\mathbf{W}^{[2]})^\top = [-0.269 \cdot 1.0 + 0.269 \cdot 0.0,\; -0.269 \cdot (-1.0) + 0.269 \cdot 2.0] = [-0.269,\; 0.807]$$

Both entries of $\mathbf{z}^{[1]}$ are positive, so the ReLU derivative is 1
and nothing is masked:

$$\boldsymbol{\delta}^{[1]} = [-0.269,\; 0.807]$$

$$\frac{\partial J}{\partial \mathbf{W}^{[1]}} = \mathbf{x}^\top \boldsymbol{\delta}^{[1]}
= \begin{bmatrix} 1 \\ 2 \end{bmatrix}[-0.269,\; 0.807]
= \begin{bmatrix} -0.269 & 0.807 \\ -0.538 & 1.614 \end{bmatrix}$$

Read the sign of $\partial J / \partial W^{[2]}_{11} = -0.673$: increasing
that weight decreases the loss, which is right, since it feeds the correct
class.

## In code

```python
def backward(self, grad):
    for layer in reversed(self.layers):
        grad = layer.backward(grad)
```

Each layer receives the derivative of the loss with respect to its own
output and returns the derivative with respect to its input. That contract
is all the loop needs to know, which is why layers can be reordered or
swapped freely.

`Linear.backward` is the three equations above:

```python
def backward(self, grad_out):
    self.dW = self.x.T @ grad_out
    self.db = grad_out.sum(axis=0)
    return grad_out @ self.W.T
```

`self.x` was saved during the forward pass, because equation 2 needs it.
This is why training uses more memory than inference: every layer holds its
input until the backward pass consumes it.

## Vanishing and exploding gradients

Unroll equation 4 through $L$ layers and the gradient at layer 1 is a
product of $L$ Jacobians. If the typical factor is below 1 the product
shrinks geometrically, and the early layers stop learning. If it is above 1
the product grows and the update overflows.

The three practical answers, all covered later: ReLU, whose derivative is
exactly 1 where it is active (chapter 04); initialization chosen to keep the
per layer factor near 1 (chapter 07); and normalization layers, which
rescale activations during training and are the reason networks with dozens
of layers train at all.

## Checking it

Every derivation above can be wrong in a way that still runs. The test for
that is finite differences:

$$\frac{\partial J}{\partial w} \approx \frac{J(w + h) - J(w - h)}{2h}$$

Central difference, error $O(h^2)$, with $h = 10^{-5}$. Compare against the
analytic gradient by relative error:

$$\text{err} = \frac{|a - n|}{\max(|a| + |n|, \epsilon)}$$

Below $10^{-7}$ is correct. Above $10^{-4}$ means a bug. Two conditions
matter: run the check in float64, because float32 rounding noise is the same
size as the difference being measured, and turn dropout off, because a
different random mask on each evaluation makes the comparison meaningless.

```
python -m pytest tests/test_gradcheck.py -v
```

`src/scratch/gradcheck.py` checks a random sample of entries from every
parameter array, for ReLU, tanh and sigmoid networks of two to four layers.
When you add a layer type, add it here first.

## Next

[07. Initialization](07-initialization.md), which decides whether these
gradients start out useful.
