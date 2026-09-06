# 02. The perceptron

One unit, one weight vector, one output. Everything in the rest of the guide
is this repeated and stacked.

## The unit

Given an input $\mathbf{x} \in \mathbb{R}^{n}$, weights $\mathbf{w} \in
\mathbb{R}^{n}$ and a bias $b$:

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b$$

$$a = g(z)$$

$z$ is the pre-activation and $a$ is the activation. Rosenblatt's 1958
perceptron used the step function

$$g(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$$

## What the bias does

Without $b$ the boundary $\mathbf{w}^\top\mathbf{x} = 0$ passes through the
origin, which forces every decision surface through a single point for no
good reason. The bias shifts it:

$$\mathbf{w}^\top \mathbf{x} = -b$$

In one dimension, $w x + b = 0$ puts the threshold at $x = -b/w$, and without
$b$ it is stuck at zero.

Some texts fold the bias into the weights by appending a constant 1 to every
input. That is tidy on paper. Keeping $b$ separate is better in code, because
weight decay should shrink weights and leave biases alone, and because the
shapes stay obvious.

## The perceptron learning rule

For each misclassified sample, with $y \in \{0, 1\}$ and prediction
$\hat{y}$:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}, \qquad b \leftarrow b + \eta (y - \hat{y})$$

If the true label is 1 and the unit said 0, the input is added to the weight
vector, which raises $\mathbf{w}^\top\mathbf{x}$ for that sample and any
similar one. If the true label is 0 and the unit said 1, the input is
subtracted. Correct samples change nothing.

Rosenblatt proved this converges in a finite number of updates whenever the
data is linearly separable. The proof gives no bound you can use in
practice, and it says nothing about data that is not separable, where the
rule simply never settles.

## Why the step function had to go

The step function has derivative zero everywhere it is defined, and no
derivative at all at $z = 0$. Gradient descent needs $\partial J / \partial
w_i$, and here that is zero, so there is no signal saying which way to move.
The perceptron rule works around this with a hand written update, but the
trick does not survive stacking: with two layers there is no way to say how
much of the output error belongs to a hidden unit.

Replacing the step with a smooth function fixes both problems at once. The
sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

is the usual first choice, and its derivative $\sigma(z)(1 - \sigma(z))$ is
nonzero everywhere. Once every piece of the network is differentiable, the
chain rule works through the whole thing, and that is backpropagation.

## XOR, and why one layer is not enough

Minsky and Papert pointed out in 1969 that a single perceptron cannot
compute XOR. Four points:

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single unit needs one straight line with the two 1s on one side and the
two 0s on the other. Suppose such a $(\mathbf{w}, b)$ exists. From the two
positive cases, $w_1 + b > 0$ and $w_2 + b > 0$, so $w_1 + w_2 + 2b > 0$.
From the two negative cases, $b < 0$ and $w_1 + w_2 + b < 0$. Subtracting
the last from the sum gives $b > 0$, which contradicts $b < 0$. No such line
exists.

Two layers solve it. Let the hidden units compute OR and NAND, and let the
output unit compute their AND:

$$h_1 = \text{OR}(x_1, x_2), \quad h_2 = \text{NAND}(x_1, x_2), \quad y = \text{AND}(h_1, h_2)$$

The hidden layer has re-expressed the input in coordinates where the problem
is linearly separable. That is what every hidden layer does, and on MNIST it
is the same story with 784 inputs instead of 2.

This result stalled the field for over a decade, not because it was wrong,
but because nobody had a way to train the second layer. Backpropagation,
popularized by Rumelhart, Hinton and Williams in 1986, was that way.

## In the code

`src/scratch/layers.py` has no Perceptron class. A layer of $n$ perceptrons
sharing an input is exactly `Linear(n_in, n_out)`, where column $j$ of
$\mathbf{W}$ is the weight vector of unit $j$ and $b_j$ is its bias. Writing
them one at a time would be the same arithmetic at a fraction of the speed.

## Next

[03. Structure and the forward pass](03-structure-and-forward-pass.md) turns
one unit into a network.
