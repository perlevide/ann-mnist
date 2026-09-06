# 04. Activation functions

Code: `src/scratch/activations.py`

Without $g$ the network collapses to one linear map (chapter 01). With $g$
it does not. The choice of $g$ then decides how well gradients survive the
trip back through the layers.

## Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

Derivative, worth doing once by hand. Write $\sigma = (1 + e^{-z})^{-1}$ and
use the chain rule:

$$\sigma'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Split the fraction:

$$\sigma'(z) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}
= \sigma(z) \cdot \left(1 - \sigma(z)\right)$$

using $\dfrac{e^{-z}}{1 + e^{-z}} = \dfrac{1 + e^{-z} - 1}{1 + e^{-z}} = 1 - \sigma(z)$.

So the derivative costs nothing once you have the output. Two problems come
with it.

The maximum of $\sigma(1 - \sigma)$ is at $\sigma = 0.5$, where it equals
0.25. Every sigmoid layer therefore multiplies the backward signal by at
most $1/4$, and by far less once $|z|$ is large. Through $k$ layers the
gradient is scaled by at most $4^{-k}$, so at $k = 10$ it has been divided by
a million. This is the vanishing gradient problem, and it is why deep
sigmoid networks were nearly untrainable before 2010.

The output is also always positive, so every input to the next layer has the
same sign. The gradient with respect to a whole weight vector then tends to
point in one direction for all of its entries at once, and the optimizer has
to zig-zag instead of moving diagonally.

Implementing it needs one piece of care. `1 / (1 + np.exp(-z))` overflows for
very negative $z$. The branch used in the code evaluates the algebraically
equal form $e^{z} / (1 + e^{z})$ when $z < 0$, so the exponent is never
positive.

## Tanh

$$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} \in (-1, 1), \qquad \tanh'(z) = 1 - \tanh^2(z)$$

Zero centered, which removes the second sigmoid problem, and the maximum
derivative is 1 instead of 0.25, which softens the first. Saturation is still
there: outside roughly $|z| > 3$ the derivative is near zero. Note also
$\tanh(z) = 2\sigma(2z) - 1$, so it is a rescaled sigmoid rather than a
different idea.

## ReLU

$$\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

The derivative is exactly 1 on the positive side. No shrinking, no matter
how many layers, which is what makes deep networks trainable. It is also
about as cheap as an operation gets, and it produces exact zeros, so
roughly half the units are inactive for any given input and the later layers
see a sparse signal.

Three things to know.

At $z = 0$ the function is not differentiable. The code returns 0 there. Any
value in $[0, 1]$ is a valid subgradient and the choice has no measurable
effect, because exact zeros essentially never occur in floating point.

A unit whose pre-activation is negative for every input in the dataset
receives zero gradient forever and never recovers. This is a dead unit.
Large learning rates cause it, because one oversized step can push a bias
far enough negative that nothing brings it back.

The output is not zero centered, since it is never negative. In practice
this matters much less than the sigmoid version of the same problem, because
the gradient does not shrink.

## Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \le 0\end{cases}, \qquad \alpha \approx 0.01$$

A dead unit now leaks a small gradient and can come back. On MNIST the
difference is within noise, and it is worth reaching for when a run shows a
large fraction of units stuck at zero.

## Softmax

Softmax is different in kind. It is not elementwise, and it belongs to the
output layer:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

Every output is positive and they sum to 1, so the vector reads as a
distribution over the classes. It is treated in chapter 05, together with
the loss it is always paired with.

The one thing to fix here is overflow. $e^{1000}$ is infinity in float32,
and MNIST logits do get large late in training. Subtract the row maximum
first:

$$\frac{e^{z_i - m}}{\sum_j e^{z_j - m}} = \frac{e^{-m} e^{z_i}}{e^{-m}\sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The value is unchanged and every exponent is now at most zero. That is what
`softmax` in `activations.py` does, and it is not optional.

## Measured

```
python experiments.py --study activation
```

Five epochs, 20000 training samples, otherwise identical settings:

| activation | train acc | val acc | val loss |
|---|---|---|---|
| relu | 0.9943 | 0.9605 | 0.1521 |
| tanh | 0.9971 | 0.9622 | 0.1270 |
| leaky_relu | 0.9967 | 0.9628 | 0.1367 |
| sigmoid | 0.9574 | 0.9385 | 0.2010 |

Sigmoid is clearly behind after the same number of epochs. Tanh keeps up with
ReLU here, which is what you should expect from a network only three layers
deep. The gap opens with depth, and this network is too shallow to show it.
Try `--hidden 128 128 128 128 128 128` if you want to watch sigmoid fail
properly.

## Choosing

ReLU for hidden layers unless you have a reason. Leaky ReLU if units are
dying. Tanh in shallow networks and in recurrent architectures, where the
bounded output helps. Sigmoid only for a single output that has to be a
probability, and never buried in the middle of a deep stack. No activation
on the output layer of a classifier, because the loss applies softmax
itself.

## Next

[05. Loss functions](05-loss-functions.md).
