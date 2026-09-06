# 05. Loss functions

Code: `src/scratch/losses.py`

The loss is the number training minimizes. Choose it badly and the gradients
are uninformative no matter how good the architecture is.

## Cross entropy

The network outputs logits $\mathbf{z} \in \mathbb{R}^{10}$. Softmax turns
them into probabilities:

$$p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

With true class $k$, the cross entropy loss for one sample is

$$L = -\sum_{i} y_i \log p_i = -\log p_k$$

since the one hot target $\mathbf{y}$ has a single nonzero entry. Only the
probability assigned to the correct class appears.

The shape of $-\log p_k$ explains the behaviour. At $p_k = 1$ the loss is 0.
At $p_k = 0.5$ it is 0.69. At $p_k = 0.01$ it is 4.6, and it grows without
bound as $p_k \to 0$. A confidently wrong prediction is punished hard, and
that is deliberate.

An untrained 10 class model should sit near $\log 10 \approx 2.303$. If your
first epoch does not start close to that number, something is wrong before
training even begins: the labels are misaligned, the initialization is
broken, or the inputs are not scaled.

Over a batch, take the mean:

$$J = -\frac{1}{N}\sum_{n=1}^{N} \log p^{(n)}_{k_n}$$

The mean rather than the sum, so that changing the batch size does not
change the scale of the gradient and force you to retune the learning rate.

### Where it comes from

Cross entropy is not an arbitrary choice. Treat the model as defining
$P(y \mid \mathbf{x}; \theta)$ and ask for the $\theta$ that makes the
observed labels most likely:

$$\theta^* = \arg\max_\theta \prod_{n=1}^{N} P(y_n \mid \mathbf{x}_n; \theta)$$

Products of many small numbers underflow, and sums are easier to
differentiate, so take the logarithm, which is monotone and does not move
the maximum:

$$\theta^* = \arg\max_\theta \sum_{n} \log P(y_n \mid \mathbf{x}_n; \theta)
= \arg\min_\theta \left(-\sum_{n} \log p^{(n)}_{k_n}\right)$$

The right hand side is the cross entropy. Minimizing it is maximum
likelihood estimation.

## The fused gradient

This is the single most useful derivation in the guide, so here it is in
full.

Start from $L = -\log p_k$ and $p_i = e^{z_i} / S$ with $S = \sum_j e^{z_j}$.
Write $L = -z_k + \log S$, because $\log p_k = z_k - \log S$.

Differentiate with respect to $z_i$. The first term contributes $-1$ when
$i = k$ and 0 otherwise. The second term:

$$\frac{\partial \log S}{\partial z_i} = \frac{1}{S}\frac{\partial S}{\partial z_i} = \frac{e^{z_i}}{S} = p_i$$

Together:

$$\boxed{\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = k]}$$

In vector form, $\nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}$. Predicted
distribution minus true distribution. Nothing else.

Two reasons this matters.

It is stable. Computing softmax and cross entropy as separate steps produces
a $1/p_k$ factor in the derivative of the log, which overflows exactly when
the model is confidently wrong, which is exactly when you need the gradient.
Fused, the $1/p_k$ cancels analytically and never appears in floating point.

It is cheap. The softmax Jacobian is a full $C \times C$ matrix,
$\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$. Multiplying by the
cross entropy gradient collapses it to a subtraction. Doing that algebra
once at write time saves it at every step of training.

This is why `SoftmaxCrossEntropy` is one class, and why the network's last
layer emits raw logits. PyTorch does the same thing: `nn.CrossEntropyLoss`
takes logits, not probabilities. Passing it softmax output is a common bug
that leaves the model training slowly rather than failing outright.

## Mean squared error

$$J = \frac{1}{2N}\sum_{n} \|\hat{\mathbf{y}}^{(n)} - \mathbf{y}^{(n)}\|^2, \qquad \frac{\partial J}{\partial \hat{\mathbf{y}}^{(n)}} = \frac{\hat{\mathbf{y}}^{(n)} - \mathbf{y}^{(n)}}{N}$$

The factor $1/2$ exists only so the 2 from the derivative cancels.

MSE is the right loss for regression and the wrong one for classification.
Put a sigmoid output in front of it and the gradient with respect to the
pre-activation is

$$\frac{\partial L}{\partial z} = (a - y) \cdot \sigma'(z)$$

That $\sigma'(z)$ is at most 0.25 and near zero when the unit is saturated.
A model that predicts 0.999 when the answer is 0 gets almost no gradient,
which is the opposite of what should happen. Cross entropy has no such
factor, as the boxed result above shows. That is the whole argument.

## Which one

Multi-class, one label per sample: softmax with cross entropy. Multi-label,
where several classes can be true at once: a sigmoid per output with binary
cross entropy. Regression: MSE, or mean absolute error when outliers matter
and you want them to count linearly instead of quadratically.

## Next

[06. Backpropagation](06-backpropagation.md). The gradient derived here is
the starting point, and the chain rule carries it back through the layers.
