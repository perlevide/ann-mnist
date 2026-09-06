# 08. Optimizers

Code: `src/scratch/optimizers.py`

Backpropagation says which way is downhill. The optimizer decides how far to
step and whether to remember where it has been.

## Gradient descent

$$\theta \leftarrow \theta - \eta \nabla_\theta J(\theta)$$

The gradient points in the direction of steepest increase, so the minus sign
goes downhill. $\eta$ is the learning rate, and it is the hyperparameter that
matters most.

Too small and training crawls. Too large and the step overshoots the
minimum, the loss oscillates or diverges, and the run prints NaN a few
epochs in. There is no formula for the right value. The practical method is
to try 0.001, 0.01, 0.1 and watch the first two epochs.

## Batch, stochastic and mini batch

Full batch uses all 54000 samples per update: an exact gradient, one update
per epoch, and far too slow.

Stochastic gradient descent uses one sample per update: 54000 updates per
epoch, a very noisy gradient, and no use of vectorized hardware.

Mini batch uses 32 to 512 samples. The gradient is an unbiased estimate of
the true one, the variance falls as $1/\sqrt{N}$, and the matrix products are
large enough to keep BLAS busy. Everything in practice is mini batch, even
when the papers say SGD.

The noise is not purely a cost. It lets the iterate escape shallow local
minima and narrow crevices that a full batch gradient would settle into, and
small batches generalize slightly better on many problems for this reason.

## Momentum

Plain SGD in a ravine, steep in one direction and shallow in another, will
bounce across the steep walls while inching along the floor. Momentum fixes
that by averaging the recent gradients:

$$\mathbf{v} \leftarrow \mu \mathbf{v} + \nabla_\theta J, \qquad \theta \leftarrow \theta - \eta \mathbf{v}$$

with $\mu = 0.9$. The components that keep changing sign cancel in the
average; the ones that keep pointing the same way add up. With a constant
gradient $\mathbf{g}$, the velocity converges to a geometric series:

$$\mathbf{v}_\infty = \mathbf{g}\sum_{k=0}^{\infty}\mu^k = \frac{\mathbf{g}}{1 - \mu}$$

so $\mu = 0.9$ multiplies the effective step by 10 along a consistent
direction. That is where the speedup comes from, and it is also why raising
$\mu$ and $\eta$ together usually diverges.

Nesterov momentum evaluates the gradient after the momentum step rather than
before it, so the correction is aimed at where the parameters are heading.
In practice the gain is small.

## RMSProp

Different parameters need different step sizes. Divide by a running root
mean square of each coordinate's own gradient:

$$s \leftarrow \rho s + (1 - \rho) g^2, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s} + \epsilon} g$$

A coordinate with consistently large gradients gets small steps; a rarely
active one gets relatively larger ones. $\epsilon = 10^{-8}$ only prevents
division by zero.

The step size ends up roughly $\eta$ in magnitude regardless of the gradient,
which is a feature during training and a limitation at the end: the iterate
settles into a band of width about $\eta$ around the minimum rather than
landing on it. `tests/test_optimizers.py` demonstrates exactly this.

## Adam

Adam (Kingma and Ba, 2014) keeps both averages:

$$m \leftarrow \beta_1 m + (1 - \beta_1) g$$
$$s \leftarrow \beta_2 s + (1 - \beta_2) g^2$$
$$\hat{m} = \frac{m}{1 - \beta_1^t}, \qquad \hat{s} = \frac{s}{1 - \beta_2^t}$$
$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{s}} + \epsilon}\hat{m}$$

Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$.

### The bias correction

Both averages start at zero, which drags them toward zero for the first
steps. Unrolling the recursion for $m$ with $m_0 = 0$:

$$m_t = (1 - \beta_1)\sum_{i=1}^{t} \beta_1^{t-i} g_i$$

If the gradients are drawn from a distribution with mean $\mathbb{E}[g]$,

$$\mathbb{E}[m_t] = \mathbb{E}[g]\,(1 - \beta_1)\sum_{i=1}^{t}\beta_1^{t-i} = \mathbb{E}[g]\,(1 - \beta_1^t)$$

using the finite geometric sum. The estimate is short by exactly the factor
$(1 - \beta_1^t)$, so dividing by it removes the bias. At $t = 1$ with
$\beta_1 = 0.9$ that factor is 0.1, so the correction is a factor of 10 and
the first step is a normal size instead of a tenth of one. As $t$ grows,
$\beta_1^t \to 0$ and the correction fades. The same argument applies to $s$
with $\beta_2$.

`test_adam_first_step_size_is_close_to_the_learning_rate` in the test suite
is this paragraph as an assertion.

## Learning rate schedules

A large rate early moves fast; a small rate late settles. Decaying the rate
gets both:

$$\eta_t = \eta_0 \gamma^{t}$$

with $\gamma$ around 0.95 per epoch. Cosine annealing and step decay are the
other common shapes. `--lr-decay 0.95` in the training scripts applies the
exponential form.

## Measured

```
python experiments.py --study optimizer
```

| optimizer | train acc | val acc | val loss |
|---|---|---|---|
| sgd | 0.9729 | 0.9413 | 0.1925 |
| sgd + momentum | 0.9943 | 0.9605 | 0.1521 |
| nesterov | 0.9837 | 0.9512 | 0.1872 |
| rmsprop | 0.9770 | 0.9447 | 0.1990 |
| adam | 0.9927 | 0.9617 | 0.1282 |

Note that each optimizer here uses its own conventional learning rate, 0.05
for the SGD family and 0.001 for RMSProp and Adam, because comparing them at
a single rate would only measure which one happens to like that rate.

Momentum is the clear win over plain SGD, and Adam has the lowest validation
loss. Nesterov landing below plain momentum is run to run variation at five
epochs, not a real ordering.

## Choosing

Adam at 0.001 when you want something that works without tuning. SGD with
momentum 0.9 and a decaying rate when you have time to tune and want the
last fraction of a percent, which is still the recipe behind many published
image results. Adam's fast early progress makes it the better default while
you are changing the model itself.

## Next

[09. Regularization](09-regularization.md), for when the model fits the
training set too well.
