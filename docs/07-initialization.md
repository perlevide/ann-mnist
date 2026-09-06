# 07. Initialization

Code: `src/scratch/initializers.py`

Where the weights start decides whether training moves at all.

## Zeros do not work

Set every weight to zero and every unit in a layer computes the same output,
receives the same gradient, and takes the same update. They stay identical
forever, so a layer of 256 units has the expressive power of one unit. This
is symmetry breaking, and randomness is what breaks it.

The measurement:

```
python experiments.py --study init
```

| init | train acc | val acc | val loss |
|---|---|---|---|
| zeros | 0.1111 | 0.1092 | 2.3031 |
| normal (std 0.01) | 0.9855 | 0.9583 | 0.1424 |
| xavier | 0.9960 | 0.9652 | 0.1260 |
| he | 0.9943 | 0.9605 | 0.1521 |

The zeros row is worth staring at. Accuracy 0.109 is chance on ten classes,
and loss 2.3031 is $\log 10$, the loss of a model that outputs a uniform
distribution. Five epochs of training moved nothing.

Biases are a different case. They are usually initialized to zero and that is
fine, because the weights have already broken the symmetry.

## Why the scale matters

Take one unit with $n$ inputs and no bias, $z = \sum_{i=1}^{n} w_i x_i$.
Assume the $w_i$ are independent of the $x_i$, each with mean zero, and let
$\text{Var}(w_i) = \sigma_w^2$ and $\text{Var}(x_i) = \sigma_x^2$. Variance
of a sum of independent terms is the sum of the variances:

$$\text{Var}(z) = \sum_{i=1}^{n} \text{Var}(w_i x_i) = n \sigma_w^2 \sigma_x^2$$

For the variance to hold steady from layer to layer you need
$n\sigma_w^2 = 1$, that is

$$\sigma_w^2 = \frac{1}{n_{\text{in}}}$$

Miss this and the effect compounds. If each layer scales the variance by
0.5, then after 10 layers the signal is $2^{-10}$ of what it was, which is
where "the activations went to zero" comes from. A factor of 2 the other way
overflows just as fast.

## Xavier, for tanh and sigmoid

Glorot and Bengio (2010) asked for the variance to hold in both directions:
forward it wants $1/n_{\text{in}}$, backward it wants $1/n_{\text{out}}$.
Neither is possible at once, so take the harmonic compromise:

$$\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

Implemented here as a uniform distribution. A uniform on $[-a, a]$ has
variance $a^2/3$, so setting $a^2/3 = 2/(n_{\text{in}} + n_{\text{out}})$
gives

$$a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$$

which is the constant 6 that appears in the code without explanation in most
tutorials.

## He, for ReLU

Xavier's derivation assumes $g$ is roughly linear near zero, which is true
for tanh and false for ReLU. ReLU sets half its inputs to zero. If $z$ is
symmetric about zero then

$$\text{Var}(\text{ReLU}(z)) \approx \tfrac{1}{2}\text{Var}(z)$$

so each layer halves the variance. Compensate with a factor of 2 (He et al.,
2015):

$$\text{Var}(w) = \frac{2}{n_{\text{in}}}, \qquad w \sim \mathcal{N}\left(0, \sqrt{\tfrac{2}{n_{\text{in}}}}\right)$$

## Which to use

He with ReLU or leaky ReLU, Xavier with tanh or sigmoid.
`default_for(activation)` in the code picks this for you.

The 0.9652 against 0.9605 in the table above is inside run to run noise, not
evidence that Xavier beats He. Three layers is not deep enough for the
difference to show. The comparison that does show is either of them against
`normal`, and both against `zeros`.

## A sanity check

After building a model, push one batch through it and print the standard
deviation of each layer's activations. They should stay in the same
neighbourhood, roughly 0.5 to 1.5, rather than shrinking layer by layer.

```python
x = splits["x_train"][:256]
for layer in model.layers:
    x = layer.forward(x, training=False)
    print(f"{layer!r:<28} std {x.std():.3f}")
```

A decreasing sequence means the initialization is too small for the depth,
and gradients will vanish before the first epoch finishes.

## Next

[08. Optimizers](08-optimizers.md).
