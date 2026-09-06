# 01. What a neural network is

A neural network is a function with adjustable numbers in it. You feed in an
input, it produces an output, and the numbers inside decide what output you
get. Training means changing those numbers until the outputs are the ones
you want.

That is the whole idea. Everything else is detail about how the function is
shaped and how the numbers get changed.

## The problem MNIST poses

An MNIST image is 28 by 28 pixels, each pixel a byte from 0 to 255. Flatten
it and you have a point in 784 dimensional space. There are ten classes, so
what you need is a function

$$f: \mathbb{R}^{784} \to \{0, 1, \dots, 9\}$$

Writing that function by hand is hopeless. Nobody can state the rule that
separates a 4 from a 9 in terms of 784 pixel values, and every attempt to
write such rules by hand runs into handwriting that breaks them.

So instead of writing $f$, you write a family of functions with parameters
$\theta$, and search the family for a member that works:

$$f_\theta: \mathbb{R}^{784} \to \mathbb{R}^{10}$$

The search needs three things. A way to measure how wrong the current
$\theta$ is, which is the loss function. A way to tell which direction to
move $\theta$ to reduce that measure, which is the gradient. And a rule for
how far to move, which is the optimizer.

## Why layers

The simplest family is linear:

$$f_\theta(\mathbf{x}) = \mathbf{x}\mathbf{W} + \mathbf{b}$$

This is logistic regression once you attach a softmax, and on MNIST it
reaches about 92 percent. Its limit is structural. A linear function can
only carve the input space with flat boundaries, and the boundary between
handwritten 4s and 9s is not flat.

Stacking two linear layers does not help, because

$$(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
= \mathbf{X}(\mathbf{W}_1\mathbf{W}_2) + (\mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2)$$

which is again a single linear map with weights $\mathbf{W}_1\mathbf{W}_2$.
Depth without a nonlinearity buys nothing.

Insert a nonlinear function $g$ between the layers and the composition stops
collapsing:

$$f_\theta(\mathbf{X}) = g(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Now the boundaries can bend. The universal approximation theorem says that
one hidden layer of this form, made wide enough, can approximate any
continuous function on a bounded region to any accuracy you like. The
theorem is reassuring but not practical: it says nothing about how wide, and
it says nothing about whether gradient descent will find the right weights.
In practice depth is cheaper than width, which is why two or three hidden
layers beat one enormous one.

You can check the depth claim yourself:

```
python experiments.py --study depth
```

On a 20000 sample subset with five epochs, the linear model reaches 0.859
validation accuracy and one hidden layer reaches 0.958. The second hidden
layer adds much less, and the third almost nothing. That flattening is
normal for MNIST with fully connected layers, and it is where convolutional
networks start to pull ahead.

## Where the biology went

The name comes from a 1943 model of a neuron by McCulloch and Pitts: inputs
arrive weighted, they are summed, and the cell fires if the sum crosses a
threshold. The vocabulary stuck. Units are still called neurons and the
nonlinearity is still called an activation.

The resemblance ends there. Real neurons spike in time, real synapses are
not trained by backpropagation, and no biological mechanism resembling the
chain rule running backwards through a network has been found. It is safer
to treat a neural network as applied linear algebra with a nonlinearity than
as a model of a brain.

## What actually gets learned

After training, look at the first weight matrix:

```
python train_scratch.py
```

and open `out/fig_weights_scratch.png`. Each column of $\mathbf{W}^{[1]}$ has
784 entries, so it reshapes back to 28 by 28 and can be viewed as an image.
Some of them come out as stroke shapes and blobs sitting where digits differ
from one another. Nobody designed those. They are what gradient descent
settled on.

## Next

[02. The perceptron](02-the-perceptron.md) starts from the single unit, which
is the smallest version of everything above.
