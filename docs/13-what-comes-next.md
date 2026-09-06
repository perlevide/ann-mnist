# 13. What comes after the MLP

## The limit you just hit

The first weight matrix is $784 \times 256$. Every pixel gets its own weight
into every hidden unit, which means the model has no idea that pixel 100 and
pixel 101 are next to each other. Shuffle every image with the same fixed
permutation of the 784 positions and this network trains to the same
accuracy. The spatial structure is available and the architecture cannot use
it.

Two consequences. The parameter count grows with image size: at 224 by 224
in colour, one hidden layer of 256 units would need 38 million weights in the
first layer alone. And a digit shifted two pixels right is, to this model, a
completely different input, so it has to learn the same shape again at every
position.

## Convolutional networks

A convolutional layer replaces the full matrix with a small kernel, say 3 by
3, slid across the image. Three properties follow.

Local connectivity: each output depends on a small neighbourhood, which is
where the information about a stroke actually is.

Weight sharing: the same kernel applies everywhere, so an edge detector
learned in one corner works in all of them. A 3 by 3 kernel with 32 output
channels has 288 weights regardless of image size.

Translation equivariance: shift the input and the feature map shifts with
it. Pooling then makes the result approximately invariant to small shifts.

On MNIST a small CNN reaches 0.4 to 0.8 percent test error against the 1.7
percent here, with fewer parameters. Backpropagation through a convolution
is the same four equations from chapter 06, applied to a different linear
operator.

## Normalization layers

Batch normalization standardizes each layer's pre-activations across the
batch, then applies a learned scale and shift. It keeps the activation
distribution stable as the weights move, which allows higher learning rates
and makes deep networks much less sensitive to initialization. Layer
normalization does the same across features instead of the batch and is what
transformers use.

## Residual connections

Adding the input of a block to its output, $y = f(x) + x$, gives the
gradient a path that skips the block entirely. That is what made networks of
50 and 100 layers trainable, and it is a direct answer to the vanishing
gradient product from chapter 06.

## A reasonable next sequence

Add data augmentation to this project and watch the validation gap close.
Then write a convolutional layer in NumPy, forward and backward, and check it
with `gradcheck.py`. Then build the same CNN in PyTorch and compare. Then
move to a dataset where the images are not centered and cropped for you,
CIFAR-10 or Fashion-MNIST, where the classes actually collide.

After that the path forks by interest: sequence models and attention on one
side, and on the other the deployment questions, quantization, ONNX export
and running a small network on a microcontroller.

## Where this project stops

Everything here is a fully connected network trained with mini batch
gradient descent on a clean, centered, single channel dataset. That covers
the mechanics completely, and the mechanics do not change. A transformer is
trained by the same four lines in chapter 10.

## Next

[14. Running on a GPU](14-running-on-a-gpu.md), which is a practical
chapter rather than a theoretical one, and worth reading before you assume
bigger hardware will make any of the above faster.
