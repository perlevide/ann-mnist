# 10. The training loop

Code: `src/scratch/network.py`, `src/data.py`

## The loop

```python
for epoch in range(epochs):
    for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size, rng):
        logits = self.forward(x_batch, training=True)
        loss = self.loss_fn.forward(logits, y_batch)
        self.backward(self.loss_fn.backward())
        optimizer.step(self.params_and_grads())
```

Forward, loss, backward, update. Everything in the previous nine chapters is
one of those four lines.

An epoch is one pass over the training data. A step, or iteration, is one
mini batch. With 54000 training samples and a batch of 128 there are 422
steps per epoch, so 20 epochs is 8440 parameter updates.

## Shuffling

`iterate_minibatches` permutes the indices at the start of each epoch. If the
order never changes, the sequence of gradients repeats exactly, the noise
becomes periodic, and the model can pick up on the ordering. Shuffling costs
one permutation per epoch.

Evaluation loops do not shuffle. It changes no metric and it makes per sample
debugging harder.

## The three splits

| split | size | used for |
|---|---|---|
| train | 54,000 | gradient updates |
| validation | 6,000 | choosing hyperparameters, early stopping |
| test | 10,000 | one final number, reported once |

MNIST ships with 60000 training and 10000 test images. The validation split
is carved out of the training set, here 10 percent.

The rule about the test set is not bureaucracy. Every time you look at test
accuracy and change something because of it, you have used the test set to
fit a decision, and the number stops being an estimate of performance on
unseen data. Use validation for every choice, and touch the test set at the
end.

## Reading the numbers

From the default run:

```
epoch   1/20  loss 0.2235  acc 0.9726  val_loss 0.1366  val_acc 0.9598
epoch   5/20  loss 0.0287  acc 0.9941  val_loss 0.1065  val_acc 0.9748
epoch  10/20  loss 0.0048  acc 0.9992  val_loss 0.1063  val_acc 0.9777
epoch  20/20  loss 0.0001  acc 1.0000  val_loss 0.1113  val_acc 0.9817
```

Epoch 1 already at 0.96 validation is normal for MNIST; it is an easy
dataset. The training loss falls by three orders of magnitude while
validation loss flattens at about 0.10 and then rises slightly. That divergence
is overfitting, and it starts around epoch 7.

Reported training accuracy is measured after the epoch with dropout off, so
it is comparable with validation accuracy. The running mean of the batch
losses printed as `loss` is measured during the epoch, while the weights are
still changing, so it is slightly pessimistic. Both conventions are in
common use, and the only mistake is comparing one against the other.

## Hyperparameters

| name | default | effect |
|---|---|---|
| learning rate | 0.05 | step size; the one to tune first |
| batch size | 128 | gradient noise and speed |
| epochs | 20 | with early stopping, an upper bound |
| hidden sizes | 256, 128 | capacity |
| activation | relu | see chapter 04 |
| optimizer | sgd with momentum 0.9 | see chapter 08 |
| weight decay | 0 | L2 strength |
| dropout | 0 | drop probability |

A sensible order to search: learning rate first, spanning powers of ten;
then architecture; then regularization; then the rest. Change one thing at a
time and keep the seed fixed, otherwise you are measuring noise.

## Seeds and reproducibility

Weight initialization, batch order and dropout masks all draw random
numbers. `seed_everything(seed)` fixes them so that two runs with the same
settings produce the same numbers, which is what makes a comparison
meaningful.

Fixed seed does not mean the result is representative. A difference of a few
tenths of a percent between two settings at one seed is usually noise. If a
difference matters, run three seeds and look at the spread.

## Batch size and learning rate move together

A larger batch gives a less noisy gradient, so it tolerates and often needs a
larger learning rate. The common rule of thumb is linear scaling: double the
batch, double the rate. It holds over a moderate range and breaks at very
large batches.

Batch size also sets memory use, since every layer keeps its input for the
backward pass.

## Common failures

Loss stays at 2.30 and accuracy at 0.10, meaning the model outputs a uniform
distribution: usually a learning rate near zero, all-zero initialization, or
labels that do not line up with the inputs.

Loss becomes NaN: learning rate too high, or a log of zero somewhere. Check
that softmax is computed with the max subtracted.

Loss falls but validation accuracy does not: overfitting, chapter 09.

Training accuracy stuck around 0.90: the model is effectively linear.
Check that an activation actually sits between the linear layers.

Validation accuracy above training accuracy: normal early in training when
dropout is on, since training accuracy is measured with the noise and
validation without it.

## Next

[11. Evaluation](11-evaluation.md).
