# 11. Evaluation

Code: `src/metrics.py`, `src/plots.py`, `evaluate.py`

## Accuracy is not enough

The default run reaches 0.9827 on the test set. That is 173 wrong out of
10000, and accuracy alone says nothing about which 173.

MNIST is close to balanced, roughly 1000 images per class, so accuracy is at
least honest here. On an unbalanced problem it stops being informative: a
detector for a condition present in 1 percent of cases scores 99 percent by
answering no every time.

## The confusion matrix

Row is the true class, column is the predicted class. The diagonal is
correct, everything else is a specific mistake.

```
python evaluate.py --model models/scratch_mlp.npz
```

writes `out/eval_confusion_test.csv`, and `out/fig_confusion_scratch.png`
shows the same thing as a heatmap.

From the default run, the most frequent mistakes:

```
7 -> 9: 9
5 -> 3: 9
9 -> 4: 7
9 -> 7: 5
6 -> 4: 5
```

These are the pairs you would predict from the shapes. A 4 with a closed top
is a 9, a 5 with a rounded lower loop is a 3, and 7 against 9 depends on
whether the upper stroke closes. The model's errors are concentrated where
the classes genuinely overlap, which is a sign that it has learned something
about digit shape rather than about pixel positions.

## Precision, recall, F1

For one class, counting that class as positive:

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

Precision asks how often a prediction of this class is right. Recall asks
how much of this class the model found. They trade off, and $F_1$, their
harmonic mean, is low unless both are decent.

From the run:

```
 class  precision   recall       f1  support
     0     0.9858   0.9929   0.9893      980
     1     0.9930   0.9947   0.9938     1135
     ...
     9     0.9753   0.9782   0.9767     1009

accuracy 9827/10000 = 0.9827
macro f1 0.9825
```

Class 1 is the easiest, class 9 the hardest. That ordering is stable across
runs and across model types, because it comes from the data.

Macro F1 averages the per class scores with equal weight, so every class
counts the same regardless of size. Micro averaging weights by support and
on a single label problem equals accuracy.

## Look at the errors

```
out/fig_errors_scratch.png
```

is a grid of misclassified test digits with the true and predicted labels.
Look at it before changing anything. Many MNIST errors are digits that a
person would also hesitate over, and no tuning fixes those. If instead you
see clean, obvious digits being missed, the bug is upstream of the model:
preprocessing, label alignment, or normalization.

## Loss against accuracy

The two do not always move together:

```
epoch  8/20  loss 0.0131  val_loss 0.1029  val_acc 0.9778
epoch 20/20  loss 0.0001  val_loss 0.1113  val_acc 0.9817
```

Validation loss got worse while validation accuracy got better. Accuracy
only cares which class has the largest logit; loss also cares how confident
the model is. Late in training the model becomes more confident on the ones
it gets right and more confidently wrong on the ones it does not, and the
loss rises even as a few more predictions cross the line. Early stopping on
loss and early stopping on accuracy pick different epochs, so decide which
one you care about before you start.

## What to expect on MNIST

| model | test error |
|---|---|
| linear classifier | about 8 percent |
| MLP, this project | 1.7 percent |
| MLP with augmentation, tuned | about 1 percent |
| convolutional network | 0.4 to 0.8 percent |
| CNN with augmentation and ensembling | about 0.2 percent |

The remaining fraction of a percent includes genuinely ambiguous images.
MNIST has been solved for a long time, and results below about 0.3 percent
are largely about how carefully the last hundred images were handled.

## Beyond the test set

Test accuracy measures performance on data drawn the same way as the
training data. Your own handwriting, photographed with a phone, is not drawn
that way: different stroke width, different centering, different contrast,
possibly dark ink on light paper rather than the reverse.

`predict.py` handles the obvious part of the gap by converting to 28 by 28
grayscale, optionally inverting, and centering by center of mass, which is
how MNIST itself was normalized. A model at 98 percent on the test set will
still do noticeably worse on your own digits, and that gap is the honest
measure of what it learned.

## Next

[12. From NumPy to PyTorch](12-numpy-to-pytorch.md).
