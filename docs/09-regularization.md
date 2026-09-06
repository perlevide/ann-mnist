# 09. Regularization

Code: `src/scratch/layers.py`, `src/scratch/losses.py`, `src/scratch/optimizers.py`

## The symptom

From the run in `out/scratch_summary.json`:

```
epoch  1  loss 0.2235  acc 0.9726  val_loss 0.1366  val_acc 0.9598
epoch  8  loss 0.0131  acc 0.9984  val_loss 0.1029  val_acc 0.9778
epoch 20  loss 0.0001  acc 1.0000  val_loss 0.1113  val_acc 0.9817
```

Training loss reaches $10^{-4}$ and training accuracy reaches exactly 1.000.
Validation loss bottoms out around epoch 7 and then drifts upward while
validation accuracy keeps creeping up slowly. The model has memorized the
training set. With 235,146 parameters and 54,000 samples, more than four
parameters per sample, it has room to.

Overfitting is the gap between training and validation performance.
Underfitting is when both are bad, and it means the model is too small or
trained for too few epochs. They call for opposite responses, so read the
gap before changing anything.

## L2, also called weight decay

Add the squared size of the weights to the loss:

$$J_{\text{reg}} = J + \frac{\lambda}{2}\sum_{l}\|\mathbf{W}^{[l]}\|_F^2$$

The gradient picks up one extra term:

$$\frac{\partial J_{\text{reg}}}{\partial \mathbf{W}} = \frac{\partial J}{\partial \mathbf{W}} + \lambda \mathbf{W}$$

so the update becomes

$$\mathbf{W} \leftarrow \mathbf{W} - \eta\left(\frac{\partial J}{\partial \mathbf{W}} + \lambda\mathbf{W}\right) = (1 - \eta\lambda)\mathbf{W} - \eta\frac{\partial J}{\partial \mathbf{W}}$$

Every step multiplies the weights by $(1 - \eta\lambda)$ before the gradient
is applied, which is where the name weight decay comes from. Typical
$\lambda$ is $10^{-4}$.

Small weights mean the function changes slowly as the input changes, so the
model cannot carve a sharp boundary around one memorized training point.
That is the mechanism.

Only weight matrices are penalized. Shrinking biases just pulls the whole
decision surface toward the origin without reducing what the model can
represent. In the code the `is_weight` flag in `params_and_grads` carries
this distinction to the optimizer.

L1, $\lambda\sum|w|$, is the other classical choice. Its gradient is
$\lambda\,\text{sign}(w)$, a constant pull toward zero that does not shrink
as $w$ does, so weights actually reach zero and the solution is sparse. L2
is the usual choice for neural networks; L1 is useful when you want feature
selection.

## Dropout

During training, zero each activation independently with probability $p$
(Srivastava et al., 2014). Each mini batch therefore trains a different
random subnetwork, and no unit can rely on any particular other unit being
present. The network is forced to spread its representation instead of
building fragile chains of co-adapted units.

Inverted dropout keeps the arithmetic consistent. A unit survives with
probability $1 - p$, so scaling the survivors by $1/(1-p)$ during training
leaves the expected value equal to the undropped value:

$$\mathbb{E}[\tilde{a}] = (1 - p)\cdot\frac{a}{1 - p} + p \cdot 0 = a$$

Evaluation then needs no rescaling at all and simply returns the input,
which is why `Dropout.forward` checks the `training` flag first.

Backward, the same mask applies, since $\partial \tilde{a} / \partial a$ is
the mask itself.

Common values are 0.2 to 0.5, and dropout goes on hidden layers only. Never
on the output. Forgetting to switch it off at evaluation time makes accuracy
jump around between calls, which is the giveaway.

## Early stopping

Track validation loss and keep the weights from the epoch where it was
lowest. Effectively free, and it needs no extra hyperparameter beyond how
long to wait before giving up.

The PyTorch trainer in `src/torchmlp/engine.py` does this by keeping the best
`state_dict` and reloading it at the end. In the run above, epoch 16 of 20
was best.

## Data augmentation

More data beats every regularizer. When you cannot collect more, generate
it: small rotations of a few degrees, shifts of one or two pixels, mild
scaling and elastic distortions. The label does not change, and the model
learns that these variations do not matter.

MNIST responds well to this, and elastic distortions were how the early
results below 0.5 percent error were reached. This project does not implement
augmentation; it is a good exercise to add.

## Measured

```
python experiments.py --study regularization
```

| setting | train acc | val acc | val loss |
|---|---|---|---|
| none | 0.9943 | 0.9605 | 0.1521 |
| L2 1e-4 | 0.9951 | 0.9628 | 0.1336 |
| dropout 0.2 | 0.9875 | 0.9613 | 0.1277 |
| both | 0.9845 | 0.9598 | 0.1404 |

Look at the training column as much as the validation column. Dropout pulls
train accuracy down from 0.9943 to 0.9875 while validation stays level and
validation loss improves, which is regularization doing its job. Five epochs
on a 20000 sample subset is too short for the gains to be large, and
stacking both is already slightly too much for a model this size.

## Order of operations

When validation accuracy is far below training accuracy: add data, then
augment, then dropout or L2, then shrink the model. When both are low: make
the model bigger or train longer, and check that the learning rate is not so
small that nothing is moving.

## Next

[10. The training loop](10-training-loop.md).
