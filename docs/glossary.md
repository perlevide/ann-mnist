# Glossary

English, Vietnamese, and what the term means here.

| English | Tiếng Việt | Meaning |
|---|---|---|
| activation | hàm kích hoạt | the nonlinear function $g$ applied after a linear layer |
| accuracy | độ chính xác | fraction of predictions that are correct |
| backpropagation | lan truyền ngược | the chain rule run backwards through the network |
| batch | lô | a group of samples processed together |
| bias | độ lệch | the additive term $b$ in $\mathbf{x}\mathbf{W} + \mathbf{b}$ |
| confusion matrix | ma trận nhầm lẫn | table of true class against predicted class |
| convergence | hội tụ | the loss settling instead of continuing to fall |
| cross entropy | entropy chéo | the loss $-\log p_k$ for the true class $k$ |
| dropout | bỏ ngẫu nhiên | zeroing random activations during training |
| epoch | lượt huấn luyện | one full pass over the training data |
| feedforward | truyền thẳng | a network with no cycles |
| forward pass | lượt truyền xuôi | computing the output from the input |
| gradient | gradient, đạo hàm | vector of partial derivatives of the loss |
| gradient descent | hạ gradient | moving parameters opposite the gradient |
| hidden layer | lớp ẩn | any layer between input and output |
| hyperparameter | siêu tham số | a setting you choose, not one that is learned |
| initialization | khởi tạo | the starting values of the weights |
| label | nhãn | the correct answer for a sample |
| layer | lớp | one linear map plus its activation |
| learning rate | tốc độ học | step size $\eta$ in the update rule |
| logit | logit | raw output before softmax |
| loss function | hàm mất mát | the scalar being minimized |
| mini batch | lô nhỏ | a batch of 32 to 512 samples |
| momentum | quán tính | running average of gradients used in the update |
| neuron, unit | nơ ron, đơn vị | one weighted sum plus activation |
| normalization | chuẩn hóa | rescaling data to zero mean and unit variance |
| one hot | mã hóa one-hot | a vector with 1 at the true class and 0 elsewhere |
| optimizer | bộ tối ưu | the rule that updates parameters from gradients |
| overfitting | quá khớp | fitting the training set at the cost of new data |
| parameter | tham số | a weight or bias, learned during training |
| perceptron | perceptron | a single unit with a step activation |
| regularization | điều chuẩn | anything added to reduce overfitting |
| ReLU | ReLU | $\max(0, z)$ |
| sigmoid | hàm sigmoid | $1/(1 + e^{-z})$ |
| softmax | hàm softmax | turns logits into a probability distribution |
| test set | tập kiểm tra | held out data, used once at the end |
| training set | tập huấn luyện | data the gradients are computed on |
| underfitting | dưới khớp | the model is too weak to fit even the training set |
| validation set | tập kiểm định | held out data used to choose hyperparameters |
| vanishing gradient | gradient tiêu biến | gradients shrinking to nothing in early layers |
| weight | trọng số | an entry of $\mathbf{W}$ |
| weight decay | suy giảm trọng số | L2 penalty, which shrinks weights each step |
