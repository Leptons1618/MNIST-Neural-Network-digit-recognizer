# MNIST Neural Network Digit Recognizer

To begin, you can download the dataset from the following link: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## Formulas Employed in the Implementation

The neural network (NN) employs a straightforward two-layer architecture. The input layer, denoted as \(a^{[0]}\), comprises 784 units, corresponding to the 784 pixels present in each 28x28 input image. The hidden layer \(a^{[1]}\) consists of 10 units utilizing the ReLU activation function. Lastly, the output layer \(a^{[2]}\) comprises 10 units, each corresponding to one of the ten digit classes. The softmax activation function is applied to this layer.

**Forward Propagation**

The forward propagation involves the following steps:

\[
\begin{align*}
Z^{[1]} &= W^{[1]} X + b^{[1]} \\
A^{[1]} &= g_{\text{ReLU}}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} &= g_{\text{softmax}}(Z^{[2]})
\end{align*}
\]

**Backward Propagation**

The backward propagation involves the following computations:

\[
\begin{align*}
dZ^{[2]} &= A^{[2]} - Y \\
dW^{[2]} &= \frac{1}{m} dZ^{[2]} A^{[1]T} \\
dB^{[2]} &= \frac{1}{m} \sum {dZ^{[2]}} \\
dZ^{[1]} &= W^{[2]T} dZ^{[2]} \cdot g^{[1]\prime} (z^{[1]}) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} A^{[0]T} \\
dB^{[1]} &= \frac{1}{m} \sum {dZ^{[1]}}
\end{align*}
\]

**Parameter Updates**

The parameters are updated using the following equations:

\[
\begin{align*}
W^{[2]} &:= W^{[2]} - \alpha dW^{[2]} \\
b^{[2]} &:= b^{[2]} - \alpha db^{[2]} \\
W^{[1]} &:= W^{[1]} - \alpha dW^{[1]} \\
b^{[1]} &:= b^{[1]} - \alpha db^{[1]}
\end{align*}
\]

**Variables and Shapes**

During the forward propagation:

- \(A^{[0]} = X\): 784 x m
- \(Z^{[1]} \sim A^{[1]}\): 10 x m
- \(W^{[1]}\): 10 x 784 (as \(W^{[1]} A^{[0]} \sim Z^{[1]}\))
- \(B^{[1]}\): 10 x 1
- \(Z^{[2]} \sim A^{[2]}\): 10 x m
- \(W^{[1]}\): 10 x 10 (as \(W^{[2]} A^{[1]} \sim Z^{[2]}\))
- \(B^{[2]}\): 10 x 1

During the backward propagation:

- \(dZ^{[2]}\): 10 x m (\(A^{[2]}\))
- \(dW^{[2]}\): 10 x 10
- \(dB^{[2]}\): 10 x 1
- \(dZ^{[1]}\): 10 x m (\(A^{[1]}\))
- \(dW^{[1]}\): 10 x 10
- \(dB^{[1]}\): 10 x 1
