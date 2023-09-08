# MNIST Neural Network Digit Recognizer

To begin, you can download the dataset from the following link: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

# ## Overview
_**Note:** This project is still in progress._

This project is an implementation of a neural network (NN) that recognizes handwritten digits from the MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale image of a handwritten digit. The goal of this project is to train a NN to recognize the digit in each image.

# ## Implementation

The implementation of the NN is written in Python 3.6. The following libraries are used:

- numpy
- matplotlib
- pandas
- TensorFlow
- Keras
- Seaborn
- scikit-learn

There are 4 notebooks in this project:

- `simple-mnist-nn.ipynb`: This notebook contains the implementation of a simple NN with 2 layers. The first layer is a hidden layer with 784 units and the second layer is an output layer with 10 units. The hidden layer uses the `ReLU` activation function and the output layer uses the `softmax` activation function. The NN is trained using the Adam optimizer and the categorical cross-entropy loss function. In this implementation, the NN is trained using gradient descent and stochastic gradient descent. No regularization, optimization, or cross entropy loss functions or any other techniques are used in this implementation.
  
- `mnist-3-layer-mlp-nn.ipynb`: The notebook also contains the implementation of a simple NN with 3 layers. The first layer is a hidden layer with 784 units, the second layer is a hidden layer with 10 units, and the third layer is an output layer with 10 units. The hidden layers use the `ReLU` activation function and the output layer uses the `softmax` activation function.

- `FNN_MNIST.ipynb`: This notebook uses `Keras` and `TensorFlow` to implement a fully connected NN (`FNN`) with 3 layers. The first layer is a hidden layer with 784 units, the second layer is a hidden layer with 10 units, and the third layer is an output layer with 10 units. The hidden layers use the `ReLU` activation function and the output layer uses the `softmax` activation function. The NN is trained using the `Adam optimizer` and the `categorical cross-entropy loss function`.
  
- `CNN_MNIST.ipynb`: This notebook uses `Keras` and `TensorFlow` to implement a convolutional NN (`CNN`) with 3 layers. The first layer is a convolutional layer with 32 filters, the second layer is a convolutional layer with 64 filters, and the third layer is a fully connected layer with 128 units. The convolutional layers use the `ReLU` activation function and the fully connected layer uses the `softmax` activation function. The NN is trained using the `Adam optimizer` and the `categorical cross-entropy loss function`.

## Formulas Employed in the `simple-mnist-nn.ipynb` Notebook

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
dW^{[2]} &= \frac{1}{m} dZ^{[2]} A^{[1]^\intercal} \\
dB^{[2]} &= \frac{1}{m} \sum {dZ^{[2]}} \\
dZ^{[1]} &= W^{[2]^\intercal} dZ^{[2]} \cdot g^{[1]\prime} (z^{[1]}) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} A^{[0]^\intercal} \\
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

## Formulas Employed in the `mnist-3-layer-mlp-nn.ipynb` Notebook

The neural network (NN) employs a straightforward three-layer architecture. The input layer, denoted as \(a^{[0]}\), comprises 784 units, corresponding to the 784 pixels present in each 28x28 input image. The first hidden layer \(a^{[1]}\) consists of 10 units utilizing the ReLU activation function. The second hidden layer \(a^{[2]}\) also comprises 10 units and uses the ReLU activation function. Lastly, the output layer \(a^{[3]}\) comprises 10 units, each corresponding to one of the ten digit classes. The softmax activation function is applied to this layer.

**Forward Propagation**

The forward propagation involves the following steps:

\[
\begin{align*}
Z^{[1]} &= W^{[1]} X + b^{[1]} \\
A^{[1]} &= g_{\text{ReLU}}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} &= g_{\text{ReLU}}(Z^{[2]}) \\
Z^{[3]} &= W^{[3]} A^{[2]} + b^{[3]} \\
A^{[3]} &= g_{\text{softmax}}(Z^{[3]})
\end{align*}
\]

**Backward Propagation**

The backward propagation involves the following computations:

\[
\begin{align*}
dZ^{[3]} &= A^{[3]} - Y \\
dW^{[3]} &= \frac{1}{m} dZ^{[3]} A^{[2]^\intercal} \\
dB^{[3]} &= \frac{1}{m} \sum {dZ^{[3]}} \\
dZ^{[2]} &= W^{[3]^\intercal} dZ^{[3]} \cdot g^{[2]\prime} (z^{[2]}) \\
dW^{[2]} &= \frac{1}{m} dZ^{[2]} A^{[1]^\intercal} \\
dB^{[2]} &= \frac{1}{m} \sum {dZ^{[2]}} \\
dZ^{[1]} &= W^{[2]^\intercal} dZ^{[2]} \cdot g^{[1]\prime} (z^{[1]}) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} A^{[0]^\intercal} \\
dB^{[1]} &= \frac{1}{m} \sum {dZ^{[1]}}
\end{align*}
\]

**Parameter Updates**

The parameters are updated using the following equations:

\[
\begin{align*}
W^{[3]} &:= W^{[3]} - \alpha dW^{[3]} \\
b^{[3]} &:= b^{[3]} - \alpha db^{[3]} \\
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
- \(W^{[2]}\): 10 x 10 (as \(W^{[2]} A^{[1]} \sim Z^{[2]}\))
- \(B^{[2]}\): 10 x 1
- \(Z^{[3]} \sim A^{[3]}\): 10 x m
- \(W^{[3]}\): 10 x 10 (as \(W^{[3]} A^{[2]} \sim Z^{[3]}\))
- \(B^{[3]}\): 10 x 1

During the backward propagation:

- \(dZ^{[3]}\): 10 x m (\(A^{[3]}\))
- \(dW^{[3]}\): 10 x 10
- \(dB^{[3]}\): 10 x 1
- \(dZ^{[2]}\): 10 x m (\(A^{[2]}\))
- \(dW^{[2]}\): 10 x 10
- \(dB^{[2]}\): 10 x 1
- \(dZ^{[1]}\): 10 x m (\(A^{[1]}\))
- \(dW^{[1]}\): 10 x 10
- \(dB^{[1]}\): 10 x 1

## Formulas Employed in the `FNN_MNIST.ipynb` Notebook

Here we use `Keras` and `TensorFlow` to implement a fully connected NN (`FNN`), so everything is done by the library.

## Formulas Employed in the `CNN_MNIST.ipynb` Notebook

Same as above, we use `Keras` and `TensorFlow` to implement a convolutional NN (`CNN`), so everything is done by the library.

# ## Results

The following table summarizes the results obtained from the 4 notebooks:

| Notebook | Accuracy |
| --- | --- |
| `simple-mnist-nn.ipynb` | 0.888 |
| `mnist-3-layer-mlp-nn.ipynb` | 0.945 |
| `FNN_MNIST.ipynb` | 0.962 |
| `CNN_MNIST.ipynb` | 0.993 |

# ## References

- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng

- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by Francois Chollet

- [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen

- [Machine Learning](https://www.coursera.org/learn/machine-learning) by Andrew Ng

- [Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/course/machinelearning/) by Kirill Eremenko and Hadelin de Ponteves


