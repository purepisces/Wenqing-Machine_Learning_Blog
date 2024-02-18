# Multi-Layer Perceptron (MLP)

## Overview
A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network (ANN). An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.

## Structure of MLP

### Input Layer
- The input layer receives the input signal to be processed.
    - Example:
      For a dataset with 3 features, the input layer will have 3 input neurons.

### Hidden Layers
- MLPs have one or more hidden layers.
- The number of neurons and layers is a hyperparameter and varies based on the complexity of the task.
    - Example:
      A simple problem might have 1 hidden layer with 10 neurons.

### Output Layer
- The output layer of an MLP is responsible for producing the final output.
    - Example:
      For binary classification, the output layer will have 1 neuron.
      For multi-class classification with 4 classes, it will have 4 neurons.

## Activation Functions
- Common activation functions include ReLU, Sigmoid, and Tanh.
    - ReLU: Used for hidden layers.
    - Sigmoid and Softmax: Commonly used in the output layer for binary and multi-class classification, respectively.

## Training MLPs
- MLPs are trained using backpropagation.
- The most common training algorithm is gradient descent.

## Applications
- MLPs are used in various fields like speech recognition, image recognition, and machine translation.

## Example Code Snippet (Python)
```python
import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:

    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use Relu activations for the layer.
        """

        self.layers = [Linear(2, 3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layer followed by the activation layer to get the model output.
        """

        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """
        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)
        if self.debug:
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:

    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """

        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)

        Z1 = self.layers[2].forward(A1)
        A2 = self.layers[3].forward(Z1)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """

        dLdZ1 = self.layers[3].backward(dLdA2)
        dLdA1 = self.layers[2].backward(dLdZ1)

        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0

class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer with specified shapes.
        Use ReLU activation function for all the linear layers including the output layer.
        """
        self.layers = [
            Linear(2, 4), ReLU(),
            Linear(4, 8), ReLU(),
            Linear(8, 8), ReLU(),
            Linear(8, 4), ReLU(),
            Linear(4, 2), ReLU()
        ]
        self.debug = debug

    def forward(self, A):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """
        if self.debug:
            self.A = [A]

        for layer in self.layers:
            A = layer.forward(A)
            if self.debug:
                self.A.append(A)

        return A

    def backward(self, dLdA):
        """
        Implement backpropagation through the model.
        """
        if self.debug:
            self.dLdA = [dLdA]

        for layer in reversed(self.layers):
            dLdA = layer.backward(dLdA)
            if self.debug:
                self.dLdA.insert(0, dLdA)

        return dLdA
```
