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
from sklearn.neural_network import MLPClassifier

# Creating an MLP for binary classification
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

# Fitting the model
mlp.fit(X_train, y_train)
