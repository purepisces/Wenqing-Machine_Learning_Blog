# Feedforward Neural Network (FFNN)
## Definition:
An FFNN comprises multiple layers: input, one or more hidden layers, and an output layer. In FFNN, data flows in a single direction—straight from input to output, without cycles. When multiple hidden layers are used, the neural network is referred to as a "deep neural network", leading to the domain known as "deep learning".

## Feedforward Process:
Given a known input vector `x` and weights throughout the network, we can compute the output vector `ŷ`.

## Basic Structure:
- **Input Layer**:
  - The number of neurons typically corresponds to the size of the input data.
    - Example: For images of 28x28 pixels (like in the MNIST dataset), the input layer has 28×28=784 neurons, one per pixel.

- **Hidden Layers**:
  - The number of neurons and the number of hidden layers are hyperparameters that can be tuned. They don't have a direct relationship with the size of the input or output but are chosen based on model performance, problem complexity, and the risk of overfitting.
  - Each neuron in the first hidden layer has weights from all neurons in the input layer and a unique bias.
- **Output Layer**:
  - The number of neurons depends on the desired output format or classification classes.
  - Binary Classification: Single neuron denoting the probability of the positive class.
  - Multi-class classification: Neurons match the number of classes. E.g., a digit recognition task would have 10 output neurons.

> Note: The output layer's neuron count often differs from the input layer. Each layer can have a distinct number of neurons, indicating its 'size' or capacity.
>
## Intricacies of Neurons:
### Understanding Weights and Biases
- **Weights (often denoted as \(w\) or \(W\))**: These are the multipliers applied to input values, representing the importance of the connection between neurons.
- **Bias (often denoted as \(b\) or \(B\))**: This is an offset added to the weighted sum of inputs. Conceptually, it's similar to the intercept in a linear regression equation.

### Inside a Neuron: Two-Step Transformation

Within each neuron, the output is derived by first taking a weighted sum of its inputs, adding a bias term, and subsequently passing this aggregate through an activation function: 

1. **Linear Combination**: Each neuron in a neural network interacts with its preceding layer through weights and biases. Inputs are scaled by weights, summed, and then offset by a bias.
    - Formula: $z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
3. **Non-linear Activation**: The linear sum $z$ undergoes a non-linear transformation through an activation function, enabling the network to learn complex patterns.
   - Common functions: ReLU, Sigmoid, Tanh, Softmax
   - Neuron's Output: After processing the result $z$ through the activation function $f$, we obtain the neuron's output, denoted as \( a \). This is expressed mathematically as: $a = f(z)$
  
   
**Significance**: By cascading these transformations across layers, neural networks can represent highly sophisticated functions. Training adjusts the neuron's weights and biases, optimizing them to reduce the disparity between predicted outputs and actual data, typically utilizing optimization algorithms like gradient descent.

### Parameter Counting

**Fully Connected Networks**: These are networks where each neuron connects to every other neuron in the subsequent layer. In cnn, we only have some of these weights are non-zero and most of them are 0, which means each neuron is connected only to a few other neurons in the previous layer. This is a special case.

**For the First Hidden Layer**:
- **Weights**: Each neuron in the first hidden layer is connected to every neuron (or input feature) in the input layer. Thus, if there are `n` neurons in the input layer, a neuron in the first hidden layer will have `n` weights associated with it.
- **Bias**: Each neuron in the first hidden layer also has its unique bias that adjusts its output.
  
**For Adjacent Layers**:
Considering two adjacent layers in a network: 
- Layer $L-1$ with $N$ neurons
- Layer $L$ with $M$ neurons
- Assuming each neuron in $L$ is connected to every neuron in $L-1$
  
Total Parameters((weights + biases):
  $N \times M (weights) + M (biases) = N \times M + M$(since Each neuron in layer $L$ get a different bias)






![](image.png)


Reference:
https://youtube.com/watch?v=jTzJ9zjC8nU





















