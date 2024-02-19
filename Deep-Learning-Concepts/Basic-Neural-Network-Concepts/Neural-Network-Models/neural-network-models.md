# Neural Network Models

## Class attributes:

- $layers$: a list storing all linear and activation layers in the correct order.

## Class methods:

- $forward$: The `forward` method takes input data $A0$ and applies transformations corresponding to the layers (linear and activation) sequentially as `self.layers[i].forward` for $i = 0, ..., l − 18$ where $l$ is the total number of layers, to compute output $Al$.

- $backward$: The $backward$ method takes in $dLdAl$, how changes in loss $L$ affect model output $Al$, and performs back-propagation from the last layer to the first layer by calling `self.layers[i].backward` for $i = l − 1, ..., 0$. It does not return anything. Note that activation and linear layers don’t need to be treated differently as both take in the derivative of the loss with respect to the layer’s output and give back the derivative of the loss with respect to the layer’s input.

Please consider the following class structure:
```python
class Model:
        def __init__(self):
            self.layers = # TODO
        def forward(self, A):
            l = len(self.layers)
            for i in range(l):
                A = # TODO - keep modifying A by passing it through a layer
            return A
        def backward(self, dLdA):
            l = len(self.layers)
            for i in reversed(range(l)):
                dLdA = # TODO - keep modifying dLdA by passing it backwards through a layer
            return dLdA
```
Note that the A mentioned in the for loop in the forward pseudo code above is written so to maintain the same name of the variable containing the current output. In case of linear layers, it is the same as the output that was written as Z in the linear layer section. The case with dLdA mentioned in the backward pseudo code is similar. In the case of activation functions, it will be the same as what was mentioned as dLdZ in the activation functions section after the current dLdA is passed through the activation layer’s backward function.
We will start by building a shallow network with 0 hidden layer in subsection 7.1, and then a slightly deeper network with 1 hidden layer in subsection 7.2. Finally, we will build a deep neural network with 4 hidden layers in subsection 7.3. Note: all models have one additional layer for the output mapping, i.e. the total number of layers l for a model with 1 hidden layer is actually 2.
