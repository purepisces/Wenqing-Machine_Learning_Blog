# Tanh Activation Function

## Overview:
The tanh (hyperbolic tangent) function is a popular nonlinear activation function used in artificial neural networks, similar to the sigmoid function but with a range from -1 to 1. This makes it more suitable for cases where the model needs to deal with negative inputs efficiently.

## Mathematical Expression:
The tanh function is defined mathematically as:

$$\text{tanh}(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

where $e$ is the base of the natural logarithm, and $x$ represents the input to the function.

## Function Characteristics:
- **Range**: The output of the tanh function is bounded between -1 and 1, inclusive.
- **Shape**: It has an S-shaped curve, similar to the sigmoid function, but stretched vertically to fit the new range.
- **Output Interpretation**: Values near 1 indicate high positive activation, values near -1 indicate high negative activation, and values around 0 indicate low or no activation.

## Example:
Consider the same neuron receiving inputs with values 0.5 and 0.8, weights 0.4 and 0.6, and bias 0.1. The pre-activated output (x) is:

$x = (0.5 \times 0.4) + (0.8 \times 0.6) + 0.1 = 0.68$

Applying the tanh function gives the activated output:

$\text{activated output} = \text{tanh}(0.68) \approx 0.591$

This output then serves as input to subsequent neurons.

## Visualization:

<img src="tanh_activation_forward.png" alt="tanh_activation_forward" width="400" height="300"/>

## Tanh Class Implementation:

### Tanh Forward Equation

In forward propagation, the pre-activation features $Z$ are passed through the tanh function to get the post-activation values $A$.

$$\begin{align}
A &= \text{tanh.forward}(Z) \\
&= \text{tanh}(Z) \\
&= \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
\end{align}$$

### Tanh Backward Equation

In backward propagation, we calculate how changes in $Z$ affect the loss, given the changes in $A$.

$$\begin{align}
\frac{dL}{dz} &= \text{tanh.backward}(dLdA) \\
&= dLdA \odot (1 - \text{tanh}^2(Z)) \\
&= dLdA \odot (1 - A^2)
\end{align}$$

Here's a Python class implementation:

```python
import numpy as np
class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - np.power(self.A, 2)
        dLdZ = dLdA * dAdZ
        return dLdZ
```


## Visualization:

<img src="tanh.png" alt="tanh" width="400" height="300"/>


## Reference:
- [Watch the video on YouTube](https://www.youtube.com/watch?v=u0VsKSoSM4Y)
- CMU_11785_Introduction_To_Deep_Learning
