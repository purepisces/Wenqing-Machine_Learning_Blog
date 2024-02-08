# Sigmoid Activation Function

## Overview:
The sigmoid function is a widely used nonlinear activation function in artificial neural networks. It effectively maps input values to a range between 0 and 1, making it particularly useful for models where the output needs to be interpreted as a probability.

## Mathematical Expression:
The sigmoid function is mathematically defined as:

$\text{sigmoid}(x) = \frac{e^x}{e^x+1} = \frac{1}{1+e^{-x}}$

where $e$ is the base of the natural logarithm, and $x$ represents the input to the function, typically the weighted sum of inputs to a neuron.

## Function Characteristics:
- **Range**: The output of the sigmoid function is bounded between 0 and 1, inclusive.
- **Shape**: It has an S-shaped curve (sigmoid curve).
- **Output Interpretation**: Values near 1 indicate a high degree of activation, while values near 0 indicate low activation. This can be intuitively understood as the probability of the neuron being activated.

## Application in Neural Networks:
In the context of neural networks, the sigmoid function is applied to each neuron within a layer. Specifically, for a fully connected layer, the function is applied to the weighted sum of inputs for each neuron, producing the neuron's activation level. This activated output is then passed on to the next layer in the network.

The use of sigmoid allows for a clear interpretation of neuron activation levels:
- **Closer to 1**: Indicates higher activation.
- **Closer to 0**: Indicates lower activation.

Thus, the sigmoid function serves as a mechanism to introduce non-linearity into the network, enabling it to learn complex patterns and make decisions that go beyond simple linear boundaries.

## Example:
Consider a neuron in a hidden layer receiving inputs from two neurons in the input layer with values 0.5 and 0.8. If the weights of these inputs are 0.4 and 0.6 respectively, and the neuron's bias is 0.1, the pre-activated output (x) would be calculated as:

$x = (0.5 \times 0.4) + (0.8 \times 0.6) + 0.1 = 0.68$

Applying the sigmoid function gives the activated output:

$\text{activated output} = \frac{1}{1 + e^{-0.68}} \approx 0.6637$

This activated output is then used as input to the neurons in the subsequent layer.

## Visualization:

<img src="sigmoid_activation.png" alt="sigmoid_activation" width="400" height="300"/>

## Derivation of the Sigmoid Function's Derivative

The equation $\frac{\partial \sigma(Z)}{\partial Z} = \sigma(Z) \cdot (1 - \sigma(Z))$ is derived from applying the chain rule to the sigmoid function, which is given by $\sigma(Z) = \frac{1}{1 + e^{-Z}}$. Here's a step-by-step breakdown of how this derivative is obtained:

1. **Express the sigmoid function**: The sigmoid function can be rewritten for convenience in derivative calculation as:
   $$\sigma(Z) = (1 + e^{-Z})^{-1}$$

2. **Apply the chain rule**: The chain rule in calculus is a formula to compute the derivative of the composition of two or more functions. In this case, we have an outer function $u^{-1}$ (where $u = 1 + e^{-Z}$) and an inner function $1 + e^{-Z}$. The chain rule states that if you have a function $h(u) = u^{-1}$ and $u(Z) = 1 + e^{-Z}$, then:
   $$\frac{d}{dZ} h(u) = \frac{dh}{du} \cdot \frac{du}{dZ}$$

3. **Compute $\frac{du}{dZ}$**: The derivative of $u(Z) = 1 + e^{-Z}$ with respect to $Z$ is:
   $$\frac{du}{dZ} = -e^{-Z}$$

4. **Compute $\frac{dh}{du}$**: The derivative of $h(u) = u^{-1}$ with respect to $u$ is:
   $$\frac{dh}{du} = -u^{-2} = -(1 + e^{-Z})^{-2}$$

5. **Combine using the chain rule**: Now, combining these results using the chain rule gives:
   $$\frac{d\sigma}{dZ} = -(1 + e^{-Z})^{-2} \cdot (-e^{-Z})$$

6. **Simplify**: This simplifies to:
   $$\frac{d\sigma}{dZ} = \frac{e^{-Z}}{(1 + e^{-Z})^2}$$

7. **Express in terms of $\sigma(Z)$**: Notice that $\sigma(Z) = \frac{1}{1 + e^{-Z}} \implies 1 + e^{-Z} = \frac{1}{\sigma(Z)}$
   
   Therefore
   $$e^{-Z} = \frac{1}{\sigma(Z)} - 1$$.
   $$(1 + e^{-Z})^2 = \left(\frac{1}{\sigma(Z)}\right)^2$$
   Substituting these into the derivative gives:
   $$\frac{d\sigma}{dZ} = \frac{e^{-Z}}{(1 + e^{-Z})^2} = \frac{\frac{1}{\sigma(Z)} - 1}{(\frac{1}{\sigma(Z)})^2} = \sigma(Z) \cdot (1 - \sigma(Z))$$
   $$\frac{d\sigma}{dZ} = \sigma(Z) - \sigma^2(Z)$$

## Sigmoid Class Implementation:

### Sigmoid Forward Equation

During forward propagation, pre-activation features $Z$ are passed to the activation function Sigmoid to calculate their post-activation values $A$.


$$\begin{align}
A &= \text{sigmoid.forward}(Z) \\
&= \sigma(Z) \\
&= \frac{1}{1 + e^{-Z}}
\end{align}$$

<img src="sigmoid_activation_forward.png" alt="sigmoid_activation_forward" width="400" height="300"/>

### Sigmoid Backward Equation

Backward propagation helps us understand how changes in pre-activation features $Z$ affect the loss, given
how changes in post-activation values $A$ affect the loss.


$$\begin{align}
\frac{dL}{dz} &= \text{sigmoid.backward}(dLdA) \\
&= dLdA \odot \frac{\partial A}{\partial Z} \\
&= dLdA \odot (\sigma(Z) - \sigma^2(Z)) \\
&= dLdA \odot (A - A \odot A)
\end{align}$$


Below is a Python class implementation of the sigmoid activation function, which includes both the forward pass (calculating the activated output) and the backward pass (calculating the gradient for backpropagation).

```python
import numpy as np
class Sigmoid:
    """
    Sigmoid Activation Function:
    - 'forward' function applies the sigmoid activation.
    - 'backward' function computes the gradient for backpropagation.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ
```



## Key Takeaway:
The sigmoid function is crucial for transforming linear inputs into outputs that can be interpreted as probabilities, thereby providing a probabilistic foundation to the activation levels within a neural network.

## Reference:
- [Watch the video on YouTube](https://www.youtube.com/watch?v=KOhbp3EIRlM)
