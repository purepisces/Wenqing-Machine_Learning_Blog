# GELU Activation Function

## Overview:
The GELU (Gaussian Error Linear Unit) activation function is based on the cumulative distribution function of the standard Gaussian distribution. It is widely used in neural networks, particularly those involving natural language processing.

## Mathematical Expression:
The GELU function is mathematically expressed as follows:

$$
A = \text{gelu.forward}(Z) = Z\Phi(Z)
$$

where $\Phi(Z)$ is the cumulative distribution function of the standard Gaussian distribution, given by:

$$
\Phi(Z) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{Z}{\sqrt{2}}\right)\right]
$$

Here, $\text{erf}$ is the error function.

## GELU Forward Equation
The forward pass of GELU can be calculated as:

$$A = \text{gelu.forward}(Z) \\
= Z \Phi(Z) \\
= Z \int_{-\infty}^{Z} \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{x^2}{2} \right) dx \\
= \frac{1}{2} Z \odot \left[ 1 + \text{erf} \left( \frac{Z}{\sqrt{2}} \right) \right]$$

## GELU Backward Equation
For the backward pass, the derivative of $A$ with respect to $Z$ is needed:

$$\begin{align*}
\frac{dA}{dZ} &= \frac{d}{dZ} Z \Phi(Z) \\
&= \Phi(Z) + Z \Phi'(Z) \\
&= \Phi(Z) + Z P(X = Z) \\
&= \frac{1}{2} \left[ 1 + \text{erf} \left( \frac{Z}{\sqrt{2}} \right) \right] + \frac{Z}{\sqrt{2\pi}} \odot \exp \left( -\frac{Z^2}{2} \right)
\end{align*}$$


where $\Phi'(Z)$ is the probability density function of the standard Gaussian distribution, resulting in:

$$
\frac{dA}{dZ} = \frac{1}{2}\left[1 + \text{erf}\left(\frac{Z}{\sqrt{2}}\right)\right] + \frac{Z}{\sqrt{2\pi}}\exp\left(-\frac{Z^2}{2}\right)
$$

This is the expression used to implement the backward function of GELU:

$$
\frac{\partial L}{\partial Z} = \text{gelu.backward}(dLdA) = dLdA \frac{dA}{dZ}
$$

$$\begin{align*}
\frac{\partial L}{\partial Z} &= \text{gelu.backward}(\text{d}L\text{d}A) \\
&= \text{d}L\text{d}A \odot \frac{\partial A}{\partial Z} \\
&= \text{d}L\text{d}A \odot \left[ \frac{1}{2} \left( 1 + \text{erf}\left(\frac{Z}{\sqrt{2}}\right) \right) + \frac{Z}{\sqrt{2\pi}} \odot \exp\left(-\frac{Z^2}{2}\right) \right]
\end{align*}$$

## Implementation:
Below is a Python class implementation for the GELU activation function:

```python
import numpy as np
from scipy.special import erf

class GELU:
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1 + erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        dAdZ = 0.5 * (1 + erf(self.Z / np.sqrt(2))) + (self.Z * np.exp(-0.5 * self.Z**2)) / np.sqrt(2*np.pi)
        dLdZ = dLdA * dAdZ
        return dLdZ
```
