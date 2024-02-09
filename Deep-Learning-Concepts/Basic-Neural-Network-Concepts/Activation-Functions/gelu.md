# GELU Activation Function

## Overview:
The GELU (Gaussian Error Linear Unit) activation function is defined in terms of the cumulative distribution function of the standard Gaussian distribution $\Phi(Z) = P(X \leq Z)$ where $X \sim \mathcal{N}(0,1)$:


## Mathematical Expression:
The GELU function is mathematically expressed as follows:

$$A = \text{gelu.forward}(Z) = \frac{1}{2} Z \odot \left[ 1 + \text{erf} \left( \frac{Z}{\sqrt{2}} \right) \right]$$

Here, erf refers to the error function which is frequently seen in probability and statistics. It can also take complex arguments but will take real ones here. Hint: Search the docs of the math and scipy libraries for help with implementation.

## GELU Forward Equation
The forward pass of GELU can be calculated as:

$$\begin{align*}
A &= \text{gelu.forward}(Z) \\
&= Z \Phi(Z) \\
&= Z \int_{-\infty}^{Z} \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{x^2}{2} \right) dx \\
&= \frac{1}{2} Z \odot \left[ 1 + \text{erf} \left( \frac{Z}{\sqrt{2}} \right) \right]
\end{align*}$$

## GELU Backward Equation
For the gelu.backward part where we calculate $\frac{\partial A}{\partial Z}$, the GELU equation given above needs to be differentiated with respect to $Z$:

$$\begin{align*}
\frac{dA}{dZ} &= \frac{d}{dZ} Z \Phi(Z) \\
&= \Phi(Z) + Z \Phi'(Z) \\
&= \Phi(Z) + Z P(X = Z) \\
&= \frac{1}{2} \left[ 1 + \text{erf} \left( \frac{Z}{\sqrt{2}} \right) \right] + \frac{Z}{\sqrt{2\pi}} \odot \exp \left( -\frac{Z^2}{2} \right)
\end{align*}$$

This gives us the final expression to implement the backward function:

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
