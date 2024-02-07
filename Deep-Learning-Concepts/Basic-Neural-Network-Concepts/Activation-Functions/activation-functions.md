I use relu.md to illstrate the main idea behind activation functions [ReLU](Deep-Learning-Concepts/Basic-Neural-Network-Concepts/Activation-Functions/relu.md).

In the realm of machine learning, engineers have the liberty to select any differentiable function to serve as an activation function. The inclusion of non-linear elements within a neural network ($f_{NN}$) is crucial for its capability to model non-linear phenomena. In the absence of activation functions, the output of an $f_{NN}$ remains linear irrespective of its depth, due to the inherent linearity of the equation $A · W + b$.

Activation functions can either take scalar or vector arguments. Scalar activations apply a function to a single number. Scalar activation functions are applied individually to each element of a vector, maintaining a direct relationship between each input and its corresponding output, which simplifies the computation of derivatives. Popular choices of scalar activation functions are **Sigmoid, ReLU, Tanh, and GELU**, as shown in the following figure.

<img src="activation-funcs.png" alt="activation-funcs" width="600" height="350"/>

#### Example:
Consider a vector $X = [x1, x2, x3]$ and applying the ReLU function $ReLU(x) = max(0, x)$. The ReLU function is applied to each vector element:

- $ReLU(x1) = max(0, x1)$
- $ReLU(x2) = max(0, x2)$
- $ReLU(x3) = max(0, x3)$

For $X = [-1, 5, -3]$, the output after applying ReLU is $[0, 5, 0]$, as ReLU converts negative inputs to 0.

#### Simplifying Derivative Calculation:
The one-to-one correspondence in scalar activations like ReLU simplifies derivative calculations:

- For $x1 = -1$, derivative $ReLU'(x1) = 0$ because $x1$ is negative.
- For $x2 = 5$, derivative $ReLU'(x2) = 1$ as $x2$ is positive.
- For $x3 = -3$, derivative $ReLU'(x3) = 0$ due to $x3$ being negative.

This element-wise approach allows for straightforward computation of derivatives, a key aspect in neural network optimization through backpropagation.

On the other hand, vector activation functions like the **Softmax** involve outputs that are interdependent on all input elements, complicating the derivative computation process. This document will guide you in implementing both scalar and vector activation functions, with an emphasis on the **Softmax** function for vector activations.


- **Class attributes**:
  - Activation functions have no trainable parameters.
  - Variables stored during forward-propagation to compute derivatives during back-propagation: layer output $A$.

- **Class methods**:
  - $forward$: The forward method takes in a batch of data $Z$ of shape $N \times C$(representing $N$ samples where each sample has $C$ features), and applies the activation function to $Z$ to compute output $A$ of shape $N \times C$.
  - $backward$: The backward method takes in `dLdA`, a measure of how the post-activations (output) affect the loss. Using this and the derivative of the activation function itself, the method calculates and returns `dLdZ`, how changes in pre-activation features (input) `Z` affect the loss `L`. In the case of scalar activations, `dLdZ` is computed as:
    $
    dLdZ = dLdA \odot \frac{\partial A}{\partial Z}
    $



## Reference:
- CMU_11785_Introduction_To_Deep_Learning


