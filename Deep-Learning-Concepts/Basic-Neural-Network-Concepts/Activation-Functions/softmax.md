# Softmax 

The Softmax activation function is a vector activation function that is mostly applied at the end of neural network to convert a vector or raw outputs to a probability distribution in which the output elements sum up to 1. However, it can also be applied in the middle of a neural network like other activation functions discussed before this.

## Softmax Forward Equation

Given a $C$-dimensional input vector $Z$, whose $m$-th element is denoted by $z_m$, $softmax.forward(Z)$ will give a vector $A$ whose $m$-th element $a_m$ is given by:

$$a_m = \frac{\exp(z_m)}{\\sum\limits_{k=1}^{C} \exp(z_k)}$$

Here $Z$ was a single vector. Similar calculations can be done for batch of $N$ vectors.

## Softmax Backward Equation

As discussed in the description of the backward method for vector activations earlier in the section, the first step in backpropagating the derivatives is to calculate the Jacobian for each vector in the batch. Let’s take the example of an input vector $Z$ (a row of the input data matrix) and corresponding output vector $A$ (a row of the output matrix calculated by softmax.forward). The Jacobian $J$ is a $C \times C$ matrix. Its element at the $m$-th row and $n$-th column is given by:


$$J_{mn} = 
\begin{cases} 
a_m(1 - a_m) & \text{if } m = n \\
-a_m a_n & \text{if } m \neq n 
\end{cases}$$


where $a_m$ refers to the $m$-th element of the vector $A$.

Now the derivative of the loss with respect to this input vector, i.e., $dLdZ$ is $1 × C$ vector and is calculated
as:

$$\frac{dL}{dZ} = \frac{dL}{dA} \cdot J$$

Similar derivative calculation can be done for all the $N$ vectors in the batch and the resulting vectors can
be stacked up vertically to give the final $N \times C$ derivatives matrix.

```python
class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        exp_z = np.exp(Z)
        self.A = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N, C = dLdA.shape

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(dLdA)

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i, :] = np.dot(dLdA[i, :], J)

        return dLdZ
```
## Example Using Softmax Activation Function in Neural Networks

Consider a neural network layer with 2 samples in a batch ($N=2$) and each sample has 3 features ($C=3$). This setup could represent the logits for 3 classes in a classification problem.

Our batch of input vectors $Z$ is given by:

$$Z = \begin{pmatrix} Z^{(1)} & Z^{(2)} \end{pmatrix}
= \begin{pmatrix}
1 & 2 & 3 \\
2 & 2 & 1
\end{pmatrix}$$
$$

### Applying Softmax Activation

Softmax is applied to each sample $Z^{(i)}$ to produce the output vector $A^{(i)}$. The Softmax function for an element $z_j$ in vector $Z^{(i)}$ is defined as:

$$\text{Softmax}(z_j) = \frac{e^{z_j}}{\sum\limits_{k=1}^{C} e^{z_k}}$$ 

After applying Softmax to each row of $Z$, we get:

$$A = \begin{pmatrix} A^{(1)} & A^{(2)} \end{pmatrix}
= \begin{pmatrix}
0.09 & 0.24 & 0.67 \\
0.42 & 0.42 & 0.16
\end{pmatrix}$$
$$

### Jacobian Matrix for Softmax

For Softmax, the Jacobian $J^{(i)}$ for a single sample $A^{(i)}$ has elements:

$$J^{(i)}_{jk} = \begin{cases}
    A^{(i)}_j (1 - A^{(i)}_j) & \text{if } j = k \\
    -A^{(i)}_j A^{(i)}_k & \text{otherwise}
\end{cases}$$

For our first sample $A^{(1)}$, the Jacobian $J^{(1)}$ would be a 3x3 matrix where each element is computed using the above rule.

Calculating $J^{(1)}$

For the first sample, $A^{(1)} = [0.09, 0.24, 0.67]$.  

Using the Softmax derivative formula:

$$ For \ \ j = k: J^{(1)}_{jj} = A^{(1)}_j (1 - A^{(1)}_j)$$

$$ For \ \ j \neq k: J^{(1)}_{jk} = -A^{(1)}_j A^{(1)}_k$$

Thus, we have:

$$J^{(1)}_{11} = 0.09 \times (1 - 0.09) = 0.0819$$

$$J^{(1)}_{22} = 0.24 \times (1 - 0.24) = 0.1824$$

$$J^{(1)}_{33} = 0.67 \times (1 - 0.67) = 0.2211$$


And for 

$$j \neq k$$


$$J^{(1)}_{12} = J^{(1)}{21} = -0.09 \times 0.24 = -0.0216$$

$$J^{(1)}_{13} = J^{(1)}{31} = -0.09 \times 0.67 = -0.0603$$

$$J^{(1)}_{23} = J^{(1)}{32} = -0.24 \times 0.67 = -0.1608$$


So, $J^{(1)}$ is

$$J^{(1)} = \begin{pmatrix}
0.0819 &-0.0216 & -0.0603 \\
-0.0216 &0.1824 & -0.1608 \\
-0.0603 &-0.1608 &0.2211
\end{pmatrix}$$

Similarly

$$J^{(2)} = \begin{pmatrix}
0.2436 & -0.1764 & -0.0672 \\
-0.1764 & 0.2436 & -0.0672 \\
-0.0672 & -0.0672 & 0.1344
\end{pmatrix}$$

Computing the Gradient $dLdZ^{(i)}$
Let's assume we have the gradient of the loss with respect to the activation output $dLdA$ for our batch as:

$$dLdA= \begin{pmatrix}
0.1 & -0.2 & 0.1\\
-0.1 & 0.3 & -0.2
\end{pmatrix}$$

The gradient $dLdZ^{(i)}$ for each sample is calculated by multiplying the corresponding row of $dLdA$ by the Jacobian $J^{(i)}$:

$$dLdZ^{(i)} =dLdA^{(i)} \cdot J^{(i)}$$

This operation would be performed for each sample, and the resulting vectors $dLdZ^{(1)}$ and $dLdZ^{(2)}$ are then stacked to form the final gradient matrix $dLdZ$ for the whole batch.

### Specific Calculations

Due to the complexity of the Softmax derivatives and for brevity, detailed computations of each element of $J^{(1)}$ and $J^{(2)}$ are omitted here. However, the general process involves:

Calculating $J^{(1)}$ and $J^{(2)}$ using $A^{(1)}$ and $A^{(2)}$, respectively.

Multiplying $dLdA^{(1)} = [0.1, -0.2, 0.1]$ by $J^{(1)}$ to get $dLdZ^{(1)}$.

Multiplying $dLdA^{(2)} = [-0.1, 0.3, -0.2]$ by $J^{(2)}$ to get $dLdZ^{(2)}$.

Stacking $dLdZ^{(1)}$ and $dLdZ^{(2)}$ vertically to form $dLdZ$


This example illustrates the process of computing the gradient of the loss with respect to the inputs for a layer using a vector activation function, where the interdependence of the inputs in producing the outputs requires the computation of a full Jacobian matrix for each sample.


> Note: dLdZ is used in the backward pass because it directly relates the loss to the parameters we want to optimize (weights and biases) through $Z$ since $Z = W \cdot A_{prev} + b$, and followed by $A = f(Z)$, where $f$ is the activation function.
> In the case of scalar activations, $dLdZ$ is computed as:
> $$dLdZ = dLdA \odot \frac{\partial A}{\partial Z}$$
> In the case of vector activation function, $dLdZ$ is computed as: For each input vector $Z^{(i)}$ of size $1 \times C$ and its corresponding output vector $A^{(i)}$ (also $1 \times C$ within the batch, the Jacobian matrix $J^{(i)}$ must be computed individually. This matrix holds dimensions $C \times C$. Consequently, the gradient $dLdZ^{(i)}$ for each sample in the batch is determined by:
> $$dLdZ^{(i)} = dLdA^{(i)} \cdot J^{(i)}$$
> 
## Reference:
- CMU_11785_Introduction_To_Deep_Learning
