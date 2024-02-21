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


where $a_m$ refers to the m-th element of the vector $A$.

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
