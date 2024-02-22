Consider we have a simple case with a batch size $N=1$ (i.e., a single input vector) and three classes $C=3$. Let's assume the output of the softmax forward pass (vector $A$) for this input is $[0.1, 0.2, 0.7]$. This would mean the input vector $Z$ to the softmax function resulted in these probabilities after the forward pass.

Now, assume we have the gradient of the loss with respect to the output of the softmax layer $\frac{dL}{dA}$ as $[0.01, -0.03, 0.02]$. We want to find the gradient of the loss with respect to the input of the softmax layer $\frac{dL}{dZ}$ using the backward equation and the process of filling in the Jacobian matrix $J$.

Given:

$A = [0.1, 0.2, 0.7]$
$\frac{dL}{dA} = [0.01, -0.03, 0.02]$
$C = 3$
We will:

Initialize the Jacobian matrix $J$ of size $C \times C$ (which is $3 \times 3$ in this case) with all zeros.
Fill in the Jacobian matrix according to the given rules:
If $m = n$, $J_{mn} = a_m(1 - a_m)$
If $m \neq n$, $J_{mn} = -a_m a_n$
Calculate $\frac{dL}{dZ}$ as $\frac{dL}{dA} \cdot J$.
Let's perform these steps with actual calculations.

The Jacobian matrix $J$ computed for the softmax output vector $A = [0.1, 0.2, 0.7]$ is:

$$J = \begin{bmatrix}
0.09 &amp; -0.02 &amp; -0.07 \\
-0.02 &amp; 0.16 &amp; -0.14 \\
-0.07 &amp; -0.14 &amp; 0.21
\end{bmatrix}$$


This matrix is filled according to the rules specified:

The diagonal elements ($J_{mm}$) are calculated as $a_m(1 - a_m)$, representing the derivative of the softmax output with respect to itself.
The off-diagonal elements ($J_{mn}$ where $m \neq n$) are negative products of the probabilities $-a_m a_n$, representing the derivative of a softmax output with respect to the other outputs.
Given the gradient of the loss with respect to the softmax output $\frac{dL}{dA} = [0.01, -0.03, 0.02]$, the gradient of the loss with respect to the input of the softmax layer $\frac{dL}{dZ}$ is computed as:

$$\frac{dL}{dZ} = \frac{dL}{dA} \cdot J = [0.0001, -0.0078, 0.0077]$$

This result provides the gradients needed to update the input vector $Z$ during the backpropagation process, effectively informing how changes to $Z$ would affect the loss, allowing for optimization of the neural network's weights.

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


