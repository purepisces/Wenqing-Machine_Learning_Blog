6.5 Softmax 
The Softmax activation function is a vector activation function that is mostly applied at the end of neural network to convert a vector or raw outputs to a probability distribution in which the output elements sum up to 1. However, it can also be applied in the middle of a neural network like other activation functions discussed before this.

6.5.1 Softmax Forward Equation

Given a C-dimensional input vector Z, whose m-th element is denoted by zm, softmax.forward(Z) will give a vector A whose m-th element am is given by:


$$a_m = \frac{\exp(z_m)}{\\sum\limits_{k=1}^{C} \exp(z_k)}$$

Here Z was a single vector. Similar calculations can be done for batch of N vectors.

6.5.2 Softmax Backward Equation
As discussed in the description of the backward method for vector activations earlier in the section, the first step in backpropagating the derivatives is to calculate the Jacobian for each vector in the batch. Let’s take the example of an input vector Z (a row of the input data matrix) and corresponding output vector A (a row of the output matrix calculated by softmax.forward). The Jacobian J is a C × C matrix. Its element at the m-th row and n-th column is given by:


$$J_{mn} = 
\begin{cases} 
a_m(1 - a_m) & \text{if } m = n \\
-a_m a_n & \text{if } m \neq n 
\end{cases}$$


where $a_m$ refers to the m-th element of the vector A.
Now the derivative of the loss with respect to this input vector, i.e., dLdZ is 1 × C vector and is calculated
as:

$$\frac{dL}{dZ} = \frac{dL}{dA} \cdot J$$
