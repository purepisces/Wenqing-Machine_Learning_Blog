# MSE Loss

MSE stands for Mean Squared Error, and is often used to quantify the prediction error for regression problems. Regression is a problem of predicting a real-valued label given an unlabeled example. Estimating house price based on features such as area, location, the number of bedrooms and so on is a classic regression problem.

## MSE Loss Forward Equation

We first calculate the squared error $SE$ between the model outputs $A$ and the ground-truth values $Y$:

$SE(A, Y) = (A - Y) \odot (A - Y)$ 

Then we calculate the sum of the squared error $SSE$, where $\iota_N$, $\iota_C$ are column vectors of size $N$ and $C$ which contain all 1s:

$SSE(A,Y) = \iota_{N}^{T} \cdot SE(A,Y) \cdot \iota_{C}$

Here, we are calculating the sum of all elements of the $N \times C$ matrix $SE(A, Y)$. The first pre multiplication with $\iota_{N}^{T}$ sums across rows. Then, the post multiplication of this product with $\iota_{C}$ sums the row sums across columns to give the final sum as a single number.

Lastly, we calculate the per-component Mean Squared Error $MSE$ loss:

$MSELoss(A, Y) = \frac{SSE(A, Y)}{N \cdot C}$

## MSE Loss Backward Equation

$MSELoss.backward() = 2 \cdot \frac{(A - Y)}{N \cdot C}$

The expression for the backward pass of the Mean Squared Error (MSE) loss, $2 \cdot \frac{(A - Y)}{N \cdot C}$, is derived from the derivative of the MSE loss function with respect to the model predictions $A$. Here's a step-by-step explanation:

Mean Squared Error Loss Function
The MSE loss function is defined as:
$MSELoss(A, Y) = \frac{1}{N \cdot C} \sum\limits_{i=1}^{N} \sum\limits_{j=1}^{C} (A_{ij} - Y_{ij})^2$

where:

$A$ are the predictions from the model.
$Y$ are the actual ground-truth values.
$N$ is the number of samples in the batch.
$C$ is the number of output dimensions per sample (in regression, this is usually 1).
Derivative of MSE Loss
To update the model parameters (in this case, through backpropagation), we need to know how changes in $A$ affect the loss. This is given by the derivative of the loss function with respect to $A$, denoted as $\frac{\partial MSELoss}{\partial A}$.

The derivative of the squared term $(A - Y)^2$ with respect to $A$ is $2 \cdot (A - Y)$. This comes from basic calculus rules, where the derivative of a function $f(x) = x^2$ is $f'(x) = 2x$.

Incorporating the Mean:
Since the MSE loss involves a mean (i.e., dividing the sum of squared errors by $N \cdot C$), when we differentiate the loss with respect to $A$, we must also take this division into account. This leads to the full derivative expression:

$\frac{\partial MSELoss}{\partial A} = 2 \cdot \frac{(A - Y)}{N \cdot C}$

This expression tells us how the loss changes with small changes in $A$, for each element of $A$. The factor of $2$ comes from the derivative of the squared term, and the division by $N \cdot C$ comes from the mean operation in the MSE formula.

Intuition
The gradient $\frac{\partial MSELoss}{\partial A}$ points in the direction of steepest increase of the loss function. By moving in the opposite direction (i.e., subtracting this gradient from the predictions $A$), we can reduce the loss, which is the goal of training the model.

In summary, the $2 \cdot \frac{(A - Y)}{N \cdot C}$ formula for $MSELoss.backward()$ is derived from differentiating the MSE loss function with respect to the predictions $A$, taking into account both the squared error and the mean operation. This gradient is used in the optimization process to adjust the model parameters in a way that minimizes the loss.

```python
import numpy as np
class MSELoss:
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        se = (A - Y) ** 2
        sse = np.sum(se)
        mse = sse / (A.shape[0] * A.shape[1])
        return mse

    def backward(self):
        dLdA = 2 * (self.A - self.Y) / (self.A.shape[0] * self.A.shape[1])
        return dLd
```
