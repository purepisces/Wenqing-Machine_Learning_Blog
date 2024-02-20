# MSE Loss

MSE, or Mean Squared Error, is a widely used metric for evaluating the prediction error in regression problems. In regression, the goal is to predict a continuous value, such as estimating a house's price based on its features (e.g., area, location, number of bedrooms).

## MSE Loss Forward Equation

The computation begins with calculating the squared error ($SE$) between the model's predictions ($A$) and the actual ground-truth values ($Y$):

$$SE(A, Y) = (A - Y) \odot (A - Y)$$

Next, we determine the sum of the squared errors ($SSE$). Here, $\iota_N$ and $\iota_C$ represent column vectors filled with 1s, of sizes $N$ and $C$ respectively:

$$SSE(A,Y) = \iota_{N}^{T} \cdot SE(A,Y) \cdot \iota_{C}$$

This operation sums all elements in the $SE(A, Y)$ matrix, which has dimensions $N \times C$. The multiplication by $\iota_{N}^{T}$ aggregates the errors across rows, and subsequent multiplication by $\iota_{C}$ sums these across columns, yielding the total error as a single scalar.

The Mean Squared Error ($MSE$) loss per component is then computed as:

$$MSELoss(A, Y) = \frac{SSE(A, Y)}{N \cdot C}$$

## MSE Loss Backward Equation
During backpropagation, the gradient of the MSE loss with respect to the model's outputs ($A$) is needed for updating model parameters:

$$MSELoss.backward() = 2 \cdot \frac{(A - Y)}{N \cdot C}$$

### Derivative of MSE Loss

The MSE loss function is defined as:

$$MSELoss(A, Y) = \frac{1}{N \cdot C} \sum\limits_{i=1}^{N} \sum\limits_{j=1}^{C} (A_{ij} - Y_{ij})^2$$

where:

$A$: Model predictions.

$Y$: Ground-truth values.

$N$: Number of samples in the batch.

$C$: Number of output dimensions per sample, typically 1 in regression tasks.

To update the model parameters (in this case, through backpropagation), we need to know how changes in $A$ affect the loss. This is given by the derivative of the loss function with respect to $A$, denoted as $\frac{\partial MSELoss}{\partial A}$.

$$\frac{\partial MSELoss}{\partial A} = 2 \cdot \frac{(A - Y)}{N \cdot C}$$

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
import numpy as np

class MSELoss:
    def forward(self, A, Y):
        # Store the predictions (A) and ground truth values (Y) for backward computation
        self.A = A
        self.Y = Y
        # Compute the squared error between predictions and ground truth
        se = (A - Y) ** 2
        # Sum the squared errors to get the total squared error
        sse = np.sum(se)
        # Compute the Mean Squared Error by dividing the total squared error by the number of elements
        mse = sse / (A.shape[0] * A.shape[1])
        return mse

    def backward(self):
        # Compute the gradient of the loss with respect to the predictions (A)
        dLdA = 2 * (self.A - self.Y) / (self.A.shape[0] * self.A.shape[1])
        return dLdA

```
## Example
Let's walk through a specific example of applying the Mean Squared Error (MSE) loss in a regression scenario. Suppose we are trying to predict house prices based on some features. For simplicity, we'll consider a case where our model predicts prices for two houses based on a single feature (like area in square feet), so our batch size $N$ is 2 and the number of features (or classes in this context) $C$ is 1.

### Given Data:

- Model outputs ($A$): Predicted prices of two houses, say $[300,000; 500,000]$ (in some currency, let's assume USD for simplicity). This can be represented as a 2x1 matrix (since $N=2$ and $C=1$):

$$A = \begin{bmatrix}
300,000 \\
500,000
\end{bmatrix}$$

- Ground-truth values ($Y$): Actual prices of the two houses, say $[350,000; 450,000]$. This is also a 2x1 matrix:

$$Y = \begin{bmatrix}
350,000 \\
450,000
\end{bmatrix}$$

### Forward Pass (Calculating MSE Loss):

1. **Calculate Squared Error ($SE$):**

$$ SE(A, Y) = (A - Y) \odot (A - Y) = \begin{bmatrix}
(300,000 - 350,000)^2\\
(500,000 - 450,000)^2
\end{bmatrix}
= \begin{bmatrix}
2500 \times 10^6 \\
2500 \times 10^6
\end{bmatrix} $$

2. **Sum of Squared Error ($SSE$):**
   Since we only have one feature, the $SSE$ is simply the sum of all elements in $SE$:

$$
SSE(A,Y) = \sum SE(A,Y) = 2500 \times 10^6 + 2500 \times 10^6 = 5000 \times 10^6
$$

3. **Mean Squared Error ($MSE$):**

$$
MSELoss(A, Y) = \frac{SSE(A, Y)}{N \cdot C} = \frac{5000 \times 10^6}{2 \times 1} = 2500 \times 10^6
$$

### Backward Pass (Calculating Gradient):

The gradient of the loss with respect to the predictions ($A$) can be calculated as:


$$\frac{\partial MSELoss}{\partial A} = 2 \cdot \frac{(A - Y)}{N \cdot C} = 2 \cdot \frac{\begin{bmatrix}
300,000 - 350,000 \\
500,000 - 450,000
\end{bmatrix}}{2 \times 1} = \begin{bmatrix}
-50,000 \\
50,000
\end{bmatrix}$$

This gradient tells us how to adjust our predictions to reduce the loss. For the first house, since the gradient is negative, we need to increase our prediction to reduce the loss, and for the second house, since the gradient is positive, we need to decrease our prediction.

