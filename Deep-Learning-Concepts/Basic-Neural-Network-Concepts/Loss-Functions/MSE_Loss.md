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

Let's walk through a specific example of applying the Mean Squared Error (MSE) loss in a regression scenario. Suppose we are trying to predict house prices based on some features. For simplicity, we'll consider a case where our model predicts prices for two houses based on multiple features, such as area in square feet and number of bedrooms, so our batch size $N$ is 2 and the number of features $C$ is 2.

### Given Data:

- Model outputs ($A$): Predicted prices and number of bedrooms for two houses. Let's say the model predicts the following for each house (price in USD, and bedrooms as a count). This can be represented as a 2x2 matrix (since $N=2$ and $C=2$):

$$
A = \begin{bmatrix}
300,000 & 3 \\
500,000 & 4
\end{bmatrix}
$$

Here, the first column represents the predicted prices for two houses, and the second column represents the predicted number of bedrooms.

- Ground-truth values ($Y$): Actual prices and number of bedrooms for the two houses. This is also a 2x2 matrix:

$$
Y = \begin{bmatrix}
350,000 & 4 \\
450,000 & 3
\end{bmatrix}
$$

### Forward Pass (Calculating MSE Loss):

1. **Calculate Squared Error ($SE$):**

$$
SE(A, Y) = (A - Y) \odot (A - Y) = \begin{bmatrix}
(300,000 - 350,000)^2 & (3 - 4)^2 \\
(500,000 - 450,000)^2 & (4 - 3)^2
\end{bmatrix} = \begin{bmatrix}
2500 \times 10^6 & 1 \\
2500 \times 10^6 & 1
\end{bmatrix}
$$

2. **Sum of Squared Error ($SSE$):**
   Sum all elements in $SE$:

$$
SSE(A, Y) = \sum SE(A, Y) = 2 \times (2500 \times 10^6) + 2 \times 1 = 5000 \times 10^6 + 2
$$

3. **Mean Squared Error ($MSE$):**

$$
MSELoss(A, Y) = \frac{SSE(A, Y)}{N \cdot C} = \frac{5000 \times 10^6 + 2}{2 \times 2} = \frac{2500 \times 10^6 + 1}{2}
$$

### Backward Pass (Calculating Gradient):

The gradient of the loss with respect to the predictions ($A$) can be calculated as:

$$
\frac{\partial MSELoss}{\partial A} = 2 \cdot \frac{(A - Y)}{N \cdot C} = 2 \cdot \frac{\begin{bmatrix}
300,000 - 350,000 & 3 - 4 \\
500,000 - 450,000 & 4 - 3
\end{bmatrix}}{2 \times 2} = \frac{1}{2} \begin{bmatrix}
-50,000 & -1 \\
50,000 & 1
\end{bmatrix}
$$

This gradient matrix provides guidance on how to adjust each of the predictions (price and number of bedrooms) for each house to minimize the loss. The negative values indicate that the predictions need to be increased, and positive values suggest a decrease in the predictions to reduce the error.

## Reference:

- CMU_11785_Introduction_To_Deep_Learning
