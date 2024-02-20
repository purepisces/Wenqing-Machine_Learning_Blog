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
