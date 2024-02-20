# MSE Loss

MSE stands for Mean Squared Error, and is often used to quantify the prediction error for regression problems. Regression is a problem of predicting a real-valued label given an unlabeled example. Estimating house price based on features such as area, location, the number of bedrooms and so on is a classic regression problem.

## MSE Loss Forward Equation

We first calculate the squared error $SE$ between the model outputs $A$ and the ground-truth values $Y$:

$SE(A, Y) = (A - Y) \odot (A - Y)$ 

Then we calculate the sum of the squared error $SSE$, where $l_N$, $l_C$ are column vectors of size $N$ and $C$ which contain all 1s:

$SSE(A,Y) = \iota_{TN} \cdot SE(A,Y) \cdot \iota_{C}$

Here, we are calculating the sum of all elements of the $N \times C$ matrix $SE(A, Y)$. The first pre multiplication with $l^T_N$ sums across rows. Then, the post multiplication of this product with $l_C$ sums the row sums across columns to give the final sum as a single number.

Lastly, we calculate the per-component Mean Squared Error $MSE$ loss:

$MSELoss(A, Y) = \frac{SSE(A, Y)}{N \cdot C}$

## MSE Loss Backward Equation

$MSELoss.backward() = 2 \cdot \frac{(A - Y)}{N \cdot C}$

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

