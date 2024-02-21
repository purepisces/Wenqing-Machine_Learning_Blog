# Loss Functions
Different loss functions may become useful depending on the type of neural network and type of data you are using, like Cross-Entropy Loss for classification, Mean Squared Error for regression. It is important to know how these are calculated, and how they will be used to update your network. 

## Loss Class

### Class Attributes:
- $A$: Stores model prediction to compute back-propagation.
- $Y$: Stores desired output to compute back-propagation.

### Class Methods:
- $forward$: 
  - Parameters: $A$ (model prediction), $Y$ (desired output)
  - Returns: Loss value $L$
  - Description: Calculates and returns a scalar loss value $L$ quantifying the mismatch between the network output and the desired output.
  
- $backward$: 
  - Returns: $dLdA$ (how changes in model outputs affect loss $L$)
  - Description: Calculates and returns `dLdA`, which represents how changes in model outputs $A$ affect the loss $L$. It enables downstream computation for back-propagation.


Please consider the following class structure:
```python
class Loss:
        def forward(self, A, Y):
            self.A = A
            self.Y = Y
            self.    # TODO (store additional attributes as needed)
            N      = # TODO,  this is the first dimension of A and Y
            C      = # TODO,  this is the second dimension of A and Y
            # TODO

            return L

        def backward(self):
            dLdA = # TODO

            return dLdA
```

| Code Name | Math      | Type    | Shape | Meaning                                 |
|-----------|-----------|---------|-------|-----------------------------------------|
| N         | $N$   | scalar  | -     | batch size                              |
| C         | $C$   | scalar  | -     | number of classes                       |
| A         | $A$   | matrix  | $N \times C$ | model outputs                        |
| Y         | $Y$   | matrix  | $N \times C$ | ground-truth values                   |
| L         | $L$   | scalar  | -     | loss value                              |
| dLdA      | $\frac{\partial L}{\partial A}$ | matrix  | $N \times C$ | how changes in model outputs affect loss |

> Note: In the context of regression tasks, the dimension corresponding to the number of classes, $C$, simplifies to 1. This is because regression problems involve predicting a single continuous variable, rather than selecting from multiple categories. Conversely, in classification scenarios, $C$ represents the total number of distinct classes or categories into which each input can be classified, and thus can vary based on the specific problem at hand.
>

The loss function topology is visualized in the follwing Figure, whose reference persists throughout this document.

<img src="Loss_Function_Topology.png" alt="Loss_Function_Topology" width="400" height="300"/>

## Example

In this example, we illustrate the use of the $Loss$ class for a classification task with 3 classes. Consider a scenario where our batch size ($N$) is 2, meaning we process two examples at a time.

### Model Outputs (A)

The model outputs, denoted as $A$, are the predicted probabilities for each class. For our example with a batch size of 2 and 3 classes, $A$ could look like:

$$A = 
\begin{bmatrix}
0.7 & 0.2 & 0.1\\
0.1 & 0.8 & 0.1
\end{bmatrix}
$$

This matrix signifies the network's predictions:
- For the first example, the probabilities for Class 1, Class 2, and Class 3 are 0.7, 0.2, and 0.1, respectively, suggesting a prediction of Class 1.
- For the second example, the probabilities are 0.1, 0.8, and 0.1, indicating a prediction of Class 2.

### Ground-Truth Values (Y)

The ground-truth values, $Y$, are represented in a one-hot encoded format:


$$Y = 
\begin{bmatrix}
1 & 0 & 0\\
0 & 0 & 1
\end{bmatrix}
$$

Here, the first row $[1, 0, 0]$ indicates that the first example belongs to Class 1, and the second row $[0, 0, 1]$ shows the second example belongs to Class 3.

### Using $A$ and $Y$ in Loss Calculation

The loss function employs the matrices $A$ and $Y$ to compute the loss value $L$. For example, employing the Cross-Entropy Loss function would involve calculating the loss for each individual example by comparing the predicted probabilities in $A$ against the actual labels in $Y$. Subsequently, these individual losses are averaged across all examples within the batch to yield a single scalar value for the loss, $L$.

The $forward$ method within the $Loss$ class is tasked with computing this scalar loss value, $L$, utilizing $A$ and $Y$. Following this, the $backward$ method calculates the gradient of the loss with respect to the model outputs, denoted as $\frac{\partial L}{\partial A}$. This gradient is crucial for the back-propagation process, enabling the update of model parameters during training.

## Reference:

- CMU_11785_Introduction_To_Deep_Learning

