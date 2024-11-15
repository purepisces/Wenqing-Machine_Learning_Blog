# Multiclass SVM Loss

The **Multiclass Support Vector Machine (SVM) Loss** is a commonly used loss function for classification tasks. It ensures that the score for the correct class exceeds the scores for all incorrect classes by a fixed margin, denoted as $\Delta$.


## Definition

For a single training example $(x_i, y_i)$, where $x_i$ is the input and $y_i$ is the correct class label, the loss is defined as:

$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

Here:

-   $s_j$ is the score for the $j$-th class.
-   $s_{y_i}$ is the score for the correct class.
-   $\Delta$ is the margin parameter, typically set to 1.

The loss penalizes the model whenever the score for an incorrect class is within $\Delta$ of the correct class score.


## Example Calculation

Suppose we have three classes, and the scores for an input are $s = [13, -7, 11]$, with the true class being $y_i = 0$. Using a margin $\Delta = 10$, the loss is computed as:

$$L_i = \max(0, -7 - 13 + 10) + \max(0, 11 - 13 + 10)$$

-   The first term evaluates to $0$, as the result is negative.
-   The second term evaluates to $8$, as the score difference falls short of the margin by 8.

Thus, the total loss for this example is $L_i = 8$.


## Full Loss with Regularization

To improve generalization and address weight ambiguity, we add an **L2 regularization penalty** to the loss function. The regularization term penalizes large weights, encouraging smaller and more distributed weight values.

The full loss for a dataset with $N$ training examples is:

$$L = \frac{1}{N} \sum_i L_i + \lambda R(W)$$

Where:

-   $R(W) = \sum_k \sum_l W_{k,l}^2$ is the L2 regularization term.
-   $\lambda$ is a hyperparameter controlling the regularization strength.

The complete expanded form is:

$$L = \frac{1}{N} \sum_i \sum_{j \neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k \sum_l W_{k,l}^2$$


## Key Concepts

### Hinge Loss

The $\max(0, -)$ function in the loss is referred to as **hinge loss**, which penalizes violations of the margin constraint. In some implementations, **squared hinge loss** ($\max(0, -)^2$) is used, which applies a quadratic penalty for stronger violations.

### Regularization

The regularization term $R(W)$ helps:

-   Prevent overfitting by discouraging large weights.
-   Improve generalization performance on test data.

Typically, regularization is applied to weights ($W$) rather than biases ($b$).


## Practical Considerations

### Setting the Margin Parameter ($\Delta$)

-   $\Delta$ is usually set to 1 by convention.
-   The tradeoff between the data loss and regularization loss is controlled by the hyperparameter $\lambda$.

### Scaling Invariance

Scaling the weights $W$ does not affect the hinge loss, as score differences remain proportional. Therefore, $\lambda$ plays a crucial role in controlling weight magnitudes.

## Implementation in Python

Below are examples of Python implementations for the Multiclass SVM Loss.

### Unvectorized Implementation

```python
def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in range(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i
```



### Vectorized Implementation

```python
def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
```



### Fully Vectorized Implementation

```python
def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment
```

## Summary

The Multiclass SVM Loss:

-   Ensures a margin $\Delta$ between correct and incorrect class scores.
-   Penalizes violations of this margin using hinge loss.
-   Incorporates regularization to improve generalization and prevent overfitting.

With its intuitive margin-based approach, the Multiclass SVM Loss remains a foundational method in classification tasks.

## Reference:
- [https://cs231n.github.io/classification/](https://cs231n.github.io/linear-classify/)
