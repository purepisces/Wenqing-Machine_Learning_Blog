# SVM Classifier

The **Support Vector Machine (SVM)** classifier is a popular approach for classification tasks, especially for its ability to maximize the separation margin between classes. This document details the **Multiclass Support Vector Machine (SVM) Loss**, its underlying principles, and practical considerations.

## Multiclass Support Vector Machine Loss

The Multiclass SVM Loss ensures that the score for the correct class exceeds the scores for all incorrect classes by a fixed margin, denoted as $\Delta$. The loss for a single example $(x_i, y_i)$ is defined as:

$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

Here:

-   $s_j$ is the score for the $j$-th class.
-   $s_{y_i}$ is the score for the correct class.
-   $\Delta$ is the margin parameter, typically set to 1.

### Example Calculation

For three classes with scores $s = [13, -7, 11]$ and the correct class $y_i = 0$:

-   $\Delta = 10$
-   Loss calculation:

$$L_i = \max(0, -7 - 13 + 10) + \max(0, 11 - 13 + 10) = 0 + 8 = 8$$

This penalty of 8 arises because the correct class score (13) is not sufficiently higher than the incorrect class score (11) by at least the margin $\Delta$.

### Full Loss with Regularization

To improve generalization and resolve weight ambiguity, we add an **L2 regularization term** to the loss:

$$R(W) = \sum_k \sum_l W_{k,l}^2$$

The complete SVM loss becomes:

$$L = \frac{1}{N} \sum_i L_i + \lambda R(W)$$

Here:

-   $N$ is the number of training examples.
-   $\lambda$ controls the regularization strength.

### Hinge Loss

The $\max(0, -)$ function in the SVM loss is called **hinge loss**, and penalizes scores that do not satisfy the margin constraint. **Squared hinge loss** ($\max(0, -)^2$) applies a stronger penalty and is an alternative to hinge loss.


## SVM Principles: Max Margin

In SVMs, the objective is to find a hyperplane that separates two classes while maximizing the margin—the distance between the closest points of each class to the hyperplane. This is achieved by minimizing $|w|$ under the constraints:

-   $w \cdot x + b = +1$: positive class boundary.
-   $w \cdot x + b = -1$: negative class boundary.

The margin is $\frac{2}{|w|}$.


## Regularization in SVM

Regularization prevents overfitting by discouraging large weights:

-   Encourages smaller and more evenly distributed weights.
-   Improves generalization performance.
-   Usually applied to weights ($W$), not biases ($b$).


## Practical Considerations

### Setting the Margin Parameter ($\Delta$)

-   Typically set to $\Delta = 1$.
-   The parameter $\lambda$ primarily controls the tradeoff between data loss and regularization loss.

### Relation to Binary SVM

In binary SVM:

$$L_i = C \max(0, 1 - y_i w^T x_i) + R(W)$$

Here:

-   $C$ controls the balance between data loss and regularization.
-   $y_i \in {-1, 1}$.

The multiclass SVM is equivalent to binary SVM for two classes, with $C \propto \frac{1}{\lambda}$.


## SVM Code Implementation

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

-   SVM loss ensures class scores maintain a margin $\Delta$ between the correct and incorrect classes.
-   Regularization ($R(W)$) prevents overfitting and resolves weight ambiguity.
-   The balance between data loss and regularization is controlled by $\lambda$.

## Reference:
- [https://cs231n.github.io/classification/](https://cs231n.github.io/linear-classify/)
