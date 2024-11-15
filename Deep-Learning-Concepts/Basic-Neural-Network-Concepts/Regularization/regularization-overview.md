# Regularization Overview

Regularization is a crucial concept in machine learning, introduced to address overfitting and improve the generalization of models. It modifies the loss function to include an additional term that penalizes complex or large weights, encouraging the model to learn simpler and more generalizable representations.

## Why Regularization?

When training a model, it is possible to find a set of weights **W** that perfectly classify every training data point, resulting in zero loss. However, such a solution may not generalize well to unseen data. For example, scaling **W** by any positive factor (e.g., multiplying **W** by $\lambda$, where $\lambda > 1$) will still yield zero loss because it uniformly increases all score differences while maintaining classification margins. This leads to ambiguity in the weight selection.

Regularization resolves this ambiguity by introducing a penalty term, which discourages overly large weights and promotes simpler solutions.

## Regularization Formulation

The most common form of regularization is the squared **L2** norm, often referred to as the **L2 penalty** or **weight decay**, which applies a quadratic penalty on the weights: 

$$R(W) = \sum_k \sum_l W_{k,l}^2$$

Here, every element of the weight matrix **W** is squared and summed. The **regularization loss** is added to the **data loss** (computed on the training dataset) to form the full loss function:

$$L = \underbrace{\frac{1}{N} \sum_i L_i}_\text{data loss} + \underbrace{\lambda R(W)}_\text{regularization loss}$$

or, fully expanded:

$$L = \frac{1}{N} \sum_i \sum_{j \neq y_i} \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) + \lambda \sum_k \sum_l W_{k,l}^2$$

-   **Data Loss**: Measures how well the model predicts the correct labels.
-   **Regularization Loss**: Penalizes large weights to improve generalization.
-   **$\lambda$**: A hyperparameter controlling the balance between data loss and regularization loss. This value is typically chosen through cross-validation.

## Benefits of Regularization

### 1. **Improved Generalization**

Regularization prevents the model from fitting noise in the training data, leading to better performance on unseen data. For instance, it encourages the model to consider all input dimensions rather than focusing on a few dominant features.

Example:

-   Input vector: $x = [1,1,1,1]$
-   Weight vectors: $w_1 = [1,0,0,0]$ and $w_2 = [0.25,0.25,0.25,0.25]$
-   Both yield the same dot product: $w_1^T x = w_2^T x = 1$
-   However, $w_1$ has an L2 penalty of 1.0, while $w_2$ has a lower penalty of 0.5, as the weights are smaller and more diffuse.

Regularization thus encourages solutions like $w_2$, which leverage all input features more evenly, improving generalization.


### 2. **Controlled Model Complexity**

By penalizing large weights, regularization discourages the model from becoming overly complex, which can lead to overfitting.

### 3. **Max Margin Property**

In Support Vector Machines (SVMs), adding the L2 penalty introduces the desirable **max margin** property, where the model maximizes the margin between classes.


## Practical Considerations

-   **Weights vs. Biases**: Regularization typically applies to weights **W**, not biases **b**, as weights control the input influence strength. In practice, this distinction often has negligible effects.
-   **Achieving Zero Loss**: Achieving zero loss with regularization is unrealistic because it would require a pathological scenario where $W=0$.
-   **Hyperparameter Tuning**: The regularization strength $\lambda$ is selected via cross-validation to balance the tradeoff between data loss and regularization loss.

## Example in Code

Below is an implementation of the Multiclass Support Vector Machine (SVM) loss with regularization:

```python
def svm_loss(W, X, y, reg):
    """
    Computes the SVM loss with regularization.
    - W: Weight matrix (e.g., 10 x 3073 for CIFAR-10)
    - X: Input data matrix (e.g., 3073 x 50000 for CIFAR-10)
    - y: Labels (e.g., 50000-D array)
    - reg: Regularization strength
    """
    # Compute scores
    scores = X.dot(W.T)
    correct_scores = scores[np.arange(len(y)), y].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_scores + 1)
    margins[np.arange(len(y)), y] = 0  # Do not consider correct class

    # Data loss
    data_loss = np.sum(margins) / X.shape[0]

    # Regularization loss
    reg_loss = reg * np.sum(W ** 2)

    # Total loss
    total_loss = data_loss + reg_loss
    return total_loss` 
```

## Summary

Regularization is a fundamental technique in machine learning that improves model generalization by discouraging large weights. By penalizing model complexity, regularization resolves weight ambiguity, prevents overfitting, and encourages simpler, more robust models. The tradeoff between data loss and regularization loss, controlled by the hyperparameter $\lambda$, ensures a balance between accuracy and generalization.

## Reference:
- [https://cs231n.github.io/classification/](https://cs231n.github.io/linear-classify/)
- linear-classification.md in Machine Learning Blog
