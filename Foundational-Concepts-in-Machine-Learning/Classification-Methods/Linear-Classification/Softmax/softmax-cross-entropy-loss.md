# Softmax and Cross-Entropy Loss

The **Softmax Cross-Entropy Loss** is a widely used loss function in classification tasks, particularly for multiclass problems. It combines the **Softmax function** to convert raw scores into probabilities and the **Cross-Entropy Loss** to measure the difference between the predicted and true distributions.


## Softmax Function

The Softmax function transforms a vector of raw scores into a probability distribution, where each value is non-negative, and the sum is 1. Given a vector of scores $z$, the Softmax function is defined as:

$$f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

Where:

-   $z_j$ is the raw score for the $j$-th class.
-   $f_j(z)$ is the normalized probability for the $j$-th class.

The Softmax function outputs probabilities that can be interpreted as the model’s confidence in each class.


## Cross-Entropy Loss

The Cross-Entropy Loss measures the difference between two probability distributions: the true distribution $p$ and the predicted distribution $q$. For a single training example, it is defined as:

$$H(p, q) = -\sum_x p(x) \log q(x)$$

For classification, $p$ is typically a one-hot encoded vector representing the true class, and $q$ is the predicted probability distribution from the Softmax function. In practice, the loss is computed as:

$$L_i = -\log\left(\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}\right)$$

Where:

-   $f_{y_i}$ is the raw score for the correct class.
-   $f_j$ is the raw score for the $j$-th class.

This can be rewritten as:

$$L_i = -f_{y_i} + \log \sum_j e^{f_j}$$

The full dataset loss is the mean of $L_i$ over all training examples, along with an optional regularization term $R(W)$ for weight penalties:

$$L = \frac{1}{N} \sum_i L_i + \lambda R(W)$$


## Probabilistic Interpretation

The Softmax Cross-Entropy Loss has a natural probabilistic interpretation. The probability of the correct class $y_i$ for input $x_i$ is given by:

$$P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}$$

The Cross-Entropy Loss minimizes the negative log-likelihood of the correct class, making it equivalent to **Maximum Likelihood Estimation (MLE)**.

Additionally, the regularization term $R(W)$ can be interpreted as a prior, making the optimization a form of **Maximum a Posteriori (MAP) Estimation**.


## Information Theory View

From an information-theoretic perspective, the Cross-Entropy Loss minimizes the **KL divergence** between the true distribution $p$ (where all probability is concentrated on the correct class) and the predicted distribution $q$. The cross-entropy can be expressed as:

$$H(p, q) = H(p) + D_{KL}(p \| q)$$

Since $H(p) = 0$ for a one-hot true distribution, minimizing the cross-entropy is equivalent to minimizing the KL divergence.


## Practical Considerations: Numerical Stability

The exponential calculations in the Softmax function can lead to numerical instability when working with large scores. To address this, the scores are normalized by subtracting the maximum value in the score vector:

$$f = f - \max_j f_j$$

This adjustment ensures that the largest value in the score vector becomes 0, reducing the risk of overflow. In Python, this can be implemented as:

```python
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
```

## Key Differences from SVM Loss

-   The Softmax classifier interprets scores as (unnormalized) log probabilities, while the SVM treats scores as class margins.
-   The Cross-Entropy Loss encourages the correct class probability to approach 1, while the SVM ensures the correct class score exceeds others by a margin.

For instance:

-   If scores are $[10, -100, -100]$:
    -   The Softmax Loss will be much smaller compared to scores $[10, 9, 9]$ because the correct class is more confidently predicted.
    -   The SVM Loss would consider both scenarios equivalent if the margin constraint is satisfied.


## Python Implementation

Below is an example Python implementation for computing the Softmax Cross-Entropy Loss.

### Unvectorized Implementation

```python
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops).
    - W: weights (D x C)
    - X: data (N x D)
    - y: labels (N)
    - reg: regularization strength
    """
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # Normalize for numerical stability
        probabilities = np.exp(scores) / np.sum(np.exp(scores))
        loss += -np.log(probabilities[y[i]])
        for j in range(num_classes):
            dW[:, j] += (probabilities[j] - (j == y[i])) * X[i]
    
    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    return loss, dW
```
### Vectorized Implementation

```python
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    - W: weights (D x C)
    - X: data (N x D)
    - y: labels (N)
    - reg: regularization strength
    """
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)  # Normalize for numerical stability
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    loss = -np.sum(np.log(probabilities[np.arange(num_train), y]))
    
    probabilities[np.arange(num_train), y] -= 1
    dW = X.T.dot(probabilities)
    
    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    return loss, dW
```

## Summary

-   The Softmax function converts raw scores into probabilities.
-   The Cross-Entropy Loss measures the difference between the predicted probabilities and the true class labels.
-   This loss function is interpretable, probabilistic, and widely used for multiclass classification tasks.
-   Numerical stability is crucial in implementation to avoid computational errors.

## Reference:
- [https://cs231n.github.io/classification/](https://cs231n.github.io/linear-classify/)
