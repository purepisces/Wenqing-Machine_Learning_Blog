# Softmax Regression

Softmax regression, also known as multinomial logistic regression, is a generalization of logistic regression for multiclass classification problems. While logistic regression is used for binary classification, softmax regression is used when there are more than two classes.

### Key Points of Softmax Regression:

- **Multiclass Classification**: Softmax regression is used when the dependent variable can take on more than two categories. For instance, classifying digits (0-9) in the MNIST dataset.

- **Generalization of Logistic Regression**: While logistic regression predicts probabilities for two classes, softmax regression predicts the probabilities of each class over all possible classes.

- **Softmax Function**: The core of softmax regression is the softmax function, which is used to convert raw prediction scores (logits) into probabilities. The softmax function is defined as follows:

  $$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

  where $\mathbf{z}$ is the input vector of raw scores (logits), $K$ is the number of classes, and $\sigma(\mathbf{z})_i$ is the probability that the input belongs to class $i$.

- **Loss Function**: The loss function used in softmax regression is the cross-entropy loss, which measures the performance of a classification model whose output is a probability value between 0 and 1. The cross-entropy loss for a single sample is given by:

  $$ \ell_{\mathrm{softmax}}(\mathbf{z}, y) = -\log(\sigma(\mathbf{z})_y) $$

  where $y$ is the true class label.

### Differences from Linear and Logistic Regression:

- **Linear Regression**: Used for predicting a continuous dependent variable. It fits a linear relationship between the independent variables and the dependent variable.

  $$h(\mathbf{x}) = \mathbf{x}^\top \mathbf{\theta}$$

- **Logistic Regression**: Used for binary classification. It applies the logistic (sigmoid) function to linear predictions to model the probability of the binary outcomes.

  $$P(y=1|\mathbf{x}) = \sigma(\mathbf{x}^\top \mathbf{\theta}) = \frac{1}{1 + e^{-\mathbf{x}^\top \mathbf{\theta}}}$$

- **Softmax Regression**: Extends logistic regression to multiple classes. It uses the softmax function to convert linear predictions into probabilities for each class.

  $$P(y=i|\mathbf{x}) = \sigma(\mathbf{z})i = \frac{e^{\mathbf{x}^\top \mathbf{\theta}i}}{\sum_{j=1}^{K} e^{\mathbf{x}^\top \mathbf{\theta}_j}}$$

In summary, softmax regression is a type of logistic regression tailored for multiclass classification tasks. It uses the softmax function to predict the probabilities of each class, allowing for the classification of instances into multiple categories.

```python
def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = theta.shape[1]
 
    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        # Compute the logits
        logits = X_batch @ theta

        # Compute the softmax probabilities
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Create a one-hot encoded matrix of the true labels
        I_y = np.zeros_like(probabilities)
        I_y[np.arange(y_batch.size), y_batch] = 1

        # Compute the gradient
        gradient = X_batch.T @ (probabilities - I_y) / y_batch.size

        # Update the parameters
        theta -= lr * gradient
   
    ### END YOUR CODE
```
