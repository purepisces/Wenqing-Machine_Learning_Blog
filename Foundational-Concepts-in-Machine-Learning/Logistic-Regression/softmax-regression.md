# Softmax Regression

Softmax regression, also known as multinomial logistic regression, is a generalization of logistic regression for multiclass classification problems. While logistic regression is used for binary classification, softmax regression is used when there are more than two classes.

### Key Points of Softmax Regression:

- **Multiclass Classification**: Softmax regression is used when the dependent variable can take on more than two categories. For instance, classifying digits (0-9) in the MNIST dataset.

- **Generalization of Logistic Regression**: While logistic regression predicts probabilities for two classes, softmax regression predicts the probabilities of each class over all possible classes.

- **Softmax Function**: The core of softmax regression is the softmax function, which is used to convert raw prediction scores (logits) into probabilities. The softmax function is defined as follows:

  $$\sigma(\mathbf{z})i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

  where $\mathbf{z}$ is the input vector of raw scores (logits), $K$ is the number of classes, and $\sigma(\mathbf{z})_i$ is the probability that the input belongs to class $i$.

- **Loss Function**: The loss function used in softmax regression is the cross-entropy loss, which measures the performance of a classification model whose output is a probability value between 0 and 1. The cross-entropy loss for a single sample is given by:

  $$\ell_{\mathrm{softmax}}(\mathbf{z}, y) = -\log(\sigma(\mathbf{z})_y)$$

  where $y$ is the true class label.

### Differences from Linear and Logistic Regression:

- **Linear Regression**: Used for predicting a continuous dependent variable. It fits a linear relationship between the independent variables and the dependent variable.

  $$h(\mathbf{x}) = \mathbf{x}^\top \mathbf{\theta}$$

- **Logistic Regression**: Used for binary classification. It applies the logistic (sigmoid) function to linear predictions to model the probability of the binary outcomes.

  $$P(y=1|\mathbf{x}) = \sigma(\mathbf{x}^\top \mathbf{\theta}) = \frac{1}{1 + e^{-\mathbf{x}^\top \mathbf{\theta}}}$$

- **Softmax Regression**: Extends logistic regression to multiple classes. It uses the softmax function to convert linear predictions into probabilities for each class.

  $$P(y=i|\mathbf{x}) = \sigma(\mathbf{z})i = \frac{e^{\mathbf{x}^\top \mathbf{\theta}i}}{\sum_{j=1}^{K} e^{\mathbf{x}^\top \mathbf{\theta}_j}}$$

In summary, softmax regression is a type of logistic regression tailored for multiclass classification tasks. It uses the softmax function to predict the probabilities of each class, allowing for the classification of instances into multiple categories.

## Softmax(a.k.a. cross-entropy) loss:

Implement the softmax (a.k.a. cross-entropy) loss as defined in `softmax_loss()` function in `src/simple_ml.py`.  Recall (hopefully this is review, but we'll also cover it in lecture on 9/1), that for a multi-class output that can take on values $y \in \{1,\ldots,k\}$, the softmax loss takes as input a vector of logits $z \in \mathbb{R}^k$, the true class $y \in \{1,\ldots,k\}$ returns a loss defined by

$$\begin{equation}
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.
\end{equation}$$

Note that as described in its docstring, `softmax_loss()` takes a _2D array_ of logits (i.e., the $k$ dimensional logits for a batch of different samples), plus a corresponding 1D array of true labels, and should output the _average_ softmax loss over the entire batch.  Note that to do this correctly, you should _not_ use any loops, but do all the computation natively with numpy vectorized operations (to set expectations here, we should note for instance that our reference solution consists of a single line of code).

Note that for "real" implementation of softmax loss you would want to scale the logits to prevent numerical overflow, but we won't worry about that here (the rest of the assignment will work fine even if you don't worry about this). 

```python
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # Formula for one training sample: \begin{equation} \ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y. \end{equation}
    
    # Compute the log of the sum of exponentials of logits for each sample
    log_sum_exp = np.log(np.sum(np.exp(Z), axis = 1))
    # Extract the logits corresponding to the true class for each sample
    # np.arange(Z.shape[0]) generates array [0, 1, 2, ..., batch_size-1]
    # Z[np.arange(Z.shape[0]), y] = Z[[row_indices], [col_indices]]
    # This selects the logits Z[i, y[i]] for each i which is each row
    correct_class_logits = Z[np.arange(Z.shape[0]), y]
    losses = log_sum_exp - correct_class_logits
    return np.mean(losses)
    ### END YOUR CODE
```
Example
```python
import numpy as np

# Logits for a batch of 3 samples and 4 classes
Z = np.array([[2.0, 1.0, 0.1, 0.5],
              [1.5, 2.1, 0.2, 0.7],
              [1.1, 1.8, 0.3, 0.4]])

# True labels for the 3 samples
y = np.array([0, 1, 2])

# np.arange(Z.shape[0]) creates an array [0, 1, 2]
row_indices = np.arange(Z.shape[0])
print("Row indices:", row_indices)  # Output: [0 1 2]

# y is [0, 1, 2]
print("True class labels:", y)  # Output: [0 1 2]

# Advanced indexing: Z[np.arange(Z.shape[0]), y] selects Z[0, 0], Z[1, 1], Z[2, 2]
correct_class_logits = Z[row_indices, y]

print("Correct class logits:", correct_class_logits)
# Output: [2.0, 2.1, 0.3]
```

### Math Prove CMU 10714 For Loss function #2: softmax / cross-entropy loss

Let's convert the hypothesis function to a "probability" by exponentiating and normalizing its entries (to make them all positive and sum to one)

$$z_i = p(\text{label} = i) = \frac{\exp(h_i(x))}{\sum_{j=1}^k \exp(h_j(x))} \Longleftrightarrow z \equiv \text{softmax}(h(x))$$

Then let's define a loss to be the (negative) log probability of the true class: this is called softmax or cross-entropy loss

$$\ell_{ce}(h(x), y) = - \log p(\text{label} = y) = - h_y(x) + \log \sum_{j=1}^k \exp(h_j(x))$$


### Math Prove By Myself

**Equation for All Training Examples**:

$$H(Y, P) = -\sum_{i=1}^k Y_i \log(P_i) = H(Y, \sigma(z)) = -\sum\limits_{i=1}^k Y_i \log(\sigma(z)_i)$$

**Equation for One Training Example**:

$$H(Y, \sigma(z)) = -\log(\sigma(z)y) = -\log\left( \frac{\exp(z_y)}{\sum\limits_{j=1}^k \exp(z_j)} \right)$$


**Simplified Equation for One Training Example**:

$$H(Y, \sigma(z)) = -z_y + \log\left( \sum\limits_{j=1}^k \exp(z_j) \right)$$

#### Softmax Function

The softmax function converts logits (raw scores) into probabilities. For a vector of logits $z$ of length $k$, the softmax function $\sigma(z)$ is defined as:

$$\sigma(z)i = \frac{\exp(z_i)}{\sum\limits_{j=1}^k \exp(z_j)}$$

for $i = 1, \ldots, k$.

#### Cross-Entropy Loss

The cross-entropy loss measures the difference between the true labels and the predicted probabilities. For a true label vector $Y$ (one-hot encoded) and a predicted probability vector $P$ (output of the softmax function), the cross-entropy loss $H(Y, P)$ is defined as:

$$H(Y, P) = -\sum_{i=1}^k Y_i \log(P_i)$$

#### Connection Between Softmax and Cross-Entropy

When using the softmax function as the final layer in a neural network for multi-class classification, the predicted probability vector $P$ is given by:

$$P_i = \sigma(z) i = \frac{\exp(z_i)}{\sum\limits_{j=1}^k \exp(z_j)}$$

The cross-entropy loss then becomes:

$$H(Y, \sigma(z)) = -\sum_{i=1}^k Y_i \log(\sigma(z)_i)$$

For a single training example where the true class is $y$, $Y$ is a one-hot encoded vector where $Y_y = 1$ and $Y_i = 0$ for $i \neq y$. Thus, the cross-entropy loss simplifies to:

$$H(Y, \sigma(z)) = -\log(\sigma(z)y) = -\log\left( \frac{\exp(z_y)}{\sum\limits_{j=1}^k \exp(z_j)} \right)$$

Using properties of logarithms, this can be rewritten as:

$$H(Y, \sigma(z)) = -\left( \log(\exp(z_y)) - \log\left( \sum\limits_{j=1}^k \exp(z_j) \right) \right)$$

$$H(Y, \sigma(z)) = -z_y + \log\left( \sum\limits_{j=1}^k \exp(z_j) \right)$$

## Stochastic gradient descent for softmax regression

In this question you will implement stochastic gradient descent (SGD) for (linear) softmax regression.  In other words, as discussed in lecture on 9/1, we will consider a hypothesis function that makes $n$-dimensional inputs to $k$-dimensional logits via the function

$$\begin{equation}
h(x) = \Theta^T x
\end{equation}$$

where $x \in \mathbb{R}^n$ is the input, and $\Theta \in \mathbb{R}^{n \times k}$ are the model parameters.  Given a dataset $\{(x^{(i)} \in \mathbb{R}^n, y^{(i)} \in \{1,\ldots,k\})\}$, for $i=1,\ldots,m$, the optimization problem associated with softmax regression is thus given by

$$\begin{equation}
\minimize_{\Theta} \; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(\Theta^T x^{(i)}, y^{(i)}).
\end{equation}$$

Recall from class that the gradient of the linear softmax objective is given by

$$\begin{equation}
\nabla_\Theta \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T
\end{equation}$$

where

$$\begin{equation}
z = \frac{\exp(\Theta^T x)}{1^T \exp(\Theta^T x)} \equiv \normalize(\exp(\Theta^T x))
\end{equation}$$

(i.e., $z$ is just the normalized softmax probabilities), and where $e_y$ denotes the $y$th unit basis, i.e., a vector of all zeros with a one in the $y$-th position.

We can also write this in the more compact notation we discussed in class.  Namely, if we let $X \in \mathbb{R}^{m \times n}$ denote a design matrix of some $m$ inputs (either the entire dataset or a minibatch), $y \in \{1,\ldots,k\}^m$ a corresponding vector of labels, and overloading $\ell_{\mathrm{softmax}}$ to refer to the average softmax loss, then

$$\begin{equation}
\nabla_\Theta \ell_{\mathrm{softmax}}(X \Theta, y) = \frac{1}{m} X^T (Z - I_y)
\end{equation}$$

where

$$\begin{equation}
Z = \normalize(\exp(X \Theta)) \quad \mbox{(normalization applied row-wise)}
\end{equation}$$

denotes the matrix of logits, and $I_y \in \mathbb{R}^{m \times k}$ represents a concatenation of one-hot bases for the labels in $y$.

Using these gradients, implement the `softmax_regression_epoch()` function, which runs a single epoch of SGD (one pass over a data set) using the specified learning rate / step size `lr` and minibatch size `batch`.  As described in the docstring, your function should modify the `Theta` array in-place.  After implementation, run the tests.

**Code:**
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

### Math Prove CMU 10714 For The gradient of the softmax objective

So, how do we compute the gradient for the softmax objective?
$$\nabla_{\theta} \ell_{ce}(\theta^T x, y) = ? $$

Let's start by deriving the gradient of the softmax loss itself: for vector $h \in \mathbb{R}^k$

$$\frac{\partial \ell_{ce}(h, y)}{\partial h_i} = \frac{\partial}{\partial h_i} \left( -h_y + \log \sum_{j=1}^k \exp h_j \right) = -1 (i = y) + \frac{\exp h_i}{\sum_{j=1}^k \exp h_j}$$

So, in vector form: 
$$\nabla_h \ell_{ce}(h, y) = z - e_y, \text{ where } z = \text{softmax}(h)$$


So how do we compute the gradient $\nabla_{\theta} \ell_{ce} (\theta^T x, y)$?

- The chain rule of multivariate calculus ... but the dimensions of all the matrices and vectors get pretty cumbersome

**Approach #1 (a.k.a. the right way)**: Use matrix differential calculus, Jacobians, Kronecker products, and vectorization

**Approach #2 (a.k.a. the hacky quick way that everyone actually does)**: Pretend everything is a scalar, use the typical chain rule, and then rearrange / transpose matrices/vectors to make the sizes work 😱 (and check your answer numerically)

**The slide I'm embarrassed to include...**

Let's compute the "derivative" of the loss:

$$\frac{\partial}{\partial \theta} \ell_{ce} (\theta^T x, y) = \frac{\partial \ell_{ce} (\theta^T x, y)}{\partial \theta^T x} \frac{\partial \theta^T x}{\partial \theta} = (z - e_y)(x), \text{ where } z = \text{softmax}(\theta^T x)$$

where $(z - e_y) (k\text{-dimensional}), (x) (n\text{-dimensional})$

So to make the dimensions work...

$$\nabla_{\theta} \ell_{ce} (\theta^T x, y) \in \mathbb{R}^{n \times k} = x (z - e_y)^T$$

Same process works if we use "matrix batch" form of the loss

$$\nabla_{\theta} \ell_{ce} (X \theta, y) \in \mathbb{R}^{n \times k} = X^T (Z - I_y), \quad Z = \text{softmax}(X \theta)$$


### Math Prove By Myself(Gradient of the Softmax Loss with Respect to Parameters)

#### Softmax Function and Loss
Given logits $z$ and the true label $y$, the softmax function and the corresponding loss function are defined as follows:

##### Softmax Function:
$$\sigma(z_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

Let's denote the softmax output (probability) for class $i$ as $p_i = \sigma(z_i)$.

##### Softmax Loss:
The softmax loss for the true class $y$ is:
$$\ell_{\mathrm{softmax}}(z, y) = \log \left( \sum_{i=1}^k \exp(z_i) \right) - z_y$$

#### Gradient Derivation
To derive the gradient of the softmax loss with respect to the parameters \( \Theta \), we need to follow these steps:

##### Gradient of the Loss with Respect to Logits:
We need to compute $\frac{\partial \ell_{\mathrm{softmax}}}{\partial z_i}$ for each $i$.

For $i = y$ (the true class):

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial z_y} = \frac{\partial}{\partial z_y} \left( \log \left( \sum_{i=1}^k \exp(z_i) \right) - z_y \right) = \frac{\exp(z_y)}{\sum_{i=1}^k \exp(z_i)} - 1 = p_y - 1$$

For $i \neq y$:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial z_i} = \frac{\partial}{\partial z_i} \log \left( \sum_{i=1}^k \exp(z_i) \right) = \frac{\exp(z_i)}{\sum_{i=1}^k \exp(z_i)} = p_i$$

Combining these, we get:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial z_i} = p_i - \delta_{iy}$$

where $\delta_{iy}$ is the Kronecker delta, which is 1 if $i = y$ and 0 otherwise.

##### Gradient with Respect to Parameters $\Theta$:

Using the chain rule, the gradient of the loss with respect to $\Theta$ is:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta} = \frac{\partial \ell_{\mathrm{softmax}}}{\partial z} \cdot \frac{\partial z}{\partial \Theta}$$

We know that $z = \Theta^T x$, so:

$$\frac{\partial z_i}{\partial \Theta_{jk}} = \frac{\partial (\Theta_{ki} x_k)}{\partial \Theta_{jk}} = x_j \delta_{ik}$$

Thus, for a single input $x$, the gradient is:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{jk}} = \sum_i \frac{\partial \ell_{\mathrm{softmax}}}{\partial z_i} \frac{\partial z_i}{\partial \Theta_{jk}} = \sum_i (p_i - \delta_{iy}) x_j \delta_{ik}$$

This simplifies to:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{jk}} = x_j (p_k - \delta_{ky})$$

##### Matrix Form
In matrix form, this becomes:
$$\nabla_{\Theta} \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T$$

where $z = \sigma(\Theta^T x)$ is the vector of softmax probabilities and $e_y$ is the one-hot encoded vector for the true class $y$.


------------------

The Kronecker delta, denoted as $\delta_{ij}$, is a function of two variables (usually integers) that is 1 if the variables are equal and 0 otherwise. It is named after the German mathematician Leopold Kronecker. Mathematically, it is defined as:

$$\delta_{ij} = 
\begin{cases} 
1 & \text{if } i = j \\
0 & \text{if } i \neq j 
\end{cases}$$

### Example of $\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta}$

Consider a simple case with:

- $n = 2$ features,
- $k = 3$ classes.

The input vector $x$ and parameter matrix $\Theta$ are given by:

$$x = \begin{pmatrix}
x_1 \\
x_2
\end{pmatrix}$$

$$\Theta = \begin{pmatrix}
\Theta_{11} & \Theta_{12} & \Theta_{13} \\
\Theta_{21} & \Theta_{22} & \Theta_{23}
\end{pmatrix}$$

Let's assume the true class $y$ is 2 (i.e., the second class).

#### Calculate Logits $z$

First, compute the logits $z$:

$$z = \Theta^T x = \begin{pmatrix}
\Theta_{11} & \Theta_{21} \\
\Theta_{12} & \Theta_{22} \\
\Theta_{13} & \Theta_{23}
\end{pmatrix} \begin{pmatrix}
x_1 \\
x_2
\end{pmatrix} = \begin{pmatrix}
\Theta_{11} x_1 + \Theta_{21} x_2 \\
\Theta_{12} x_1 + \Theta_{22} x_2 \\
\Theta_{13} x_1 + \Theta_{23} x_2
\end{pmatrix}$$

#### Calculate Softmax Probabilities $\sigma(z)$

Next, compute the softmax probabilities:

$$\sigma(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)}$$

Let:

$$z_1 = \Theta_{11} x_1 + \Theta_{21} x_2, \quad z_2 = \Theta_{12} x_1 + \Theta_{22} x_2, \quad z_3 = \Theta_{13} x_1 + \Theta_{23} x_2$$

Then, the softmax probabilities are:

$$\sigma(z_1) = \frac{\exp(z_1)}{\exp(z_1) + \exp(z_2) + \exp(z_3)}$$

$$\sigma(z_2) = \frac{\exp(z_2)}{\exp(z_1) + \exp(z_2) + \exp(z_3)}$$

$$\sigma(z_3) = \frac{\exp(z_3)}{\exp(z_1) + \exp(z_2) + \exp(z_3)}$$

#### Calculate $\delta_{ky}$

Given that the true class $y = 1$ (second class), the one-hot encoded vector $e_y$ is:

$$e_y = \begin{pmatrix}
0 \\
1 \\
0
\end{pmatrix}$$

#### Partial Derivative $\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{jk}}$

We want to compute the partial derivative for each element $\Theta_{jk}$:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{jk}} = x_j (\sigma(z_k) - \delta_{ky})$$

Let's compute a few of these explicitly using the symbolic expressions:

##### For $j = 1$, $k = 1$:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{11}} = x_1 (\sigma(z_1) - \delta_{1y}) = x_1 \left(\frac{\exp(z_1)}{\exp(z_1) + \exp(z_2) + \exp(z_3)} - 0\right) = x_1 \left(\frac{\exp(z_1)}{\sum_{j=1}^3 \exp(z_j)}\right)$$

##### For $j = 1$, $k = 2$:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{12}} = x_1 (\sigma(z_2) - \delta_{2y}) = x_1 \left(\frac{\exp(z_2)}{\exp(z_1) + \exp(z_2) + \exp(z_3)} - 1\right) = x_1 \left(\frac{\exp(z_2)}{\sum_{j=1}^3 \exp(z_j)} - 1\right)$$

##### For $j = 1$, $k = 3$:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{13}} = x_1 (\sigma(z_3) - \delta_{3y}) = x_1 \left(\frac{\exp(z_3)}{\exp(z_1) + \exp(z_2) + \exp(z_3)} - 0\right) = x_1 \left(\frac{\exp(z_3)}{\sum_{j=1}^3 \exp(z_j)}\right)$$

##### For $j = 2$, $k = 1$:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{21}} = x_2 (\sigma(z_1) - \delta_{1y}) = x_2 \left(\frac{\exp(z_1)}{\exp(z_1) + \exp(z_2) + \exp(z_3)} - 0\right) = x_2 \left(\frac{\exp(z_1)}{\sum_{j=1}^3 \exp(z_j)}\right)$$

##### For $j = 2$, $k = 2$:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{22}} = x_2 (\sigma(z_2) - \delta_{2y}) = x_2 \left(\frac{\exp(z_2)}{\exp(z_1) + \exp(z_2) + \exp(z_3)} - 1\right) = x_2 \left(\frac{\exp(z_2)}{\sum_{j=1}^3 \exp(z_j)} - 1\right)$$

##### For $j = 2$, $k = 3$:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{23}} = x_2 (\sigma(z_3) - \delta_{3y}) = x_2 \left(\frac{\exp(z_3)}{\exp(z_1) + \exp(z_2) + \exp(z_3)} - 0\right) = x_2 \left(\frac{\exp(z_3)}{\sum_{j=1}^3 \exp(z_j)}\right)$$

#### Summary of Gradients

Summarizing these, the gradients for the elements of $\Theta$ are:

$$\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta} = \begin{pmatrix}
\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{11}} & \frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{12}} & \frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{13}} \\
\frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{21}} & \frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{22}} & \frac{\partial \ell_{\mathrm{softmax}}}{\partial \Theta_{23}}
\end{pmatrix} = \begin{pmatrix}
x_1 \left(\frac{\exp(z_1)}{\sum_{j=1}^3 \exp(z_j)}\right) & x_1 \left(\frac{\exp(z_2)}{\sum_{j=1}^3 \exp(z_j)} - 1\right) & x_1 \left(\frac{\exp(z_3)}{\sum_{j=1}^3 \exp(z_j)}\right) \\
x_2 \left(\frac{\exp(z_1)}{\sum_{j=1}^3 \exp(z_j)}\right) & x_2 \left(\frac{\exp(z_2)}{\sum_{j=1}^3 \exp(z_j)} - 1\right) & x_2 \left(\frac{\exp(z_3)}{\sum_{j=1}^3 \exp(z_j)}\right)
\end{pmatrix}$$


### Example with Matrix Dimensions

Let's use an example to illustrate this.

#### Weight Matrix $\Theta$:

$$\Theta = \begin{pmatrix}
\Theta_{11} & \Theta_{12} & \Theta_{13} \\
\Theta_{21} & \Theta_{22} & \Theta_{23}
\end{pmatrix}$$

Here, $\Theta$ is a $2 \times 3$ matrix for a model with 2 features and 3 classes.

#### Input Vector $x$:

$$x = \begin{pmatrix}
x_1 \\
x_2
\end{pmatrix}$$

#### Softmax Probabilities $\sigma(z)$:

$$\sigma(z) = \begin{pmatrix}
\sigma(z_1) \\
\sigma(z_2) \\
\sigma(z_3)
\end{pmatrix}$$

This is a $3 \times 1$ vector.

#### One-Hot Encoded True Class $e_y$:

If the true class \(y\) is 2 (i.e., the second class), then:

$$e_y = \begin{pmatrix}
0 \\
1 \\
0
\end{pmatrix}$$

#### Gradient Matrix $\nabla_\Theta \ell_{\mathrm{softmax}}$:

The gradient matrix is given by:

$$\nabla_\Theta \ell_{\mathrm{softmax}} = x (\sigma(z) - e_y)^T$$

#### Calculating the Outer Product $x (\sigma(z) - e_y)^T$:

$$\sigma(z) - e_y = \begin{pmatrix}
\sigma(z_1) \\
\sigma(z_2) \\
\sigma(z_3)
\end{pmatrix} - \begin{pmatrix}
0 \\
1 \\
0
\end{pmatrix} = \begin{pmatrix}
\sigma(z_1) \\
\sigma(z_2) - 1 \\
\sigma(z_3)
\end{pmatrix}$$

The outer product $x (\sigma(z) - e_y)^T$ is:

$$x (\sigma(z) - e_y)^T = \begin{pmatrix}
x_1 \\
x_2
\end{pmatrix} \begin{pmatrix}
\sigma(z_1) & \sigma(z_2) - 1 & \sigma(z_3)
\end{pmatrix} = \begin{pmatrix}
x_1 \sigma(z_1) & x_1 (\sigma(z_2) - 1) & x_1 \sigma(z_3) \\
x_2 \sigma(z_1) & x_2 (\sigma(z_2) - 1) & x_2 \sigma(z_3)
\end{pmatrix}$$

#### Updating the Weight Matrix $\Theta$

Using the gradient matrix, we update the weight matrix $Theta$ as follows:

$$\Theta := \Theta - \eta \nabla_\Theta \ell_{\mathrm{softmax}}$$

For our example, if the learning rate $\eta$ is 0.01, the updated weights would be:

$$\Theta = \Theta - 0.01 \begin{pmatrix}
x_1 \sigma(z_1) & x_1 (\sigma(z_2) - 1) & x_1 \sigma(z_3) \\
x_2 \sigma(z_1) & x_2 (\sigma(z_2) - 1) & x_2 \sigma(z_3)
\end{pmatrix}$$

#### Conclusion

The gradient matrix $\nabla_\Theta \ell_{\mathrm{softmax}}$ is separate from the weight matrix $\Theta$. It is used to compute the direction and magnitude of updates needed to minimize the loss function. The weight matrix $\Theta$ is updated iteratively using the gradient matrix during the training process through gradient descent or other optimization algorithms.

------------------------------
🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟

### Gradient for a Single Example

Recall from the gradient for a single example:

$$\nabla_\Theta \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T$$

where:

- $x$ is the input vector for a single example.
- $z$ is the vector of softmax probabilities for the input example.
- $e_y$ is the one-hot encoded vector for the true class $y$.

### Extending to Multiple Examples (Batch)

$$\nabla_\Theta \ell_{\mathrm{softmax}}(X \Theta, y) = \frac{1}{m} X^T (Z - I_y)$$

When we have multiple examples, we can represent the inputs as a matrix $X$ where each row is an input vector $x_i$ for the $i$-th example. Similarly, we can represent the true labels as a matrix $I_y$ where each row is the one-hot encoded vector $e_{y_i}$ for the $i$-th label.

The matrix $Z$ is the matrix of softmax probabilities for all examples, where each row contains the softmax probabilities for the corresponding example.

### Notation and Dimensions

Let's define the matrices and their dimensions:

- $X \in \mathbb{R}^{m \times n}$: Design matrix of $m$ input vectors, each with $n$ features.
- $\Theta \in \mathbb{R}^{n \times k}$: Weight matrix, mapping $n$ features to $k$ classes.
- $Z \in \mathbb{R}^{m \times k}$: Matrix of softmax probabilities for $m$ examples and $k$ classes.
- $I_y \in \mathbb{R}^{m \times k}$: One-hot encoded matrix for the true labels of $m$ examples and $k$ classes.


### Computing the Gradient

The average softmax loss gradient over $m$ examples can be computed by summing the gradients for each example and then dividing by $m$:

$$\nabla_\Theta \ell_{\mathrm{softmax}} = \frac{1}{m} \sum_{i=1}^m x_i (z_i - e_{y_i})^T$$

In matrix form, this can be represented as:

$$\nabla_\Theta \ell_{\mathrm{softmax}}(X \Theta, y) = \frac{1}{m} X^T (Z - I_y)$$

### Why This Works

- **Matrix $X$**:
  Each row of $X$ is an input vector $x_i$.

- **Matrix $Z$**:
  Each row of $Z$ is the softmax probabilities $z_i$ for the corresponding input $x_i$.

- **Matrix $I_y$**:
  Each row of $I_y$ is the one-hot encoded vector $e_{y_i}$ for the true class $y_i$.

- **Matrix Multiplication**:
  $X^T (Z - I_y)$ computes the sum of outer products $x_i (z_i - e_{y_i})^T$ for all examples in the batch.
  Dividing by $m$ gives the average gradient.

## Training MNIST with softmax regression

Although it's not a part of the tests, now that you have written this code, you can also try training a full MNIST linear classifier using SGD.  For this you can use the `train_softmax()` function in the `src/simple_ml.py` file (we have already written this function for you, so you don't need to write it yourself, though you can take a look to see what it's doing).  

You can see how this works using the following code.  For reference, as seen below, our implementation runs in ~3 seconds on Colab, and achieves 7.97% error.

```python
def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    # h: (np.ndarray[np.float32]): 2D numpy array of shape (batch_size x num_classes), containing the logit predictions for each class.
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, slr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    # X_tr.shape[1]: the number of features in the training data
    # y_tr.max()+1 : the number of classes
    # weight matrix theta's shape (number of features x number of classes)
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        # Computes the loss and error for the entire training dataset
        # X_tr @ theta ((num_examples x number of features)@(number of features x number of classes)) = (num_examples x num_classes)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))
```


### Explain np.mean(h.argmax(axis=1) != y)
```python
import numpy as np

# Predicted logits for 5 samples and 3 classes
h = np.array([[0.2, 0.5, 0.3],
              [0.1, 0.3, 0.6],
              [0.7, 0.2, 0.1],
              [0.4, 0.4, 0.2],
              [0.1, 0.6, 0.3]])

# True class labels
y = np.array([1, 2, 0, 1, 2])

# Predicted classes
predicted_classes = h.argmax(axis=1)
print("Predicted classes:", predicted_classes)
# Output: [1 2 0 0 1]

# Comparison of predicted classes with true classes
comparison = predicted_classes != y
print("Comparison (predicted != true):", comparison)
# Output: [False False False  True  True]

# Calculate the mean (classification error rate)
error_rate = np.mean(comparison)
print("Error rate:", error_rate)
# Output: 0.4 since (0 + 0 + 0 + 1 + 1) / 5 = 2 / 5 = 0.4
```

## Softmax Regression as a Linear Classifier

Let's work through a specific example to illustrate why softmax regression (multinomial logistic regression) is considered a linear classifier. We'll use a simple 2D dataset for easy visualization.

### Example Setup
Suppose we have a dataset with two features $X_1$ and $X_2$ and three classes (A, B, and C). Let's denote the input feature vector as $\mathbf{x} = [X_1, X_2]$.

### Softmax Regression Model

#### Linear Transformation
We have a weight matrix $\Theta$ of shape $(2, 3)$, where each column corresponds to the weights for one of the three classes.

Let's assume:

$$\Theta = \begin{bmatrix}
\theta_{11} & \theta_{12} & \theta_{13} \\
\theta_{21} & \theta_{22} & \theta_{23}
\end{bmatrix}$$

#### Logit Calculation
The logits for each class are computed as:

$$Z = \mathbf{x} \Theta = [X_1, X_2] \begin{bmatrix}
\theta_{11} & \theta_{12} & \theta_{13} \\
\theta_{21} & \theta_{22} & \theta_{23}
\end{bmatrix} = [Z_1, Z_2, Z_3]$$

where:

$$Z_1 = \theta_{11} X_1 + \theta_{21} X_2$$

$$Z_2 = \theta_{12} X_1 + \theta_{22} X_2$$

$$Z_3 = \theta_{13} X_1 + \theta_{23} X_2$$

#### Softmax Function
The softmax function converts these logits into probabilities:

$$P(y=i \mid \mathbf{x}) = \frac{e^{Z_i}}{e^{Z_1} + e^{Z_2} + e^{Z_3}}$$

### Decision Boundaries
The decision boundaries are where the classifier is indifferent between two classes. For example:

- The boundary between class A and class B is where $Z_1 = Z_2$:
  
  $$\theta_{11} X_1 + \theta_{21} X_2 = \theta_{12} X_1 + \theta_{22} X_2$$
  
  Rearranging, we get:

  $$(\theta_{11} - \theta_{12}) X_1 + (\theta_{21} - \theta_{22}) X_2 = 0$$
  
  This is a linear equation representing a line in the $X_1$-$X_2$ plane.

- Similarly, the boundary between class B and class C is:

  $$(\theta_{12} - \theta_{13}) X_1 + (\theta_{22} - \theta_{23}) X_2 = 0$$

- The boundary between class A and class C is:
  
  $$(\theta_{11} - \theta_{13}) X_1 + (\theta_{21} - \theta_{23}) X_2 = 0$$

## Code Implementation:
- [Softmax Regression](../../Code-Implementation/Softmax-Regression)
  
## Reference:
- CMU 10714 Deep Learning Systems


