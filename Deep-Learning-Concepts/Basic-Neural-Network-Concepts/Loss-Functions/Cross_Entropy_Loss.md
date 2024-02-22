To understand why the derivative of the cross-entropy loss 
$- \sum\limits_{i=1}^{C} Y_i \log(\sigma(A_i))$ with respect to the logits $A_i$ is $\sigma(A_i) - Y_i$, let's delve into the mathematical details. This derivation involves applying the chain rule for derivatives and leveraging the specific mathematical properties of the softmax and logarithm functions.

### Cross-Entropy Loss and Softmax
The cross-entropy loss for a single sample when using the softmax function for a multi-class classification problem is given by:

$$H(Y, \sigma(A)) = - \sum_{i=1}^{C} Y_i \log(\sigma(A_i))$$
where:
- $Y_i$ is the true label for class $i$, which is 1 for the correct class and 0 for all other classes in one-hot encoding.
- $\sigma(A_i)$ is the softmax function applied to the logit $A_i$, representing the predicted probability that the sample belongs to class $i$.

### Softmax Function
The softmax function for a logit $A_i$ is defined as:

$$\sigma(A_i) = \frac{e^{A_i}}{\sum\limits_{j=1}^{C} e^{A_j}}$$

### Derivation of the Gradient
To find the gradient of the cross-entropy loss with respect to the logits $A_i$, we need to compute the derivative $\frac{\partial H}{\partial A_i}$. This involves applying the chain rule to the composition of the logarithm and the softmax function.

#### Step 1: Apply the Chain Rule
First, note that we need to apply the chain rule for the derivative of a composite function, which in this case is the logarithm of the softmax output:

$$\frac{\partial H}{\partial A_i} = - \sum\limits_{k=1}^{C} Y_k \frac{\partial \log(\sigma(A_k))}{\partial A_i}$$

#### Step 2: Derivative of the Logarithm of Softmax
The derivative of $\log(\sigma(A_k))$ with respect to $A_i$ involves two cases: when $i=k$ and when $i \neq k$.

When $i=k$, using the derivative of the logarithm $\frac{\partial \log(x)}{\partial x} = \frac{1}{x}$ and the definition of softmax, we get:

$$\frac{\partial \log(\sigma(A_i))}{\partial A_i} = \frac{1}{\sigma(A_i)} \cdot \sigma(A_i) \cdot (1 - \sigma(A_i)) = 1 - \sigma(A_i)$$

When $i \neq k$, the derivative involves the softmax function for a different class $k$, and the result is:

$$\frac{\partial \log(\sigma(A_k))}{\partial A_i} = - \sigma(A_i)$$

#### Step 3: Combine the Cases
Combining these two cases and considering the effect of the one-hot encoding of $Y_k$ (which is 0 for all $k \neq i$ and 1 for $k=i$), the summation simplifies to:

$$\frac{\partial H}{\partial A_i} = -Y_i (1 - \sigma(A_i)) + \sum_{k \neq i} Y_k \sigma(A_i)$$

Given that $Y_i$ can only be 1 for the true class and 0 otherwise, and $\sum_{k \neq i} Y_k = 0$ (since for the true class $i$, all other $Y_k$ are 0), this further simplifies to:

$$\frac{\partial H}{\partial A_i} = -Y_i + Y_i \sigma(A_i) + 0 = \sigma(A_i) - Y_i$$

### Conclusion
The derivative of the cross-entropy loss with respect to the logits $A_i$ thus simplifies to $\sigma(A_i) - Y_i$, indicating the difference between the predicted probability for class $i$ and the actual class label. This elegant result is fundamental in training neural networks for classification tasks, as it directly relates the gradient to the discrepancy between the model's predictions and the true labels.

# TEMP
The equation $\frac{\partial \log(\sigma(A_k))}{\partial A_i} = -\sigma(A_i)$ when $i \neq k$ comes from understanding the behavior of the softmax function and the logarithmic function in the context of taking derivatives.

Let's break it down:

**Softmax Function:** The softmax function for a class $k$ is defined as $\sigma(A_k) = \frac{e^{A_k}}{\sum_{j=1}^{C} e^{A_j}}$ where $A_k$ is the logit for class $k$ and $C$ is the total number of classes.

**Derivative of Softmax:** When taking the derivative of $\sigma(A_k)$ with respect to $A_i$ where $i \neq k$, we need to apply the quotient rule because $\sigma(A_k)$ is a ratio. The derivative will have two parts:
- The numerator $e^{A_k}$ will be treated as a constant (because we are differentiating with respect to $A_i$, and $i \neq k$), so its derivative is $0$.
- The denominator $\sum_{j=1}^{C} e^{A_j}$ includes $e^{A_i}$, and its derivative with respect to $A_i$ is $e^{A_i}$.

This results in the derivative of the softmax for $i \neq k$ being negative because the derivative of the denominator will be subtracted (due to the quotient rule), and since the numerator's derivative is $0$, we are left with $\frac{\partial \sigma(A_k)}{\partial A_i} = -\sigma(A_k) \cdot \sigma(A_i)$ for $i \neq k$.

**Logarithm of Softmax:** When you take the derivative of $\log(\sigma(A_k))$ with respect to $A_i$, by the chain rule, this derivative is the derivative of the log function ($\frac{1}{\sigma(A_k)}$) times the derivative of its argument ($\sigma(A_k)$ with respect to $A_i$).

**Combining the Two:** For $i \neq k$, the derivative of $\log(\sigma(A_k))$ with respect to $A_i$ is the product of $\frac{1}{\sigma(A_k)}$ (from the derivative of the log function) and $-\sigma(A_k) \cdot \sigma(A_i)$ (from the derivative of the softmax function), which simplifies to $\frac{\partial \log(\sigma(A_k))}{\partial A_i} = -\sigma(A_i)$ for $i \neq k$.

This result essentially indicates that the rate of change of the log likelihood for class $k$ with respect to the logit of a different class $i$ is influenced negatively by the predicted probability of class $i$, $\sigma(A_i)$, reflecting how increases in $A_i$ decrease the log likelihood of the true class $k$ when $k \neq i$.


# Cross-Entropy Loss

Cross-entropy loss is one of the most commonly used loss function for probability-based classification problems. 

## Cross-Entropy Loss Forward Equation

Firstly, we use softmax function to transform the raw model outputs $A$ into a probability distribution consisting of $C$ classes proportional to the exponentials of the input numbers.

$\iota_N$, $\iota_C$ are column vectors of size $N$ and $C$ which contain all 1s. 

$$\text{softmax}(A) = \sigma(A) = \frac{\exp(A)}{\sum\limits_{j=1}^{C} \exp(A_{ij})}$$



Now, each row of A represents the model’s prediction of the probability distribution while each row of Y represents target distribution of an input in the batch.
Then, we calculate the cross-entropy H(A,Y) of the distribution Ai relative to the target distribution Yi for i = 1,...,N:

$$\text{crossentropy} = H(A, Y) = (-Y \circ \log(\sigma(A))) \cdot \mathbf{\iota}_C$$

Remember that the output of a loss function is a scalar, but now we have a column matrix of size N. To transform it into a scalar, we can either use the sum or mean of all cross-entropy.

Here, we choose to use the mean cross-entropy as the cross-entropy loss as that is the default for PyTorch as well:

$$\text{sumcrossentropyloss} := \mathbf{\iota}_N^T \cdot H(A, Y) = SCE(A, Y)$$

$$\text{meancrossentropyloss} := \frac{SCE(A, Y)}{N}$$

insert img

Cross-Entropy Loss Backward Equation

$$\text{xent.backward}() = \frac{\sigma(A) - Y}{N}$$


## Example

To illustrate the cross-entropy loss, let's consider a specific example with a small dataset. Imagine we have a simple classification problem with three classes (C=3) and we are working with a batch of two samples ($N=2$). The raw output scores ($A$) from the model for these two samples and the corresponding true labels ($Y$) could be as follows:

- Raw model outputs for the two samples ($A$):
  - Sample 1: $[2.0, 1.0, 0.1]$
  - Sample 2: $[0.1, 2.0, 1.9]$

- True class distributions ($Y$, one-hot encoded):
  - Sample 1: $[0, 1, 0]$  (class 2 is the true class)
  - Sample 2: $[1, 0, 0]$  (class 1 is the true class)

Let's go through the cross-entropy loss computation step by step for this example:

### 1. Apply Softmax

First, we apply the softmax function to the raw outputs to get the predicted probabilities for each class.

For Sample 1, the softmax calculation would be:
$$\sigma(A_1) = \left[\frac{e^{2.0}}{e^{2.0} + e^{1.0} + e^{0.1}}, \frac{e^{1.0}}{e^{2.0} + e^{1.0} + e^{0.1}}, \frac{e^{0.1}}{e^{2.0} + e^{1.0} + e^{0.1}}\right]$$

For Sample 2, similarly:
$$\sigma(A_2) = \left[\frac{e^{0.1}}{e^{0.1} + e^{2.0} + e^{1.9}}, \frac{e^{2.0}}{e^{0.1} + e^{2.0} + e^{1.9}}, \frac{e^{1.9}}{e^{0.1} + e^{2.0} + e^{1.9}}\right]$$

### 2. Compute Cross-Entropy Loss

Next, we compute the cross-entropy loss for each sample. The loss for a single sample is given by:
$$H(A_i, Y_i) = -\sum_{c=1}^{C} Y_{ic} \log(\sigma(A_{ic}))$$

For Sample 1:
$$H(A_1, Y_1) = -[0 \times \log(\sigma(A_{11})) + 1 \times \log(\sigma(A_{12})) + 0 \times \log(\sigma(A_{13}))]$$

For Sample 2:
$$H(A_2, Y_2) = -[1 \times \log(\sigma(A_{21})) + 0 \times \log(\sigma(A_{22})) + 0 \times \log(\sigma(A_{23}))]$$

### 3. Calculate Mean Cross-Entropy Loss

Finally, we calculate the mean of these losses to get the mean cross-entropy loss for the batch:
$$\text{meancrossentropyloss} = \frac{H(A_1, Y_1) + H(A_2, Y_2)}{2}$$

### 4. Cross-Entropy Loss Backward

For backpropagation, the gradient of the cross-entropy loss with respect to the raw model outputs before applying softmax is given by:
$$\frac{\partial \text{Loss}}{\partial A} = \frac{\sigma(A) - Y}{N}$$

For each sample in the batch, we compute:
- For Sample 1: $\frac{\sigma(A_1) - Y_1}{2}$
- For Sample 2: $\frac{\sigma(A_2) - Y_2}{2}$

This gives us the gradients that we need to backpropagate through the network.

Based on the calculations:

- The softmax probabilities for the two samples are approximately:
  - Sample 1: [0.659, 0.242, 0.099]
  - Sample 2: [0.073, 0.487, 0.440]

- The cross-entropy losses for the two samples are:
  - Sample 1: 1.417
  - Sample 2: 2.620

- The mean cross-entropy loss for this batch is approximately 2.019.

- The gradients of the loss with respect to the raw model outputs ($A$) are:
  - For Sample 1: [0.330, -0.379, 0.049]
  - For Sample 2: [-0.464, 0.243, 0.220]

These results give us the predicted probabilities for each class using the softmax function, the individual cross-entropy losses for each sample, the overall mean cross-entropy loss for the batch, and the gradients required for the backward pass. The negative values in the gradients indicate the direction in which we should adjust our model's parameters to reduce the loss, while the positive values suggest the opposite.

```python
class CrossEntropyLoss:
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        self.softmax = self.softmax(A)
        crossentropy = -Y * np.log(self.softmax)
        L = np.sum(crossentropy) / A.shape[0]
        return L

    def backward(self):
        dLdA = (self.softmax - self.Y) / self.A.shape[0]
        return dLdA
```

