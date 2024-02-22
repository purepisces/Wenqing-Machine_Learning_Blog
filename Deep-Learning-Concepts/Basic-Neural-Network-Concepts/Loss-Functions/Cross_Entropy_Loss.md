## Derivation of the Gradient $\frac{\partial \text{Loss}}{\partial A}$

The gradient $\frac{\partial \text{Loss}}{\partial A}$ signifies how the loss changes with slight variations in the raw output scores $A$. This is crucial for backpropagation, as it guides how the model's weights should be adjusted to minimize the loss.

###Step 1: Differentiate Cross-Entropy Loss with Respect to $\sigma(A)$

The first step involves differentiating the cross-entropy loss with respect to the softmax probabilities, $\sigma(A)$. For a specific class $c$, this differentiation yields:

$\frac{\partial H(A, Y)}{\partial \sigma(A)_c} = -\frac{Y_c}{\sigma(A)_c}$

This equation indicates that the rate of change of the loss with respect to the predicted probability of class $c$ is inversely proportional to $\sigma(A)_c$, scaled by the true label $Y_c$.

### Step 2: Chain Rule to Relate $\frac{\partial \text{Loss}}{\partial A}$

To connect the change in loss with the raw output scores $A$, we apply the chain rule of calculus, which in the context of functions of functions, states that:

$\frac{\partial \text{Loss}}{\partial A} = \frac{\partial \text{Loss}}{\partial \sigma(A)} \cdot \frac{\partial \sigma(A)}{\partial A}$

The term $\frac{\partial \sigma(A)}{\partial A}$ is derived from the softmax function and represents how changes in the raw scores $A$ affect the softmax probabilities $\sigma(A)$. For the softmax function, this derivative is particularly interesting because it takes a different form when differentiating with respect to the score of the correct class versus the scores of other classes.

### Step 3: Combine Terms and Simplify

By substituting the derivatives from Steps 1 and 2 into the chain rule expression, and after some algebraic simplification, we arrive at the gradient of the cross-entropy loss with respect to the raw output scores $A$:

$\frac{\partial \text{Loss}}{\partial A} = \sigma(A) - Y$ 

This result shows that the gradient is simply the difference between the predicted probabilities and the true labels. This makes intuitive sense: if the predicted probability for the correct class is too low (or high for incorrect classes), the raw scores need to be adjusted in a way that increases (or decreases) these probabilities.

### Step 4: Scaling by Batch Size

In practice, when working with batches of data, we often compute the mean loss over all samples in the batch to stabilize training and make the loss less dependent on the batch size. Consequently, the gradient of the mean loss with respect to $A$ is scaled by the reciprocal of the batch size $N$:

$\frac{\partial \text{Mean Loss}}{\partial A} = \frac{\sigma(A) - Y}{N}$

This scaling ensures that the magnitude of the updates does not directly scale with the number of samples, providing a more consistent update step across different batch sizes.

### Conclusion

The derivation of $\frac{\sigma(A) - Y}{N}$ for the gradient of the cross-entropy loss with respect to the raw model outputs $A$ highlights the direct relationship between the difference in predicted probabilities and true labels, and how this difference informs the adjustments needed in the model's parameters. This gradient is a cornerstone of learning in neural networks, guiding the optimization process towards models that better approximate the underlying data distribution.

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

