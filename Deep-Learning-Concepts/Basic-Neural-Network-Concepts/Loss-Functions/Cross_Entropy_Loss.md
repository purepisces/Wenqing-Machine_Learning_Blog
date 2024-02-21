
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

To provide a specific example of how cross-entropy loss works, let's consider a classification problem where we have a dataset consisting of images, and our task is to classify these images into three categories (C = 3): Cat, Dog, and Bird.

### Forward Pass

Suppose our model has made predictions for a single image in the batch (N = 1), and the raw output scores from the model (logits) for this image are $A = [2.0, 1.0, 0.1]$. These scores correspond to the model's confidence in each class (Cat, Dog, Bird, respectively).

**Softmax Activation:** We first apply the softmax function to convert these logits into probabilities.

$$\sigma(A) = \frac{\exp(A)}{\sum\limits_{j=1}^{3} \exp(A_{j})}$$

Substituting the values, we get:

$$\sigma(A) = \frac{[\exp(2.0), \exp(1.0), \exp(0.1)]}{\exp(2.0) + \exp(1.0) + \exp(0.1)}$$

Let's compute this:

$$\sigma(A) = [\sigma(A){\text{Cat}}, \sigma(A){\text{Dog}}, \sigma(A){\text{Bird}}]$$

**Target Distribution:** Assume the true label for this image is "Cat", so the target distribution $Y$ is a one-hot encoded vector: 

$$Y = [1, 0, 0]$$

**Cross-Entropy Loss:** Now, we calculate the cross-entropy loss:

$$H(A, Y) = -[1, 0, 0] \circ \log([\sigma(A){\text{Cat}}, \sigma(A){\text{Dog}}, \sigma(A){\text{Bird}}])$$

Only the first component matters (because the other components of $Y$ are zero):

$$H(A, Y) = -\log(\sigma(A)_{\text{Cat}})$$

### Backward Pass

For the backward pass, we compute the gradient of the loss with respect to the softmax probabilities, which will be used to update the model parameters.

$$\frac{\partial H(A, Y)}{\partial A} = \frac{\sigma(A) - Y}{N}$$

In this case, since we have only one example (N = 1), the gradient simplifies to:

$$\frac{\partial H(A, Y)}{\partial A} = \sigma(A) - Y$$

This gradient tells us how we should adjust our logits $A$ to decrease the loss, thereby improving our model's predictions.

Let's calculate the softmax probabilities, the cross-entropy loss, and the gradients for this example using Python:

For this example, the softmax probabilities for the classes (Cat, Dog, Bird) are approximately $[0.659, 0.242, 0.099]$ respectively. This means the model assigns a 65.9% probability to the "Cat" class, which is the correct class in this case.

The cross-entropy loss for this prediction is approximately $0.417$. This value quantifies how far off the prediction is from the actual label. A lower loss value indicates a better prediction.

The gradient of the loss with respect to the logits $A$ is approximately $[-0.341, 0.242, 0.099]$. This gradient tells us how to adjust the logits in order to reduce the loss. Specifically, we need to increase the logit corresponding to the "Cat" class (since its gradient is negative, indicating we should increase the logit to decrease the loss) and decrease the logits for the "Dog" and "Bird" classes (since their gradients are positive).

These computed values will guide the optimization algorithm in updating the model parameters to improve its predictions over training iterations.
