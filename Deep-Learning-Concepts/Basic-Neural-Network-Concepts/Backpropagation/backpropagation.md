# Version 1 Backpropagation 
Backpropagation is the algorithm used to calculate the gradient of the loss function with respect to each parameter (weight) in a neural network. It allows the network to update these weights in a way that minimizes the loss function, enabling the network to learn from the data.

## Backpropagation Overview

#### Forward Pass:
- Input data passes through the network, and the output is calculated.
- Activations and intermediate values (such as $Z_i$) are stored for use in the backward pass.

#### Backward Pass:
- The loss is computed using the output of the network and the true labels.
- Gradients of the loss with respect to each parameter are calculated using the chain rule of calculus.
- These gradients are used to update the parameters (weights) using an optimization algorithm like gradient descent.

### Backpropagation with Respect to $Z_i$ and $W_i$

In backpropagation, we need to calculate gradients with respect to both the activations $Z_i$ and the weights $W_i$. Here’s how this is done:

#### Gradients with Respect to Activations $Z_i$
The gradients with respect to the activations $Z_i$ are intermediate steps in the backpropagation process. They are used to calculate the gradients with respect to the weights. Specifically, for layer $i$, we compute $\frac{\partial \ell}{\partial Z_i}$, which represents how the loss changes with respect to the activations of that layer.

#### Gradients with Respect to Weights $W_i$
The ultimate goal is to compute the gradients of the loss with respect to the weights $W_i$, denoted as $\frac{\partial \ell}{\partial W_i}$. These gradients tell us how to adjust the weights to minimize the loss.

Refer to slides - [CMU 10714 Manual-Neural-Nets PDF](manual_neural_nets.pdf)

## Example to illustrate backpropagation from CMU Deep Learning Systems hw0 question 5: SGD for a two-layer neural network

Let's consider the case of a simple two-layer neural network. Specifically, for input $x \in \mathbb{R}^n$, we'll consider a two-layer neural network (without bias terms) of the form

$$\begin{equation}
z = W_2^T \mathrm{ReLU}(W_1^T x)
\end{equation}$$

where $W_1 \in \mathbb{R}^{n \times d}$ and $W_2 \in \mathbb{R}^{d \times k}$ represent the weights of the network (which has a $d$-dimensional hidden unit), and where $z \in \mathbb{R}^k$ represents the logits output by the network. We again use the softmax / cross-entropy loss, meaning that we want to solve the optimization problem

$$\begin{equation}
minimize_{W_1, W_2} \;\; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(W_2^T \mathrm{ReLU}(W_1^T x^{(i)}), y^{(i)}).
\end{equation}$$

Or alternatively, overloading the notation to describe the batch form with matrix $X \in \mathbb{R}^{m \times n}$, this can also be written

$$\begin{equation}
minimize_{W_1, W_2} \;\; \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y).
\end{equation}$$

Using the chain rule, we can derive the backpropagation updates for this network. Specifically, let

$$\begin{equation}
\begin{split}
Z_1 \in \mathbb{R}^{m \times d} & = \mathrm{ReLU}(X W_1) \\
G_2 \in \mathbb{R}^{m \times k} & = normalize(\exp(Z_1 W_2)) - I_y \\
G_1 \in \mathbb{R}^{m \times d} & = \mathrm{1}\{Z_1 > 0\} \circ (G_2 W_2^T)
\end{split}
\end{equation}$$

where $\mathrm{1}\{Z_1 > 0\}$ is a binary matrix with entries equal to zero or one depending on whether each term in $Z_1$ is strictly positive and where $\circ$ denotes elementwise multiplication. Then the gradients of the objective are given by

$$\begin{equation}
\begin{split}
\nabla_{W_1} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} X^T G_1 \\
\nabla_{W_2} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} Z_1^T G_2. \\
\end{split}
\end{equation}$$

**Note:** If the details of these precise equations seem a bit cryptic to you (prior to the 9/8 lecture), don't worry too much. These _are_ just the standard backpropagation equations for a two-layer ReLU network: the $Z_1$ term just computes the "forward" pass while the $G_2$ and $G_1$ terms denote the backward pass. But the precise form of the updates can vary depending upon the notation you've used for neural networks, the precise ways you formulate the losses, if you've derived these previously in matrix form, etc. If the notation seems like it might be familiar from when you've seen deep networks in the past, and makes more sense after the 9/8 lecture, that is more than sufficient in terms of background (after all, the whole _point_ of deep learning systems, to some extent, is that we don't need to bother with these manual calculations). But if these entire concepts are _completely_ foreign to you, then it may be better to take a separate course on ML and neural networks prior to this course, or at least be aware that there will be substantial catch-up work to do for the course.

### Neural networks in machine learning

Recall that neural networks just specify one of the "three" ingredients of a machine learning algorithm, also need:
- Loss function: still cross entropy loss, like last time
- Optimization procedure: still SGD, like last time

In other words, we still want to solve the optimization problem

$$\min_{\theta} \frac{1}{m} \sum_{i=1}^{m} \ell_{ce}(h_{\theta}(x^{(i)}), y^{(i)})$$

using SGD, just with $h_{\theta}(x)$ now being a neural network.

Requires computing the gradients $\nabla_{\theta} \ell_{ce}(h_{\theta}(x^{(i)}), y^{(i)})$ for each element of $\theta$.

### The gradient(s) of a two-layer network part1

Let's work through the derivation of the gradients for a simple two-layer network, written in batch matrix form, i.e.,

$$\nabla_{\{W_1, W_2\}} \ell_{ce}(\sigma(XW_1)W_2, y)$$

The gradient w.r.t. $W_2$ looks identical to the softmax regression case:

$$\frac{\partial \ell_{ce}(\sigma(XW_1)W_2, y)}{\partial W_2} = \frac{\partial \ell_{ce}}{\partial \sigma(XW_1)W_2} \cdot \frac{\partial \sigma(XW_1)W_2}{\partial W_2}
= (S - I_y) \cdot \sigma(XW_1), \quad [S = \text{softmax}(\sigma(XW_1)W_2)]$$

so (matching sizes) the gradient is

$$\nabla_{W_2} \ell_{ce}(\sigma(XW_1)W_2, y) = \sigma(XW_1)^T (S - I_y)$$


> **Prove  $S - I_y$**
> 
> **1. First Prove $\ell_{ce}(h(x), y) = -h_y(x) + \log \sum_{j=1}^{k} \exp(h_j(x))$**
> 
> Let's convert the hypothesis function to a "probability" by exponentiating and normalizing its entries (to make them all positive and sum to one)
> 
> $$z_i = p(\text{label} = i) = \frac{\exp(h_i(x))}{\sum_{j=1}^{k} \exp(h_j(x))} \quad \Longleftarrow \quad z \equiv \text{softmax}(h(x))$$
> 
> Then let's define a loss to be the (negative) log probability of the true class: this is called softmax or cross-entropy loss
> 
> $$\ell_{ce}(h(x), y) = -\log p(\text{label} = y) = -h_y(x) + \log \sum_{j=1}^{k} \exp(h_j(x))$$
> 
> **2. Second prove $S - I_y$**
> 
>Let's start by deriving the gradient of the softmax loss itself: for vector $h \in \mathbb{R}^k$
> 
>$$\frac{\partial \ell_{ce} (h, y)}{\partial h_i} = \frac{\partial}{\partial h_i} \left( -h_y + \log \sum_{j=1}^{k} \exp h_j \right)$$
>$$= -1 \{ i = y \} + \frac{\exp h_i}{\sum_{j=1}^{k} \exp h_j}$$
> 
>So, in vector form:
> 
>$$\nabla_h \ell_{ce} (h, y) = z - e_y, \text{ where } z = \text{softmax}(h)$$
> 
>In “matrix batch” form:
> 
>$$\nabla_h \ell_{ce} (X\theta, y) = S - I_y, \text{ where } S = \text{softmax}(X\theta)$$

### The gradient(s) of a two-layer network part2

Deep breath and let's do the gradient w.r.t. \( W_1 \)...

$$\frac{\partial \ell_{ce} (\sigma(XW_1) W_2, y)}{\partial W_1} = \frac{\partial \ell_{ce} (\sigma(XW_1) W_2, y)}{\partial \sigma(XW_1) W_2} \cdot \frac{\partial \sigma(XW_1) W_2}{\partial \sigma(XW_1)} \cdot \frac{\partial \sigma(XW_1)}{\partial XW_1} \cdot \frac{\partial XW_1}{\partial W_1}$$

$$= (S - I_y) \cdot W_2 \cdot \sigma'(XW_1) \cdot X$$

and so the gradient is

$$\nabla_{W_1} \ell_{ce} (\sigma(XW_1) W_2, y) = X^T \left( (S - I_y) W_2^T \circ \sigma'(XW_1) \right)$$

where $\circ$ denotes elementwise multiplication

$$\sigma'(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}$$

### Backpropagation "in general"

> 🌟🌟🌟🌟🌟🌟 In the slides explanation, the  neural network is Z1->W1->Z2->W2->Z3, however, in the assignment, the neural network is X->W1->Z1->W2->Z2. So the formula will be index different, but the logic is same. For example in assignment, $G_1 \in \mathbb{R}^{m \times d}  = \mathrm{1}\{Z_1 > 0\} \circ (G_2 W_2^T)$, but in slides $G_i$ is $G_{i+1}W_i$.

There is a method to this madness ... consider our fully-connected network:

$$ Z_{i+1} = \sigma_i(Z_i W_i), \quad i = 1, \ldots, L $$

Then (now being a bit terse with notation)

$$ \frac{\partial \ell(Z_{L+1}, y)}{\partial W_i} = \frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \frac{\partial Z_L}{\partial Z_{L-1}} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} \cdot \frac{\partial Z_{i+1}}{\partial W_i} $$

$$ G_{i+1} =\frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \frac{\partial Z_L}{\partial Z_{L-1}} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} $$

Then we have a simple "backward" iteration to compute the $G_i$'s

$$ G_i = G_{i+1} \cdot \frac{\partial Z_{i+1}}{\partial Z_i} = G_{i+1} \cdot \frac{\partial \sigma_i(Z_i W_i)}{\partial Z_i W_i} \cdot \frac{\partial Z_i W_i}{\partial Z_i} = G_{i+1} \cdot \sigma'(Z_i W_i) \cdot W_i $$

### Computing the real gradients

To convert these quantities to "real" gradients, consider matrix sizes

$$G_i = \frac{\partial \ell(Z_{L+1}, y)}{\partial Z_i} = \nabla_{Z_i} \ell(Z_{L+1}, y) \in \mathbb{R}^{m \times n_i}$$

so with "real" matrix operations

$$G_i = G_{i+1} \cdot \sigma'(Z_i W_i) \cdot W_i = \left( G_{i+1} \circ \sigma'(Z_i W_i) \right) W_i^T$$

Similar formula for actual parameter gradients $\nabla_{W_i} \ell(Z_{L+1}, y) \in \mathbb{R}^{n_i \times n_{i+1}}$

$$\frac{\partial \ell(Z_{L+1}, y)}{\partial W_i} = G_{i+1} \cdot \frac{\partial \sigma_i(Z_i W_i)}{\partial Z_i W_i} \cdot \frac{\partial Z_i W_i}{\partial W_i} = G_{i+1} \cdot \sigma'(Z_i W_i) \cdot Z_i$$

$$\implies \nabla_{W_i} \ell(Z_{L+1}, y) = Z_i^T \left( G_{i+1} \circ \sigma'(Z_i W_i) \right)$$

> 🌟🌟🌟🌟🌟🌟 Note that in the assignment, $G_1 \in \mathbb{R}^{m \times d}  = \mathrm{1}\{Z_1 > 0\} \circ (G_2 W_2^T)$. This is because in the slides explanation, the  neural network is Z1->W1->Z2->W2->Z3, however, in the assignment, the neural network is X->W1->Z1->W2->Z2. So $G_i$ is not $G_{i+1}W_i$ but $G_{i+1} W_{i+1}$

### Backpropagation: Forward and backward passes

Putting it all together, we can efficiently compute all the gradients we need for a neural network by following the procedure below

**Forward pass**
1. Initialize: $Z_1 = X$
   
   Iterate: $Z_{i+1} = \sigma_i(Z_i W_i), \quad i = 1, \ldots, L$

**Backward pass}**

2. Initialize: $G_{L+1} = \nabla_{Z_{L+1}} \ell(Z_{L+1}, y) = S - I_y$
   
   Iterate: $G_i = \left( G_{i+1} \circ \sigma'_i(Z_i W_i) \right) W_i^T, \quad i = L, \ldots, 1$

And we can compute all the needed gradients along the way

$$\nabla_{W_i} \ell(Z_{L+1}, y) = Z_i^T \left( G_{i+1} \circ \sigma'_i(Z_i W_i) \right)$$

"Backpropagation" is just chain rule + intelligent caching of intermediate results


### A closer look at these operations

What is really happening with the backward iteration?

$$\frac{\partial \ell(Z_{L+1}, y)}{\partial W_i} = \frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} \cdot \frac{\partial Z_{i+1}}{\partial W_i}$$

$$ G_{i+1} =\frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \frac{\partial Z_L}{\partial Z_{L-1}} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} $$

Each layer needs to be able to multiply the "incoming backward" gradient $G_{i+1}$ by its derivatives, $\frac{\partial Z_{i+1}}{\partial W_i}$, an operation called the "vector Jacobian product."

This process can be generalized to arbitrary computation graphs: this is exactly the process of automatic differentiation we will discuss in the next lecture.

### Code Implementation(pay attention to backpropagation)
```python
def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        # Forward pass: Compute Z1 and Z2 (the output logits)
        Z1 = np.maximum(0, X_batch @ W1)  # ReLU activation
        Z2 = Z1 @ W2

        # Compute softmax probabilities
        exp_logits = np.exp(Z2)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Create a one-hot encoded matrix of the true labels
        I_y = np.zeros((len(y_batch), num_classes))
        I_y[np.arange(len(y_batch)), y_batch] = 1

        # Backward pass: Compute gradients G2 and G1
        G2 = probs - I_y
        G1 = (Z1 > 0).astype(np.float32) * (G2 @ W2.T)
        # Compute the gradients for W1 and W2
        grad_W1 = X_batch.T @ G1 / batch
        grad_W2 = Z1.T @ G2 / batch

        # Perform the gradient descent step(Update the weights)
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2
    ### END YOUR CODE
```


# Version 2 Backpropagation 

In our previous discussion, we delved into how a neural network can fit a complex model to data, such as determining the effectiveness of various drug dosages against a virus. We saw that while low and high dosages were ineffective, a medium dosage proved beneficial. The neural network achieved this by adjusting a basic activation function into a "green squiggle" that matched our data points through manipulation of weights and biases.


<img src="fit_squiggle.png" alt="fit_squiggle" width="400" height="300"/>

Remember, the neural network starts with identical activation functions but using different weights and biases on the connections, it flips and stretches the activation functions into new shapes, which are then added together to get a squiggle that is shifted to fit the data.

<img src="new_shape.png" alt="new_shape" width="400" height="300"/>
<img src="shift_fit.png" alt="shift_fit" width="400" height="300"/>

However, we did not talk about how to estimate the weights and biases, so let’s talk about how back propagation optimizes the weights and biases in this and other neural networks.


<img src="how_optimize.png" alt="how_optimize" width="400" height="300"/>

In this part, we talk about the main ideas of backpropagation:

1. **Using the chain rule to calculate derivatives**

   \(\frac{dSSR}{dbias} = \frac{dSSR}{dPredicted} \cdot \frac{dPredicted}{dbias}\)

   ![step1](step1.png)

2. **Plugging the derivatives into gradient descent to optimize parameters**

   ![step2](step2.png)

In the next part, we’ll talk about how the chain rule and gradient descent apply to multiple parameters simultaneously and introduce some fancy notation, then we will go completely bonkers with the chain rule and show how to optimize all 7 parameters simultaneously in this neural network.

<img src="7_para.png" alt="7_para" width="400" height="300"/>

Note: conceptually, backpropagation starts with the last parameter and works its way backwards to estimate all of the other parameters. However, we can discuss all of the main ideas behind backprogagation by just estimating the last Bias, b3.

<img src="last_para.png" alt="last_para" width="400" height="300"/>

In order to start from the back, let’s assume that we already have optimal values for all of the parameters except for the last bias term, $b_3$. The parameter values that have already been optimized are marked green, and unoptimized parameters will be red. Also note, to keep the math simple, let’s assume dosages go from 0(low) to 1(high).

<img src="assume.png" alt="assume" width="400" height="300"/>

Now, if we run dosages from 0 to 1 through the connection to the top node in the hidden layer then we get x-axis coordinates for the activation function that are all inside this red box and when we plug the x-axis coordinates into the activation function, which, in this example is the soft plus activation function, we get the corresponding y-axis coordinates and this blue curve. Then we multiply the y-axis coordinates on the blue curve by -1.22 and we get the final blue curve.

<img src="softplus.png" alt="softplus" width="400" height="300"/> <img src="final_blue_curve.png" alt="final_blue_curve" width="400" height="300"/>

Then same operation for run dosages from 0 to 1 through the connection to the bottom node in the hidden layer, and get the final orange curve. Then add the blue and orange curves together to get the green squiggle.

<img src="get_squiggle.png" alt="get_squiggle" width="400" height="300"/>

Now we are ready to add the final bias, b3, to the green squiggle. because we don’t yet know the optimal value for b3, we have to give it an initial value. And because bias terms are frequently initialized to 0, we will set b3 = 0. Now, adding 0 to all of the y-axis coordinates on the green squiggle leaves it right where it is. However, that means the green squiggle is pretty far from the data that we observed. We can quantify how good the green squiggle fits the data by calculating the sum of the squared residuals.

<img src="b30.png" alt="b30" width="400" height="300"/>

A residual is the difference between the observed and predicted values.

$Residual = Observed - Predicted$

<img src="cal_residual.png" alt="cal_residual" width="400" height="300"/>

By calculation when $b_3$= 0, the SSR = 20.4. And we can plot it in the following graph, where y-axis is SSR, x-axis is the bias $b_3$.

<img src="b3_SSR.png" alt="b3_SSR" width="400" height="300"/>

Now, if we increase b3 to 1, then we add 1 to the y-axis coordinates on the green squiggle and shift the green squiggle up 1 and we end up with shorter residuals.

<img src="vary_b3_ssr.png" alt="vary_b3_ssr" width="400" height="300"/>

And if we had time to plug in tons of values for b3, we would get this pink curve and we could find the lowest point, which corresponds to the value for b3 that results in the lowest SSR, here.

<img src="lowest_ssr.png" alt="lowest_ssr" width="400" height="300"/>

However, instead of plugging in tons of values to find the lowest point in the pink curve, we use gradient descent to find it relatively quickly. And that means we need to find the derivative of the sum of the squared residuals with respect to b3.

<img src="lowest_by_gradient.png" alt="lowest_by_gradient" width="400" height="300"/>

$SSR = \sum\limits_{i=1}^{n=3} (Observed_i - Predicted_i)^2$


Each predicted value comes from the green squiggle, Predicted_i = green squiggle_i, and the green squiggle comes from the last part of the neural network. In other words, the green squiggle is the sum of the blue and orange curves plus b3.

$Predicted_i = \text{green squiggle}_i = \text{blue} + \text{orange} + b_3$

<img src="green_last.png" alt="green_last" width="400" height="300"/>

Now remember, we want to use gradient descent to optimize b3 and that means we need to take the derivative of the SSR with respect to b3. And because the SSR are linked to b3 by the predicted values, we can use the chain rule so solve for the derivative of the sum of the squared residuals with respect to b3.
The chain rule says that the derivative of the SSR respect to b3 is the derivative of the SSR with respect to the predicted values times the derivative of the predicted values with respect to b3.

Now let’s solve for the first part, the derivative of the SSR with respect to the predicted values:

$\frac{d SSR}{d b_3} = \frac{d SSR}{d Predicted} \cdot \frac{d Predicted}{d b_3}$

$\frac{d SSR}{d Predicted} = \frac{d}{d Predicted} \sum\limits_{i=1}^{n=3} (Observed_i - Predicted_i)^2 = \sum\limits_{i=1}^{n=3} 2 \cdot (Observed_i - Predicted_i) \cdot (-1) = \sum\limits_{i=1}^{n=3} -2 \cdot (Observed_i - Predicted_i)$

$\frac{d}{d Predicted} (Observed - Predicted) = -1$


<img src="ssr_predicted.png" alt="ssr_predicted" width="400" height="300"/>


Now let’s solve for the second part, the derivative of the Predicted values with respect to b3. Remember, the blue and orange curves were created before we got to b3, so the derivative of the blue curve with respect to b_3 is 0, because the blue curve is independent of b_3.


 $\frac{d Predicted}{d b_3} = \frac{d}{d b_3} (\text{green squiggle}) = \frac{d}{d b_3} (blue + orange + b_3) = 0 + 0 + 1 = 1$


<img src="creat_before.png" alt="creat_before" width="400" height="300"/>

<img src="predicted_b3.png" alt="predicted_b3" width="400" height="300"/>


Therefore

$\frac{d SSR}{d b_3} = \frac{d SSR}{d Predicted} \cdot \frac{d Predicted}{d b_3} = \sum\limits_{i=1}^{n=3} -2 \cdot (Observed_i - Predicted_i) \cdot 1$

And that means we can pug this derivative into gradient descent to find the optimal value for $b_3$.

<img src="find_optimal_b3.png" alt="find_optimal_b3" width="400" height="300"/>

$\frac{d SSR}{d b_3} = \sum\limits_{i=1}^{n=3} -2 \cdot (Observed_i - Predicted_i) \cdot 1$


Let’s see how we can use this equation with gradient descent

$$\begin{align*}
\frac{d SSR}{d b_3} &= \sum_{i=1}^{n=3} -2 \cdot (Observed_i - Predicted_i) \cdot 1 \\
&= -2 \cdot (Observed_1 - Predicted_1) \cdot 1 \\
&\quad - 2 \cdot (Observed_2 - Predicted_2) \cdot 1 \\
&\quad - 2 \cdot (Observed_3 - Predicted_3) \cdot 1 \\
&= -2 \cdot (0 - (-2.6)) \cdot 1 \\
&\quad - 2 \cdot (1 - (-1.6)) \cdot 1 \\
&\quad - 2 \cdot (0 - (-2.61)) \cdot 1 \\
&= -15.7
\end{align*}$$



Remember, we get the predicted values on the green squiggle by running the dosages through the neural network.

<img src="predict_green.png" alt="predict_green" width="400" height="300"/>


Now we just do the math and get -15.7 and that corresponds to the slope for when b3=0

<img src="slope_b30.png" alt="slope_b30" width="400" height="300"/>

Now we plug the slope into the gradient descent equation for step size, and in this example, we’ll set the learning rate to 0.1. And then we use the step size to calculate the new value for b3.

$$\begin{align*}
\text{Step size} &= \text{slope} \times \text{learning rate} \\
&= -15.7 \times 0.1 \\
&= -1.57 \\
\\
\text{New } b_3 &= \text{Old } b_3 - \text{Step size} \\
&= 0 - (-1.57) \\
&= 1.57
\end{align*}$$


Changing b3 to 1.57 shifts the green squiggle up and that shrinks the residuals.

<img src="b3_157.png" alt="b3_157" width="400" height="300"/>


Now, plugging in the new predicted values and doing the math gives us -6.26 which corresponds to the slope when b3 = 1.57. Then calculate the step size and the new value for b3 which is 2.19. Changing b3 to 2.19 shifts the green squiggle up further and that shrinks the residuals even more.

<img src="b3_2191.png" alt="b3_2191" width="300" height="200"/> <img src="b3_2192.png" alt="b3_2192" width="300" height="200"/> <img src="b3_2193.png" alt="b3_2193" width="300" height="200"/>


Now we just keep taking steps until the step size is close to 0. And because the spa size is close to 0 when b3 = 2.61, we decide that 2.16 is the optimal value for b3.

<img src="b3_261.png" alt="b3_261" width="400" height="300"/>

So the main ideas for back propagation are that when a parameter is unknown like b3, we use the chain rule to calculate the derivative of the sum of the squared residuals(SSR) with respect to the unknown parameter, which in this case was b3. Then we initialize the unknown parameter with a number and in this case we set b3 = 0 and used gradient descent to optimize the unknown parameter.

<img src="main_idea.png" alt="main_idea" width="400" height="300"/> <img src="optimize.png" alt="optimize" width="400" height="300"/>

## Reference:
- [Watch the video on YouTube](https://www.youtube.com/watch?v=IN2XmBhILt4)
