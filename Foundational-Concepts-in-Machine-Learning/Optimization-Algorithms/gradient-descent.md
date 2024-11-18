If you are not familiar with some calculus concept, please refer to [Calculus](Foundational-Concepts-in-Machine-Learning/Calculus/calculus.md).

# First Version: Gradient Descent Explanation 

In the realms of statistics, machine learning, and data science, optimization plays a crucial role. Whether it's fitting a line in linear regression, adjusting the curve in logistic regression, or forming clusters in t-SNE, gradient descent emerges as a powerful tool to optimize various models.

## Optimizing Different Models with Gradient Descent

- **Linear Regression**: Optimizes the intercept and slope of a line.
- **Logistic Regression**: Adjusts the curve (or "squiggle") to classify data points.
- **t-SNE**: Optimizes data points into meaningful clusters.

<img src="optimize_line.png" alt="optimize_line" width="300" height="200"/> <img src="optimize_squiggle.png" alt="optimize_squiggle" width="300" height="200"/> <img src="optimize_clusters.png" alt="optimize_clusters" width="300" height="200"/>

The cool thing is that gradient descent can optimize all these things and much more. So if we learn how to optimize this line using gradient descent then we will have learned the strategy that optimizes this squiggle and these clusters and many more of the optimization problems we have in statistics, machine learning and data science.

## Understanding Gradient Descent with a Simple Example

<img src="optimize_this_line.png" alt="optimize_this_line" width="350" height="200"/>


So let’s start with a simple data set, on the x-axis we have weight and on the y-axis we have height. $\text{Predicted height} = \text{intercept} + slope \times \text{weight}$, so let’s learn how gradient descent can fit a line to data by finding the optimal values for the intercept and the slope. Actually, we’ll start by using gradient descent to find the intercept, then once we understand how gradient descent works, we’ll use it to solve for the intercept and the slope.

So for now, let’s just plug in the least squares estimate for the slope, 0.64. And we’ll use gradient descent to find the optimal value for the intercept.

The first thing we do is pick a random value for the intercept. This is just an initial guess that gives gradient descent something to improve upon. In this case, we’ll use 0, but any number will do. And that gives us the equation for this line.

In this example, we will evaluate how well this line fits the data with the sum of the squared residuals. Note: In machine learning lingo, the sum of the squared residuals is a type of loss function.

<img src="ssr.png" alt="ssr" width="350" height="200"/>


We’ll start by calculating the residual.

<img src="cal_residual.png" alt="cal_residual" width="350" height="200"/>


And the SSR is 3.1, now we can draw this in the graph where x-axis represents intercept, y-axis represent sum of squared residuals, the red point in the graph represents the sum of the squared residuals when the intercept = 0.

<img src="cal_ssr.png" alt="cal_ssr" width="350" height="200"/>

And for increasing values for the intercept, we get these points. 

<img src="increase_intercept.png" alt="increase_intercept" width="350" height="200"/>

Of the points that we calculated for the graph, this one has the lowest sum of squared residuals, but is it the best we can do? And what if the best value for the intercept is somewhere between these values? 

<img src="lowest_ssr.png" alt="lowest_ssr" width="350" height="200"/> <img src="lowest_ssr2.png" alt="lowest_ssr2" width="350" height="200"/>


A slow and painful method for finding the minimal sum of the squared residuals is to plug and chug a bunch more values for the intercept. But gradient descent is way more efficient!


Gradient descent only does a few calculations far from the optimal solution and increases the number of calculations closer to the optimal value. In other words, gradient descent identifies the optimal value by taking big steps when it is far away and baby steps when it is close.

So let’s get back to using gradient descent to find the optimal value for the intercept, starting from a random value. In this case, the random value was 0. When we calculated the sum of the squared residuals, the first residual was the difference between the observed height, which was 1.4 and the predicted height, which came from the equation for this line.

$\text{Predicted height} = \text{intercept} + 0.64 \times \text{weight}$
$\text{Sum of squared residuals} = (\text{observed} - \text{predicted})^2 = (1.4 - \text{predicted})^2 = \left(1.4 - (\text{intercept} + 0.64 \times \text{weight})\right)^2$


Since the individual weighs 0.5, we replace weight with 0.5. 

So for this individual, 
= $(1.4-(intercept + 0.64 * 0.5))^2$
And now we can plug in any value for the intercept and get a new predicted height. And we will repeat this operation for other points. The equation becomes 

$Sum of squared residuals = (1.4-(intercept + 0.64 * 0.5))^2 + (1.9-(intercept + 0.64 * 2.3))^2 + (3.2-(intercept + 0.64 * 2.9))^2$
  
Now we can easily plug in any value for the intercept and get the sum of the squared residuals.

<img src="diff_intercept.png" alt="diff_intercept" width="350" height="200"/>


Thus, we now have an equation for this curve, and we can take the derivative of this function and determine the slope at any value for the intercept.

So now let’s take the derivative of the sum of the squared residuals with respect to the intercept. 
The derivative of the sum of the squared residuals with respect to the intercept equals the derivative of the first part plus the derivative of the second part plus the derivative of the third part.

<img src="derivative1.png" alt="derivative1" width="350" height="200"/>

$\frac{d}{d \text{intercept}} \text{Sum of Squared Residuals} = \frac{d}{d \text{intercept}} \left[(1.4 - (\text{intercept} + 0.64 \times 0.5))^2\right] + \frac{d}{d \text{intercept}} \left[(1.9 - (\text{intercept} + 0.64 \times 2.3))^2\right] + \frac{d}{d \text{intercept}} \left[(3.2 - (\text{intercept} + 0.64 \times 2.9))^2\right]$

Let’s start by taking the derivative of the first part, to take the derivative, we need to apply the chain rule:

$\frac{d}{d \text{intercept}} \left(1.4 - (\text{intercept} + 0.64 \times 0.5)\right)^2 = 2 \left(1.4 - (\text{intercept} + 0.64 \times 0.5)\right) \times \left(-1\right) = -2 \left(1.4 - (\text{intercept} + 0.64 \times 0.5)\right)$

So $\frac{D}{d \text{ intercept}} \text{Sum of Squared Residuals} = -2 \left(1.4 - (\text{intercept} + 0.64 \times 0.5)\right) - 2 \left(1.9 - (\text{intercept} + 0.64 \times 2.3)\right) - 2 \left(3.2 - (\text{intercept} + 0.64 \times 2.9)\right)$

Now that we have the derivative, gradient descent will use it to find where the sum of squared residuals is lowest.


<img src="gd_find.png" alt="gd_find" width="350" height="200"/>

Note: If we were using least squares to solve for the optimal value for the intercept, we would simply find where the slope of the curve = 0. In contrast, gradient descent finds the minimum value by taking steps from an initial guess until it reaches the best value. This makes gradient descent very useful when it is not possible to solve for where the derivative = 0, and this is why gradient descent can be used in so many different situations.
Remember, we started by setting the intercept to a random number. In this case, that was 0. So we plug 0 into the derivative and we get -5.7.

$\frac{D}{d \text{ intercept}} \text{Sum of Squared Residuals}  = -2(1.4-(0+ 0.64 x 0.5))
-2(1.9-(0 + 0.64 x 2.3))
-2(3.2-(0+ 0.64 x 2.9)) =-5.7$                                                  
So when the intercept = 0, the slope of the curve = -5.7

<img src="slope57.png" alt="slope57" width="350" height="200"/>


Note, the closer we get to the optimal value for the intercept, the closer the slope of the curve gets to 0. This means that when the slope of the curve is close to 0, then we should take baby steps, because we are close to the optimal value. And when the slope is far from 0, then we should take big steps because we are far from the optimal value. However, if we take a super huge step then we would increase the sum of the squared residuals. So the size of the step should be related to the slope, since it tells us if we should take a baby step or a big step, but we need to make sure the big step is not too big.
Gradient descent determines the step size by multiplying the slope by a small number called the learning rate.

<img src="lr.png" alt="lr" width="350" height="200"/>

Step size = -5.7x0.1 = -0.57
When the intercept equals 0, the step size equals -0.57. With the step size, we can calculate a new intercept. The new intercept is the old intercept minus the step size

$New intercept = old intercept - step size
= 0 - (-0.57) = 0.57$

In one big step, we moved much closer to the optimal value for the intercept.

<img src="move_closer.png" alt="move_closer" width="350" height="200"/>


Going back to the original data and the original line with the intercept = 0, we can see how much the residuals shrink when the intercept equals 0.57. 
Now let’s take another step closer to the optimal value for the intercept. To take another step, we go back to the derivative and plug in the new intercept(0.57)


$\frac{D}{d \text{ intercept}} \text{Sum of Squared Residuals} = -2 \left(1.4 - \left(\text{intercept} + 0.64 \times 0.5\right)\right) - 2 \left(1.9 - \left(\text{intercept} + 0.64 \times 2.3\right)\right) - 2 \left(3.2 - \left(\text{intercept} + 0.64 \times 2.9\right)\right)$

$\frac{D}{d \text{ intercept}} \text{Sum of Squared Residuals} = -2 \left(1.4 - \left(0.57 + 0.64 \times 0.5\right)\right) - 2 \left(1.9 - \left(0.57 + 0.64 \times 2.3\right)\right) - 2 \left(3.2 - \left(0.57 + 0.64 \times 2.9\right)\right) = -2.3$

And that tells us the slope of the curve = -2.3.
Now let’s calculate the step size, 
Step size = slope x learning rate 
= -2.3 X 0.1 = -0.23.
And the new intercept equals 0.8.
New intercept = old intercept - step size = 0.57-(-0.23) = 0.8
Now we can compare the residuals when the intercept equals 0.57 to when the intercept equals 0.8. Overall, the sum of the squared residuals is getting smaller.
Notice that the first step was relatively large, compared to the second step.

<img src="step_change.png" alt="step_change" width="350" height="200"/>

Now let’s calculate the derivative at the new intercept(0.8), and we get -0.9

$\frac{D}{d \text{ intercept}} \text{Sum of Squared Residuals}  = 
-2(1.4-(0.8+ 0.64 x 0.5))
-2(1.9-(0.8 + 0.64 x 2.3))
+ -2(3.2-(0.8 + 0.64 x 2.9))
=-0.9$

Step size = slope x learning rate = -0.9 x0.1 = -0.09, The step size equals -0.09 and the new intercept equals 0.89.
new intercept = old intercept -step size = 0.8-(-0.09)=0.89
Now we increase the intercept from 0.8 to 0.89, then we take another step and the new intercept equals 0.92. And then we take another step, and the new intercept equals 0.94. And then we take another step, and the new intercept equals 0.95.

<img src="intercept_95.png" alt="intercept_95" width="350" height="200"/>

Notice how each step gets smaller and smaller the closer we get to the bottom of the curve. After 6 steps, the gradient descent estimate for the intercept is 0.95. Note: The Least Squares estimate for the intercept is also 0.95. So we know that gradient descent has done its job, but without comparing its solution to a gold standard, how does gradient descent know to stop taking steps?

Gradient descent stops when the step size is very close to 0.
Step size = slope x learning rate
The step size will be very close to 0 when the slope is very close to 0.
In practice, the minimum step size equals 0.001 or smaller.
So if this slope equals 0.009, then we would plug in 0.009 for the slope and 0.1 for the learning rate.
Step size = slope x learning rate 
= 0.009 x 0.1= 0.0009
Which is smaller than 0.001, so gradient descent would stop.

<img src="stepsize_00009.png" alt="stepsize_00009" width="350" height="200"/>


That said, gradient descent also includes a limit on the number of steps it will take before giving up. In practice, the maximum number of steps = 1000 or greater.
So even if the step size is large, if here have been more than the maximum number of steps, gradient descent will stop.

Let’s review what we’ve learned so far, the first thing we did is decide to use the sum of the squared residuals as the loss function to evaluate how well a line fits the data.

$\text{Sum of Squared Residuals} = (1.4-(intercept + 0.64 x 0.5))^2
(1.9-(intercept + 0.64 x 2.3))^2
(3.2-(intercept + 0.64 x 2.9))^2$    

<img src="first_thing.png" alt="first_thing" width="350" height="200"/>



Then we took the derivative of the sum of the squared residuals. In other words, we took the derivative of the loss function.

$\frac{D}{d \text{intercept}} \text{Sum of Squared Residuals} = -2(1.4-(intercept + 0.64 x 0.5))
-2(1.9-(intercept + 0.64 x 2.3))
-2(3.2-(intercept + 0.64 x 2.9))$


<img src="derivative_loss.png" alt="derivative_loss" width="350" height="200"/>

Then we picked a random value for the intercept, in this case we set the intercept = 0.

<img src="intercept0.png" alt="intercept0" width="350" height="200"/>


Then we calculated the derivative when the intercept equals zero, plugged that slope into the step size calculation and then calculated the new intercept, the difference between the old intercept and the step size.

<img src="cal_new_intercept.png" alt="cal_new_intercept" width="350" height="200"/>


Lastly, we plugged the new intercept into the derivative and repeated everything until step size was close to 0.


<img src="lastly.png" alt="lastly" width="350" height="200"/>

Now that we understand how gradient descent can calculate the intercept, let’s talk about how to estimate he intercept and the slope.

<img src="est_inter_slope.png" alt="est_inter_slope" width="350" height="200"/>

Just like before, we will use the sum of the square residuals as the loss function.

<img src="ssr_loss.png" alt="ssr_loss" width="350" height="200"/>

This is a 3-D graph of the loss function for different values for the intercept and the slope.

<img src="3d.png" alt="3d" width="350" height="200"/>

This axis is the sum of the squared residuals, this axis represents different values for the slope, and this axis represents different values for the intercept. We want to find the values for the intercept and slope that give us the minimum sum of the squared residuals.

<img src="axis1.png" alt="axis1" width="350" height="200"/> <img src="axis2.png" alt="axis2" width="350" height="200"/> <img src="axis3.png" alt="axis3" width="350" height="200"/>
<img src="find_min.png" alt="find_min" width="350" height="200"/>



So, just like before, we need to take the derivative of this function.

$\text{Sum of Squared Residuals} = (1.4-(\text{intercept} + slope x 0.5))^2
(1.9-(intercept + slope x 2.3))^2
(3.2-(intercept + slope x 2.9))^2$   

And just like before, we’ll take the derivative with respect to the intercept, but unlike before, we’ll also take the derivative with respect to the slope.

D/d intercept sum of squared residuals
D/d slope sum of squared residuals

<img src="deri_slope.png" alt="deri_slope" width="350" height="200"/>


We’ll start by taking the derivative with respect to the intercept, just like before, we take the derivative of each part.


$\text{Sum of Squared Residuals}  = (1.4-(intercept + slope x 0.5))^2
(1.9-(intercept + slope x 2.3))^2
(3.2-(intercept + slope x 2.9))^2$ 

D/d intercept Sum of squared residuals = D/d intercept (1.4-(intercept + slope x 0.5))^2
D/d intercept (1.9-(intercept + slope x 2.3))^2
D/d intercept (3.2-(intercept + slope x 2.9))^2     

<img src="deri_wrt_intercept.png" alt="deri_wrt_intercept" width="350" height="200"/>


$$\begin{align*}
\frac{D}{d\text{intercept}} \, \text{Sum of Squared Residuals} = &amp; \frac{d}{d \, \text{intercept}} \left(1.4 - (\text{intercept} + \text{slope} \times 0.5)\right)^2 \\
&amp;+ \frac{d}{d\text{intercept}} \left(1.9 - (\text{intercept} + \text{slope} \times 2.3)\right)^2 \\
&amp;+ \frac{d}{d\text{intercept}} \left(3.2 - (\text{intercept} + \text{slope} \times 2.9)\right)^2 \\
= &amp; -2 \left(1.4 - (\text{intercept} + \text{slope} \times 0.5)\right) \\
&amp;- 2 \left(1.9 - (\text{intercept} + \text{slope} \times 2.3)\right) \\
&amp;- 2 \left(3.2 - (\text{intercept} + \text{slope} \times 2.9)\right)
\end{align*}$$

So this whole thing is the derivative of the sum of the squared residuals with respect to the intercept.
Now let’s take the derivative of the sum of the squared residuals with respect to the slope.

$$\begin{align*}
\frac{D}{d\text{slope}} \, \text{Sum of Squared Residuals} = &amp; \frac{d}{d \, \text{slope}} \left(1.4 - (\text{intercept} + \text{slope} \times 0.5)\right)^2 \\
&amp;+ \frac{d}{d\text{slope}} \left(1.9 - (\text{intercept} + \text{slope} \times 2.3)\right)^2 \\
&amp;+ \frac{d}{d\text{slope}} \left(3.2 - (\text{intercept} + \text{slope} \times 2.9)\right)^2 \\
= &amp; -2 \times 0.5 \left(1.4 - (\text{intercept} + \text{slope} \times 0.5)\right) \\
&amp;- 2 \times 2.3 \left(1.9 - (\text{intercept} + \text{slope} \times 2.3)\right) \\
&amp;- 2 \times 2.9 \left(3.2 - (\text{intercept} + \text{slope} \times 2.9)\right)
\end{align*}$$

<img src="deri_wrt_slope.png" alt="deri_wrt_slope" width="350" height="200"/>


When you have 2 or more derivatives of the same function, they are called a gradient. We will use this gradient to descend to lowest point in the loss function, which, in this case, is the sum of the squared residuals. Thus, this is why this algorithm is called gradient descent.

<img src="total_deri.png" alt="total_deri" width="350" height="200"/>


Just like before, we will start by picking a random number for the intercept. In this case we’ll set the intercept = 0 and we’ll pick a random number for the slope. In this case we’ll set the slope = 1.
Thus, this line with intercept = 0 and slope =1, is where we will start.
Now let’s plug in 0 for the intercept and 1 for the slope, and that gives us two slopes. Now we plug the slopes into the step size formula and multiply by the learning rate, which this time we set to 0.01. 

Step Size_Intercept = Slope x Learning Rate
Step Size_Slope = Slope x Learning Rate

Note: The larger learning rate that we used in the first example doesn’t work this time. Even after a bunch of steps, gradient descent doesn’t arrive at the correct answer. This means that gradient descent can be very sensitive to the learning rate.
The good news is that in practice, a reasonable learning rate can be determined automatically by starting large and getting smaller with each step. So in theory, you shouldn’t have to worry too much about the learning rate.
Anyway, we do the math and get two step sizes.
Step Size_Intercept = Slope x Learning Rate = -1.6x0.01=-0.016
Step Size_Slope = Slope x Learning Rate = -0.8x0.01=-0.008

<img src="step_size.png" alt="step_size" width="350" height="200"/>

Now we calculate the new intercept and new slope by plugging in the old intercept and the old slope and the step sizes.
New Intercept = Old Intercept - Step Size = 0-(-0.016)=0.016
New slope = Old slope - Step Size = 1-(-0.008)=1.008
And we end up with a new intercept and a new slope
This is the line we started with(slope = 1 and intercept = 0) and this is the line new line(with slope = 1.008 and intercept = 0.016) after the first step.

<img src="new_line.png" alt="new_line" width="350" height="200"/>

Now we just repeat what we did until all of the steps sizes are very small or we reach the maximum number of steps.
This is the best fitting line, with intercept = 0.95 and slope = 0.64, the same values we get from least squares.

<img src="best_fit_line.png" alt="best_fit_line" width="350" height="200"/>

We now know how gradient descent optimizes two parameters, the slope and intercept. If we had more parameters, then we’d just take more derivatives and everything else stays the same.
Note: the sum of the squared residuals is just one type of loss function, however, there are tons of other loss functions that work with other types of data. Regardless of which loss function you use, gradient descent works the same way.

<img src="loss_function.png" alt="loss_function" width="350" height="200"/>

Step 1: Take the derivative of the loss function for each parameter in it. In fancy machine learning lingo, take the gradient of the loss function.
Step 2: Pick random values for the parameters.
Step 3: plug the parameter values into the derivatives(ahem, the gradient)
Step 4: calculate the step sizes: step size = slope x learning rate
Step 5: calculate the new parameters:
New parameter = old parameter - step size
Now go back to step 3 and repeat until step size is very small, or you reach the maximum number of steps.

## Stochastic Gradient Descent

One last thing, in out example, we only had three data points, so the math didn’t take very long, but when you have millions of data points, it can take a long time. So there is a thing called stochastic gradient descent that uses a randomly selected subset of the data at every step rather than the full dataset. This reduces the time spent calculating the derivatives of the loss function.

<img src="stochastic_gradient_descent.png" alt="stochastic_gradient_descent" width="350" height="200"/>

## Reference:
- [Watch the video on YouTube](https://www.youtube.com/watch?v=sDv4f4s2SB8)

# Second Version: Gradient Descent Explanation 
Highly Recommend reading  [optimization](optimization-overview.md) for detail.

Gradient Descent is a fundamental optimization algorithm used to minimize loss functions in machine learning. By iteratively adjusting model parameters in the direction of the steepest descent (negative gradient), we aim to reach the loss function's minimum. Gradient Descent serves as the backbone of neural network training, underpinning more advanced optimization techniques. 

## Computing the Gradient

Before applying Gradient Descent, we need to compute the gradient of the loss function with respect to the parameters. There are two main approaches:

### Numerical Gradient (Finite Differences)

The **numerical gradient** approximates the gradient using the finite difference formula:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_i, \dots, x_n)}{h}$$

```python
def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
```

While straightforward, numerical gradients are computationally expensive and prone to truncation errors. For better accuracy, the **centered difference formula** can be used:

$$\frac{f(x + h) - f(x - h)}{2h}$$

This reduces truncation error, achieving second-order accuracy.

### Analytical Gradient (Using Calculus)

The **analytical gradient** leverages calculus to derive exact expressions for the gradient. For example, for the SVM loss:

$$\nabla_{w_{y_i}} L_i = - \left( \sum_{j \neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i$$

Here, the gradient is computed directly from the formula, making it efficient and precise. However, errors in implementation can occur, so **gradient checking** (comparing with numerical gradients) is often performed.

### Vanilla Gradient Descent

The basic Gradient Descent algorithm iteratively updates parameters as follows:

$$w \gets w - \eta \nabla_w L$$

Where:

-   $w$ are the parameters (weights),
-   $\eta$ is the learning rate,
-   $\nabla_w L$ is the gradient of the loss $L$ with respect to $w$.

Implementation:

```python
# Vanilla Gradient Descent

while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update
```

### Mini-Batch Gradient Descent

For large datasets, computing gradients over the entire dataset is inefficient. Instead, **mini-batches** are used to approximate the full gradient. This accelerates convergence and is commonly used in practice:

```python
# Vanilla Minibatch Gradient Descent

while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```


### Stochastic Gradient Descent (SGD)

An extreme case of mini-batch Gradient Descent, **SGD** updates parameters using a single training example at a time. Though computationally less efficient than mini-batches, it often achieves good results with modern frameworks that optimize vectorized operations.


### Learning Rate

The **learning rate** ($\eta$) determines the step size for parameter updates. Its choice significantly impacts convergence:

-   **Small $\eta$**: Slow but steady progress.
-   **Large $\eta$**: Faster convergence but risks overshooting the minimum.

Tuning the learning rate is critical and often done experimentally.

### Practical Considerations

1.  **Gradient Checking**: Compare the analytical gradient with the numerical gradient to ensure correctness.
2.  **Batch Size**: Typically chosen based on memory constraints or standard values like 32, 64, or 128.
3.  **Step Size**: Requires careful tuning to balance convergence speed and stability.


### Summary

Gradient Descent is a cornerstone of optimization in machine learning, enabling parameter updates by following the gradient of the loss function. Variants like mini-batch Gradient Descent and SGD adapt it to practical use cases. Mastery of Gradient Descent and its nuances is essential for training neural networks effectively.

## Reference:
- [https://cs231n.github.io/optimization-1/](https://cs231n.github.io/optimization-1/)
