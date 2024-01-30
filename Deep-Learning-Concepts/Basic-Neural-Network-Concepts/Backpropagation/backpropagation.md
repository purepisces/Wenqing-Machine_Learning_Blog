Backpropagation

In part1, inside the black box, we started with a simple dataset that showed whether or not different drug dosages were effective against a virus. The low and high dosages were not effective, but the medium dosage was effective. Then we talked about how a neural network fits a green a green squiggle to this dataset.

<img src="fit_squiggle.png" alt="fit_squiggle" width="400" height="300"/>

Remember, the neural network starts with identical activation functions but using different weights and biases on the connections, it flips and stretches the activation functions into new shapes, which are then added together to get a squiggle that is shifted to fit the data.

<img src="new_shape.png" alt="new_shape" width="400" height="300"/>
<img src="shift_fit.png" alt="shift_fit" width="400" height="300"/>

However, we did not talk about how to estimate the weights and biases, so let’s talk about how back propagation optimizes the weights and biases in this and other neural networks.

<img src="how_optimize.png" alt="how_optimize" width="400" height="300"/>


In this part, we talk about the main ideas of back propagation:
Using the chain rule to calculate derivatives

$\frac{dSSR}{dbias} = \frac{dSSR}{dPredicted} \cdot \frac{dPredicted}{dbias}$



<img src="step1.png" alt="step1" width="400" height="300"/>

2. Plugging the derivatives into gradient descent to optimize parameters

<img src="step2.png" alt="step2" width="400" height="300"/>


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

Step size = slope x learning rate = -15.7 x 0.1 = -1.57

New b3 = Old b3 - Step Size = 0 - (-1.57) = 1.57

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

<img src="b3_2191.png" alt="b3_2191" width="400" height="300"/>

<img src="b3_2192.png" alt="b3_2192" width="400" height="300"/>

<img src="b3_2193.png" alt="b3_2193" width="400" height="300"/>


Now we just keep taking steps until the step size is close to 0. And because the spa size is close to 0 when b3 = 2.61, we decide that 2.16 is the optimal value for b3.

<img src="b3_261.png" alt="b3_261" width="400" height="300"/>

So the main ideas for back propagation are that when a parameter is unknown like b3, we use the chain rule to calculate the derivative of the sum of the squared residuals(SSR) with respect to the unknown parameter, which in this case was b3. Then we initialize the unknown parameter with a number and in this case we set b3 = 0 and used gradient descent to optimize the unknown parameter.

<img src="main_idea.png" alt="main_idea" width="400" height="300"/>
<img src="optimize.png" alt="optimize" width="400" height="300"/>


