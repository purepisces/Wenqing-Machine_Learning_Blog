Backpropagation

In part1, inside the black box, we started with a simple dataset that showed whether or not different drug dosages were effective against a virus. The low and high dosages were not effective, but the medium dosage was effective. Then we talked about how a neural network fits a green a green squiggle to this dataset.
Insert fit_squiggle png
Remember, the neural network starts with identical activation functions but using different weights and biases on the connections, it flips and stretches the activation functions into new shapes, which are then added together to get a squiggle that is shifted to fit the data.
Insert new_shape png
Insert shift_fit png
However, we did not talk about how to estimate the weights and biases, so let’s talk about how back propagation optimizes the weights and biases in this and other neural networks.
Insert how_optimize png

In this part, we talk about the main ideas of back propagation:
Using the chain rule to calculate derivatives
d SSR/ d bias = d SSR/d Predicted x d Predicted/d bias
Insert step1 png
2. Plugging the derivatives into gradient descent to optimize parameters

Insert step2 png

In the next part, we’ll talk about how the chain rule and gradient descent apply to multiple parameters simultaneously and introduce some fancy notation, then we will go completely bonkers with the chain rule and show how to optimize all 7 parameters simultaneously in this neural network.

Insert 7_para png

Note: conceptually, backpropagation starts with the last parameter and works its way backwards to estimate all of the other parameters. However, we can discuss all of the main ideas behind backprogagation by just estimating the last Bias, b3.

Insert last_para png

In order to start from the back, let’s assume that we already have optimal values for all of the parameters except for the last bias term, b3. The parameter values that have already been optimized are marked green, and unoptimized parameters will be red. Also note, to keep the math simple, let’s assume dosages go from 0(low) to 1(high).
Insert assume png

Now, if we run dosages from 0 to 1 through the connection to the top node in the hidden layer then we get x-axis coordinates for the activation function that are all inside this red box and when we plug the x-axis coordinates into the activation function, which, in this example is the soft plus activation function, we get the corresponding y-axis coordinates and this blue curve. Then we multiply the y-axis coordinates on the blue curve by -1.22 and we get the final blue curve.
Insert softplus png insert final_blue_curve png
Then same operation for run dosages from 0 to 1 through the connection to the bottom node in the hidden layer, and get the final orange curve. Then add the blue and orange curves together to get the green squiggle.

Insert get_squiggle png

Now we are ready to add the final bias, b3, to the green squiggle. because we don’t yet know the optimal value for b3, we have to give it an initial value. And because bias terms are frequently initialized to 0, we will set b3 = 0. Now, adding 0 to all of the y-axis coordinates on the green squiggle leaves it right where it is. However, that means the green squiggle is pretty far from the data that we observed. We can quantify how good the green squiggle fits the data by calculating the sum of the squared residuals.

Insert b30 png

A residual is the difference between the observed and predicted values.
Residual = (Observed - Predicted)
Insert cal_residual png

By calculation when b3= 0, the SSR = 20.4. And we can plot it in the following graph, where y-axis is SSR, x-axis is the bias b3.
Insert b3_SSR png

Now, if we increase b3 to 1, then we add 1 to the y-axis coordinates on the green squiggle and shift the green squiggle up 1 and we end up with shorter residuals.
Insert vary_b3_ssr png

And if we had time to plug in tons of values for b3, we would get this pink curve and we could find the lowest point, which corresponds to the value for b3 that results in the lowest SSR, here.
Insert lowest_ssr png

However, instead of plugging in tons of values to find the lowest point in the pink curve, we use gradient descent to find it relatively quickly. And that means we need to find the derivative of the sum of the squared residuals with respect to b3.

Insert lowest_by_gradient png

SSR = \sum_{I=1}^{n=3}(Obseved_i - Predicted_i)^2
Each predicted value comes from the green squiggle, Predicted_i = green squiggle_i, and the green squiggle comes from the last part of the neural network. In other words, the green squiggle is the sum of the blue and orange curves plus b3.

Predicted_i = green squiggle_i = blue + orange + b3


Insert green_last png
