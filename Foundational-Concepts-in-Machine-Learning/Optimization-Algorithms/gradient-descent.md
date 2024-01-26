If you are not farmiliar with some calculus concept, please refer to [Calculus](Foundational-Concepts-in-Machine-Learning/Calculus/calculus.md).

Garden descent:

In statistics, machine learning and other data science fields, we optimize a lot of stuff. When we fit a line with linear regression, we optimize the intercept and slope. When we use logistic regression, we optimize a squiggle. And when we use t-SNE, we optimize clusters.

<img src="optimize_line.png" alt="optimize_line" width="300" height="200"/> <img src="optimize_squiggle.png" alt="optimize_squiggle" width="300" height="200"/> <img src="optimize_clusters.png" alt="optimize_clusters" width="300" height="200"/>

The cool thing is that gradient descent can optimize all these things and much more. So if we learn how to optimize this line using gradient descent then we will have learned the strategy that optimizes this squiggle and these clusters and many more of the optimization problems we have in statistics, machine learning and data science.

<img src="optimize_this_line.png" alt="optimize_this_line" width="350" height="200"/>

So let’s start with a simple data set, on the x-axis we have weight and on the y-axis we have height. Predicted height = intercept + slope * height, so let’s learn how gradient descent can fit a line to data by finding the optimal values for the intercept and the slope. Actually, we’ll start by using gradient descent to find the intercept, then once we understand how gradient descent works, we’ll use it to solve for the intercept and the slope.

So for now, let’s just plug in the least squares estimate for the slope, 0.64. And we’ll use gradient descent to find the optimal value for the intercept.

The first thing we do is pick a random value for the intercept. This is just an initial guess that gives gradient descent something to improve upon. In this case, we’ll use 0, but any number will do. And that gives us the equation for this line.

In this example, we will evaluate how well this line fits the data with the sum of the squared residuals. Note: In machine learning lingo, the sum of the squared residuals is a type of loss function.

<img src="ssr.png" alt="ssr" width="350" height="200"/>


We’ll start by calculating the residual.

Insert cal_residual png

And the SSR is 3.1, now we can draw this in the graph where x-axis represents intercept, y-axis represent sum of squared residuals, the red point in the graph represents the sum of the squared residuals when the intercept = 0.

Insert cal_ssr png

And for increasing values for the intercept, we get these points. 
Insert increase_intercept png

Of the points that we calculated for the graph, this one has the lowest sum of squared residuals, but is it the best we can do? And what if the best value for the intercept is somewhere between these values? 
Insert lowest_ssr png
Insert lowest_ssr2 png

A slow and painful method for finding the minimal sum of the squared residuals is to plug and chug a bunch more values for the intercept. But gradient descent is way more efficient!


Gradient descent only does a few calculations far from the optimal solution and increases the number of calculations closer to the optimal value. In other words, gradient descent identifies the optimal value by taking big steps when it is far away and baby steps when it is close.

So let’s get back to using gradient descent to find the optimal value for the intercept, starting from a random value. In this case, the random value was 0. When we calculated the sum of the squared residuals, the first residual was the difference between the observed height, which was 1.4 and the predicted height, which came from the equation for this line.
Predicted height = intercept + 0.64 * weight
Sum of squared residuals = (observed-predicted)^2 
= (1.4-predicted)^2
= (1.4-(intercept + 0.64 * weight))^2
Since the individual weighs 0.5, we replace weight with 0.5. So for this individual, 
= (1.4-(intercept + 0.64 * 0.5))^2
And now we can plug in any value for the intercept and get a new predicted height. And we will repeat this operation for other points. The equation becomes 

Sum of squared residuals = (1.4-(intercept + 0.64 * 0.5))^2
+ (1.9-(intercept + 0.64 * 2.3))^2
+ (3.2-(intercept + 0.64 * 2.9))^2
Now we can easily plug in any value for the intercept and get the sum of the squared residuals.
Insert diff_intercept png

Thus, we now have an equation for this curve, and we can take the derivative of this function and determine the slope at any value for the intercept.

So now let’s take the derivative of the sum of the squared residuals with respect to the intercept. 
The derivative of the sum of the squared residuals with respect to the intercept equals the derivative of the first part plus the derivative of the second part plus the derivative of the third part.

Insert derivative1 png

D/d intercept sum of squared residuals = d/d intercept (1.4-(intercept + 0.64 x 0.5))^2 
d/d intercept(1.9-(intercept + 0.64 * 2.3))^2 
d/d intercept (3.2-(intercept + 0.64 * 2.9))^2

Let’s start by taking the derivative of the first part, to take the derivative, we need to apply the chain rule:
d/d intercept (1.4-(intercept + 0.64 x 0.5))^2 
=2(1.4-(intercept + 0.64 x 0.5))x(-1)
= -2(1.4-(intercept + 0.64 x 0.5))

So D/d intercept sum of squared residuals = -2(1.4-(intercept + 0.64 x 0.5))
-2(1.9-(intercept + 0.64 x 2.3))
+ -2(3.2-(intercept + 0.64 x 2.9))

Now that we have the derivative, gradient descent will use it to find where the sum of squared residuals is lowest.

Insert gd_find graph

Note: If we were using least squares to solve for the optimal value for the intercept, we would simply find where the slope of the curve = 0. In contrast, gradient descent finds the minimum value by taking steps from an initial guess until it reaches the best value. This makes gradient descent very useful when it is not possible to solve for where the derivative = 0, and this is why gradient descent can be used in so many different situations.
Remember, we started by setting the intercept to a random number. In this case, that was 0. So we plug 0 into the derivative and we get -5.7.

D/d intercept sum of squared residuals = -2(1.4-(0+ 0.64 x 0.5))
-2(1.9-(0 + 0.64 x 2.3))
-2(3.2-(0+ 0.64 x 2.9)) =-5.7                                                      So when the intercept = 0, the slope of the curve = -5.7

Insert slope57 png

Note, the closer we get to the optimal value for the intercept, the closer the slope of the curve gets to 0. This means that when the slope of the curve is close to 0, then we should take baby steps, because we are close to the optimal value. And when the slope is far from 0, then we should take big steps because we are far from the optimal value. However, if we take a super huge step then we would increase the sum of the squared residuals. So the size of the step should be related to the slope, since it tells us if we should take a baby step or a big step, but we need to make sure the big step is not too big.
Gradient descent determines the step size by multiplying the slope by a small number called the learning rate.

Insert lr png
Step size = -5.7x0.1 = -0.57
When the intercept equals 0, the step size equals -0.57. With the step size, we can calculate a new intercept. The new intercept is the old intercept minus the step size
New intercept = old intercept - step size
= 0 - (-0.57) = 0.57

In one big step, we moved much closer to the optimal value for the intercept.
Insert move_closer png

Going back to the original data and the original line with the intercept = 0, we can see how much the residuals shrink when the intercept equals 0.57. 
Now let’s take another step closer to the optimal value for the intercept. To take another step, we go back to the derivative and plug in the new intercept(0.57)


D/d intercept sum of squared residuals = -2(1.4-(intercept + 0.64 x 0.5))
-2(1.9-(intercept + 0.64 x 2.3))
-2(3.2-(intercept + 0.64 x 2.9))

D/d intercept sum of squared residuals = -2(1.4-(0.57+ 0.64 x 0.5))
-2(1.9-(0.57 + 0.64 x 2.3))
+ -2(3.2-(0.57 + 0.64 x 2.9))
=-2.3

And that tells us the slope of the curve = -2.3.
Now let’s calculate the step size, 
Step size = slope x learning rate 
= -2.3 X 0.1 = -0.23.
And the new intercept equals 0.8.
New intercept = old intercept - step size = 0.57-(-0.23) = 0.8
Now we can compare the residuals when the intercept equals 0.57 to when the intercept equals 0.8. Overall, the sum of the squared residuals is getting smaller.
Notice that the first step was relatively large, compared to the second step.

Insert step_change png
Now let’s calculate the derivative at the new intercept(0.8), and we get -0.9


D/d intercept sum of squared residuals = 
-2(1.4-(0.8+ 0.64 x 0.5))
-2(1.9-(0.8 + 0.64 x 2.3))
+ -2(3.2-(0.8 + 0.64 x 2.9))
=-0.9

Step size = slope x learning rate = -0.9 x0.1 = -0.09, The step size equals -0.09 and the new intercept equals 0.89.
new intercept = old intercept -step size = 0.8-(-0.09)=0.89
Now we increase the intercept from 0.8 to 0.89, then we take another step and the new intercept equals 0.92. And then we take another step, and the new intercept equals 0.94. And then we take another step, and the new intercept equals 0.95.

Insert intercept_95 png

Notice how each step gets smaller and smaller the closer we get to the bottom of the curve. After 6 steps, the gradient descent estimate for the intercept is 0.95. Note: The Least Squares estimate for the intercept is also 0.95. So we know that gradient descent has done its job, but without comparing its solution to a gold standard, how does gradient descent know to stop taking steps?

Gradient descent stops when the step size is very close to 0.
Step size = slope x learning rate
The step size will be very close to 0 when the slope is very close to 0.
In practice, the minimum step size equals 0.001 or smaller.
So if this slope equals 0.009, then we would plug in 0.009 for the slope and 0.1 for the learning rate.
Step size = slope x learning rate 
= 0.009 x 0.1= 0.0009
Which is smaller than 0.001, so gradient descent would stop.

Insert stepsize_00009 png

That said, gradient descent also includes a limit on the number of steps it will take before giving up. In practice, the maximum number of steps = 1000 or greater.
So even if the step size is large, if here have been more than the maximum number of steps, gradient descent will stop.

Let’s review what we’ve learned so far, the first thing we did is decide to use the sum of the squared residuals as the loss function to evaluate how well a line fits the data.

Sum of squared residuals = (1.4-(intercept + 0.64 x 0.5))^2
(1.9-(intercept + 0.64 x 2.3))^2
(3.2-(intercept + 0.64 x 2.9))^2      
insert  first_thing graph                           


Then we took the derivative of the sum of the squared residuals. In other words, we took the derivative of the loss function.

D/d intercept sum of squared residuals = -2(1.4-(intercept + 0.64 x 0.5))
-2(1.9-(intercept + 0.64 x 2.3))
-2(3.2-(intercept + 0.64 x 2.9))

Insert derivative_loss graph
Then we picked a random value for the intercept, in this case we set the intercept = 0.

Insert intercept0 graph

Then we calculated the derivative when the intercept equals zero, plugged that slope into the step size calculation and then calculated the new intercept, the difference between the old intercept and the step size.

Insert cal_new_intercept png

Lastly, we plugged the new intercept into the derivative and repeated everything until step size was close to 0.

Insert lastly png



<img src="cal_residual.png" alt="cal_residual" width="350" height="200"/>


