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

<img src="cal_residual.png" alt="cal_residual" width="350" height="200"/>


