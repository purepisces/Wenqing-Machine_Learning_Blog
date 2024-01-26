# Fitting a line to data aka least squares aka linear regression

Which line is better for fitting the data?
A horizontal line that cuts through the average y value of our data is probably the worst fit of all, however, it gives us a good starting point for talking about how to find the optimal line to fit our data.

<img src="multiple_line.png" width="400" height="350" alt="multiple_line">


We can measure how well this line fits the data by seeing how close it is to the data points.

<img src="calculate_distance1.png" width="400" height="350" alt="calculate_distance1">

However, if we simply make $(b-y_1)+(b-y_2)+(b-y_3)+(b-y_4)+(b-y_5)+…$, $(b-y_4)$ and $(b-y_5)$ the value will be negative, which will subtract form total and make the overall fit appear better than it really is.
Back in the day, when they first working this out, they probably tried taking the absolute value of everything and then discovered that it made the math pretty tricky.

<img src="calculate_distance2.png" width="400" height="350" alt="calculate_distance2">

So they ended up squaring each term. Squaring ensures that each term is positive.
Here is the equation that shows the total distance the data points have from the horizontal line. $(b-y_1)^2 + (b-y_2)^2+(b-y_3)^2+(b-y_4)^2+…+ = 24.62$, this is our measure of how well this line fits the data. It’s called the “sum of squared residual”, because the residuals are the differences between the real data and the line, and we are summing the square of these values.

<img src="calculate_distance3.png" width="400" height="350" alt="calculate_distance3">
To find that sweet spot, let’s start with the generic line equation, which is $$y = a * x+ b$$ , a is the slope of the line, and b is the y-intercept of the line. We want to find the optimal values for a and b so that we minimize the sum of squared residuals.
In more general math terms, sum of squared residuals = $$((a * x_1+b)-y_1)^2+((a * x_2+b)-y_2)^2+…$$ which is calculating the distance between the line and the observed value.

<img src="SSR.png" width="400" height="350" alt="SSR">


Since we want the line that will give us the smallest sum of squares, this method for finding the best values for “a” and “b” is called “Least Squares”.

<img src="least_squares.png" width="400" height="350" alt="least_squares">
If we plotted the sum of squared residuals vs each rotation, we will get this function. How do we find the optimal rotation for the line?
We take the derivative of this function, the derivative tells us the slope of the function at every point. The slope at the best point(the “least squares”) is zero. Remember, the different rotations are just different values for “a” the slope) and “b”(the intercept.

<img src="best_point.png" width="400" height="350" alt="best_point">

Taking the derivatives of both the slope and the intercepts tells us where the optimal values are for the best fit.

<img src="optimal_value.png" width="400" height="350" alt="optimal_value">
No one ever solves this problem by hand, this is done on a computer. This is done on a computer, so for most people, it is not essential to know how to take these derivatives.

Big important concept:
We want to minimize the square of the distance between the observed values and the line.
<img src="big_important_concept1.png" width="400" height="350" alt="big_important_concept1">

We do this by taking the derivative and finding where it is equal to 0. The final line minimizes the sums of squares(it gives the “least squares”) between it and the real data. In this case, the line is defined by the following equation y = 0.77*x + 0.66.
<img src="big_important_concept2.png" width="400" height="350" alt="big_important_concept2">
<img src="big_important_concept22.png" width="400" height="350" alt="big_important_concept22">

## Why Both Derivatives are Set to Zero in Minimizing SSR in Linear Regression

The key to understanding why we set both the derivative with respect to `a` (the slope) and the derivative with respect to `b` (the intercept) to zero in the context of minimizing the Sum of Squared Residuals (SSR) in linear regression lies in the principles of multivariable optimization.

### Simplified Single Variable Function Example

First, consider a function `f(x) = x^2`, which is a simple parabola.

- **Function**: `f(x) = x^2`
- **Derivative**: The derivative `f'(x) = 2x`

#### Analyzing the Derivative

- **At `x = 0`**: The derivative `f'(0) = 0`. This is where the function has its minimum.
- **At `x = 1`**: The derivative `f'(1) = 2`. Since this is not zero, it implies:
  - If the derivative is positive, decreasing `x` slightly will decrease `f(x)`.
  - If the derivative is negative, increasing `x` slightly will decrease `f(x)`.

### Application to SSR in Linear Regression

In linear regression, SSR is a function of two variables, `a` and `b`:

$SSR(a, b) = \sum((yi - axi - b)^2)$

#### Importance of Partial Derivatives

- The partial derivatives with respect to `a` and `b` indicate how SSR changes as `a` and `b` are varied.
- A non-zero derivative indicates that a small adjustment in that variable will lead to a decrease in SSR.

#### Setting Both Derivatives to Zero

- Setting both derivatives to zero is essential to find the minimum SSR. This means that the SSR cannot be further decreased by changing either `a` or `b`.
- This approach helps identify the optimal slope (`a`) and intercept (`b`) for the best-fit line in the data.

### Conclusion

In summary, a non-zero derivative at a point indicates that moving in the direction opposite to the sign of the derivative will decrease the function value (SSR in this case). In the context of linear regression, we need both derivatives (with respect to $a$ and $b$ to be zero to ensure we've found the minimum SSR, where the line best fits the data. By setting both the derivative with respect to `a` and `b` to zero, we ensure that we have found the point where SSR is minimized with respect to both variables. This point represents the optimal values of `a` (slope) and `b` (intercept) for the linear regression line that best fits the given data.


## Reference:
- [YouTube Video](https://www.youtube.com/watch?v=PaFPbb66DxQ)
