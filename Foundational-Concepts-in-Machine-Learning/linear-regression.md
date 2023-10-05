# Linear Regression

Machine learning has a myriad of applications, and one of its primary tasks is **prediction**. Linear regression predicts a dependent variable using one or more independent variables. It assumes a linear relationship between them.

## Model:
$y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n + \epsilon$

Where:
- $y$: Dependent variable.
- $x_1, ..., x_n$: Independent variables.
- $\beta_0, ..., \beta_n$: Coefficients.
- $\epsilon$: Error term.

## Objective:
Minimize the squared differences between observed and predicted values.
$\text{Min} \ \sum_i (\beta_0 + \beta_1 x_{i1} + ... + \beta_n x_{in} - y_i)^2$

## Simple Linear Regression: An Example

Consider a scenario where:
- $x$ represents the independent variable (e.g., height).
- $y$ represents the dependent variable we want to predict (e.g., weight).
  
We can express it as:
$y = g(x)$

In many cases, we have a training dataset comprising both $x$ and $y$ values. Our goal is to infer the function $g$. This task of finding the right function $g$ is termed **regression**.
And the easiest way to do this is to assume that this function g is a linear function g(x) = alpha x+beta. Hence the name linear regression. Alpha is the slope of this line and beta is the intercept.

To do linear regression, we just need to pick the alpha and beta that makes this line fit the data as much as possible.

One way to quantify the fit of a line to a bunch of data points is to consider where the point in the training data set is and where it should be according to this line. Take the square of the difference and then take the sum over all data points.

$\text{Min} \ \sum_i (\alpha x_i + \beta - y_i)^2$
To find optimal values for $\alpha$ and $\beta$, we compute the gradient of the above expression, set it to zero, and solve for both $\alpha$ and $\beta$. The gradient is given by:
$\left[ \sum_i 2x_i (\alpha x_i + \beta - y_i), \ 2\sum_i (\alpha x_i + \beta - y_i) \right] = 0$


## Assumptions:
- Linearity between variables.
- Observations are independent.
- Constant error variance.
- In multiple regression, predictors aren't highly correlated.
- Errors are normally distributed.

## Use Cases:
Linear regression is used in various fields and for tasks like predicting sales, estimating growth trends, and as part of more complex machine learning processes.

