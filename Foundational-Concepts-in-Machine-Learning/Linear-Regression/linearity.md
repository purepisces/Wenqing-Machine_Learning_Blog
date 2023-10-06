# Linearity in Linear Regression

## Overview

In the context of linear regression, the assumption of linearity implies that the relationship between the independent variable(s) and the dependent variable is linear. This means that the change in the dependent variable is proportional to the change in the independent variable(s).

## Why is Linearity Important?

Linear regression aims to fit a straight line (or a hyperplane in multiple regression) to the data points. If the underlying relationship between the variables is inherently non-linear, then a linear model may not provide a good fit, leading to inaccurate predictions and misleading insights.


> Note: A line is a hyperplane in 2-dimensional space, and a plane is a hyperplane in 3-dimensional space. The term "hyperplane" is a generalization that describes a subspace of one dimension less than its ambient space. Example for ambient space: If you have a line or a plane inside a 3-dimensional space (like the space we're familiar with in our physical world), then that 3D space is the "ambient space" for the line or plane.
>
> Note: simple linear regression involves predicting a response variable based on a single predictor variable, multiple regression predicts a response based on two or more predictors. Simple Linear Regression: $Y = \beta_0 + \beta_1X_1$ Multiple Regression: $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p$
>

## How to Check for Linearity?

1. **Scatter Plots**: A simple scatter plot of the dependent variable against each independent variable can provide a visual check. A roughly straight-line pattern suggests a linear relationship.

2. **Residual Plots**: Plotting residuals (the differences between observed and predicted values) against predicted values can also provide insights. If residuals are scattered randomly without any specific pattern, it suggests that the linearity assumption holds.

A residual plot is a graphical representation used to visualize these residuals. On this plot:

The x-axis typically represents the predicted values (or sometimes the observed values or the values of an independent variable).
The y-axis represents the residuals.
By plotting the residuals against the predicted values, you can gauge the appropriateness of a linear model:

Random scatter: If the residuals seem to be randomly scattered around the horizontal axis (y=0), this suggests that a linear model is appropriate.
Patterns or trends: If there are discernible patterns (like a curve or a funnel shape), this indicates that the model might not capture some of the underlying relationships, suggesting potential non-linearity or issues with the model's assumptions.

3. **Correlation Coefficient**: A high absolute value of the correlation coefficient (close to 1 or -1) between two variables indicates a strong linear relationship.

The correlation coefficient, often represented as $r$, is a statistic that measures the strength and direction of the linear relationship between two variables. It can take values between -1 and 1, inclusive. 
The term "strength" refers to how closely data points adhere to a linear trend.
Here's what the values signify:
- $r = 1$ : Perfect positive linear relationship.
- $r = -1$: Perfect negative linear relationship.
- $r = 0$: No linear relationship.
- $r$ between 0 and 1 (or 0 and -1): The strength of the positive (or negative) linear relationship.

The mathematical derivation of the correlation coefficient is grounded in the covariance of two variables relative to their standard deviations. The formula for the Pearson correlation coefficient for a sample is:

$r = \frac{\Sigma (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\Sigma (x_i - \bar{x})^2 \Sigma (y_i - \bar{y})^2}}$

Where:
- $x_i$ and $y_i$: Individual data points.
- $\bar{x}$ and $\bar{y}$: Mean values of $x$ and $y$ respectively.

Variance: Variance is a measure of the dispersion or spread of a set of data points around their mean. It captures the average of the squared differences from the Mean. Mathematically, for a variable X \text{Variance(X)} = \Sigma \frac{(x_i - \bar{x})^2}{n}

Covariance: Covariance measures the joint variability of two variables. If the variables tend to increase and decrease together, the covariance is positive. If one variable tends to increase when the other decreases, the covariance is negative. For variables X and y
\text{Cov(X, Y)} = \Sigma \frac{(x_i - \bar{x})(y_i - \bar{y})}{n}
Thus, in a perfect positive linear relationship, the combined variability of 
�
X and 
�
Y (i.e., the covariance) is the highest it can possibly be, given their individual variabilities (i.e., their variances).

In simpler terms:

- The numerator captures how changes in one variable are associated with changes in another. If they tend to increase and decrease together, the value is positive; if one tends to increase as the other decreases, the value is negative.
- The denominator is a normalization factor ensuring the value falls between -1 and 1.

For a perfect linear relationship, whether positive or negative, the variance in one variable is perfectly explained by the variance in the other variable. Thus, the numerator will be equal to the denominator, resulting in $r=1$ (for perfect positive) or $r=-1$ (for perfect negative).

## Remedies for Non-Linearity

If the linearity assumption is violated, consider the following approaches:

1. **Data Transformation**: Applying mathematical transformations like logarithms, square roots, or polynomials can sometimes linearize relationships.

2. **Adding Polynomial Terms**: If there's a curvilinear relationship, adding polynomial terms (like squared or cubic terms) to the model can capture the non-linearity.

3. **Using Non-linear Models**: If the data is inherently non-linear, it might be appropriate to consider non-linear regression models or other machine learning algorithms.

## Conclusion

The linearity assumption is fundamental to linear regression. Violations can lead to suboptimal model performance. Always visualize and test your data for linearity before fitting a linear regression model.

