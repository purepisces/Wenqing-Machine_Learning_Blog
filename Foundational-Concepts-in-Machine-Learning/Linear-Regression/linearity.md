# Linearity in Linear Regression

## Overview

In the context of linear regression, the assumption of linearity implies that the relationship between the independent variable(s) and the dependent variable is linear. This means that the change in the dependent variable is proportional to the change in the independent variable(s).

## Why is Linearity Important?

Linear regression aims to fit a straight line (or a hyperplane in multiple regression) to the data points. If the underlying relationship between the variables is inherently non-linear, then a linear model may not provide a good fit, leading to inaccurate predictions and misleading insights.


> Note: A line is a hyperplane in 2-dimensional space, and a plane is a hyperplane in 3-dimensional space. The term "hyperplane" is a generalization that describes a subspace of one dimension less than its ambient space. Example for ambient space: If you have a line or a plane inside a 3-dimensional space (like the space we're familiar with in our physical world), then that 3D space is the "ambient space" for the line or plane.
>
> Note: simple linear regression involves predicting a response variable based on a single predictor variable, multiple regression predicts a response based on two or more predictors. Simple Linear Regression: $Y = \beta_0 + \beta_1X_1$ Multiple Regression:$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p$
>

## How to Check for Linearity?

1. **Scatter Plots**: A simple scatter plot of the dependent variable against each independent variable can provide a visual check. A roughly straight-line pattern suggests a linear relationship.

2. **Residual Plots**: Plotting residuals (the differences between observed and predicted values) against predicted values can also provide insights. If residuals are scattered randomly without any specific pattern, it suggests that the linearity assumption holds.

3. **Correlation Coefficient**: A high absolute value of the correlation coefficient (close to 1 or -1) between two variables indicates a strong linear relationship.

## Remedies for Non-Linearity

If the linearity assumption is violated, consider the following approaches:

1. **Data Transformation**: Applying mathematical transformations like logarithms, square roots, or polynomials can sometimes linearize relationships.

2. **Adding Polynomial Terms**: If there's a curvilinear relationship, adding polynomial terms (like squared or cubic terms) to the model can capture the non-linearity.

3. **Using Non-linear Models**: If the data is inherently non-linear, it might be appropriate to consider non-linear regression models or other machine learning algorithms.

## Conclusion

The linearity assumption is fundamental to linear regression. Violations can lead to suboptimal model performance. Always visualize and test your data for linearity before fitting a linear regression model.

