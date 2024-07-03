## Question 4
**To start Linear Regression, you would need to make some assumptions. What are those assumptions?**

To start a Linear Regression model, there are some fundamental assumptions that you need to make:
- The model should have a multivariate normal distribution
- There should be no auto-correlation
- Homoscedasiticity, i.e, the dependent variable’s variance should be similar to all of the data
- There should be a linear relationship
- There should be no or almost no multicollinearity present

## Question 5
**What is multicollinearity and how will you handle it in your regression model?**

If there is a correlation between the independent variables in a regression model, it is known as multicollinearity. Multicollinearity is an area of concern as independent variables should always be independent. When you fit the model and analyze the findings, a high degree of correlation between variables might present complications.

There are various ways to check and handle the presence of multicollinearity in your regression model. One of them is to calculate the Variance Inflation Factor (VIF). If your model has a VIF of less than 4, there is no need to investigate the presence of multicollinearity. However, if your VIF is more than 4, an investigation is very much required, and if VIF is more than 10, there are serious concerns regarding multicollinearity, and you would need to correct your regression model.
