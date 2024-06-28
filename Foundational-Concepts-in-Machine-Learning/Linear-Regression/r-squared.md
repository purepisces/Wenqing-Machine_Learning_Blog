# Understanding R Squared (R²)

### What is Correlation (R)?
- Correlation measures the strength of a relationship between two quantitative variables (e.g., weight and size).
- Values close to 1 or -1 indicate a strong relationship.
- Values close to 0 suggest a weak or no relationship.
  
### Why R² Matters?
- R², is a metric of correlation like R, which measures the strength of a relationship, but it's easier to interpret.
- While R = 0.7 might not seem twice as strong as R = 0.5, R² = 0.7 is what it looks like, 1.4 times as good as R² = 0.5
- R² provides an intuitive and straightforward calculation for understanding the proportion of variation explained by the relationship.

> R² directly tells us the percentage of variation explained by the model, making it easier to understand how much of the outcome is accounted for by the predictor(s). For example, an R² of 0.49 means 49% of the variation is explained by the model, while an R² of 0.25 means 25% is explained.

### Example: Mouse Weight Prediction

1. **Initial Data Plotting**
   - Plot mouse weight (y-axis) against mouse IDs (x-axis).
   - Calculate and plot the mean weight.
   - Calculate variance as the sum of squared differences from the mean.
     $\text{The variation of the data} = \text{Sum(weight for mouse i } - \text{mean})^2$
     The difference between each data point is squared so that points below the mean don’t cancel out points above the mean.


<img src="variation-of-ID.png" alt="variation-of-ID" width="400" height="300"/>

2. **Reordering Data by Size**
   - Reorder mice by size without changing the mean and variance. The distances between the dots and the line have not changed, just their order.

<img src="variation-of-Size.png" alt="variation-of-Size" width="400" height="300"/>

3. **Better Prediction with a Fitted Line**
   - Fit a line to the size-weight data.
   - Use this line for more accurate weight predictions based on size.

 <img src="fit-a-line.png" alt="fit a line" width="400" height="300"/>


4. **Quantifying the Improvement with R²**
   - $R² = \frac{(Var(mean) - Var(line))}{Var(mean)}$
       Ranges from 0 to 1, with higher values indicating better predictions.


 <img src="quantify-R-square.png" alt="quantify R square" width="350" height="300"/> <img src="Var(mean).png" alt="Var(mean)" width="320" height="300"/>  <img src="Var(line).png" alt="Var(line)" width="320" height="300"/>

R² ranges from 0 to 1 because the variation around the line will never be greater than the variation around the mean and will never be less than 0. This division also makes R² a percentage.

5. **Examples**
   - High R² (e.g., 0.81) implies a strong relationship, like size and weight.
   - Low R² (e.g., 0.06) suggests a weak relationship, like sniffing time and weight.

An R² of 0.81 means there is 81% less variation around the line than the mean, or the size/weight relationship accounts for 81% of the total variation. This means that most of the variation in the data is explained by the size/weight relationship.

<img src="example-r-squared.png" alt="example-r-squared" width="400" height="300"/>

In another example, we’re comparing two possibly uncorrelated variables, which is comparing mouse weight and time spent sniffing a rock. We find R² = 0.06, thus there is only 6% less variation around the line than the mean or the sniff/weight relationship accounts for 6% of the total variation. This means only 6% of the variation is explained by this relationship. This means that hardly any of the variation in the data is explained by the sniff/weight relationship, indicating a very weak correlation.

<img src="r-squared-6.png" alt="r-squared-6" width="400" height="300"/>

### Interpreting R²
- Statistically significant R² = 0.9: 90% of the variation is explained by the relationship.
- Statistically significant R² = 0.01: Only 1% of the variation is explained.

When someone says, “The statistically significant R² was 0.9,” you can think, “Very good! The relationship between the two variables explains 90% of the variation in the data!” Conversely, if R² = 0.01, you can think, “Who cares if that relationship is significant, only 1% of the variation is explained; something else must explain the remaining 99%.”

> Even if the statistical tests show that the relationship between the two variables is statistically significant, the practical importance of this relationship might be minimal if the R² value is very low. In other words, statistical significance does not necessarily imply that the relationship is meaningful or substantial in explaining the variation in the data.

### Relation to R
- R² is the square of R.
- A high R (e.g., 0.9) squared gives a high R² (e.g., 0.81).
- R² provides a clearer comparison and is easier to interpret (e.g., R² of 0.7² is twice as good as 0.5²).

When someone says, “The statistically significant R was 0.9,” you can think, “0.9 times 0.9 = 0.81. Very good! The relationship between the two variables explains 81% of the variation in the data!” For R = 0.5, you can think, “0.5 times 0.5 = 0.25. The relationship accounts for 25% of the variation in the data. That’s good if there are a million other things accounting for the remaining 75%, bad if there is only one thing.”

R² is easier to interpret than plain old R. For example, R = 0.7 compared to R = 0.5:
- R² = 0.7² = 0.49: 49% of the original variation is explained.
- R² = 0.5² = 0.25: 25% of the original variation is explained.

With R², it is clear that the first correlation is roughly twice as good as the second.

> The relationship between R and the proportion of variance explained is non-linear. A correlation of R = 0.7 does not mean it is 1.4 times better than R = 0.5.
> 
> Squaring the correlation coefficient (R) to get R² linearizes this relationship, making it more intuitive to compare the strengths.

### Key Points
- R² indicates the percentage of variation explained by the relationship.
- R² does not indicate the direction of the correlation.
- Square the value of R to get R² for easier interpretation.
  
R² does not indicate the direction of the correlation because squared numbers are never negative. If the direction of the correlation isn’t obvious, you can say, “the two variables were positively (or negatively) correlated with R² = ...”

### Main Ideas of R²

- R² is the percentage of variation explained by the relationship between two variables.
- If someone gives you a value for R, square it to understand the percentage of variation explained.


## Understanding Why Var(line) ≤ Var(mean)

### Concepts of Variance and Linear Regression

#### Variance Around the Mean (Var(mean))

Variance measures the average squared deviation of each data point from the mean of the dataset. The formula for variance is:

$$\text{Var(mean)} = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2$$

where $y_i$ are the data points, $\bar{y}$ is the mean, and $N$ is the number of data points.

#### Variance Around the Fitted Line (Var(line))

In linear regression, we fit a line to the data that minimizes the sum of the squared differences between the actual data points and the predicted values (i.e., the residuals). The formula for the variance around the fitted line (residual sum of squares) is:

$$\text{Var(line)} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

where $\hat{y}_i$ are the predicted values from the regression line.

### Why Var(line) ≤ Var(mean)

#### Linear Regression Minimizes Residuals

The goal of linear regression is to find the line that minimizes the sum of squared residuals (differences between observed and predicted values). This process is known as the "least squares" method. By minimizing these residuals, the variance around the fitted line (Var(line)) is minimized compared to the variance around the mean (Var(mean)).

#### Decomposition of Total Variance

The total variance in the data (Var(mean)) can be decomposed into two parts: the variance explained by the regression model (explained variance) and the variance not explained by the model (residual variance). Mathematically:

$$\text{Total Variance (Var(mean))} = \text{Explained Variance} + \text{Residual Variance (Var(line))}$$

Since both explained variance and residual variance are non-negative, the residual variance (Var(line)) must be less than or equal to the total variance (Var(mean)).

### Visual Intuition

#### Variation Around the Mean

If you consider the mean line (a horizontal line at the mean value of the dependent variable), the distances (squared differences) of each data point from this mean line are relatively large.

#### Variation Around the Fitted Line

When you fit a regression line, you adjust the line to be as close as possible to all data points, thus minimizing the squared differences (residuals). This reduces the overall distances compared to the mean line.

### Formal Proof Using Statistical Properties

#### Sum of Squares

The total sum of squares (SST) measures the total variance in the dependent variable and is defined as:

$$SST = \sum (y_i - \bar{y})^2$$

The residual sum of squares (SSR) measures the variance not explained by the model:

$$SSR = \sum (y_i - \hat{y}_i)^2$$

#### Relationship

The explained sum of squares (SSE) measures the variance explained by the model:

$$SSE = \sum (\hat{y}_i - \bar{y})^2$$

By definition:

$$SST = SSE + SSR$$

### Conclusion

Since the linear regression model is designed to minimize the residual variance (SSR or Var(line)), it follows that the variance around the fitted line (Var(line)) will always be less than or equal to the variance around the mean (Var(mean)). This ensures that the fitted line provides a better or equal fit to the data compared to simply using the mean as a predictor.

Thus, the improvement in fit provided by the regression model (as measured by R²) indicates how much of the total variance is explained by the model, with Var(line) being a crucial component in this comparison.

## Example: Predicting Mouse Weight from Size

Suppose we have data on 5 mice, where we measure their size (independent variable, X) and weight (dependent variable, Y). Here's the data:

| Mouse ID | Size (X) | Weight (Y) |
|----------|----------|------------|
| 1        | 10       | 15         |
| 2        | 20       | 25         |
| 3        | 30       | 35         |
| 4        | 40       | 45         |
| 5        | 50       | 55         |

### Step 1: Calculate the Mean of Y

$$\bar{Y} = \frac{15 + 25 + 35 + 45 + 55}{5} = \frac{175}{5} = 35$$

### Step 2: Calculate the Total Sum of Squares (SST)

$$SST = \sum (Y_i - \bar{Y})^2 = (15 - 35)^2 + (25 - 35)^2 + (35 - 35)^2 + (45 - 35)^2 + (55 - 35)^2$$

$$SST = (-20)^2 + (-10)^2 + 0^2 + 10^2 + 20^2 = 400 + 100 + 0 + 100 + 400 = 1000$$

### Step 3: Calculate the Variance around the Mean (Var(mean))

$$\text{Var(mean)} = \frac{SST}{N} = \frac{1000}{5} = 200$$

### Step 4: Fit a Linear Regression Line

Assume the fitted regression line is:

$$\hat{Y} = 10 + 0.9X$$

### Step 5: Calculate the Predicted Values (\(\hat{Y}\))

| Mouse ID | Size (X) | Weight (Y) | Predicted Weight (\(\hat{Y}\)) |
|----------|----------|------------|--------------------------------|
| 1        | 10       | 15         | 10 + 0.9(10) = 19              |
| 2        | 20       | 25         | 10 + 0.9(20) = 28              |
| 3        | 30       | 35         | 10 + 0.9(30) = 37              |
| 4        | 40       | 45         | 10 + 0.9(40) = 46              |
| 5        | 50       | 55         | 10 + 0.9(50) = 55              |

### Step 6: Calculate the Residual Sum of Squares (SSR)

$$SSR = \sum (Y_i - \hat{Y})^2 = (15 - 19)^2 + (25 - 28)^2 + (35 - 37)^2 + (45 - 46)^2 + (55 - 55)^2$$

$$SSR = (-4)^2 + (-3)^2 + (-2)^2 + (-1)^2 + 0^2 = 16 + 9 + 4 + 1 + 0 = 30$$

### Step 7: Calculate the Variance around the Fitted Line (Var(line))

$$\text{Var(line)} = \frac{SSR}{N} = \frac{30}{5} = 6$$

### Step 8: Calculate R² using the Formula

$$R^2 = \frac{\text{Var(mean)} - \text{Var(line)}}{\text{Var(mean)}}$$

$$R^2 = \frac{200 - 6}{200} = \frac{194}{200} = 0.97$$

### Interpretation

- **R² Value**: The R² value is 0.97.
- **Meaning**: This means that 97% of the variation in mouse weight can be explained by the mouse size. Only 3% of the variation is due to other factors not accounted for by the model.

### Summary

In this example, the high R² value (0.97) indicates a strong linear relationship between the size of the mice and their weight. This means our linear regression model does an excellent job explaining the variability in the weight of the mice based on their size.

## Reference:
- [YouTube Video](https://www.youtube.com/watch?v=bMccdk8EdGo)
