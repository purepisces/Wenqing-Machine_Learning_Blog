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

## Reference:
- [YouTube Video](https://www.youtube.com/watch?v=bMccdk8EdGo)
