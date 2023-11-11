# Correlation

## Introduction to Correlation

Correlation quantifies the strength of relationships between variables. It ranges from -1 (strong negative relationship) to +1 (strong positive relationship).

### Defining Correlation Values

- **Correlation Value**: Indicates the strength of the relationship.
  - Small Value: Weak relationship.
  - Moderate Value: Moderate relationship.
  - Large Value: Strong relationship.
- **Maximum Correlation**: The value of 1 indicates a perfect positive linear relationship. Correlation = 1 is when a straight line with a positive slope can go though the centre of every data point. That means if someone gave us a value for gene x, then we would guess that gene Y had a value in a very very narrow range.
  
### Characteristics of Correlation
- **Scale Independence**: Correlation does not depend on the scale of data.
- **P-Value**: A measure of confidence in the correlation. A small p-value indicates a low probability of the observed correlation occurring by chance.

## Scale Independence in Correlation

### Principle

- Correlation is independent of the data scale. This means that the actual numerical values or units of the data do not affect the correlation coefficient.

### Characteristics of Correlation

- **Correlation = 1**: Achieved when a straight line with a positive slope can pass through all data points.
  - **Slope Variability**: This condition holds true regardless of whether the slope of the line is steep or gentle.
  - **Data Scale Irrelevance**: The scale of the data does not influence this correlation value.

### Data Quantity and Correlation

- **Impact of Data Quantity**: The amount of data impacts the confidence in the correlation, but not the correlation value itself.
  - **Example of Minimal Data**: With just two data points, drawing a straight line with a positive slope will result in a correlation of 1. However, this high correlation should be interpreted with caution due to the limited data.

### Caution with Small Data Sets

- **Limited Data Interpretation**: A high correlation value (like 1) derived from a very small dataset may not be reliable for making predictions or inferences. More data points generally provide a more robust basis for correlation analysis.

> Note: When analyzing correlations in small datasets, our confidence in the findings is low because as more points are added, the less likely it is that a straight line can accurately represent their relationship. Correlation focuses on linear relationships, and a p-value assesses the probability that a similar or stronger relationship could occur by chance in random data. A very small p-value, like $2.2 \times 10^{-16}$, indicates a very low probability of the observed relationship occurring by chance, thereby increasing our confidence in the correlation.
>

<img src="p-value.png" width="450" height="300" alt="p-value">

### Correlation Extremes and Confidence

**Maximum Correlation:** Occurs with a perfect positive linear relationship, where more data enhances confidence in its utility.

<img src="max-correlation.png" width="400" height="300" alt="max-correlation"> <img src="most-confidence.png" width="400" height="300" alt="most-confidence">


**Negative Correlation (-1):** Indicates a strong inverse relationship, with a negatively sloped line passing through all data points. More data and a smaller p-value increase confidence in this relationship. Just like before, as long as a straight line goes through all of the data and the slope of the line is negative, correlation = -1 when the slope is large and when the slope is small.


<img src="small-p-value.png" width="300" height="250" alt="small-p-value"> <img src="slope-large.png" width="300" height="250" alt="slope-large"> <img src="slope-small.png" width="300" height="250" alt="slope-small">

So far we’ve seen that when the slope of the line is negative, the strongest relationship has correlation = -1 and when the slope of the line is positive, the strongest relationship has correlation = 1.

<img src="strong-relationship.png" width="450" height="300" alt="strong-relationship">


**Zero Correlation (0):** Implies no linear relationship; a value on one axis doesn't predict the value on the other.

> Note:When the correlation value isn't zero, we can make inferences using the correlation line, but these inferences become more precise as the correlation approaches -1 or 1. Our confidence in these inferences depends on both the amount of data collected and the p-value. With more data and a lower p-value, our confidence in the trend increases. In the left graph, we have very little confidence in the trend because we have very little data and the p-value = 0.8. In the middle, we have moderate confidence in the trend because we have more data and the p-value = 0.08, on the right, we have a lot of confidence in the trend because we have even more data and the p-value = 0.008. However, even if the sample size grows, a low correlation value (like 0.3 in the example) means our predictions remain unreliable. A larger dataset increases our confidence in these predictions, but it does not necessarily improve their accuracy. Therefore, even with high confidence due to more data, predictions can still be poor if the correlation is weak.
>

<img src="diff-p-value.png" width="450" height="300" alt="diff-p-value">

## Correlation Calculation

- **Formula**:
$\text{correlation} = \frac{\text{Covariance}(\text{Gene X}, \text{Gene Y})}{\sqrt{\text{Variance}(\text{Gene X})} \times \sqrt{\text{Variance}(\text{Gene Y})}} = \frac{\sum (X_i - \overline{X})(Y_i - \overline{Y})}{\sqrt{\sum (X_i - \overline{X})^2 \sum (Y_i - \overline{Y})^2}}$

<img src="Covariance of Gene X and Y.png" width="300" height="300" alt="Covariance of Gene X and Y"> <img src="Variance-Gene-X.png" width="300" height="300" alt="Variance-Gene-X"> <img src="Variance-Gene-Y.png" width="300" height="300" alt="Variance-Gene-Y">

The covariance can be any value between positive and negative infinity, depending on whether the slope of the line that represents the relationship is positive or negative, how far the data are spread out around the means and the scale of the data. Thus, when we calculate correlation, the denominator squeezes the covariance to be a number from -1 to 1. In other words, the denominator ensures that the scale of the data does not effect the correlation value, and this makes correlations much easier to interpret.
When the data all fall on a straight line with a positive or negative slope, then the covariance and the product of the square roots of the variance terms are the same and division gives us 1 or -1, depending on the slope. When the data do not all fall on a straight line with a positive or negative slope, then the covariance accounts for less of the variance in the data, and the correlation is closer to 0.
So for this graph where the correlation is 0.9 we can quantify our confidence in this relationship with a p-value. The smaller the p-value, the more confidence we can have in the guesses we make. In this case, the p-value is 0.03, which means there is 3% chance that random data could produce a similarly strong relationship, or stronger.

<img src="3-pvalue.png" width="450" height="300" alt="3-pvalue">

## R^2: A Related Measure
Even though correlation values are way easier to interpret than covariance values, they are still not super easy to interpret. For example, it’s not super obvious that this relationship, where correlation = 0.9, is twice as good at making prediction as this relationship, where correlation = 0.64.

<img src="Interpret.png" width="450" height="300" alt="interpret">

The good news is that $R^{2}$, which is related to correlation, solves this problem. Another awesome thing about R^2 is that it can quantify relationships that are more complicated than simple straight lines.

## Conclusion
In summary, correlation quantifies the strength of relationships, weak relationship will have a small correlation value, moderate relationship will have moderate correlation value and strong relationship will have large correlation value.
Insert strength-relationship graph
Correlation values go from -1, which is the strongest linear relationship with a negative slope to 1, which is the strongest linear relationship with a positive slope. In both cases, if a straight line can not go through all of the data, then we will get correlation values closer to 0 and the worse the fit, the closer the correlation values get to 0. And when there is no relationship that we can represent with a straight line, correlation = 0.
Lastly, our confidence in our inferences depends on the amount of data we have collected and the p-value. The more data we have, the smaller the p-value and the more confidence we have our inferences.

<img src="confidence.png" width="450" height="300" alt="confidence">

