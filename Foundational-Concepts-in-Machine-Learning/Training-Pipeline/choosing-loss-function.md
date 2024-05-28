# Choosing Loss Functions

Binary Classification: Involves discrete classes and often probabilistic outputs. Metrics like binary cross-entropy are suited for handling probabilities and penalizing incorrect classifications. Binary cross-entropy is used because it penalizes incorrect predictions more heavily and works well with probabilistic outputs.

Forecasting: Involves continuous data and requires metrics that measure the accuracy of predicted values relative to actual values. MAPE and SMAPE provide meaningful interpretations for continuous predictions.


## Binary Classification
### Why Binary cross-entropy is used because it penalizes incorrect predictions more heavily and works well with probabilistic outputs.

Binary Cross-Entropy (Log Loss): Measures the performance of a classification model whose output is a probability value between 0 and 1. Binary cross-entropy is defined as:

$$\text{Binary Cross-Entropy} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$$

Where:

- $N$ is the number of samples.
- $y_i$ is the actual binary label (0 or 1) of the $i$-th sample.
- $p_i$ is the predicted probability of the $i$-th sample being in class 1.




Binary cross-entropy measures how well the predicted probabilities align with the actual labels. High confidence correct predictions yield low BCE, while high confidence incorrect predictions yield high BCE.

Let's revisit the example with the corrected interpretations:

| Example | Actual Label (\(y\)) | Predicted Probability (\(p\)) | BCE Calculation                                    | BCE Value |
|---------|-----------------------|-------------------------------|---------------------------------------------------|-----------|
| 1       | 1                     | 0.9                           | $-[1 \log(0.9) + 0 \log(0.1)]$             | 0.105     |
| 2       | 0                     | 0.2                           | $-[0 \log(0.2) + 1 \log(0.8)]$                | 0.223     |
| 3       | 1                     | 0.1                           | $-[1 \log(0.1) + 0 \log(0.9)]$                | 2.303     |
| 4       | 0                     | 0.8                           | $-[0 \log(0.8) + 1 \log(0.2)]$               | 1.609     |
| 5       | 1                     | 0.5                           | $-[1 \log(0.5) + 0 \log(0.5)]$                | 0.693     |

### Interpretation of Each Example

#### Example 1:
- **Actual Label**: 1
- **Predicted Probability**: 0.9
- **Confidence**: High confidence correct prediction.
- **BCE Value**: 0.105 (Low BCE, good prediction).

#### Example 2:
- **Actual Label**: 0
- **Predicted Probability**: 0.2
- **Confidence**: High confidence correct prediction for class 0 (since \(1 - p = 0.8\)).
- **BCE Value**: 0.223 (Low to moderate BCE, good prediction).

#### Example 3:
- **Actual Label**: 1
- **Predicted Probability**: 0.1
- **Confidence**: High confidence incorrect prediction.
- **BCE Value**: 2.303 (Very high BCE, poor prediction).

#### Example 4:
- **Actual Label**: 0
- **Predicted Probability**: 0.8
- **Confidence**: High confidence incorrect prediction.
- **BCE Value**: 1.609 (High BCE, poor prediction).

#### Example 5:
- **Actual Label**: 1
- **Predicted Probability**: 0.5
- **Confidence**: Low confidence prediction (uncertain).
- **BCE Value**: 0.693 (Moderate BCE, reflects uncertainty).

### Why BCE Penalizes Incorrect Predictions Heavily

Binary cross-entropy penalizes predictions based on their confidence and correctness:

- **High confidence correct predictions** (e.g., Example 1 and Example 2) yield low BCE.
- **High confidence incorrect predictions** (e.g., Example 3 and Example 4) yield high BCE. This heavy penalty encourages the model to avoid making confident but incorrect predictions.
- **Low confidence predictions** (e.g., Example 5) yield moderate BCE, reflecting the model's uncertainty.

## MAPE and SMAPE

### Example Data
Let's consider a small dataset of actual and forecasted values:

| Time ($t$) | Actual Value ($A_t$) | Forecast Value ($F_t$) |
|------------|----------------------|------------------------|
| 1          | 100                  | 90                     |
| 2          | 150                  | 160                    |
| 3          | 200                  | 195                    |
| 4          | 50                   | 60                     |
| 5          | 80                   | 70                     |

### MAPE(Mean Absolute Percentage Error) Calculation

The MAPE formula is given by:
$$\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{A_t - F_t}{A_t} \right|$$

Where:

- $n$ = number of data points
- $A_t$ = actual value at time $t$
- $F_t$ = forecasted (predicted) value at time $t$
  
For each time period $t$:

1. Calculate the absolute percentage error:
   $$\left| \frac{A_t - F_t}{A_t} \right|$$
2. Multiply by 100 to convert it to a percentage.

Let's calculate the absolute percentage errors for each time period:

For $t = 1$:
$$\left| \frac{100 - 90}{100} \right| \times 100 = 10\%$$

For $t = 2$:
$$\left| \frac{150 - 160}{150} \right| \times 100 = \left| \frac{-10}{150} \right| \times 100 \approx 6.67\%$$

For $t = 3$:
$$\left| \frac{200 - 195}{200} \right| \times 100 = 2.5\%$$

For $t = 4$:
$$\left| \frac{50 - 60}{50} \right| \times 100 = 20\%$$

For $t = 5$:
$$\left| \frac{80 - 70}{80} \right| \times 100 = 12.5\%$$

Now, calculate the mean of these percentage errors:
$$\text{MAPE} = \frac{1}{5} (10 + 6.67 + 2.5 + 20 + 12.5) = \frac{1}{5} \times 51.67 \approx 10.33\%$$

### SMAPE(Symmetric Absolute Percentage Error) Calculation

The SMAPE formula is given by:

$$\text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \frac{|F_t - A_t|}{\left( \frac{|A_t| + |F_t|}{2} \right)}$$

Where:

- $n$ = number of data points
- $A_t$ = actual value at time $t$
- $F_t$ = forecasted (predicted) value at time $t$
  
For each time period $t$:

1. Calculate the absolute error:
   $$|F_t - A_t|$$
2. Calculate the average of the actual and forecast values:
   $$\left( \frac{|A_t| + |F_t|}{2} \right)$$
3. Divide the absolute error by the average:
   $$\frac{|F_t - A_t|}{\left( \frac{|A_t| + |F_t|}{2} \right)}$$
4. Multiply by 100 to convert it to a percentage.

Let's calculate the SMAPE for each time period:

For $t = 1$:
$$\left| \frac{90 - 100}{\left( \frac{100 + 90}{2} \right)} \right| \times 100 = \left| \frac{10}{95} \right| \times 100 \approx 10.53\%$$

For $t = 2$:
$$\left| \frac{160 - 150}{\left( \frac{150 + 160}{2} \right)} \right| \times 100 = \left| \frac{10}{155} \right| \times 100 \approx 6.45\%$$

For $t = 3$:
$$\left| \frac{195 - 200}{\left( \frac{200 + 195}{2} \right)} \right| \times 100 = \left| \frac{5}{197.5} \right| \times 100 \approx 2.53\%$$

For $t = 4$:
$$\left| \frac{60 - 50}{\left( \frac{50 + 60}{2} \right)} \right| \times 100 = \left| \frac{10}{55} \right| \times 100 \approx 18.18\%$$

For $t = 5$:
$$\left| \frac{70 - 80}{\left( \frac{80 + 70}{2} \right)} \right| \times 100 = \left| \frac{10}{75} \right| \times 100 \approx 13.33\%$$

Now, calculate the mean of these percentages:
$$\text{SMAPE} = \frac{1}{5} (10.53 + 6.45 + 2.53 + 18.18 + 13.33) = \frac{1}{5} \times 51.02 \approx 10.20\%$$

### Comparison
- **MAPE**: 10.33%
- **SMAPE**: 10.20%

Both MAPE and SMAPE provide similar results in this example, but they handle extreme values differently. In general, SMAPE tends to be more stable when actual values are very small or when there are significant outliers in the data.

## Quantile Loss

Some of the problems that involve forecasting include Marketplace forecasting, Hardware capacity planning, and Marketing.

For the regression problem, DoorDash used Quantile Loss to forecast Food Delivery demand.

The Quantile Loss is given by:

$$ L(\hat{y}, y) = \max(\alpha(\hat{y} - y), (1 - \alpha)(y - \hat{y})) $$

Where:
- $\hat{y}$ is the predicted value.
- $y$ is the actual value.
- $\alpha$ is the quantile to be estimated (e.g., 0.5 for the median).

Quantile Loss helps in providing a more comprehensive picture of the distribution of errors, which is especially useful in applications like demand forecasting where understanding the range of possible outcomes is crucial.

### Overestimation Example Calculation

Given:
- The actual value $y = 100$
- The predicted value $\hat{y} = 110$
- The quantile $\alpha = 0.75$

#### Step-by-Step Calculation

1. Calculate the two terms inside the max function:
   $\alpha(\hat{y} - y) = 0.75 \times (110 - 100) = 0.75 \times 10 = 7.5$
   $(1 - \alpha)(y - \hat{y}) = 0.25 \times (100 - 110) = 0.25 \times (-10) = -2.5$

2. Take the maximum of the two calculated terms:
   $L(\hat{y}, y) = \max(7.5, -2.5) = 7.5$

### Underestimation Example Calculation

Given:
- The actual value $y = 100$
- The predicted value \( \hat{y} = 90 \)
- The quantile \( \alpha = 0.75 \)

#### Step-by-Step Calculation

1. Calculate the two terms inside the max function:
   \[ \alpha(\hat{y} - y) = 0.75 \times (90 - 100) = 0.75 \times (-10) = -7.5 \]
   \[ (1 - \alpha)(y - \hat{y}) = 0.25 \times (100 - 90) = 0.25 \times 10 = 2.5 \]

2. Take the maximum of the two calculated terms:
   \[ L(\hat{y}, y) = \max(-7.5, 2.5) = 2.5 \]

### Interpretation

In the overestimation example:
- The loss value is \(7.5\), reflecting a heavier penalty for overestimating with \(\alpha = 0.75\).

In the underestimation example:
- The loss value is \(2.5\), reflecting a lighter penalty for underestimating with \(\alpha = 0.75\).

Since \(\alpha = 0.75\), the Quantile Loss function penalizes overestimations more heavily, making it useful in scenarios where overestimations are costlier than underestimations.
