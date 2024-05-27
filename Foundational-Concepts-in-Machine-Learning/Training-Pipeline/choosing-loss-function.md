# chossing loss function

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

## SMAPE

The SMAPE formula is given by:
$$\text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \frac{|F_t - A_t|}{\left( \frac{|A_t| + |F_t|}{2} \right)}$$

Where:

- $n$ = number of data points
- $A_t$ = actual value at time $t$
- $F_t$ = forecasted (predicted) value at time $t$

