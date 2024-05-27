It depends on the use case when deciding which loss function to use. For binary classification, the most popular is cross-entropy. In the Click Through Rate (CTR) prediction, Facebook uses Normalized Cross Entropy loss (a.k.a. logloss) to make the loss less sensitive to the background conversion rate. 

In a forecast problem, the most common metrics are the Mean Absolute Percentage Error (MAPE) and the Symmetric Absolute Percentage Error (SMAPE). For MAPE, you need to pay attention to whether or not your target value is skew, i.e., either too big or too small. On the other hand, SMAPE is not symmetric, as it treats under-forecast and over-forecast differently.


Binary Classification: Involves discrete classes and often probabilistic outputs. Metrics like binary cross-entropy are suited for handling probabilities and penalizing incorrect classifications. Binary cross-entropy is used because it penalizes incorrect predictions more heavily and works well with probabilistic outputs.

Forecasting: Involves continuous data and requires metrics that measure the accuracy of predicted values relative to actual values. MAPE and SMAPE provide meaningful interpretations for continuous predictions.

