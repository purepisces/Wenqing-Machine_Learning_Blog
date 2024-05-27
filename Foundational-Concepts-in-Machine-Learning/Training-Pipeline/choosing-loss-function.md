It depends on the use case when deciding which loss function to use. For binary classification, the most popular is cross-entropy. In the Click Through Rate (CTR) prediction, Facebook uses Normalized Cross Entropy loss (a.k.a. logloss) to make the loss less sensitive to the background conversion rate. 

In a forecast problem, the most common metrics are the Mean Absolute Percentage Error (MAPE) and the Symmetric Absolute Percentage Error (SMAPE). For MAPE, you need to pay attention to whether or not your target value is skew, i.e., either too big or too small. On the other hand, SMAPE is not symmetric, as it treats under-forecast and over-forecast differently.

