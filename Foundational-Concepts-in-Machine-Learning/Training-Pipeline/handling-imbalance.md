# Handle imbalance class distribution

In ML use cases like Fraud Detection, Click Prediction, or Spam Detection, it’s common to have imbalance labels. There are few strategies to handle them, i.e, you can use any of these strategies depend on your use case.

- Use class weights in loss function: For example, in a spam detection problem where non-spam data has 95% data compare to other spam data which has only 5%. We want to penalize more in the non-spam class. In this case, we can modify the entropy loss function using weight.
  ```python3
  //w0 is weight for class 0, 
  w1 is weight for class 1
  loss_function = -w0 * ylog(p) - w1*(1-y)*log(1-p)
  ```
### Case 1: Correct Classification (Spam Email)
- **True Label:** Spam (y = 1)
- **Predicted Probability:** p = 0.9 (high confidence for spam)
- **Class Weights:** w_0 = 1.05 (non-spam), w_1 = 20 (spam)

Since the true label is spam and the model predicts a high probability of spam, this is a correct classification.

$$\text{loss} = - w_1 \cdot y \cdot \log(p) - w_0 \cdot (1 - y) \cdot \log(1 - p)$$

$$\text{loss} = - 20 \cdot 1 \cdot \log(0.9) - 1.05 \cdot 0 \cdot \log(0.1)$$

$$\text{loss} = - 20 \cdot \log(0.9)$$

$$\text{loss} \approx - 20 \cdot (-0.105)$$

$$\text{loss} \approx 2.1$$

### Case 2: Misclassification (Spam Email)
- **True Label:** Spam (y = 1)
- **Predicted Probability:** p = 0.3 (low confidence for spam)
- **Class Weights:** w_0 = 1.05 (non-spam), w_1 = 20 (spam)

Since the true label is spam but the model predicts a low probability of spam, this is a misclassification.

$$\text{loss} = - w_1 \cdot y \cdot \log(p) - w_0 \cdot (1 - y) \cdot \log(1 - p)$$

$$\text{loss} = - 20 \cdot 1 \cdot \log(0.3) - 1.05 \cdot 0 \cdot \log(0.7)$$

$$\text{loss} = - 20 \cdot \log(0.3)$$

$$\text{loss} \approx - 20 \cdot (-0.523)$$

$$\text{loss} \approx 10.46$$

### Case 3: Correct Classification (Non-Spam Email)
- **True Label:** Non-Spam (y = 0)
- **Predicted Probability:** p = 0.1 (low confidence for spam)
- **Class Weights:** w_0 = 1.05 (non-spam), w_1 = 20 (spam)

Since the true label is non-spam and the model predicts a low probability of spam, this is a correct classification.

$$\text{loss} = - w_1 \cdot y \cdot \log(p) - w_0 \cdot (1 - y) \cdot \log(1 - p)$$

$$\text{loss} = - 20 \cdot 0 \cdot \log(0.1) - 1.05 \cdot 1 \cdot \log(0.9)$$

$$\text{loss} = - 1.05 \cdot \log(0.9)$$

$$\text{loss} \approx - 1.05 \cdot (-0.105)$$

$$\text{loss} \approx 0.11$$

### Case 4: Misclassification (Non-Spam Email)
- **True Label:** Non-Spam (y = 0)
- **Predicted Probability:** p = 0.8 (high confidence for spam)
- **Class Weights:** w_0 = 1.05 (non-spam), w_1 = 20 (spam)

Since the true label is non-spam but the model predicts a high probability of spam, this is a misclassification.

$$\text{loss} = - w_1 \cdot y \cdot \log(p) - w_0 \cdot (1 - y) \cdot \log(1 - p)$$

$$\text{loss} = - 20 \cdot 0 \cdot \log(0.8) - 1.05 \cdot 1 \cdot \log(0.2)$$

$$\text{loss} = - 1.05 \cdot \log(0.2)$$

$$\text{loss} \approx - 1.05 \cdot (-0.699)$$

$$\text{loss} \approx 0.73$$

### Summary
In these examples, you can see:

- When the model correctly classifies an email, the loss is relatively low.
- When the model misclassifies an email, the loss is higher, especially for the spam class due to the higher weight.

The increased loss for misclassifications during training encourages the model to adjust its parameters to reduce these high losses in future iterations, leading to better overall performance, particularly for the minority class (spam).
