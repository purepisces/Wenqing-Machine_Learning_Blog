# Softmax Classifier

The Softmax classifier is a popular approach for multiclass classification, extending the principles of logistic regression to multiple classes. Unlike SVM, which produces uncalibrated scores, the Softmax classifier provides normalized probabilities for each class, enabling a probabilistic interpretation.

## **Softmax Classifier Loss Function**

In the Softmax classifier:

1.  **Score Calculation**:  
    Scores are computed as:  
    $f(x_i; W) = W x_i$ 
    These scores, $f$, are interpreted as **unnormalized log probabilities** for each class.
    
2.  **Loss Definition**:  
    The Softmax classifier uses the **cross-entropy loss** to encourage the correct class to dominate the probability distribution:  
    $$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \quad \text{or equivalently:} \quad L_i = -f_{y_i} + \log\sum_j e^{f_j}$$
    Where:
    
    -   $f_{y_i}$ is the score for the correct class.
    -   $f_j$ is the score for each class in the vector of scores $f$.
3.  **Full Dataset Loss**:  
    The total loss is the mean of $L_i$ across all examples, combined with a regularization term $R(W)$:  
    $L = \frac{1}{N} \sum_i L_i + \lambda R(W)$
    


## **The Softmax Function**

The Softmax function is defined as:  
fj(z)=ezj∑kezkf_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}fj​(z)=∑k​ezk​ezj​​  
It converts raw scores into probabilities between 0 and 1, ensuring they sum to 1. Each class probability reflects the model's confidence for that class.

### **Probabilistic Interpretation**

The probability of the correct class $y_i$ given the input $x_i$ is:  
$$P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}$$

By minimizing the **negative log likelihood**, the classifier adjusts its parameters to maximize the probability assigned to the correct class. This approach can also be seen as **Maximum Likelihood Estimation (MLE)**.


## **Information-Theoretic View**

The Softmax classifier minimizes the **cross-entropy** between:

-   $p$: the true distribution (placing all probability on the correct class).
-   $q$: the predicted probabilities.

The cross-entropy is:  
$$H(p, q) = - \sum_x p(x) \log q(x)$$
This process is equivalent to minimizing the Kullback-Leibler (KL) divergence between $p$ and $q$.

## **Practical Considerations**

### **Numerical Stability**

Direct computation of the Softmax function can lead to numerical instability due to the exponential terms. A common trick is to shift the scores by their maximum value:  
$f \gets f - \max(f)$

For example, in Python:

```python
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
```

This adjustment ensures that all values remain manageable during the computation.


## **Comparison: Softmax vs. SVM**

1.  **Objective**:
    
    -   SVM focuses on separating classes by a margin.
    -   Softmax maximizes the likelihood of the correct class.
2.  **Output**:
    
    -   SVM produces scores that are uncalibrated and harder to interpret.
    -   Softmax provides normalized probabilities, offering more interpretability.
3.  **Behavior**:
    
    -   SVM stops optimizing once the margin condition is met.
    -   Softmax continually adjusts probabilities to maximize the correct class.
4.  **Example**:  
    For scores $[10, -2, 3]$ with $y_i = 0$:
    
    -   SVM ensures the margin condition is satisfied.
    -   Softmax adjusts probabilities to emphasize the correct class.


## **Regularization in Softmax Classifier**

Regularization is applied to the weights $W$ to improve generalization and prevent overfitting. The most common regularization term is the L2 norm:

$$R(W) = \sum_{k,l} W_{k,l}^2$$

The complete loss function, including regularization, is:  
$$L = \frac{1}{N} \sum_i \left( -f_{y_i} + \log\sum_j e^{f_j} \right) + \lambda \sum_{k,l} W_{k,l}^2$$


## **Summary**

-   **Score Function**: Maps input features to unnormalized class scores.
-   **Softmax Function**: Converts scores to normalized probabilities.
-   **Loss Function**: Uses cross-entropy to encourage correct classifications.
-   **Probabilistic View**: Interprets scores as relative confidence for each class.
-   **Numerical Stability**: Achieved through normalization tricks during computation.

The Softmax classifier provides a probabilistic approach to classification, balancing accuracy and interpretability.
