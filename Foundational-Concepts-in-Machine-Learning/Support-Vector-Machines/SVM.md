# Support Vector Machines (SVM)

Support Vector Machines (SVM) are a powerful tool in supervised learning, particularly for classification tasks. They aim to find a decision boundary that separates data points from different classes with the maximum margin. This section explores SVM concepts, loss functions, and the max-margin principle.


## **Multiclass Support Vector Machine Loss**

In machine learning, a commonly used loss function for SVMs is the **Multiclass Support Vector Machine (SVM) loss**. This loss encourages the correct class score to exceed all incorrect class scores by at least a fixed margin, denoted as $\Delta$.

### **Definition**

The Multiclass SVM loss for the $i$-th example is computed as:

$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

Where:

-   $s_j$ = predicted score for the $j$-th class.
-   $s_{y_i}$ = predicted score for the correct class $y_i$.
-   $\Delta$ = margin (commonly set to 1).

### **Example Calculation**

Suppose we have three classes with scores $s = [13, -7, 11]$, and the true class is the first one ($y_i = 0$). With $\Delta = 10$, the loss is calculated as:

$$L_i = \max(0, -7 - 13 + 10) + \max(0, 11 - 13 + 10)$$

The first term evaluates to zero because it is negative, while the second term results in 8. Thus, the total loss is 8.

### **Hinge Loss and Squared Hinge Loss**

The $\max(0, -)$ function is called **hinge loss**. An alternative, **squared hinge loss**, penalizes margin violations quadratically:

-   Hinge loss: $\max(0, x)$
-   Squared hinge loss: $\max(0, x)^2$

Both hinge loss types guide the model to improve its classification boundaries.


## **The Max-Margin Principle**

In SVMs, the goal is to find a hyperplane that maximizes the **margin**—the distance between the closest points from each class to the hyperplane.

### **Key Concepts**

1.  **Decision Boundary and Margins**  
    The decision boundary is defined as:  
    $w \cdot x + b = 0$
    Two marginal hyperplanes are added for the positive and negative classes:
    
    -   $w \cdot x + b = +1$ (positive margin)
    -   $w \cdot x + b = -1$ (negative margin)
2.  **Support Vectors**  
    Points on the marginal hyperplanes are called **support vectors**. These define the margin and satisfy the constraint:  
    $w \cdot x_i + b = \pm 1$
    
3.  **Maximizing the Margin**  
    The margin width is $\frac{2}{|w|}$, so maximizing the margin is equivalent to minimizing $|w|$.
    > $d = \frac{|c_1 - c_2|}{\|w\|} =  \frac{|1 - (-1)|}{\|w\|} = \frac{2}{\|w\|}$

    

## **Regularization and the Full Loss**

Regularization resolves ambiguity in weight selection and improves **generalization** by discouraging overly large weights. The most common regularization method is the **L2 penalty**, defined as:

$$R(W) = \sum_k \sum_l W_{k,l}^2$$

The full SVM loss is:

$$L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2$$

Where:

-   $N$ = number of training examples.
-   $\lambda$ = regularization strength.


## **Comparison: SVM vs. Softmax**

SVM and Softmax classifiers differ in their loss functions and optimization goals:

-   **SVM**: Focuses on margins between classes. Loss is zero once the margin is met.
-   **Softmax**: Assigns probabilities to classes and continuously optimizes the likelihood of the correct class.

Example:  
For scores $[10, -2, 3]$ with $y_i = 0$:

-   SVM enforces the margin condition ($10 > -2 + \Delta$, $10 > 3 + \Delta$).
-   Softmax encourages increasing the probability for the correct class.


## **Advantages of SVM**

-   SVMs focus on the most challenging points (support vectors), making them robust for high-dimensional spaces.
-   The use of regularization prevents overfitting and improves generalization.


## **Summary**

-   SVMs aim to separate classes with the maximum margin while minimizing a loss function.
-   The **Multiclass SVM Loss** ensures that correct class scores exceed incorrect class scores by at least $\Delta$.
-   Regularization enhances model generalization, balancing data loss and weight penalties.
-   The SVM framework is efficient for both linear and kernelized classifiers.
