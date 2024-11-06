Before reading it, go through the content in [nearest-neighbor-classifier](nearest-neighbor-classifier-overview.md).

# k-Nearest Neighbor Classifier

Relying on just the nearest data point to make a prediction may not always yield the best results. A more effective approach is often the **k-Nearest Neighbor (k-NN) Classifier**. In this method, instead of finding only the closest point in the training set, we find the **k closest points** and use them to "vote" on the label for the test point. When *k = 1*, this approach simplifies to the standard Nearest Neighbor classifier. However, higher values of **k** provide a smoothing effect that makes the classifier less sensitive to outliers.

### Example of Decision Boundaries in k-NN

The following example shows the difference between a 1-NN (Nearest Neighbor) and a 5-NN classifier. Here, 2-dimensional points are assigned to one of three classes (red, blue, or green). The colored regions represent the **decision boundaries** created by the classifier using L2 distance. White regions indicate ambiguous classifications, where class votes are tied.

<img src="5-NN classifier.jpeg" alt="5-NN classifier" width="800" height="250"/>

Notice that with a 1-NN classifier, outlier points (like a green point surrounded by blue points) can create isolated "islands" of potentially incorrect classifications. In contrast, the 5-NN classifier smooths over these irregularities, leading to better **generalization** on unseen test data (not shown here). The gray regions in the 5-NN example arise from tied votes among neighbors (e.g., 2 red, 2 blue, and 1 green neighbor).

### Choosing the Right k Value

In practice, k-NN is almost always preferred over simple nearest neighbor classification. However, selecting the optimal value for *k* is crucial, as it affects the balance between sensitivity to noise (low *k*) and generalization ability (high *k*). We'll explore this selection process in the next section.

### Validation Sets for Hyperparameter Tuning

In k-nearest neighbor classification, we need to set the value of *k*, but what value works best? Additionally, we can choose from several distance metrics, such as the L1 or L2 norm, or even others like dot products. These choices are known as **hyperparameters**, which are settings we need to decide before training the model. Many machine learning algorithms rely on hyperparameters, but selecting the best values isn't always straightforward.

A common approach is to try different values and observe what performs best. However, this must be done carefully. Specifically, **the test set should never be used to adjust hyperparameters**. The test set is meant to provide an unbiased evaluation of the model's final performance, so it should only be used once, at the very end. Tuning hyperparameters on the test set risks overfitting, where the model performs well on that specific test set but may not generalize to new data. By treating the test set as a one-time resource, we can ensure it accurately reflects the model’s ability to generalize.

> **Only evaluate on the test set once, at the end of training.**

To tune hyperparameters without touching the test set, we split our training set into two parts: a smaller **training set** and a **validation set**. For example, in the CIFAR-10 dataset, we might reserve 1,000 images from the 50,000 training images as a validation set, using the remaining 49,000 for training. This validation set serves as a "fake test set" to help us choose the best hyperparameters.

#### Example Code for Hyperparameter Tuning with Validation Set

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

After running this code, we can plot the results to see which value of _k_ performs best on the validation set. Once the optimal value is found, we lock in that choice and evaluate the model on the actual test set only once.

> **Tip**: Always split your training data into a training set and a validation set to tune hyperparameters, then evaluate on the test set a single time at the end.

### Explanation of the Hyperparameter Tuning Process

The code snippet above is part of the **hyperparameter tuning process**, which occurs **before the final model is trained and evaluated on the test set**.

In this code, we're experimenting with different values of $k$ (the number of neighbors in the k-Nearest Neighbor classifier) to see which value works best on the validation set. This helps us determine the optimal $k$ value that will likely generalize well when we evaluate the model on the test set.

#### Step-by-Step Workflow

1.  **Split Data**:
    
    -   We start by splitting the original training set into a smaller training set (49,000 examples) and a validation set (1,000 examples).
    -   The validation set acts as a "proxy test set" to help tune hyperparameters without touching the actual test set.
2.  **Hyperparameter Tuning (Before Final Training)**:
    
    -   The code iterates over different values of $k$ (1, 3, 5, etc.).
    -   For each kkk value, it trains the Nearest Neighbor model on the smaller training set and evaluates its performance on the validation set.
    -   It calculates the accuracy for each kkk on the validation set and stores the results in `validation_accuracies`.
3.  **Selecting the Best Hyperparameter**:
    
    -   Once all values of $k$ have been evaluated, we analyze `validation_accuracies` to select the best kkk based on the highest validation accuracy.
4.  **Final Training and Evaluation**:
    
    -   After determining the optimal $k$, we train the model on the full training set (all 50,000 examples) using this selected $k$ value.
    -   Finally, we evaluate the model only once on the test set to report the final performance.

This code snippet is part of the **hyperparameter tuning phase**, a preliminary step before final model training and testing.

### Cross-Validation

If the training data is limited, a single validation set might not give a reliable estimate. In such cases, **cross-validation** provides a more robust approach. In cross-validation, instead of creating a single validation set, we divide the training data into multiple **folds**. For example, in 5-fold cross-validation, we split the training data into 5 equal parts:

1.  Train on 4 parts and validate on the remaining part.
2.  Repeat this process so that each part serves as the validation set once.
3.  Calculate the average performance across all folds to obtain a more stable estimate.

#### Example of 5-Fold Cross-Validation

For each _k_ value, we train on 4 of the 5 folds and validate on the remaining fold. This process is repeated 5 times (once for each fold), and the average accuracy across folds is recorded. The plot would show accuracy on the y-axis and _k_ values on the x-axis, with error bars representing the standard deviation. The peak of the trend line indicates the optimal value of _k_. Increasing the number of folds typically makes the results less noisy.

### Practical Considerations

Cross-validation is often computationally expensive, so in practice, many prefer to use a single validation set rather than multiple folds. The training/validation split commonly ranges from 50%-90% of data for training, with the remaining data for validation. When tuning multiple hyperparameters, a larger single validation split (e.g., 30%-40%) is usually preferred to balance computational efficiency and reliable validation. However, if you’re working with a small dataset, cross-validation is generally safer for ensuring robust performance estimates. Common choices for cross-validation include 3-fold, 5-fold, or 10-fold.

> **Typical data splits**: With a training and test set provided, split the training data into folds. Use one fold for validation and the others for training. In the end, after tuning hyperparameters, evaluate once on the test set to measure final performance.


## Reference:
- https://cs231n.github.io/classification/
