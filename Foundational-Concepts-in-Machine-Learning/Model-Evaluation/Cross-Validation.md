# Cross-Validation

In machine learning, tuning hyperparameters like the number of neighbors in k-Nearest Neighbor (k-NN) is crucial for a model’s performance. **Cross-validation** provides a reliable way to tune these hyperparameters without overfitting to the test set.

### Why Cross-Validation?

Hyperparameter tuning typically requires a validation set to test different configurations. However, using a single validation set may yield unreliable results, especially with limited data. Cross-validation addresses this by creating multiple validation sets, allowing the model to validate on different portions of the data, leading to more robust performance estimates.

### How Cross-Validation Works

In **k-fold cross-validation**, we split the training data into _k_ equal parts, called **folds**. The model is trained on _k - 1_ folds and validated on the remaining fold. This process is repeated _k_ times, so each fold serves as the validation set once. The final performance is the average across all folds, giving a more stable estimate.

For example, in 5-fold cross-validation:

1.  Divide data into 5 folds.
2.  Train on 4 folds and validate on the 5th fold.
3.  Repeat for all 5 folds and average the validation performance.

This averaging reduces the noise from a single split and provides a more reliable performance metric.

> **Tip:** The choice of _k_ (e.g., 5 or 10 folds) balances computational cost and reliability. Fewer folds (e.g., 3) reduce computational cost, while more folds (e.g., 10) yield smoother estimates at the expense of higher computation.

### Choosing the Best k Value in k-NN with Cross-Validation

Cross-validation can help determine the optimal number of neighbors _k_ in k-NN:

-   Try different values of _k_ (e.g., 1, 3, 5, etc.).
-   For each value of _k_, run cross-validation to evaluate the model’s accuracy.
-   Plot the results to identify the _k_ that maximizes performance, ensuring better generalization.

#### Example of 5-Fold Cross-Validation

For each $k$ value, we train on 4 of the 5 folds and validate on the remaining fold. This process is repeated 5 times (once for each fold), and the average accuracy across folds is recorded. The plot would show accuracy on the y-axis and $k$ values on the x-axis, with error bars representing the standard deviation. The peak of the trend line indicates the optimal value of $k$. In this case, cross-validation indicates that a value of approximately $k = 7$ performs best for this dataset, as shown by the peak in the plot. Increasing the number of folds typically makes the results to a smoother, less noisy curve.

<img src="Cross-validation plot.png" alt="Cross-validation plot" width="500" height="400"/>

**Example Code:**
```python
# Assume we have Xtr_rows, Ytr as our full training data
# Xtr_rows: input data, Ytr: labels

import numpy as np
num_folds = 5
fold_size = len(Xtr_rows) // num_folds

validation_accuracies = []

for k in [1, 3, 5, 10, 20, 50, 100]:
    accuracies = []
    
    for fold in range(num_folds):
        # Split data into training and validation sets for each fold
        Xval_rows = Xtr_rows[fold * fold_size: (fold + 1) * fold_size]
        Yval = Ytr[fold * fold_size: (fold + 1) * fold_size]
        
        # Remaining data as training set
        Xtrain_rows = np.concatenate((Xtr_rows[:fold * fold_size], Xtr_rows[(fold + 1) * fold_size:]), axis=0)
        Ytrain = np.concatenate((Ytr[:fold * fold_size], Ytr[(fold + 1) * fold_size:]), axis=0)
        
        # Train the k-NN model and evaluate on validation fold
        nn = NearestNeighbor()  # Assuming we have a NearestNeighbor class
        nn.train(Xtrain_rows, Ytrain)
        Yval_predict = nn.predict(Xval_rows, k=k)  # Assuming predict accepts a k parameter
        acc = np.mean(Yval_predict == Yval)
        print(f'Fold {fold + 1} accuracy for k={k}: {acc}')
        
        accuracies.append(acc)
    
    # Store average accuracy across folds for each k
    avg_accuracy = np.mean(accuracies)
    validation_accuracies.append((k, avg_accuracy))
    print(f'Average accuracy for k={k}: {avg_accuracy}')
```
This code snippet iterates over different _k_ values, evaluates each on a 5-fold cross-validation scheme, and stores the average accuracy for each _k_.

### Practical Considerations

Cross-validation is often computationally expensive, so in practice, many prefer to use a single validation set rather than multiple folds. The training/validation split commonly ranges from 50%-90% of data for training, with the remaining data for validation. However, this depends on multiple factors: For example if the number of hyperparameters is large you may prefer to use bigger validation splits. If you’re working with a small dataset, cross-validation is generally safer for ensuring robust performance estimates. Common choices for cross-validation include 3-fold, 5-fold, or 10-fold.

-   **3-fold**: Useful for very limited data but can be less stable.
-   **5-fold**: Provides a balance between stability and computational efficiency.
-   **10-fold**: Offers smoother estimates and is often used for final model evaluation, though it requires more computation.

<img src="data splits.jpeg" alt="data splits" width="800" height="200"/>

> **Typical data splits**: First, a training and test set are defined. The training set is then divided into multiple folds (for example, 5 folds). Folds 1-4 are used as the training set, while one fold (e.g., fold 5, highlighted in yellow) serves as the validation fold for tuning hyperparameters. Cross-validation further extends this by rotating through each fold, allowing each one to serve as the validation fold in turn, a process known as 5-fold cross-validation. Once the model is fully trained and the optimal hyperparameters are selected, it is evaluated a single time on the test set (shown in red) for final performance measurement.

### Summary

-   **Cross-validation** provides a robust method for tuning hyperparameters and assessing model performance.
-   It’s particularly useful in **limited data scenarios**, where a single validation set might lead to unreliable estimates.
-   **Average performance across folds** helps select optimal hyperparameters, leading to models that generalize better to unseen data.

By incorporating cross-validation, you can confidently assess and tune models for robust, reliable performance.
