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

For each $k$ value, we train on 4 of the 5 folds and validate on the remaining fold. This process is repeated 5 times (once for each fold), and the average accuracy across folds is recorded. The plot would show accuracy on the y-axis and $k$ values on the x-axis, with error bars representing the standard deviation. The peak of the trend line indicates the optimal value of $k$. In this case, cross-validation indicates that a value of approximately $k = 7$ performs best for this dataset, as shown by the peak in the plot. Increasing the number of folds typically makes the results to a smoother, less noisy curve.

<img src="Cross-validation plot.png" alt="Cross-validation plot" width="500" height="400"/>

### Practical Considerations

Cross-validation is often computationally expensive, so in practice, many prefer to use a single validation set rather than multiple folds. The training/validation split commonly ranges from 50%-90% of data for training, with the remaining data for validation. However, this depends on multiple factors: For example if the number of hyperparameters is large you may prefer to use bigger validation splits. If you’re working with a small dataset, cross-validation is generally safer for ensuring robust performance estimates. Common choices for cross-validation include 3-fold, 5-fold, or 10-fold.

<img src="data splits.jpeg" alt="data splits" width="800" height="200"/>


> **Typical data splits**: First, a training and test set are defined. The training set is then divided into multiple folds (for example, 5 folds). Folds 1-4 are used as the training set, while one fold (e.g., fold 5, highlighted in yellow) serves as the validation fold for tuning hyperparameters. Cross-validation further extends this by rotating through each fold, allowing each one to serve as the validation fold in turn, a process known as 5-fold cross-validation. Once the model is fully trained and the optimal hyperparameters are selected, it is evaluated a single time on the test set (shown in red) for final performance measurement.

### Pros and Cons of the Nearest Neighbor Classifier

One advantage of the Nearest Neighbor (NN) classifier is its simplicity and ease of implementation. Additionally, this method requires no training time. The classifier only needs to store and possibly index the training data, making it ready for predictions immediately. However, this simplicity comes with a drawback: the **computational cost is high at test time**. To classify a new test example, the NN classifier must compare it to every single training example, which can be slow and inefficient, especially for large datasets. This is generally undesirable in practical applications, where test-time efficiency is often more critical than training-time efficiency.

In contrast, methods like deep neural networks shift this trade-off, with high computational costs during training but very low costs at test time. This is more practical since once a neural network is trained, it can classify new examples quickly and efficiently.

**Approximate Nearest Neighbor (ANN)** algorithms and libraries (such as [FLANN](https://github.com/mariusmuja/flann)) have been developed to address this issue. These algorithms enable a trade-off between the accuracy of nearest neighbor retrieval and the space/time complexity of the retrieval process. They typically rely on a preprocessing or indexing step, such as building a k-d tree or running the k-means algorithm.

The Nearest Neighbor classifier can sometimes be effective, particularly with low-dimensional data, but it struggles with high-dimensional data like images. A major issue is that images are composed of many pixels, making them high-dimensional objects. In such cases, **pixel-based distance measures (e.g., L2 distance) become unintuitive and are very different from perceptual similarities**.

For example, consider the image below:

<img src="pixel-based distance.png" alt="pixel-based distance" width="800" height="200"/>


Here, we have an original image of a face on the left, followed by three altered versions. Each altered image is at the same L2 pixel distance from the original, meaning they are "equally far" away in terms of raw pixel values. However, perceptually, they appear very different:

- The **shifted** version is a slight spatial displacement, which doesn’t change much visually but can result in a large pixel distance.
- The **messed up** version has parts of the face blocked out, making it visually quite different from the original, though it’s at the same L2 distance.
- The **darkened** version preserves the structure and features of the face but is darker, also showing the same L2 distance.

This demonstrates how pixel-based distances can fail to capture meaningful, perceptual similarity in high-dimensional data like images. The distances don’t align with our intuitive sense of similarity. Clearly, the pixel-wise distance does not correspond at all to perceptual or semantic similarity.

### Pixel-Based Distance Limitations Visualized with t-SNE

To further illustrate this limitation, we can use **t-SNE**, a visualization technique to take the CIFAR-10 images and embed them in two dimensions so that their (local) pairwise distances are best preserved. Below is a t-SNE visualization of CIFAR-10 images, where the images that appear closer together are similar in terms of L2 pixel distance:

<img src="t-SNE-pixels_embed_cifar10.jpg" alt="t-SNE-pixels_embed_cifar10" width="800" height="200"/>

In this visualization, you’ll notice that images with similar backgrounds or color distributions tend to be grouped together, even if they belong to different categories. For example, a dog on a white background might appear near a frog on a white background, despite them being different objects. This clustering happens because L2 distance, based on pixel values alone, is influenced more by color and background similarity than by the actual semantic content of the images.

Ideally, we would like all images within a particular category (e.g., all dogs, all cats) to form distinct clusters based on their content, not on irrelevant features like background or color. However, achieving this kind of clustering requires going beyond simple pixel comparisons. Advanced methods, such as convolutional neural networks (CNNs), learn feature representations that capture the actual content of images, making them more effective for image classification.

### Summary

To summarize:
-   We introduced the challenge of **Image Classification**, where the goal is to categorize a set of labeled images and predict the categories of new, unlabeled test images, then assess the accuracy of those predictions.
-   We explored a basic approach using the **Nearest Neighbor classifier**. This method involves several hyperparameters (like the choice of $k$ or the distance metric) without a straightforward way to determine their optimal values.
-   To set these hyperparameters properly, we split our training data into a training set and a **validation set** (acting as a "fake test set"). By testing different hyperparameter values on this validation set, we identify the ones that yield the best performance.
-   When data is limited, we discussed **cross-validation** as a technique to improve the reliability of hyperparameter estimates by reducing noise.
-   After identifying the best hyperparameters, we fix them and conduct a single **evaluation** on the actual test set.
-   Using Nearest Neighbor on the CIFAR-10 dataset, we achieve approximately 40% accuracy. While the method is simple and easy to implement, it requires storing the entire training set and is computationally expensive at test time.
-   We noted that using L1 or L2 distances on raw pixel data is not adequate, as these distances tend to capture background and color patterns rather than the true semantic content of images.

In upcoming lectures, we’ll explore methods to address these limitations, ultimately achieving higher accuracies (up to 90%), eliminating the need to store the training set after learning, and enabling test image evaluation in under a millisecond.


### Summary: Applying kNN in practice

When applying the k-Nearest Neighbors (kNN) algorithm in practice (hopefully not on images, or perhaps as only a baseline), follow these steps:

1.  **Data Preprocessing:** Normalize your features to achieve zero mean and unit variance. For image data, normalization is less critical since pixel distributions are typically uniform. However, we will explore data normalization in detail in later sections.
    
2.  **Dimensionality Reduction (if needed):** For very high-dimensional data, consider using techniques like Principal Component Analysis (PCA), Neighborhood Components Analysis (NCA), or Random Projections to reduce dimensionality before applying kNN.
    
3.  **Train-Validation Split:** Divide your training data randomly into training and validation sets, typically assigning 70-90% to training. If there are many hyperparameters, a larger validation set can provide better estimation.  If you are concerned about the size of your validation data, for a more robust evaluation, consider cross-validation with multiple folds if resources allow (the more folds the better, but more expensive).
    
4.  **Hyperparameter Tuning:** Train and assess the kNN classifier on the validation data (for all folds, if doing cross-validation) across different values of **k** and various distance metrics, such as L1 or L2 norms.
    
5.  **Efficient Retrieval (if needed):** If kNN is computationally intensive, use an Approximate Nearest Neighbor library (e.g., FLANN) to accelerate neighbor retrieval at a slight cost to accuracy.
    
6.  **Final Model Selection:** Take note of the hyperparameters that gave the best results. There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be burned on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.

## Reference:
- https://cs231n.github.io/classification/
