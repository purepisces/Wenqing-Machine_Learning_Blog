Before reading it, go through the content in [nearest-neighbor-classifier](nearest-neighbor-classifier-overview.md).

# k-Nearest Neighbor Classifier

Relying on just the nearest data point to make a prediction may not always yield the best results. A more effective approach is often the **k-Nearest Neighbor (k-NN) Classifier**. In this method, instead of finding only the closest point in the training set, we find the **k closest points** and use them to "vote" on the label for the test point. When *k = 1*, this approach simplifies to the standard Nearest Neighbor classifier. However, higher values of **k** provide a smoothing effect that makes the classifier less sensitive to outliers.

### Example of Decision Boundaries in k-NN

The following example shows the difference between a 1-NN (Nearest Neighbor) and a 5-NN classifier. Here, 2-dimensional points are assigned to one of three classes (red, blue, or green). The colored regions represent the **decision boundaries** created by the classifier using L2 distance. White regions indicate ambiguous classifications, where class votes are tied.

<img src="5-NN classifier.jpeg" alt="5-NN classifier" width="800" height="250"/>

Notice that with a 1-NN classifier, outlier points (like a green point surrounded by blue points) can create isolated "islands" of potentially incorrect classifications. In contrast, the 5-NN classifier smooths over these irregularities, leading to better **generalization** on unseen test data (not shown here). The gray regions in the 5-NN example arise from tied votes among neighbors (e.g., 2 red, 2 blue, and 1 green neighbor).

### Choosing the Right k Value

In practice, k-NN is almost always preferred over simple nearest neighbor classification. However, selecting the optimal value for *k* is crucial, as it affects the balance between sensitivity to noise (low *k*) and generalization ability (high *k*). We'll explore this selection process in the next section.

## Reference:
- https://cs231n.github.io/classification/
