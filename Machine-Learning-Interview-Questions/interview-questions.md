## Question 1

**You have come across some missing data in your dataset. How will you handle it?**

**Answer:**

In order to handle some missing or corrupted data, the easiest way is to just replace the corresponding rows and columns, which contain the incorrect data, with some different values. The two most useful functions in Panda for this purpose are `isnull()` and `fillna()`.

- `isnull()`: is used to find missing values in a dataset.
- `fillna()`: is used to fill missing values with 0’s.

## Question 2

**Explain Decision Tree Classification**

A decision tree uses a tree structure to generate any regression or classification models. While the decision tree is developed, the datasets are split up into ever-smaller subsets in a tree-like manner with branches and nodes. Decision trees can handle both categorical and numerical data.

## Question 3

**How is a logistic regression model evaluated?**

One of the best ways to evaluate a logistic regression model is to use a confusion matrix, which is a very specific table that is used to measure the overall performance of any algorithm.

Using a confusion matrix, you can easily calculate the Accuracy Score, Precision, Recall, and F1 score. These can be extremely good indicators for your logistic regression model.

If the recall of your model is low, then it means that your model has too many False Negatives. Similarly, if the precision of your model is low, it signifies that your model has too many False Positives. In order to select a model with a balanced precision and recall score, the F1 Score must be used.

## Reference
- https://www.turing.com/interview-questions/machine-learning
- nlp: https://intellipaat.com/blog/interview-question/nlp-interview-questions/
