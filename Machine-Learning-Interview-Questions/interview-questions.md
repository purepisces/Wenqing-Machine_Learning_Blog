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

## 🌟 Question 4
**To start Linear Regression, you would need to make some assumptions. What are those assumptions?**

To start a Linear Regression model, there are some fundamental assumptions that you need to make:
- The model should have a multivariate normal distribution
- There should be no auto-correlation
- Homoscedasiticity, i.e, the dependent variable’s variance should be similar to all of the data
- There should be a linear relationship
- There should be no or almost no multicollinearity present

## 🌟 Question 5
**What is multicollinearity and how will you handle it in your regression model?**

If there is a correlation between the independent variables in a regression model, it is known as multicollinearity. Multicollinearity is an area of concern as independent variables should always be independent. When you fit the model and analyze the findings, a high degree of correlation between variables might present complications.

There are various ways to check and handle the presence of multicollinearity in your regression model. One of them is to calculate the Variance Inflation Factor (VIF). If your model has a VIF of less than 4, there is no need to investigate the presence of multicollinearity. However, if your VIF is more than 4, an investigation is very much required, and if VIF is more than 10, there are serious concerns regarding multicollinearity, and you would need to correct your regression model.

## 🌟 Question 6
**Explain why the performance of XGBoost is better than that of SVM?**

XGBoost is an ensemble approach that employs a large number of trees. This implies that when it repeats itself, it becomes better.

If our data isn't linearly separable, SVM, being a linear separator, will need to use a Kernel to bring it to a point where it can be split. Due to there not being an ideal Kernel for every dataset, this can be limiting.

## Question 7
**Why is an encoder-decoder model used for NLP?**

An encoder-decoder model is used to create an output sequence based on a given input sequence. The final state of the encoder is used as the initial state of the decoder, and this makes the encoder-decoder model extremely powerful. This also allows the decoder to access the information that is taken from the input sequence by the encoder.

## Question 8
**What is cross validation?**

Cross validation is a concept used to evaluate models’ performances to avoid overfitting. It is an easy method to compare the predictive capabilities of models and is best suitable when limited data is available.

## Question 9

**What are the types of Machine learning?**

There are mainly three types of Machine Learning, viz:

Reinforcement learning: It is about taking the best possible action to maximize reward in a particular scenario. It is used by various software and machines to find the best path it should take in a given situation.

Supervised learning: Using labeled datasets to train algorithms to classify data easily for predicting accurate outcomes. Supervised algorithms are those that use labeled data to learn a mapping function from input variables to output variables.

Unsupervised learning: It uses ML to analyze and cluster unlabeled datasets. Unsupervised algorithms learn from unlabeled data and discover hidden patterns and structures in the data.

## Question 10

**What is Selection Bias?**

Selection Bias is a statistical error that brings about a bias in the sampling portion of the experiment. This, in turn, causes more selection of the sampling portion than other groups, which brings about an inaccurate conclusion.

## Question 11

**What is the difference between correlation and causality?**

Correlation is the relation of one action (A) to another action (B) when A does not necessarily lead to B, but Causality is the situation where one action (A) causes a result (B).

## Question 12

**What is the difference between Correlation and Covariance?**

Correlation quantifies the relationship between two random variables with three values: 0,1 and -1.

Covariance is the measure of how two different variables are related and how changes in one impact the other. 

## Question 13

**What are the differences between Type I error and Type II error?**
| Type I               | Type II                  |
|----------------------|--------------------------|
| False positive       | False negative           |
| This states that something has happened when it has not happened | This states that nothing has happened when it has actually happened |

## Question 14

**What is semi-supervised learning?**

A semi-supervised learning happens when a small amount of labeled data is introduced to an algorithm. The algorithm then studies that data and uses it on unlabeled data. Semi-supervised learning combines the efficiency of unsupervised learning and the performance of supervised learning.
w
## Question 15

**Where are semi-supervised learning applied?**

Some areas it is applied include labeling data, fraud detection, and machine translation.

## 🌟 Question 16

**What is the Bayesian Network?**

Bayesian network represents a graphical model between sets of variables. We say it probabilistic because these networks are built on a probability distribution and also use probability theory for prediction and anomaly detection. Bayesian networks are used in for reasoning, diagnostic, anomaly detection, prediction to list a few.

## 🌟 Question 17

**What is another name for a Bayesian Network?**

Casual network, Belief Network, Bayes network, Bayes net, Belief Propagation Network, etc. are some of its other names. It is a probabilistic graphical model that showcases a set of variables and their conditional dependencies.

## Question 18

**What is sensitivity?**

This is the probability that the prediction outcome of the model is true when the value is positive. It can be described as the metric for evaluating a model’s ability to predict the true positives of each available category. Sensitivity = TP / TP+FN (i.e. True Positive/True Positive + False Negative)

## Question 19

**What is specificity?**

This is the probability the prediction of the model is negative when the actual value is negative. It can be termed as the model’s ability to foretell the true negative for each category available.. Specificity = TN / TN + FP (i.e. True Negative/True Negative + False Positive)


## 🌟 Question 20

**What techniques are used to find resemblance in the recommendation system?**

Cosine and Pearson Correlation are techniques used to find resemblance in recommendation systems. Where the Pearson correlation coefficient is the covariance between two vectors divided by their standard deviation, Cosine, on the other hand, is used for measuring the similarity between two vectors.

## 🌟 Question 21

**What does the area under the ROC curve indicate?**

ROC stands for Receiver Operating Characteristic. It measures the usefulness of a test where the larger the area, the more useful the test. These areas are used to compare the effectiveness of the tests. A higher AUC (area under the curve) generally indicates that the model is better at distinguishing between the positive and negative classes. AUC values range from 0 to 1, with a value of 0.5 indicating that the model is no better than random guessing, and a value of 1 indicating perfect classification.


## 🌟 Question 22

**What is clustering?**

Clustering is a process of grouping sets of items into several groups. Items or objects must be similar within the cluster and different from other objects in other clusters. The goal of clustering is to identify patterns and similarities in the data that can be used to gain insights and make predictions. Different clustering algorithms use different methods to group data points based on their features and similarity measures, such as distance or density. Clustering is commonly used in various applications such as customer segmentation, image and text classification, anomaly detection, and recommendation systems.

## 🌟 Question 23

**List the differences between KNN and k-means clustering.**

| KNN                                | K-means clustering                    |
|------------------------------------|---------------------------------------|
| Useful for classification and regression | Used for clustering                    |
| This is a supervised technique     | This is an unsupervised technique     |





## Reference
- https://www.turing.com/interview-questions/machine-learning
- nlp: https://intellipaat.com/blog/interview-question/nlp-interview-questions/
