# Logistic Regression

## Introduction

Logistic regression is a statistical method for analyzing datasets in which there are one or more independent variables that determine an outcome. It estimates the probability that a given input point belongs to a particular class. Unlike linear regression which outputs continuous values, logistic regression outputs probabilities bounded between 0 and 1.

## Key Concepts

### Logistic Function
The core of logistic regression is the logistic function (also known as the sigmoid function), which maps any real-valued number into the range [0, 1]:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Here, $z$ is a linear combination of the input features. For binary classification, the logistic function outputs a probability that the given input point belongs to the positive class.

### Decision Boundary
The decision boundary is the threshold at which we classify the output. For binary classification, a common threshold is 0.5:
- If the output probability $\geq 0.5$, classify as class 1.
- If the output probability $< 0.5$, classify as class 0.

## Model Representation

### Hypothesis
In logistic regression, the hypothesis is defined as:

$$h_\theta(x) = \sigma(\theta^T x)$$

where $\theta$ is the vector of weights, $x$ is the input feature vector, and $\sigma$ is the sigmoid function.

### Cost Function
The cost function for logistic regression is the log-loss (also known as binary cross-entropy):

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

where $m$ is the number of training examples, $y$ is the actual label, and $h_\theta(x)$ is the predicted probability.

### Optimization
The goal is to find the parameters $\theta$ that minimize the cost function $J(\theta)$. This is typically done using optimization algorithms such as gradient descent.

## Sparse Features

In many real-world applications, the feature space is sparse, meaning that most feature values are zero. Logistic regression is particularly effective in these scenarios because:
- It can efficiently handle high-dimensional data.
- The model complexity is linear in the number of features, making it computationally feasible.

## Distributed Training

### Why Distributed Training?
With large volumes of data, training a model on a single machine can be slow and impractical. Distributed training allows for parallel processing, making it possible to handle large datasets efficiently.

### Apache Spark
Apache Spark is a powerful tool for distributed data processing. It can be used to train logistic regression models in parallel across a cluster of machines.

### Alternating Direction Method of Multipliers (ADMM)
ADMM is an optimization technique that decomposes the problem into smaller subproblems that can be solved in parallel. This is particularly useful for distributed environments where coordination between machines is necessary.

## Practical Considerations

### Handling Imbalanced Data
In binary classification problems, the classes are often imbalanced. For instance, in click-through rate (CTR) prediction, the number of clicks (positive class) is usually much smaller than the number of non-clicks (negative class). To address this imbalance:
- **Resampling**: Over-sample the minority class or under-sample the majority class in the training data.
- **Class Weights**: Assign higher weights to the minority class during training.

### Validation and Testing
It is crucial to keep the validation and test sets intact (i.e., not resampled) to ensure that performance metrics accurately reflect the model's effectiveness in real-world scenarios.

## Conclusion

Logistic regression is a robust and efficient method for binary classification, particularly effective with sparse features and large datasets. With the ability to leverage distributed training frameworks like Apache Spark and optimization techniques like ADMM, logistic regression can scale to handle the demands of modern data science applications.


# Logistic Regression and Log-Odds

## Probability and Odds

- **Probability (P)**: The likelihood that an event occurs.
- **Odds**: The ratio of the probability that the event occurs to the probability that it does not occur.

  $$\text{Odds} = \frac{P}{1 - P}$$
  

## Log-Odds (Logit)

- The log-odds is the natural logarithm of the odds.
  
  $$\text{Log-Odds} = \log \left( \frac{P}{1 - P} \right)$$

## Logistic Regression Model

- In logistic regression, the log-odds of the probability of the event occurring (e.g., the probability of a click) is modeled as a linear combination of the input features.
- The model equation is:
  
  $$\log \left( \frac{P}{1 - P} \right) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$$
  
- Here, $$\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$$ is the linear combination of features, often denoted as $z$.

## Connecting \( z \) to Log-Odds

- The term $$z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$$ represents the log-odds in the logistic regression model. Although $z$ itself does not contain a logarithm, it is the argument of the logistic function, which maps it to a probability.

