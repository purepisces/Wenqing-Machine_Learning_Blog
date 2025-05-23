**Before Reading This:**
Please refer to the following resources for background knowledge:
1. [CS231n Classification Overview](https://cs231n.github.io/classification/)
2. Wenqing’s Machine Learning Blog:
   - [Nearest Neighbor Classifier Overview](https://github.com/purepisces/Wenqing-Machine_Learning_Blog/blob/main/Foundational-Concepts-in-Machine-Learning/Nearest-Neighbor-Classifiers/nearest-neighbor-classifier-overview.md)
   - [k-Nearest Neighbor Classifier Overview](https://github.com/purepisces/Wenqing-Machine_Learning_Blog/blob/main/Foundational-Concepts-in-Machine-Learning/Nearest-Neighbor-Classifiers/knn-overview.md)

Table of Contents:

- [Linear Classification](#linear-classification)
  - [Parameterized mapping from images to label scores](#parameterized-mapping-from-images-to-label-scores)
  - [Interpreting a linear classifier](#interpreting-a-linear-classifier)
  - [Loss function](#loss-function)
    - [Multiclass Support Vector Machine loss](#multiclass-support-vector-machine-loss)
  - [Practical Considerations](#practical-considerations)
  - [Softmax classifier](#softmax-classifier)
  - [SVM vs. Softmax](#svm-vs-softmax)
  - [Summary](#summary)

# Linear Classification

Previously, we explored Image Classification, which involves assigning a single label to an image from a predefined category set. We discussed the k-Nearest Neighbor (kNN) classifier, which labels images by comparing them to labeled examples in the training set. However, kNN has some drawbacks:

-   The classifier needs to retain all training data to compare against new test data, which can be inefficient in terms of storage because datasets may easily be gigabytes in size.
-   Classifying a new image is computationally expensive, as it demands comparisons with each training image.

### Overview

Now, we’ll develop a more robust approach to image classification, eventually extending to Neural Networks and Convolutional Neural Networks (CNNs). This approach has two key elements: a **score function** that maps raw data to class scores, and a **loss function** that quantifies the agreement between predicted scores and actual labels. By minimizing the loss function with respect to the score function’s parameters, we can optimize the model.

### Parameterized mapping from images to label scores

The first step is defining a score function that maps the pixel values of an image to confidence scores for each class. Let’s walk through this with an example. Suppose we have a training dataset of images $x_i \in R^D$ paired with labels $y_i$, where $i = 1 \dots N$ and $y_i \in { 1 \dots K }$. Here, **N** represents the number of examples, **D** the dimensionality of each image, and **K** the number of categories. For instance, in CIFAR-10, there are **N** = 50,000 images, each with **D** = 32 x 32 x 3 = 3072 pixels, and **K** = 10 classes (e.g., dog, cat, car, etc). We now define the score function $f: R^D \mapsto R^K$, mapping raw image pixels to class scores.

#### Linear Classifier

 In this module we will start out with arguably the simplest possible function, a linear mapping:
 
$$f(x_i, W, b) = W x_i + b$$

In this equation, we assume that the image $x_i$ is represented as a flattened column vector with a shape of $[D \times 1]$. The matrix $W$ has dimensions $[K \times D]$, and the vector $b$ has dimensions $[K \times 1]$; together, they are the parameters of this function. For CIFAR-10, each image $x_i$​ is flattened into a column vector of shape $[3072 \times 1]$, so $W$ is $[10 \times 3072]$, and $b$ is $[10 \times 1]$. As a result, 3072 values (the raw pixel values) are input into the function, producing 10 values as outputs (the class scores). The entries in $W$ are commonly referred to as weights, while $b$ is known as the bias vector, which adjusts the output scores independently of the data in $x_i$. However, it is common for weights and parameters to be used interchangeably.

Key points:

-   Each row of **W** acts as a separate classifier, so **W** effectively evaluates multiple classifiers simultaneously. For example, if we have three classes—**ship**, **cat**, and **dog**—then row 1 in the weight matrix would contain the weights for the "ship" classifier, row 2 for the "cat" classifier, and row 3 for the "dog" classifier.
-   Our objective is to adjust **W** and **b** such that scores align with ground truth labels across the dataset.
-   A major advantage: once the parameters **W** and **b** are learned, the training set isn’t needed for classification, as new images are classified using the learned parameters alone.
-   Finally, it's worth noting that classifying a test image only requires a single matrix multiplication and addition, making it much faster than comparing the test image to every training image.

> **Foreshadowing**: CNNs will build on this framework, mapping image pixels to scores with more complex functions and parameters.

### Interpreting a linear classifier

A linear classifier determines the score for each class by calculating a weighted sum of pixel values across all three color channels in an image. The classifier’s function can assign positive or negative significance to specific colors in certain areas based on the values of these weights. For example, for a "ship" class, a strong presence of blue pixels along the edges of an image might indicate water. In this case, the classifier could have positive weights in the blue channel (indicating that more blue boosts the likelihood of a ship) and negative weights in the red and green channels (indicating that red or green decreases the likelihood of a ship).
___

<img src="imagemap.jpg" alt="imagemap" width="700" height="300"/>

This example illustrates how an image is mapped to class scores. For simplicity, we consider an image composed of just 4 monochrome pixels, and we have 3 classes represented as red (cat), green (dog), and blue (ship). Please note that the colors here denote the classes and are not related to RGB channels. We will flatten the image pixels into a column vector and perform matrix multiplication to calculate the scores for each class. It's important to point out that this specific weight matrix $W$ is not effective, as it assigns a very low score to the cat class despite the image being of a cat. In fact, this weight configuration suggests that the model is more inclined to classify it as a dog.
___


**Analogy of Images as High-Dimensional Points**

Since images can be represented as high-dimensional column vectors, we can think of each image as a point in a high-dimensional space. For example, in the CIFAR-10 dataset, each image is represented as a point in a 3072-dimensional space, corresponding to the 32x32x3 pixel dimensions. Consequently, the entire dataset can be viewed as a labeled set of these points.

We defined the score of each class as a weighted sum of all image pixels, which means that each class score is a linear function over this space. Although we can't visualize a 3072-dimensional space, we can imagine compressing all these dimensions into two for visualization purposes. This allows us to conceptualize how the classifier operates.

<img src="pixelspace.jpeg" alt="pixelspace" width="400" height="300"/>


A cartoon representation of image space, illustrating that each image is a point, and three classifiers are visualized. For instance, the red line for the car classifier indicates all points that receive a score of zero for the car class. The red arrow shows the direction of increasing scores, meaning all points to the right of the line have positive and linearly increasing scores, while those to the left have negative and linearly decreasing scores.
___

As previously mentioned, each row of the weight matrix $W$ represents a classifier for one of the classes. Geometrically, modifying a row in $W$ alters the corresponding line in pixel space, causing it to rotate in different directions. The bias terms $b$ allow our classifiers to translate these lines. Notably, without bias terms, if we plug in $x_i = 0$, the score will always be zero, regardless of the weights, forcing all lines to intersect at the origin.

**Interpretation of linear classifiers as template matching.** Another way to interpret the weights $W$ is to view each row as a _template_ (or _prototype_) for a class. The score for each class is then derived by comparing the corresponding template with the image using an _inner product_ (or _dot product_). In this context, the linear classifier performs template matching, with the templates learned from the data. This approach can be thought of as a variant of the Nearest Neighbor method, where instead of utilizing numerous training images, we rely on a single template per class (which is learned rather than drawn from the training set). In this case, the (negative) inner product serves as the distance metric instead of traditional L1 or L2 distances.
___

<img src="templates.jpg" alt="templates" width="700" height="100"/>

Looking ahead: Example learned weights after training on CIFAR-10. For instance, the ship template prominently features blue pixels, indicating that this template will yield a high score when matched against images of ships in the ocean through the inner product. 
___

Additionally, it's noteworthy that the horse template appears to include features from both left- and right-facing horses due to the dataset's diversity. This results in the linear classifier merging these variations into a single template. Similarly, the car classifier seems to combine several variations into a single template that must identify cars from all angles and colors. In this case, the template appears predominantly red, suggesting that red cars are more prevalent in the CIFAR-10 dataset. The linear classifier's simplicity makes it challenging to effectively differentiate between colors, a limitation that will be addressed in neural networks. Neural networks can create intermediate neurons in hidden layers to detect specific car types (e.g., green cars facing left, blue cars facing front) and subsequently combine these detections into a more precise car score through weighted sums.

**The Bias Trick**

Before we proceed, it's useful to mention a common technique for simplifying the representation of the parameters $W$ and $b$. Recall that we defined the score function as:

$$f(x_i, W, b) = W x_i + b$$

As we delve deeper into the material, managing two sets of parameters (the weights $W$ and the biases $b$) can become cumbersome. A widely used method is to combine these parameters into a single matrix by adding an extra dimension to the vector $x_i$, which always holds the constant value $1$—this is known as the _bias dimension_. With this adjustment, the new score function simplifies to a single matrix multiplication:

$$f(x_i, W) = W x_i$$

In our CIFAR-10 example, $x_i$ now becomes a $3073 \times 1$ vector instead of $3072 \times 1$ (with the added dimension holding the constant $1$), and $W$ is now $10 \times 3073$ instead of $10 \times 3072$. The additional column in $W$ corresponds to the bias $b$. An illustration can help clarify this:

<img src="bias trick.jpeg" alt="bias trick" width="700" height="300"/>

Illustration of the bias trick. Performing matrix multiplication and then adding a bias vector (left) is equivalent to adding a bias dimension with a constant value of 1 to all input vectors and extending the weight matrix by one column to include a bias column (right). By preprocessing our data to append ones to all vectors, we only need to learn a single weight matrix instead of managing two separate matrices for weights and biases. 

**Image Data Preprocessing**

As a brief note, in the examples above, we used raw pixel values that range from [0...255]. In machine learning, it is standard practice to normalize input features (with each pixel treated as a feature). Specifically, it is important to **center the data** by subtracting the mean from each feature. For images, this means calculating a _mean image_ across all training images and subtracting it from each image, resulting in pixel values ranging from approximately [-127 ... 127]. Another common preprocessing step is to scale each input feature so that its values fall within the range [-1, 1]. While zero-mean centering is arguably the more critical step, we will explore its significance further as we understand the dynamics of gradient descent.


### Loss function
In the previous section, we introduced a function that maps pixel values to class scores, parameterized by a set of weights $W$. While we cannot control the given data $(x_i, y_i)$, we do have control over these weights. Our goal is to adjust these weights so that the predicted class scores align well with the actual labels in the training set.

For instance, considering an image of a cat with scores for the classes "cat," "dog," and "ship," we observed that a certain set of weights performed poorly. Although the input depicted a cat, the resulting score for "cat" was very low (-96.8) compared to the other classes (437.9 for "dog" and 61.95 for "ship"). To quantify our dissatisfaction with misclassifications like this, we introduce a **loss function** (also known as the **cost function** or **objective**). This function will yield a high value if the model's predictions deviate significantly from the true labels, and a low value if the model performs well.

#### Multiclass Support Vector Machine Loss

In machine learning, various approaches define loss functions, which help quantify a model’s error on a dataset. A commonly used loss function is the **Multiclass Support Vector Machine (SVM) loss**. This loss function aims for the model to give the correct class of each example a score that exceeds all incorrect classes by a fixed margin, denoted as $\Delta$. Sometimes, it can be helpful to describe the loss function in human terms: the SVM “wants”for a certain outcome, where achieving this outcome results in a lower loss (which is good).

> -   The **"certain outcome"** is the state where the correct class score is confidently higher than all incorrect class scores by the margin $\Delta$. Achieving this outcome yields a **lower loss**, guiding the SVM to improve its classification boundaries.

To get more specific, suppose for the $i$-th example we have the image data $x_i$​ and label $y_i$, where $y_i$ identifies the correct class. The score function, which computes a vector of scores $f(x_i, W)$, assigns each class a score based on the model parameters $W$. We denote the score for the $j$-th class as $s_j = f(x_i, W)_j$. The Multiclass SVM loss for the $i$-th example is then calculated as:

$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

**Example of Multiclass SVM Loss Calculation**

Consider a case with three classes with scores $s = [13, -7, 11]$, and the true class is the first one (meaning $y_i = 0$). With a margin $\Delta = 10$, the loss function sums over all incorrect classes $(j \neq y_i)$, resulting in two terms:

$$L_i = \max(0, -7 - 13 + 10) + \max(0, 11 - 13 + 10)$$

The first term yields zero because the calculation gives a negative value, which is clamped to zero by the $\max(0, -)$ function. This zero result indicates that the correct class score (13) is already sufficiently higher than the incorrect class score (-7) by more than the margin $\Delta$. In contrast, the second term produces 8, as the difference between the correct and incorrect class scores (13 and 11, respectively) falls short of the desired margin by 8.

Thus, the SVM loss accumulates penalties whenever the correct class score is not sufficiently higher than the scores of incorrect classes by at least $\Delta$.

For linear score functions (e.g., $f(x_i; W) = W x_i$), we can also express the loss as:

$$L_i = \sum_{j \neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)$$

where $w_j$​ is the $j$-th row of the weight matrix $W$, reshaped as a column. However, this will not necessarily be the case once we start to consider more complex forms of the score function $f$.

**Terminology: Hinge Loss and Squared Hinge Loss**

The $\max(0, -)$ function, which sets thresholds at zero, is often called **hinge loss**. Some implementations use **squared hinge loss** (also called L2-SVM), where $\max(0, -)^2$ penalizes violations of the margin more strongly, quadratically instead of linearly. The standard (unsquared) hinge loss is more widely used, though squared hinge loss may perform better on certain datasets. This decision is usually guided by **cross-validation**.

> When comparing different loss functions during cross-validation, the **evaluation metric**(e.g. accuracy) should remain the same for each model, regardless of the loss function used. This ensures that the comparison is fair and that any observed differences in performance are due to the loss function rather than changes in the metric.

> Loss functions help measure the error of predictions, providing a clear target for optimization on the training data.

___

<img src="margin.jpg" alt="margin" width="700" height="100"/>


The Multiclass Support Vector Machine (SVM) aims for the score of the correct class to be greater than the scores of all other classes by at least a margin of $\Delta$. If any incorrect class score falls within this margin (the "red region") or higher, a loss is incurred. Otherwise, the loss is zero. Our goal is to find the weights that satisfy this margin constraint for as many training examples as possible, thereby minimizing the total loss across the dataset.
___
**Regularization**. When training a model, we might find a set of weights **W** that perfectly classifies every data point, achieving zero loss across all examples (i.e. all scores are so that all the margins are met, and $L_i = 0$ for all i). However, this solution is not necessarily optimal: scaling **W** by any positive factor (e.g., multiplying **W** by $\lambda$, where $\lambda > 1$) would also yield zero loss as it uniformly increases all score differences without altering the classification margins. For instance, if a correct class score exceeded the nearest incorrect class score by 15, doubling each element in **W** would result in a difference of 30, maintaining the loss at zero.

This ambiguity in the weight selection can be addressed by introducing a **regularization penalty R(W)** to the loss function, encouraging the model to prefer certain weights over others. The most common form is the  squared **L2 norm**, which penalizes large weights by applying an element-wise quadratic penalty over all parameters:

$$R(W) = \sum_k\sum_l W_{k,l}^2$$

Here, each element of **W** is squared and summed. The regularization term, unlike the data loss, depends only on the weights and not on the data. By adding this penalty, the **full Multiclass Support Vector Machine (SVM) loss** is composed of two parts: the **data loss** (the average loss $L_i$ across all examples) and the **regularization loss**. Therefore, the complete Multiclass SVM loss is defined as:

<img src="full_svm_loss.png" alt="full_svm_loss" width="500" height="100"/>
<!-- $$L =  \underbrace{ \frac{1}{N} \sum_i L_i }_\text{data loss} + \underbrace{ \lambda R(W) }_\text{regularization loss} $$ -->

or, fully expanded:

<img src="full_expanded_svm_loss.png" alt="full_expanded_svm_loss" width="700" height="100"/>
<!-- $$L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2$$ -->


In this expression, $N$ represents the total number of training examples, and $\lambda$ is a hyperparameter that controls the weight of the regularization penalty. There is no simple way of setting this hyperparameter , and it is usually set via cross-validation.

Beyond the motivation discussed above, incorporating the regularization penalty brings several valuable properties, which we’ll revisit in later sections. For instance, adding the L2 penalty introduces the desirable **max margin** property in SVMs. For a deeper dive, you can refer to the [CS229 lecture notes](http://cs229.stanford.edu/notes/cs229-notes3.pdf).
___

### Explain Support Vector Machines (SVMs) and the Max Margin Principle

In SVMs, we aim to separate two classes with a hyperplane while maximizing the **margin**—the distance between the closest points of each class to this hyperplane.

**1. Decision Boundary and Margins**

The **decision boundary** (or separating hyperplane) is defined by: $w \cdot x + b = 0$ This hyperplane divides the two classes.

To define the margin, we add two **marginal hyperplanes**:

-   $w \cdot x + b = +1$: boundary of the positive class
-   $w \cdot x + b = -1$: boundary of the negative class

**2. Support Vectors and Constraints**

The points on these marginal hyperplanes are called **support vectors**. We set the constraint: $w \cdot x_i + b = \pm 1$ for these points, ensuring they are closest to the decision boundary and define the margin.

**3. Maximizing the Margin**

The distance between the marginal hyperplanes is $\frac{2}{\|w\|}$. Maximizing this margin is equivalent to minimizing $\|w\|$.

> $d = \frac{|c_1 - c_2|}{\|w\|} =  \frac{|1 - (-1)|}{\|w\|} = \frac{2}{\|w\|}$
___
Regularization not only resolves the ambiguity in weight selection but also improves **generalization** by preventing any one dimension of the input from having an overly large influence. For instance, consider an input vector $x = [1,1,1,1]$ and two weight vectors $w_1 = [1,0,0,0]$ and $w_2 = [0.25,0.25,0.25,0.25]$ Both yield the same dot product with $x$ since $w_1^Tx = w_2^Tx = 1$, yet $w_1$​ has an L2 penalty of 1.0, while $w_2$​ has a lower penalty of 0.5. Intuitively, this is because the weights in $w_2$ are smaller and more diffuse. Since the L2 penalty prefers smaller and more diffuse weight vectors, the final classifier is encouraged to take into account all input dimensions to small amounts rather than a few input dimensions and very strongly. As we will see later in the class, this effect can improve the generalization performance of the classifiers on test images and lead to less *overfitting*.

It is common to regularize weights **W** but not biases **b**, as weights are responsible for controlling input influence strength, whereas biases are not. However, in practice this often turns out to have a negligible effect.  Finally, it's important to note that, due to the regularization penalty, achieving a loss of exactly 0.0 across all examples is impossible, as this would require the unrealistic condition of $W = 0$.

**Code**. Here is the loss function (without regularization) implemented in Python, in both unvectorized and half-vectorized form:

```python
def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in range(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i

def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment
```

The key takeaway from this section is that the SVM loss provides a specific method for assessing how well the model's predictions on the training data align with the actual labels. Moreover, achieving accurate predictions on the training set corresponds to minimizing this loss.

> Our next step is to develop a method to determine the weights that will minimize the loss.



### Practical Considerations

**Setting the Margin Parameter ($\Delta$)**

In a multiclass SVM, the margin parameter $\Delta$ determines the desired score separation between the correct class and each incorrect class. You might wonder about the optimal value of $\Delta$ and whether it needs tuning (e.g., via cross-validation). In practice, though, we typically set $\Delta = 1$ without needing to adjust it further.

Why? Because the parameters $\Delta$ and $\lambda$ both influence the **tradeoff between data loss and regularization loss**, but they do so in tandem. While $\Delta$ sets a target margin for class separation, the weights **W** (scaled by $\lambda$ through regularization) directly control the scores and their separations. If we make all elements of **W** smaller, the score differences shrink, and if we make them larger, the score differences increase proportionally. This means that the specific value of $\Delta$ is less critical because **W** can always be rescaled to match it. Thus, the main tradeoff is managed by $\lambda$, which governs how large we allow **W** to grow and, therefore, how strongly we enforce this margin.

**Relation to Binary Support Vector Machine (SVM)**

If you’re familiar with binary SVMs, you’ve likely seen a formulation like this:

$$L_i = C \max(0, 1 - y_i w^T x_i) + R(W)$$

where $C$ controls the balance between data loss (classification errors) and regularization, and $y_i$ can be either -1 or 1. Interestingly, the multiclass SVM formulation we discuss here becomes equivalent to this binary SVM if we have only two classes. The difference is in the parameter usage: $C$ in binary SVM and $\lambda$ in multiclass SVM both control the same balance but in inverse ways, with the relationship $C \propto \frac{1}{\lambda}$​.

### Softmax Classifier

The Softmax classifier is another popular approach for classification, similar to SVM but with a different loss function. If you’re familiar with **Logistic Regression** in binary classification, the Softmax classifier is its generalization to multiclass problems. Unlike the SVM, which treats the outputs $f(x_i,W)$ as (uncalibrated and possibly difficult to interpret) scores for each class, the Softmax classifier provides **normalized class probabilities** and has a probabilistic interpretation.

#### Softmax Classifier Loss Function

In the Softmax classifier:

-   The scores $f(x_i; W) = W x_i$ are calculated the same way as in the SVM.
    
-   However, we interpret these scores as **unnormalized log probabilities** for each class. 
	> If we have a probability **0.3**, its log-probability is **$\log(0.3) \approx -1.2$**.
    
-   Instead of using hinge loss (as in SVM), we use **cross-entropy loss**, defined as:
    
$$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j}$$
    
where $f_{y_i}$ is the score for the correct class, and $f_j$ represents each class score in the vector of scores $f$. This loss function encourages the classifier to place as much probability mass on the correct class as possible. And as before, the full loss for the dataset is the mean of $L_i$ over all training examples together with a regularization term $R(W)$. 

#### The Softmax Function

The function:

$$f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

is called the **softmax function**. It takes a vector of real values (scores) and converts it into a vector of values between 0 and 1 that sum to 1, representing **class probabilities**.

**Information Theory View**

The cross-entropy loss is based on the **cross-entropy** between the true distribution $p$ (which places all probability on the correct class) and the estimated distribution $q$ (the probabilities output by the Softmax classifier). The cross-entropy is:

$$H(p,q) = - \sum_x p(x) \log q(x)$$

Therefore, the Softmax classifier minimizes the cross-entropy between the predicted class probabilities$q = \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}$ and the 'true' distribution, where all probability mass is concentrated on the correct class (i.e., $p = [0, \ldots, 1, \ldots, 0]$ with a single 1 at the yiy_iyi​-th position). Additionally, since cross-entropy can be expressed as a sum of entropy and the Kullback-Leibler (KL) divergence, $H(p, q) = H(p) + D_{KL}(p \| q)$, and the entropy of the delta function ppp is zero, this objective is equivalent to minimizing the KL divergence between the two distributions (a measure of their difference). Essentially, the cross-entropy objective seeks to place all probability mass on the correct answer in the predicted distribution.

#### Probabilistic Interpretation

The Softmax classifier’s probability for the correct label $y_i$ given the input $x_i$​ is:

$$P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j} }$$

which can be interpreted as the (normalized) probability assigned to the correct label $y_i$ given the image $x_i$ and parameterized by $W$. Here, we interpret the scores $f$ as unnormalized log probabilities. By exponentiating these scores, we obtain unnormalized probabilities, and the division performs the normalization so that the probabilities sum to 1. With this probabilistic interpretation, we are minimizing the **negative log likelihood** of the correct class, essentially performing **Maximum Likelihood Estimation (MLE)**.

A nice feature of this view is that we can now also interpret the regularization term $R(W)$ in the full loss function as coming from a Gaussian prior over the weight matrix $W$, where instead of MLE we are performing the *Maximum a posteriori* (MAP) estimation. We mention these interpretations to help your intuitions, but the full details of this derivation are beyond the scope of this class.


#### Practical Consideration: Numerical Stability

When implementing the Softmax function in code, the intermediate terms $e^{f_{y_i}}$ and $\sum_j e^{f_j}$ can become very large due to the exponential calculations. Dividing such large numbers can lead to numerical instability, so it’s essential to apply a normalization trick. By multiplying both the numerator and denominator by a constant $C$ and incorporating it into the sum, we obtain the following mathematically equivalent expression:


$$\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
= \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}}
= \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j + \log C}}$$


We are free to select any value for $C$ without affecting the results, but choosing it wisely can enhance the numerical stability of the computation. A common approach is to set $\log C = -\max_j f_j$, effectively shifting the values in the vector $f$ so that the maximum value becomes zero. In code:

```python
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
```

**Potentially confusing naming conventions**: The _SVM classifier_ specifically uses the _hinge loss_, also referred to as the _max-margin loss_. In contrast, the _Softmax classifier_ employs the _cross-entropy loss_. The Softmax classifier derives its name from the _softmax function_, which transforms raw class scores into normalized positive values that sum to one, enabling the application of cross-entropy loss. It’s worth noting that referring to the 'softmax loss' is technically incorrect, as softmax is merely the squashing function. However, this shorthand is commonly used in practice.

### SVM vs. Softmax

A picture might help clarify the distinction between the Softmax and SVM classifiers:

<img src="svmvssoftmax.png" alt="svmvssoftmax" width="600" height="350"/>


An example illustrating the difference between the SVM and Softmax classifiers for a single data point: In both classifiers, the same score vector fff is computed (e.g., via matrix multiplication as shown in this section). However, the scores in fff are interpreted differently:

-   The SVM treats these scores as class scores and its loss function encourages the correct class (class 2, shown in blue) to have a score higher than the other classes by a specified margin.
-   The Softmax classifier interprets the scores as (unnormalized) log probabilities for each class. Its loss function seeks to maximize the (normalized) log probability of the correct class (or equivalently, minimize the negative log probability).

For this example, the SVM loss is 1.58, while the Softmax loss is 1.04 (using the natural logarithm, not base 2 or base 10). However, these values are not directly comparable; they are meaningful only when comparing losses within the same classifier using the same data.

**Softmax classifier provides "probabilities" for each class.** 

The Softmax classifier provides interpretable "probabilities" for each class, unlike the SVM, which produces uncalibrated scores that are harder to interpret. For instance, given an image, the SVM might output scores like $[12.5,0.6,−23.0]$ for the classes "cat," "dog," and "ship," respectively. In contrast, the Softmax classifier would transform these scores into probabilities such as $[0.9,0.09,0.01]$, allowing us to interpret the model's confidence in each class.

However, the term "probabilities" is placed in quotes because their sharpness (how peaky or diffuse they are) depends on the regularization strength $\lambda$, which controls the size of the weights $W$. For example, suppose the unnormalized log-probabilities are $[1, -2, 0]$. The Softmax function would compute:

$$[1, -2, 0] \rightarrow [e^1, e^{-2}, e^0] = [2.71, 0.14, 1] \rightarrow [0.7, 0.04, 0.26]$$

Here, the steps involve exponentiating the scores and normalizing them to sum to 1. If the regularization strength $\lambda$ were higher, the weights $W$ would be penalized more, leading to smaller logits. For example, if the weights were halved to $[0.5, -1, 0]$, the Softmax computation would yield:

$$[0.5, -1, 0] \rightarrow [e^{0.5}, e^{-1}, e^0] = [1.65, 0.37, 1] \rightarrow [0.55, 0.12, 0.33]$$

In this case, the probabilities become more diffuse, spreading across classes. With extremely strong regularization, where the weights approach very small values, the output probabilities would converge toward a uniform distribution (e.g., $[0.33, 0.33, 0.33]$ for three classes).

Thus, the Softmax probabilities are better understood as **relative confidences**. Similar to the SVM, the ordering of the scores is meaningful, but the absolute probability values (or their differences) are influenced by regularization and are not strictly interpretable.

**In practice, SVM and Softmax are usually comparable.** 

The performance difference between SVM and Softmax classifiers is typically minor, and opinions on which is better often depend on the use case. One key distinction is that the SVM optimizes a more _local_ objective, which can be viewed as either an advantage or a limitation.

For example, consider a case where the scores are $[10,−2,3]$ and the correct class is the first. An SVM with a margin of $\Delta = 1$ checks if the correct class score exceeds the other class scores by at least the margin. If this condition is met (e.g., $10 > -2+1$ and $10 > 3+1$), the loss is zero. The SVM does not concern itself with the specific values of the scores beyond this point, so it would treat $[10, -100, -100]$and $[10, 9, 9]$ as equivalent because both satisfy the margin.

In contrast, the Softmax classifier always seeks to improve the separation between the correct and incorrect classes. For the same scores, $[10,−100,−100]$ would result in a much lower loss than $[10,9,9]$ because the latter assigns relatively higher probabilities to the incorrect classes. The Softmax classifier continuously adjusts its probabilities, aiming to make the correct class's probability higher and the incorrect classes' probabilities lower. This means Softmax is "never fully satisfied," as it always tries to optimize further.

The SVM's focus on satisfying the margin without micromanaging scores can be seen as a benefit in certain situations. For instance, in a car classifier, the model may devote most of its effort to distinguishing between cars and trucks (a challenging distinction) without being distracted by frog examples, which it already classifies with low scores and that likely occupy a different part of the feature space. This targeted effort reflects the SVM's local optimization approach.

### Summary

To summarize:

-   We introduced a **score function** that maps image pixels to class scores, implemented as a linear function involving weights (**W**) and biases (**b**).
-   Unlike the kNN classifier, this **parametric approach** offers the advantage of discarding the training data after learning the parameters. Moreover, predicting a new test image is computationally efficient, requiring only a single matrix multiplication with **W**, rather than exhaustive comparisons with all training examples.
-   We discussed the **bias trick**, which integrates the bias vector into the weight matrix for simplicity, allowing us to manage a single parameter matrix.
-   We defined a **loss function** (we introduced two commonly used losses for linear classifiers: the **SVM** and the **Softmax**) that measures how compatible a given set of parameters is with respect to the ground truth labels in the training dataset. Both are designed such that accurate predictions correspond to a low loss value.

In this section, we explored how to map images to class scores using parameterized models and evaluated predictions through loss functions. The next step is to determine the parameters that minimize this loss, a process known as _optimization_, which will be covered in the following section.

## Reference:
- [https://cs231n.github.io/classification/](https://cs231n.github.io/linear-classify/)

