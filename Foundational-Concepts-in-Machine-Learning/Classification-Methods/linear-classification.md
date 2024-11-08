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
