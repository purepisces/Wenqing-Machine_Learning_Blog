# Nearest Neighbor Classifier Overview

## Introduction

The Nearest Neighbor Classifier, while rarely used in practical applications today, offers a fundamental approach to image classification. Unlike advanced models such as Convolutional Neural Networks (CNNs), this classifier operates by directly comparing images to determine similarity. This simple method can be an effective first step for understanding basic principles in classification.

### Example Dataset: CIFAR-10

One widely used image classification dataset for introductory tasks is CIFAR-10, which contains 60,000 images, each 32x32 pixels, categorized into 10 classes (e.g., "airplane," "automobile," "bird," etc.). This dataset is divided into a training set of 50,000 images and a test set of 10,000 images. Below is an example showing 10 images from each class:

<img src="CIFAR-10.jpg" alt="CIFAR-10" width="900" height="400"/>

*Left: Sample images from CIFAR-10. Right: Each test image with its top 10 nearest training set images (based on pixel-wise differences).*

In this example, given 50,000 training images (5,000 per class), the Nearest Neighbor Classifier labels each of the 10,000 test images by comparing it to the entire training set, predicting the label based on the closest match. However, this approach has limitations. For example, it only accurately predicts the class in around 3 out of 10 cases in a sample test, sometimes retrieving images of different classes due to background or pixel similarity.

## Image Comparison Using L1 Distance

For the Nearest Neighbor Classifier, a simple method to compare two images is pixel-wise comparison. Representing images as vectors (e.g., $I_1$ and $I_2$), we can calculate the **L1 distance** by summing the absolute differences across all pixels:

$$d_1 (I_1, I_2) = \sum_{p} \left| I^p_1 - I^p_2 \right|$$

In this approach, identical images yield a distance of zero, while dissimilar images have a larger distance. This method can be visualized as follows:

<img src="pixel-wise differences.jpeg" alt="pixel-wise differences" width="800" height="250"/>


*An illustration of pixel-wise differences using L1 distance for a single color channel.*

### Loading and Preparing the Data

To implement this classifier, we first load the CIFAR-10 data into four arrays: training images, training labels, test images, and test labels.

```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
```
### Training and Evaluation

With the data ready, we can create and evaluate the Nearest Neighbor Classifier:

```python
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```

### Implementation of Nearest Neighbor Classifier

Here’s an example of a Nearest Neighbor Classifier using the L1 distance:

```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

This classifier achieves 38.6% accuracy on CIFAR-10, significantly better than random guessing (10%) but still far from human performance or modern CNN models.

## L2 Distance as an Alternative

Another way to measure distance is the **L2 distance** (Euclidean distance), which is more sensitive to differences between vectors. It is calculated as:

$$d_2 (I_1, I_2) = \sqrt{\sum_{p} \left( I^p_1 - I^p_2 \right)^2}$$


In practice, you may omit the square root for computational efficiency since it doesn’t affect the nearest neighbor selection.

```python
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```

### L1 vs. L2 Distance

The choice of distance metric can influence performance. L2 distance penalizes larger differences more heavily, whereas L1 distance is more lenient. The L1 and L2 distances are specific cases of the p-norm and can be selected based on problem requirements.


## Reference:
- https://cs231n.github.io/classification/
