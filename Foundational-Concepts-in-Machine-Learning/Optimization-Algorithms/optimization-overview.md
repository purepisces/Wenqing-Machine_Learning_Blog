
Table of Contents:

- [Introduction](#intro)
- [Visualizing the loss function](#vis)
- [Optimization](#optimization)
  - [Strategy #1: Random Search](#opt1)
  - [Strategy #2: Random Local Search](#opt2)
  - [Strategy #3: Following the gradient](#opt3)
- [Computing the gradient](#gradcompute)
  - [Numerically with finite differences](#numerical)
  - [Analytically with calculus](#analytic)
- [Gradient descent](#gd)
- [Summary](#summary)

<a name='intro'></a>

### Introduction

In the previous section we introduced two key components in context of the image classification task:

1. A (parameterized) **score function** mapping the raw image pixels to class scores (e.g. a linear function)
2. A **loss function** that measured the quality of a particular set of parameters based on how well the induced scores agreed with the ground truth labels in the training data. We saw that there are many ways and versions of this (e.g. Softmax/SVM).

Concretely, recall that the linear function had the form \\( f(x_i, W) =  W x_i \\) and the SVM we developed was formulated as:

$$
L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + 1) \right] + \alpha R(W)
$$

We saw that a setting of the parameters \\(W\\) that produced predictions for examples \\(x_i\\) consistent with their ground truth labels \\(y_i\\) would also have a very low loss \\(L\\). We are now going to introduce the third and last key component: **optimization**. Optimization is the process of finding the set of parameters \\(W\\) that minimize the loss function.

**Foreshadowing:** Once we understand how these three core components interact, we will revisit the first component (the parameterized function mapping) and extend it to functions much more complicated than a linear mapping: First entire Neural Networks, and then Convolutional Neural Networks. The loss functions and the optimization process will remain relatively unchanged.

<a name='vis'></a>

### Visualizing the loss function

The loss functions we'll look at in this class are usually defined over very high-dimensional spaces (e.g. in CIFAR-10 a linear classifier weight matrix is of size [10 x 3073] for a total of 30,730 parameters), making them difficult to visualize. However, we can still gain some intuitions about one by slicing through the high-dimensional space along rays (1 dimension), or along planes (2 dimensions). For example, we can generate a random weight matrix \\(W\\) (which corresponds to a single point in the space), then march along a ray and record the loss function value along the way. That is, we can generate a random direction \\(W_1\\) and compute the loss along this direction by evaluating \\(L(W + a W_1)\\) for different values of \\(a\\). This process generates a simple plot with the value of \\(a\\) as the x-axis and the value of the loss function as the y-axis. We can also carry out the same procedure with two dimensions by evaluating the loss \\( L(W + a W_1 + b W_2) \\) as we vary \\(a, b\\). In a plot, \\(a, b\\) could then correspond to the x-axis and the y-axis, and the value of the loss function can be visualized with a color:

<div class="fig figcenter fighighlight">
  <img src="/assets/svm1d.png">
  <img src="/assets/svm_one.jpg">
  <img src="/assets/svm_all.jpg">
  <div class="figcaption">
    Loss function landscape for the Multiclass SVM (without regularization) for one single example (left,middle) and for a hundred examples (right) in CIFAR-10. Left: one-dimensional loss by only varying <b>a</b>. Middle, Right: two-dimensional loss slice, Blue = low loss, Red = high loss. Notice the piecewise-linear structure of the loss function. The losses for multiple examples are combined with average, so the bowl shape on the right is the average of many piece-wise linear bowls (such as the one in the middle).
  </div>
</div>

We can explain the piecewise-linear structure of the loss function by examining the math. For a single example we have:

$$
L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + 1) \right]
$$

It is clear from the equation that the data loss for each example is a sum of (zero-thresholded due to the \\(\max(0,-)\\) function) linear functions of \\(W\\). Moreover, each row of \\(W\\) (i.e. \\(w_j\\)) sometimes has a positive sign in front of it (when it corresponds to a wrong class for an example), and sometimes a negative sign (when it corresponds to the correct class for that example). To make this more explicit, consider a simple dataset that contains three 1-dimensional points and three classes. The full SVM loss (without regularization) becomes:

$$
\begin{align}
L_0 = & \max(0, w_1^Tx_0 - w_0^Tx_0 + 1) + \max(0, w_2^Tx_0 - w_0^Tx_0 + 1) \\\\
L_1 = & \max(0, w_0^Tx_1 - w_1^Tx_1 + 1) + \max(0, w_2^Tx_1 - w_1^Tx_1 + 1) \\\\
L_2 = & \max(0, w_0^Tx_2 - w_2^Tx_2 + 1) + \max(0, w_1^Tx_2 - w_2^Tx_2 + 1) \\\\
L = & (L_0 + L_1 + L_2)/3
\end{align}
$$

Since these examples are 1-dimensional, the data \\(x_i\\) and weights \\(w_j\\) are numbers. Looking at, for instance, \\(w_0\\), some terms above are linear functions of \\(w_0\\) and each is clamped at zero. We can visualize this as follows:

<div class="fig figcenter fighighlight">
  <img src="/assets/svmbowl.png">
  <div class="figcaption">
    1-dimensional illustration of the data loss. The x-axis is a single weight and the y-axis is the loss. The data loss is a sum of multiple terms, each of which is either independent of a particular weight, or a linear function of it that is thresholded at zero. The full SVM data loss is a 30,730-dimensional version of this shape.
  </div>
</div>

As an aside, you may have guessed from its bowl-shaped appearance that the SVM cost function is an example of a [convex function](http://en.wikipedia.org/wiki/Convex_function) There is a large amount of literature devoted to efficiently minimizing these types of functions, and you can also take a Stanford class on the topic ( [convex optimization](http://stanford.edu/~boyd/cvxbook/) ). Once we extend our score functions \\(f\\) to Neural Networks our objective functions will become non-convex, and the visualizations above will not feature bowls but complex, bumpy terrains.

*Non-differentiable loss functions*. As a technical note, you can also see that the *kinks* in the loss function (due to the max operation) technically make the loss function non-differentiable because at these kinks the gradient is not defined. However, the [subgradient](http://en.wikipedia.org/wiki/Subderivative) still exists and is commonly used instead. In this class will use the terms *subgradient* and *gradient* interchangeably.

<a name='optimization'></a>

### Optimization

To reiterate, the loss function lets us quantify the quality of any particular set of weights **W**. The goal of optimization is to find **W** that minimizes the loss function. We will now motivate and slowly develop an approach to optimizing the loss function. For those of you coming to this class with previous experience, this section might seem odd since the working example we'll use (the SVM loss) is a convex problem, but keep in mind that our goal is to eventually optimize Neural Networks where we can't easily use any of the tools developed in the Convex Optimization literature.

<a name='opt1'></a>

#### Strategy #1: A first very bad idea solution: Random search

Since it is so simple to check how good a given set of parameters **W** is, the first (very bad) idea that may come to mind is to simply try out many different random weights and keep track of what works best. This procedure might look as follows:

```python
# assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
# assume Y_train are the labels (e.g. 1D array of 50,000)
# assume the function L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in range(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

# prints:
# in attempt 0 the loss was 9.401632, best 9.401632
# in attempt 1 the loss was 8.959668, best 8.959668
# in attempt 2 the loss was 9.044034, best 8.959668
# in attempt 3 the loss was 9.278948, best 8.959668
# in attempt 4 the loss was 8.857370, best 8.857370
# in attempt 5 the loss was 8.943151, best 8.857370
# in attempt 6 the loss was 8.605604, best 8.605604
# ... (trunctated: continues for 1000 lines)
```

In the code above, we see that we tried out several random weight vectors **W**, and some of them work better than others. We can take the best weights **W** found by this search and try it out on the test set:

```python
# Assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict == Yte)
# returns 0.1555
```

With the best **W** this gives an accuracy of about **15.5%**. Given that guessing classes completely at random achieves only 10%, that's not a very bad outcome for a such a brain-dead random search solution!

**Core idea: iterative refinement**. Of course, it turns out that we can do much better. The core idea is that finding the best set of weights **W** is a very difficult or even impossible problem (especially once **W** contains weights for entire complex neural networks), but the problem of refining a specific set of weights **W** to be slightly better is significantly less difficult. In other words, our approach will be to start with a random **W** and then iteratively refine it, making it slightly better each time.

> Our strategy will be to start with random weights and iteratively refine them over time to get lower loss

**Blindfolded hiker analogy.** One analogy that you may find helpful going forward is to think of yourself as hiking on a hilly terrain with a blindfold on, and trying to reach the bottom. In the example of CIFAR-10, the hills are 30,730-dimensional, since the dimensions of **W** are 10 x 3073. At every point on the hill we achieve a particular loss (the height of the terrain).

<a name='opt2'></a>

#### Strategy #2: Random Local Search

The first strategy you may think of is to try to extend one foot in a random direction and then take a step only if it leads downhill. Concretely, we will start out with a random \\(W\\), generate random perturbations \\( \delta W \\) to it and if the loss at the perturbed \\(W + \delta W\\) is lower, we will perform an update. The code for this procedure is as follows:

```p
