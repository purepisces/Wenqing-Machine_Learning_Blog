# Mini-Batch Gradient Descent

## Overview

Mini-Batch Gradient Descent is an efficient optimization technique for training machine learning models, especially for large-scale datasets. It combines the benefits of both **Stochastic Gradient Descent (SGD)** and **Batch Gradient Descent** by computing gradients over small subsets (mini-batches) of the data.

## Motivation

In large-scale machine learning tasks (e.g., datasets with millions of examples), computing the gradient over the entire dataset for a single parameter update is computationally expensive. Mini-Batch Gradient Descent addresses this inefficiency by:

1.  Sampling a **mini-batch** of data from the training set.
2.  Computing the gradient using only this mini-batch.
3.  Updating the parameters based on this gradient.

This approach leads to faster convergence while maintaining the stability of batch gradient descent.

## Algorithm

The **Mini-Batch Gradient Descent** algorithm is straightforward and can be implemented as follows:

```python
# Vanilla Minibatch Gradient Descent

while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```

### Explanation of Parameters:

-   **data_batch**: A randomly sampled subset of the dataset (e.g., 256 examples).
-   **weights_grad**: The gradient of the loss function computed over the mini-batch.
-   **step_size**: Also known as the **learning rate**, it controls the size of updates to the weights.

### Key Characteristics:

1.  **Frequent Updates**: By performing updates more frequently (after each mini-batch), training can converge faster than batch gradient descent.
2.  **Approximation of the Full Gradient**: Although the gradient is computed for only a subset of data, it often approximates the gradient for the full dataset well.


## Benefits

-   **Faster Training**: Mini-batches allow for frequent parameter updates, leading to faster convergence.
-   **Efficient Computation**: Modern computational frameworks perform better with vectorized operations, which are more efficient when applied to mini-batches rather than individual examples.
-   **Reduced Memory Usage**: Instead of loading the entire dataset into memory, only the mini-batch needs to be processed at a time.



## Comparison to Other Gradient Descent Methods

| Feature                | Batch Gradient Descent          | Mini-Batch Gradient Descent     | Stochastic Gradient Descent      |
|------------------------|----------------------------------|---------------------------------|-----------------------------------|
| **Update Frequency**   | Once per full dataset epoch     | Once per mini-batch             | Once per example                 |
| **Convergence Speed**  | Slower                         | Faster                          | Highly variable                  |
| **Stability**          | High                           | Moderate                        | Low                              |
| **Memory Requirements**| High                           | Moderate                        | Low                              |


## Practical Considerations

1.  **Mini-Batch Size**:
    
    -   Typical sizes are powers of 2 (e.g., 32, 64, 128) due to hardware optimization for vectorized operations.
    -   Large mini-batches can lead to more stable updates but require more memory.
    -   Small mini-batches increase noise in the gradient estimation but lead to faster updates.
2.  **Step Size (Learning Rate)**:
    
    -   A crucial hyperparameter that determines the effectiveness of updates.
    -   Too large: Can cause divergence.
    -   Too small: Leads to slow convergence.
3.  **Trade-offs**:
    
    -   Mini-Batch Gradient Descent balances the computational efficiency of Batch Gradient Descent and the speed of Stochastic Gradient Descent.


## Extreme Case: Stochastic Gradient Descent (SGD)

An extreme form of Mini-Batch Gradient Descent is **Stochastic Gradient Descent**, where the batch size is reduced to a single example. While this approach enables even more frequent updates, it introduces significant noise and instability. In practice, the term "SGD" is often used interchangeably with Mini-Batch Gradient Descent, regardless of the batch size.


## Summary

Mini-Batch Gradient Descent is a widely-used optimization technique that:

-   Offers a balance between speed and stability.
-   Reduces memory usage.
-   Leverages the power of modern vectorized computations for efficiency.

By iteratively updating model parameters using small, randomly sampled subsets of the data, it enables scalable training of complex machine learning models on large datasets.
