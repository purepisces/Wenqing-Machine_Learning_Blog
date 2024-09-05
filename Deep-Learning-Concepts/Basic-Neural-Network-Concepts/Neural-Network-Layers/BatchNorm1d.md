### BatchNorm1d

`needle.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1, device=None, dtype="float32")`

Applies batch normalization over a mini-batch of inputs as described in the paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).

$$y = w \circ \frac{z_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b$$
  
but where here the mean and variance refer to to the mean and variance over the _batch_dimensions. The function also computes a running average of mean/variance for all features at each layer $\hat{\mu}, \hat{\sigma}^2$, and at test time normalizes by these quantities:

$$y = \frac{(x - \hat{mu})}{((\hat{\sigma}^2_{i+1})_j+\epsilon)^{1/2}}$$

BatchNorm uses the running estimates of mean and variance instead of batch statistics at test time, i.e.,

after `model.eval()` has been called on the BatchNorm layer's `training` flag is false.

 
To compute the running estimates, you can use the equation $$\hat{x_{new}} = (1 - m) \hat{x_{old}} + mx_{observed},$$

where $m$ is momentum.

##### Parameters

- `dim` - input dimension

- `eps` - a value added to the denominator for numerical stability.

- `momentum` - the value used for the running mean and running variance computation.

  

##### Variables

- `weight` - the learnable weights of size `dim`, elements initialized to 1.

- `bias` - the learnable bias of size `dim`, elements initialized to 0.

- `running_mean` - the running mean used at evaluation time, elements initialized to 0.

- `running_var` - the running (unbiased) variance used at evaluation time, elements initialized to 1.

Code Implementation:
```python
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # Learnable parameters
        # Both self.weight and self.bias have shape (dim,) = (features,)
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        # Running mean and variance (not learnable)
        # Both self.running_mean and self.running_var have shape (dim,) = (features,)
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # Compute mean and variance across the batch
            # The shape of x is (batch_size, features)
            batch_size = x.shape[0]
            # The shape of batch_mean is (features, )
            batch_mean = ops.divide_scalar(ops.summation(x, axes=(0,)),batch_size)
            # The batch_mean has the shape (features,). It is first reshaped to (1, features)
            # and then broadcasted to (batch_size, features).
            broadcast_batch_mean = ops.broadcast_to(ops.reshape(batch_mean, (1, -1)), x.shape)
            
            # The shape of batch_var is (features, )
            batch_var =ops.divide_scalar(ops.summation(ops.power_scalar((x - broadcast_batch_mean),2), axes=(0,)), batch_size)
            # The batch_var has the shape (features,). It is first reshaped to (1, features)
            # and then broadcasted to (batch_size, features).
            broadcast_batch_var = ops.broadcast_to(ops.reshape(batch_var, (1, -1)), x.shape)
            
            # Update running mean and variance
            # Both self.running_mean and self.running_var have shape (dim,) = (features,)
            # We must use the detached `batch_mean` and `batch_var` (i.e., using `.data`), 
            # otherwise the `requires_grad` attribute of `self.running_mean` and `self.running_var` 
            # will become `True`.
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data

            # Normalize the input
            # The shape of x_hat = (batch_size, features)
            x_hat = (x - broadcast_batch_mean) / ops.power_scalar(broadcast_batch_var + self.eps, 0.5)
        else:
            # Use running mean and variance during evaluation
            # Both self.running_mean and self.running_var have the shape (features,). 
            # They are first reshaped to (1, features) and then broadcasted to (batch_size, features).
            broadcast_running_mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1)), x.shape)
            broadcast_running_var = ops.broadcast_to(ops.reshape(self.running_var, (1, -1)), x.shape)
            
            # The shape of x_hat = (batch_size, features)
            # self.eps is a scalar value, when added to broadcast_running_var, 
            # it is automatically broadcasted to match the shape of broadcast_running_var, 
            # which is (batch_size, features).
            x_hat = (x - broadcast_running_mean) / ops.power_scalar(broadcast_running_var + self.eps, 0.5)
        
        # Both self.weight and self.bias have the shape (features,). 
        # They are first reshaped to (1, features) and then broadcasted to (batch_size, features).
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        
        # Apply learnable scale (weight) and shift (bias)
        # Element-wise multiplication of broadcast_weight and x_hat (batch_size, features)
        return broadcast_weight * x_hat + broadcast_bias
        ### END YOUR SOLUTION
```
___

### Explanation For detach
In python/needle/autograd.py
When access `batch_mean.data` and `batch_var.data`, it calls the `data` property method, which is the getter:
```python
@property
def data(self):
    return self.detach()
```
**Getter (`return self.detach()`)**: When you access `tensor.data`, it calls the `detach()` method to return a new tensor that is detached from the computational graph.
```python
@classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value
```

This code snippet defines a `classmethod` called `make_const` in the `Tensor` class (or a similar class derived from `Value`). This method is used to create a new `Tensor` (or `Value`) object that represents a constant value, meaning it doesn't require gradient computation and is not part of the computational graph.

```python
def detach(self):
    """Create a new tensor that shares the data but detaches from the graph."""
    return Tensor.make_const(self.realize_cached_data())
```
**`detach(self)`**: The `detach()` method calls `Tensor.make_const`, which creates a new tensor with the same `cached_data` (same numerical data) as the original tensor but with `requires_grad=False`. This means that the returned tensor is disconnected from the original computational graph used for automatic differentiation. Any operations on the detached tensor will not be tracked for gradients, making it independent of the computational graph.

#### Why Use `detach()`:

-   **Preventing Gradients**: By using `batch_mean.data` and `batch_var.data`, you ensure that the `running_mean` and `running_var` are updated with the raw numerical values from `batch_mean` and `batch_var` without tracking these operations in the computational graph. This prevents `self.running_mean` and `self.running_var` from requiring gradients, which is the intended behavior.

### Normalize the input $\frac{z_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})}$ 

The term $\frac{z_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})}$ normalizes the input $z_i$​ by centering it (subtracting the mean) and scaling it (dividing by the standard deviation, which is the square root of the variance plus epsilon). This process results in a normalized feature with a mean of 0 and a variance of 1, often denoted as $\hat{x}$ or $\hat{z_i}$​.
### Explanation for  Running estimates

Running estimates in the context of Batch Normalization refer to the continuously updated averages of the mean and variance of the features, which are computed over multiple mini-batches during training.

#### Running Mean and Running Variance

-   **Running Mean ($\hat{\mu}$​)**: This is an estimate of the average value of each feature across all the training data. Instead of recalculating the mean from scratch for every mini-batch, the running mean is updated incrementally using the mean of each new mini-batch. This allows the model to have a stable estimate of the mean even when the data is divided into smaller batches.
    
-   **Running Variance ($\hat{\sigma}^2$)**: This is an estimate of the variance of each feature across all the training data. Like the running mean, the running variance is updated incrementally using the variance calculated from each new mini-batch.
    

#### How Running Estimates Work

During training, for each mini-batch, the mean and variance are calculated for the features in that mini-batch. The running estimates (mean and variance) are then updated as follows:

-   **Running Mean Update**: 
$$\hat{\mu}_{\text{new}} = (1 - m) \cdot \hat{\mu}_{\text{old}} + m \cdot \mu_{\text{batch}}$$
    
-   **Running Variance Update**: 
$$\hat{\sigma}^2_{\text{new}} = (1 - m) \cdot \hat{\sigma}^2_{\text{old}} + m \cdot \sigma^2_{\text{batch}}$$
    

Where:

-   $\hat{\mu}_{\text{old}}$ and $\hat{\sigma}^2_{\text{old}}$ are the previous running estimates.
-   $\mu_{\text{batch}}$ and $\sigma^2_{\text{batch}}$​ are the mean and variance computed from the current mini-batch.
-   $m$ is the momentum, controlling how much of the current batch's statistics influence the running estimates.

#### Why Running Estimates Are Important

-   **Training Phase**: During training, the batch statistics (mean and variance of the current mini-batch) are used to normalize the data. At the same time, the running estimates are updated to reflect the current data distribution.
    
-   **Inference Phase**: When the model is in inference (evaluation) mode, batch statistics are not available because data is usually processed one sample at a time. Instead of using batch statistics, the model uses the running estimates of mean and variance to normalize the data. This ensures that the model can generalize well on new data, using the learned statistics from training.

In summary, running estimates provide a way to capture the global statistics of the data during training and apply them during inference, enabling the model to function consistently across both phases.

### Why Use Running Mean and Variance in Evaluation Mode?

1.  **Stability and Consistency**:
    
    -   During evaluation (or inference), the model needs to behave consistently, regardless of the specific inputs. If we were to compute the mean and variance on-the-fly during evaluation (like we do during training), the output could vary significantly depending on the specific batch of data being processed at that moment. This would make the model's predictions unstable.
    -   The running mean and variance are accumulated during training and represent a stable estimate of the mean and variance across the entire training data (or a large portion of it). Using these running estimates during evaluation ensures that the network's behavior is consistent and does not depend on the specific batch of data being processed at inference time.
2.  **Non-dependence on Batch Size**:
    
    -   At evaluation time, the batch size might be different from what was used during training (e.g., you might process one sample at a time). If we were to compute the batch statistics on-the-fly in evaluation mode, the model's performance could degrade because the batch statistics computed on a single sample (or a small batch) would not be representative of the overall data distribution.
    -   By using the running mean and variance, which are computed over many batches during training, we avoid this issue and ensure that the network's output remains robust.

### Why Calculate Batch Statistics During Training but Not Use Running Estimates?

1.  **Capturing the Data Distribution**:
    
    -   During training, the goal is to learn the parameters of the model (including the weights and biases of the BatchNorm layer) that work well with the data distribution. To do this, it's important to normalize the data based on the statistics of the current mini-batch. This allows the model to adapt to the actual distribution of the data it sees during training.
    -   Batch statistics (mean and variance) computed during each mini-batch are a good approximation of the data distribution within that mini-batch, and using them helps the model learn more effectively.
    
2.  **Avoiding Overfitting to Specific Batches**:
    
    -   If we were to use the running mean and variance during training instead of the batch statistics, the model might overfit to the running estimates, which are based on previous batches. This would reduce the regularization effect provided by BatchNorm, which comes from the slight noise introduced by normalizing with batch-specific statistics.
    -   By normalizing with the statistics of the current batch, BatchNorm introduces some noise that acts as a regularizer, helping to prevent overfitting.

### Summary:

-   **Training Mode**: BatchNorm normalizes the data using the batch-specific mean and variance because this allows the model to learn effectively, adapting to the distribution of the data it sees during training. During this process, the running mean and variance are updated to accumulate stable estimates that will be used during evaluation.
    
-   **Evaluation Mode**: BatchNorm uses the running mean and variance to ensure consistent and stable performance across different inputs, regardless of the batch size or data distribution. This helps to maintain the model's predictive accuracy and stability when it's deployed in the real world.
    

### Explanation of the Momentum in BatchNorm1d

In BatchNorm1d, the momentum hyperparameter controls how quickly the running estimates (mean and variance) adapt to the statistics of the current mini-batch of data. Here's a detailed explanation of how the value of momentum affects the behavior:

#### **Momentum Formula**

The running estimates for the mean $\hat{\mu}$​ and variance $\hat{\sigma}^2$ are updated using the following formula:

$$\hat{x}_{\text{new}} = (1 - m) \hat{x}_{\text{old}} + m \cdot x_{\text{observed}}$$

Where:

-   $\hat{x}_{\text{new}}$​ is the updated running estimate.
-   $m$ is the momentum value.
-   $\hat{x}_{\text{old}}$​ is the previous running estimate.
-   $x_{\text{observed}}$​ is the current observed statistic (mean or variance) from the mini-batch.

#### **Effect of Large Momentum $m \approx 1$**

-   **Rapid Adaptation:** When momentum is large (e.g., $m = 0.9$), the new running estimates are heavily influenced by the current batch statistics. This means that the running mean and variance will change quickly to reflect the statistics of the most recent mini-batches.
    
-   **Less Historical Influence:** The contribution of the previous running estimate $1 - m$ is small, so the running mean and variance quickly forget the statistics of earlier batches.
    
-   **Use Case:** Large momentum values are useful when the data distribution is changing rapidly, and you want the running statistics to adapt quickly to new data.

#### **Effect of Small Momentum $m \approx 0$**

-   **Slow Adaptation:** When momentum is small (e.g., $m=0.1$), the new running estimates are only slightly influenced by the current batch statistics. The running mean and variance will change slowly, making them more stable and less responsive to fluctuations in the mini-batch statistics.
    
-   **More Historical Influence:** The contribution of the previous running estimate is large, so the running mean and variance maintain a stronger influence from earlier batches.
    
-   **Use Case:** Small momentum values are useful when the data distribution is stable, and you want the running statistics to smooth out noise and avoid overreacting to temporary fluctuations in the batch statistics.

#### **Summary**

-   **Large Momentum ($m \approx 1$)**: Fast adaptation to recent batches, less influenced by historical data.
-   **Small Momentum ($m \approx 0$)**: Slow adaptation, more influenced by the history of the data, providing more stability to the running estimates.

### Key Differences For LayerNorm and BatchNorm

-   **Normalization Axis**:
    -   **LayerNorm**: Normalizes across the features within each individual example.
    -   **BatchNorm**: Normalizes across the batch for each feature.
-   **Mean and Variance Calculation**:
    -   **LayerNorm**: Calculates the mean and variance across the features of a single example.
    -   **BatchNorm**: Calculates the mean and variance across the batch for each feature.
