
# Regularization

Regularization is a set of techniques that can prevent overfitting in neural networks and thus improve the accuracy of a Deep Learning model when facing completely new data from the problem domain.

## Batch Normalization 

Z-score normalization is the procedure during which the feature values are rescaled so that they have the properties of a normal distribution. Let µ be the mean (the average value of the feature, averaged over all examples in the dataset) and σ be the standard deviation from the mean.
Standard scores (or z-scores) of features are calculated as follows:

$$\hat{x} = \frac{x - µ}{σ}$$

Batch normalization is a method used to make training of artificial neural networks faster and more stable through normalization of the layers’ inputs by re-centering and re-scaling. It comes from the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.

In this section, your task is to implement the forward and backward attribute functions of the BatchNorm1d class in file `batchnorm.py`.


- Class attributes:
  - `alpha`: a hyperparameter used for the running mean and running var computation.
  - `eps`: a value added to the denominator for numerical stability.
  - `Bw`: learnable parameter of a BN (batch norm) layer to scale features.
  - `Bb`: learnable parameter of a BN (batch norm) layer to shift features.
  - `dLBw`: how changes in $\gamma$ affect loss
  - `dLBb`: how changes in $\beta$ affect loss
  - `running_M`: learnable parameter, the estimated mean of the training data
  - `running_V`: learnable parameter, the estimated variance of the training data
    
- Class methods:
  - `forward`: It takes in a batch of data $Z$ computes the batch normalized data $\hat{Z}$, and returns the scaled and shifted data $\tilde{Z}$. In addition:
    * During training, forward calculates the mean and standard-deviation of each feature over the mini-batches and uses them to update the `running_M` $E[Z]$ and `running_V`$Var[Z]$, which are learnable parameter vectors trained during forward propagation. By default, the elements of $E[Z]$ are set to $0$ and the elements of $Var[Z]$ are set to 1.
    * During inference, the learnt mean `running_M` $E[Z]$ and variance `running_V`$Var[Z]$ over the entire training dataset are used to normalize $\tilde{Z}$.
  - `backward`: takes input $dLdBZ$, how changes in BN layer output affects loss, computes and stores the necessary gradients $dLdBW$, $dLdBb$ to train learnable parameters BW and Bb. Returns $dLdZ$,  how the changes in BN layer input $Z$ affect loss $L$ for downstream computation.

Please consider the following class structure:
```python
class BatchNorm1d:
    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))
        
    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        if eval == False:
            # training mode
            self.Z = Z
            self.N = None  # TODO
            self.M = None  # TODO
            self.V = None  # TODO
            self.NZ = None  # TODO
            self.BZ = None  # TODO
            self.running_M = None  # TODO
            self.running_V = None  # TODO
        else:
            # inference mode
            self.NZ = None  # TODO
            self.BZ = None  # TODO
        return self.BZ
    
    def backward(self, dLdBZ):
        self.dLdBW = None  # TODO
        self.dLdBb = None  # TODO
        dLdNZ = None  # TODO
        dLdV = None  # TODO
        dLdM = None  # TODO
        dLdZ = None  # TODO
        return dLdZ
```
## Code Implementation
```python
import numpy as np

class BatchNorm1d:
    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        self.Z = Z
        if not eval:
            self.M = np.mean(Z, axis=0, keepdims=True)
            self.V = np.var(Z, axis=0, keepdims=True)
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)

        self.BZ = self.BW * self.NZ + self.Bb
        return self.BZ

    def backward(self, dLdBZ):
        N = self.Z.shape[0]
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW
        dLdV = np.sum(dLdNZ * (self.Z - self.M) * -0.5 * (self.V + self.eps)**(-1.5), axis=0, keepdims=True)
        dLdM = np.sum(dLdNZ * -1 / np.sqrt(self.V + self.eps), axis=0, keepdims=True) + \
               dLdV * np.mean(-2 * (self.Z - self.M), axis=0, keepdims=True)

        dLdZ = (dLdNZ / np.sqrt(self.V + self.eps)) + (dLdV * 2 * (self.Z - self.M) / N) + (dLdM / N)
        return dLdZ
```
