# Neural Network Layers

## Linear Layer

Linear layers, also known as fully-connected layers, connect every input neuron to every output neuron and are commonly used in neural networks. Refer to Figure A for the visual representation of a linear layer. In this section, your task is to implement the Linear class in file `linear.py` (Additionally, to enhance clarity, we’ve marked the equations requiring conversion into code with a distinctive blue color for all the sections):

- Class attributes:
  - Learnable model parameters weight W, bias b.
  - Variables stored during forward-propagation to compute derivatives during back-propagation: layer input A, batch size N ≥ 6.
  - Variables stored during backward-propagation to train model parameters dLdW, dLdb.

- Class methods:
  - `__init__`: Two parameters define a linear layer: `in_feature (C_in)` and `out_feature (C_out)`. Zero initialize weight W and bias b based on the inputs. Refer to Table 5.1 to see how the shapes of W and b are related to the inputs (Hint: - Check the shapes of the `in_feature` and `out_feature` and create a numpy array with zeros based on the required shape of W and b given in Table 5.1).
  - `forward`: forward method takes in a batch of data A of shape N x C_in (representing N samples where each sample has C_in features), and computes output Z of shape N x C_out – each data sample is now represented by C_out features.
  - `backward`: backward method takes in input dLdZ, how changes in its output Z affect loss L. It calculates and stores dLdW, dLdb – how changes in the layer weights and bias affect loss, which are used to improve the model. It returns dLdA, how changes in the layer inputs affect loss to enable downstream computation.

Please consider the following class structure:

```python
class Linear:

    def __init__(self, in_features, out_features):
        self.W = # TODO
        self.b = # TODO

    def forward(self, A):
        self.A = # TODO
        self.N = # TODO: store the batch size
        Z = # TODO

        return Z

    def backward(self, dLdZ):
        dLdA = # TODO
        dLdW = # TODO
        dLdb = # TODO
        self.dLdW = dLdW
        self.dLdb = dLdb

        return dLdA
```

```python
#mytorch.nn.linear.py
import numpy as np

class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        Z = A.dot(self.W.T) + self.b.T
        return Z


    def backward(self, dLdZ):
        self.Ones = np.ones((dLdZ.shape[0], 1))
        dLdA = dLdZ.dot(self.W)
        self.dLdW = dLdZ.T.dot(self.A)
        self.dLdb = dLdZ.T.dot(self.Ones)
        if self.debug:
            self.dLdA = dLdA
        return dLdA
```

