# Optimizers

In deep learning, optimizers are used to adjust the parameters for a model. The purpose of an optimizer is to adjust model weights to maximize a loss function.

To recap, we built our own MLP models in Section 7 using linear class we built in Section 5 and activation classes we built in Section 6 and have seen how to do forward propagation, and backward propagation for the core components used in neural networks. Forward propagation is used for estimation, and backward propagation informs us on how changes in parameters affect loss. And in Section 8, we coded some loss functions, which are criterion we use to evaluate the quality of our model’s estimates. The last step is to improve our model using the information we learned on how changes in parameters affect loss.

## Stochastic Gradient Descent (SGD) [ mytorch.optim.SGD ]

In this section, we are going to implement Minibatch stochastic gradient descent with momentum, which we will refer to as SGD in this homework. Minibatch SGD is a version of SGD algorithm that speeds up the computation by approximating the gradient using smaller batches of the training data, and Momen- tum is a method that helps accelerate SGD by incorporating velocity from the previous update to reduce oscillations. The sgd function in PyTorch library is actually an implementation of Minibatch stochastic gradient descent with momentum.

Your task is to implement the step attribute function of the SGD class in file sgd.py: • Class attributes:
– l: list of model layers
– L: number of model layers
– lr: learning rate, tunable hyperparameter scaling the size of an update.
– mu: momentum rate μ, tunable hyperparameter controlling how much the previous updates affect the direction of current update. μ = 0 means no momentum.
– v W: list of weight velocity for each layer
– v b: list of bias velocity for each layer • Class methods:
– step: Updates W and b of each of the model layers:
∗ Because parameter gradients tell us which direction makes the model worse, we move opposite
the direction of the gradient to update parameters.
∗ When momentum is non-zero, update velocities v W and v b, which are changes in the gradient to get to the global minima. The velocity of the previous update is scaled by hyperparameter μ, refer to lecture slides for more details.
Please consider the following class structure:

```python
class SGD:
        def __init__(self, model, lr=0.1, momentum=0):
            self.l   = model.layers
            self.L   = len(model.layers)
            self.lr  = lr
            self.mu  = momentum
            self.v_W = [np.zeros(self.l[i].W.shape) for i in range(self.L)]
            self.v_b = [np.zeros(self.l[i].b.shape) for i in range(self.L)]
        def step(self):
            for i in range(self.L):
              if self.mu == 0:
                self.l[i].W = # TODO
                self.l[i].b = # TODO
              else:
                self.v_W[i] = # TODO
                self.v_b[i] = # TODO
                self.l[i].W = # TODO
                self.l[i].b = # TODO
```
| Code Name | Math | Type   | Shape | Meaning                                            |
|-----------|------|--------|-------|----------------------------------------------------|
| model     | -    | object | -     | model with layers attribute                        |
| 1         | -    | object | -     | layers attribute selected from the model           |
| L         | -    | scalar | -     | number of layers in the model                      |
| lr        | λ    | scalar | -     | learning rate hyperparameter to scale affect of new gradients |
| momentum  | μ    | scalar | -     | momentum hyperparameter to scale affect of prior gradients |
| v_W       | -    | list   | L     | list of velocity weight parameters, one for each layer |
| v_b       | -    | list   | L     | list of velocity bias parameters, one for each layer |
| v_w[i]    | vWi  | matrix | Ci+1 x Ci | velocity for layer i weight                        |
| v_b[i]    | vbi  | matrix | Ci+1 x 1  | velocity for layer i bias                          |
| l[i].w    | Wi   | matrix | Ci+1 x Ci | weight parameter for a layer                      |
| l[i].b    | bi   | matrix | Ci+1 x 1  | bias parameter for a layer                        |


## SGD Equation (Without Momentum)

$$W := W - \lambda \frac{\partial L}{\partial W}$$

$$b := b - \lambda \frac{\partial L}{\partial b}$$

## SGD Equations (With Momentum)

$$v_W := \mu v_W + \frac{\partial L}{\partial W}$$

$$v_b := \mu v_b + \frac{\partial L}{\partial b}$$

$$W := W - \lambda v_W$$

$$b := b - \lambda v_b$$


