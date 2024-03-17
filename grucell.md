# GRU Cell

In a standard RNN, a long product of matrices can cause the long-term gradients to vanish (i.e reduce to zero) or explode (i.e tend to infinity). One of the earliest methods that were proposed to solve this issue is LSTM (Long short-term memory network). GRU (Gated recurrent unit) is a variant of LSTM that has fewer parameters, offers comparable performance and is significantly faster to compute. GRUs are used for a number of tasks such as Optical Character Recognition and Speech Recognition on spectrograms using transcripts of the dialog. In this section, you are going to get a basic understanding of how the forward and backward pass of a GRU cell work.

Replicate a portion of the `torch.nn.GRUCell` interface. Consider the following class definition.

```python
class GRUCell:
    def forward(self, x, h_prev_t):
        self.x = x
        self.hidden = h_prev_t
        self.r = # TODO
        self.z = # TODO
        self.n = # TODO
        h_t = # TODO
        return h_t

    def backward(self, delta):
        self.dWrx = # TODO
        self.dWzx = # TODO
        self.dWnx = # TODO
        self.dWrh = # TODO
        self.dWzh = # TODO
        self.dWnh = # TODO
        self.dbrx = # TODO
        self.dbzx = # TODO
        self.dbnx = # TODO
        self.dbrh = # TODO
        self.dbzh = # TODO
        self.dbnh = # TODO
        return dx, dh
```

As you can see in the code given above, the `GRUCell` class has forward and backward attribute functions.

In `forward`, we calculate `h_t`. The attribute function `forward` includes multiple components:
- As arguments, `forward` expects input `x` and `h_prev_t`.
- As attributes, `forward` stores variables `x`, `hidden`, `r`, `z`, and `n`.
- As an output, `forward` returns variable `h_t`.

In `backward`, we calculate the gradient changes needed for optimization. The attribute function `backward` includes multiple components:
- As an argument, `backward` expects input `delta`.
- As attributes, `backward` stores variables `dWrx`, `dWzx`, `dWnx`, `dWrh`, `dWzh`, `dWnh`, `dbrx`, `dbzx`, `dbnx`, `dbrh`, `dbzh`, `dbnh` and calculates `dz`, `dn`, `dr`, `dh_prev_t` and `dx`.
- As an output, `backward` returns variables `dx` and `dh_prev_t`.

NOTE: Your GRU Cell will have a fundamentally different implementation in comparison to the RNN Cell (mainly in the backward method). This is a pedagogical decision to introduce you to a variety of different possible implementations, and we leave it as an exercise to you to gauge the effectiveness of each implementation.

## 4.1 GRU Cell Forward

## Table 4: GRUCell Forward Components

| Code Name | Math | Type | Shape | Meaning |
|-----------|------|------|-------|---------|
| `input_size` | $$H_{in}$$ | scalar | — | The number of expected features in the input $$x$$ |
| `hidden_size` | $$H_{out}$$ | scalar | — | The number of features in the hidden state $$h$$ |
| `x` | $$x_t$$ | vector | $$H_{in}$$ | observation at the current time-step |
| `h_prev_t` | $$h_{t-1}$$ | vector | $$H_{out}$$ | hidden state at previous time-step |
| `Wrx` | $$W_{rx}$$ | matrix | $$H_{out} \times H_{in}$$ | Weight matrix for input (for reset gate) |
| `Wzx` | $$W_{zx}$$ | matrix | $$H_{out} \times H_{in}$$ | Weight matrix for input (for update gate) |
| `Wnx` | $$W_{nx}$$ | matrix | $$H_{out} \times H_{in}$$ | Weight matrix for input (for candidate hidden state) |
| `Wrh` | $$W_{rh}$$ | matrix | $$H_{out} \times H_{out}$$ | Weight matrix for hidden state (for reset gate) |
| `Wzh` | $$W_{zh}$$ | matrix | $$H_{out} \times H_{out}$$ | Weight matrix for hidden state (for update gate) |
| `Wnh` | $$W_{nh}$$ | matrix | $$H_{out} \times H_{out}$$ | Weight matrix for hidden state (for candidate hidden state) |
| `brx` | $$b_{rx}$$ | vector | $$H_{out}$$ | bias vector for input (for reset gate) |
| `bzx` | $$b_{zx}$$ | vector | $$H_{out}$$ | bias vector for input (for update gate) |
| `bnx` | $$b_{nx}$$ | vector | $$H_{out}$$ | bias vector for input (for candidate hidden state) |
| `brh` | $$b_{rh}$$ | vector | $$H_{out}$$ | bias vector for hidden state (for reset gate) |
| `bzh` | $$b_{zh}$$ | vector | $$H_{out}$$ | bias vector for hidden state (for update gate) |
| `bnh` | $$b_{nh}$$ | vector | $$H_{out}$$ | bias vector for hidden state (for candidate hidden state) |

In mytorch/gru cell.py implement the forward pass for a GRUCell using Numpy, analogous to the Pytorch equivalent nn.GRUCell (Though we follow a slightly different naming convention than the Pytorch documentation.) The equations for a GRU cell are the following:

$$\begin{align}
& rt = \sigma(W_{rx} \cdot x_t + b_{rx} + W_{rh} \cdot h_{t-1} + b_{rh}) \tag{4} \\
& zt = \sigma(W_{zx} \cdot x_t + b_{zx} + W_{zh} \cdot h_{t-1} + b_{zh}) \tag{5} \\
& nt = \tanh(W_{nx} \cdot x_t + b_{nx} + rt \odot (W_{nh} \cdot h_{t-1} + b_{nh})) \tag{6} \\
& h_t = (1 - zt) \odot nt + zt \odot h_{t-1} \tag{7}
\end{align}$$

Derive the appropriate shape of rt, zt, nt, ht using the equation given. Note the difference between element-wise multiplication and matrix multiplication.
Please refer to (and use) the GRUCell class attributes defined in the init method, and define any more attributes that you deem necessary for the backward pass. Store all relevant intermediary values in the forward pass.
The inputs to the GRUCell forward method are x and h_prev.t represented as xt and ht−1 in the equations above. These are the inputs at time t. The output of the forward method is ht in the equations above.

There are other possible implementations for the GRU, but you need to follow the equations above for the forward pass. If you do not, you might end up with a working GRU and zero points on autolab. Do not modify the init method, if you do, it might result in lost points.

Equations given above can be represented by the following figures:

## 4.2 GRU Cell Backward

In mytorch/gru cell.py implement the backward pass for the GRUCell specified before. The backward method of the GRUCell seems like the most time-consuming task in this homework because you have to compute 14 gradients but it is not difficult if you do it the right way. This method takes as input delta, and you must calculate the gradients w.r.t the parameters and return the derivative w.r.t the inputs, xt and ht−1, to the cell. The partial derivative input you are given, delta, is the summation of: the derivative of the loss w.r.t the input of the next layer xl+1,t and the derivative of the loss w.r.t the input hidden-state at the next time-step hl,t+1. Using these partials, compute the partial derivative of the loss w.r.t each of the six weight matrices, and the partial derivative of the loss w.r.t the input xt, and the hidden state ht.

# Table 5: GRUCell Backward Components

| Code Name | Math | Type | Shape | Meaning |
|-----------|------|------|-------|---------|
| `_delta` | $$\frac{\partial L}{\partial h_t}$$ | vector | $$H_{out}$$ | Gradient of loss w.r.t $$h_t$$ |
| `dWrx` | $$\frac{\partial L}{\partial W_{rx}}$$ | matrix | $$H_{out} \times H_{in}$$ | Gradient of loss w.r.t $$W_{rx}$$ |
| `dWzx` | $$\frac{\partial L}{\partial W_{zx}}$$ | matrix | $$H_{out} \times H_{in}$$ | Gradient of loss w.r.t $$W_{zx}$$ |
| `dWnx` | $$\frac{\partial L}{\partial W_{nx}}$$ | matrix | $$H_{out} \times H_{in}$$ | Gradient of loss w.r.t $$W_{nx}$$ |
| `dWrh` | $$\frac{\partial L}{\partial W_{rh}}$$ | matrix | $$H_{out} \times H_{out}$$ | Gradient of loss w.r.t $$W_{rh}$$ |
| `dWzh` | $$\frac{\partial L}{\partial W_{zh}}$$ | matrix | $$H_{out} \times H_{out}$$ | Gradient of loss w.r.t $$W_{zh}$$ |
| `dWnh` | $$\frac{\partial L}{\partial W_{nh}}$$ | matrix | $$H_{out} \times H_{out}$$ | Gradient of loss w.r.t $$W_{nh}$$ |
| `dbrx` | $$\frac{\partial L}{\partial b_{rx}}$$ | vector | $$H_{out}$$ | Gradient of loss w.r.t $$b_{rx}$$ |
| `dbzx` | $$\frac{\partial L}{\partial b_{zx}}$$ | vector | $$H_{out}$$ | Gradient of loss w.r.t $$b_{zx}$$ |
| `dbnx` | $$\frac{\partial L}{\partial b_{nx}}$$ | vector | $$H_{out}$$ | Gradient of loss w.r.t $$b_{nx}$$ |
| `dbrh` | $$\frac{\partial L}{\partial b_{rh}}$$ | vector | $$H_{out}$$ | Gradient of loss w.r.t $$b_{rh}$$ |
| `dbzh` | $$\frac{\partial L}{\partial b_{zh}}$$ | vector | $$H_{out}$$ | Gradient of loss w.r.t $$b_{zh}$$ |
| `dbnh` | $$\frac{\partial L}{\partial b_{nh}}$$ | vector | $$H_{out}$$ | Gradient of loss w.r.t $$b_{nh}$$ |
| `dx` | $$\frac{\partial L}{\partial x_t}$$ | vector | $$H_{in}$$ | Gradient of loss w.r.t $$x_t$$ |
| `dh_prev_t` | $$\frac{\partial L}{\partial h_{t-1}}$$ | vector | $$H_{out}$$ | Gradient of loss w.r.t $$h_{t-1}$$ |

The table above lists the 14 gradients to be computed, and delta is the input of the backward function.

How to start? Given below are the equations you need to compute the derivatives for backward pass. We also recommend refreshing yourself on the rules for gradients from Lecture 5.
IMPORTANT NOTE: As you compute the above gradients, you will notice that a lot of expressions are being reused. Store these expressions in other variables to write code that is easier for you to debug. This problem is not as big as it seems. Apart from dx and dh prev t, all gradients can computed in 2-3 lines of code. For your convenience, the forward equantions are listed here:

$$ rt = \sigma(W_{rx} \cdot xt + b_{rx} + W_{rh} \cdot h_{t-1} + b_{rh}) \tag{8}$$
$$ zt = \sigma(W_{zx} \cdot xt + b_{zx} + W_{zh} \cdot h_{t-1} + b_{zh}) \tag{9}$$
$$ nt = \tanh(W_{nx} \cdot xt + b_{nx} + rt \circ (W_{nh} \cdot h_{t-1} + b_{nh})) \tag{10}$$
$$ht = (1 − zt) \circ nt + zt \circ h_{t-1} \tag{11}$$

In the backward calculation, we start from terms involved in equation 11 and work back to terms involved in equation 8.

1. Forward Eqn: $$ht = (1 − z_t) \circ n_t + z_t \circ h_{t-1}$$
   (a) $$\frac{\partial L}{\partial zt} = \frac{\partial L}{\partial ht} \times \frac{\partial ht}{\partial zt}$$
   (b) $$\frac{\partial L}{\partial nt} = \frac{\partial L}{\partial ht} \times \frac{\partial ht}{\partial nt}$$

2. Forward Eqn: $$nt = \tanh(W_{nx} \cdot x_t + b_{nx} + r_t \circ (W_{nh} \cdot h_{t-1} + b_{nh}))$$
   (a) $$\frac{\partial L}{\partial W_{nx}} = \frac{\partial L}{\partial nt} \times \frac{\partial nt}{\partial W_{nx}}$$
   (b) $$\frac{\partial L}{\partial b_{nx}} = \frac{\partial L}{\partial nt} \times \frac{\partial nt}{\partial b_{nx}}$$
   (c) $$\frac{\partial L}{\partial rt} = \frac{\partial L}{\partial nt} \times \frac{\partial nt}{\partial rt}$$
   (d) $$\frac{\partial L}{\partial W_{nh}} = \frac{\partial L}{\partial n_t} \times \frac{\partial n_t}{\partial W_{nh}}$$
   (e) $$\frac{\partial L}{\partial b_{nh}} = \frac{\partial L}{\partial n_t} \times \frac{\partial n_t}{\partial b_{nh}}$$

3. Forward Eqn: $$z_t = \sigma(W_{zx} \cdot x_t + b_{zx} + W_{zh} \cdot h_{t-1} + b_{zh})$$
   (a) $$\frac{\partial L}{\partial W_{zx}} = \frac{\partial L}{\partial z_t} \times \frac{\partial z_t}{\partial W_{zx}}$$
   (b) $$\frac{\partial L}{\partial b_{zx}} = \frac{\partial L}{\partial z_t} \times \frac{\partial z_t}{\partial b_{zx}}$$
   (c) $$\frac{\partial L}{\partial W_{zh}} = \frac{\partial L}{\partial z_t} \times \frac{\partial z_t}{\partial W_{zh}}$$
   (d) $$\frac{\partial L}{\partial b_{zh}} = \frac{\partial L}{\partial z_t} \times \frac{\partial z_t}{\partial b_{zh}}$$

4. Forward Eqn: $$r_t = \sigma(W_{rx} \cdot x_t + b_{rx} + W_{rh} \cdot h_{t-1} + b_{rh})$$
   (a) $$\frac{\partial L}{\partial W_{rx}} = \frac{\partial L}{\partial r_t} \times \frac{\partial r_t}{\partial W_{rx}}$$
   (b) $$\frac{\partial L}{\partial b_{rx}} = \frac{\partial L}{\partial r_t} \times \frac{\partial r_t}{\partial b_{rx}}$$
   (c) $$\frac{\partial L}{\partial W_{rh}} = \frac{\partial L}{\partial r_t} \times \frac{\partial r_t}{\partial W_{rh}}$$
   (d) $$\frac{\partial L}{\partial b_{rh}} = \frac{\partial L}{\partial r_t} \times \frac{\partial r_t}{\partial b_{rh}}$$

5. Terms involved in multiple forward equations:

   (a) $$\frac{\partial L}{\partial x_t} = \frac{\partial L}{\partial n_t} \times \frac{\partial n_t}{\partial x_t} + \frac{\partial L}{\partial z_t} \times \frac{\partial z_t}{\partial x_t} + \frac{\partial L}{\partial r_t} \times \frac{\partial r_t}{\partial x_t}$$
   (b) $$\frac{\partial L}{\partial h_{t-1}} = \frac{\partial L}{\partial h_t} \times \frac{\partial h_t}{\partial h_{t-1}} + \frac{\partial L}{\partial n_t} \times \frac{\partial n_t}{\partial h_{t-1}} + \frac{\partial L}{\partial z_t} \times \frac{\partial z_t}{\partial h_{t-1}} + \frac{\partial L}{\partial r_t} \times \frac{\partial r_t}{\partial h_{t-1}}$$








