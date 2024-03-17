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



