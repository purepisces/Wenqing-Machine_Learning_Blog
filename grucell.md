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
