Transformer vs RNN:
In an RNN, computations are performed sequentially. For a given sequence, the RNN processes one element at a time and maintains a hidden state that carries information from previous steps to the next. This inherent nature can make them slower for long sequences.


The transformer architecture, introduced in the paper “Attention Is All You Need,”, eliminates the sequential processing in favour of parallel processing. It uses self-attention mechanisms to weigh input elements differently and can consider all sequence elements simultaneously.
