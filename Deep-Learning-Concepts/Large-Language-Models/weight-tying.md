# Using GPT generated
# Weight Tying in Language Models

## Definition

**Weight tying** is a technique in language models where the **input token embedding layer** and the **output language modeling head (`lm_head`)** share the same weight matrix.

In a causal language model, these two components perform different operations:

* **Input embedding (`embed_tokens`)**: converts a token ID into a hidden vector.
* **Output LM head (`lm_head`)**: converts a hidden vector into logits over the vocabulary.

Although the operations are different, both operate between the same two spaces:

```text
Vocabulary Space  <---->  Hidden Representation Space
```

Because of this relationship, the same weight matrix can be used in both directions.

---

## Basic Structure

A simplified causal language model looks like:

```text
Token IDs
   ↓
Embedding Layer
   ↓
Token Embeddings
   ↓
Transformer Layers
   ↓
Hidden States
   ↓
LM Head
   ↓
Vocabulary Logits
   ↓
Next Token
```

The two weights involved in weight tying are typically:

```text
model.embed_tokens.weight
lm_head.weight
```

Both normally have shape:

```text
[vocab_size, hidden_size]
```

For example, if:

```text
vocab_size = 32,000
hidden_size = 4,096
```

then both matrices have shape:

```text
[32000, 4096]
```

---

## Input Token Embedding

The embedding layer converts a discrete token ID into a continuous vector representation.

Suppose our vocabulary contains only three tokens:

```text
0 → cat
1 → dog
2 → apple
```

and the embedding matrix is:

```text
W =

cat    [ 1.0,  0.0]
dog    [ 0.8,  0.6]
apple  [-0.5,  0.9]
```

If the input token is:

```text
dog → token_id = 1
```

the embedding layer performs a lookup:

```python
W[1]
```

and obtains:

```text
[0.8, 0.6]
```

Therefore:

```text
Token ID
   ↓
Embedding Lookup
   ↓
Hidden Vector
```

Conceptually:

```text
dog → [0.8, 0.6]
```

### Embedding as Matrix Multiplication

Although embedding is normally implemented as an efficient table lookup, it can also be understood mathematically as matrix multiplication.

The one-hot representation of `dog` is:

```text
x = [0, 1, 0]
```

Then:

```text
xW = [0.8, 0.6]
```

Therefore the input embedding operation can conceptually be written as:

[
h = xW
]

where:

* (x): one-hot token representation
* (W): embedding matrix
* (h): hidden representation

---

## Output LM Head

After the input passes through the Transformer layers, the model produces a hidden state.

For example:

```text
h = [0.7, 0.5]
```

The model now needs to answer:

> Which vocabulary token should come next?

The LM head compares this hidden state with the output weight associated with every token.

Using the same matrix:

```text
W =

cat    [ 1.0,  0.0]
dog    [ 0.8,  0.6]
apple  [-0.5,  0.9]
```

we calculate:

```text
cat score
= [0.7, 0.5] · [1.0, 0.0]
= 0.70

dog score
= [0.7, 0.5] · [0.8, 0.6]
= 0.86

apple score
= [0.7, 0.5] · [-0.5, 0.9]
= 0.10
```

Therefore:

```text
cat    → 0.70
dog    → 0.86
apple  → 0.10
```

`dog` receives the largest logit and is therefore more likely to be selected as the next token.

The complete LM head operation is:

[
\text{logits} = hW^T
]

where:

* (h): Transformer hidden state
* (W): output weight matrix
* (W^T): transpose of the weight matrix
* `logits`: one score for every vocabulary token

---

## Why Can `embed_tokens` and `lm_head` Share Weights?

At first this may seem strange because the two layers perform different operations.

The key idea is:

> **Weight tying shares the parameter matrix, not the operation.**

The input embedding performs:

```text
token → vector
```

while the LM head performs:

```text
vector → token scores
```

Mathematically:

```text
Input:

one_hot_token @ W
        ↓
hidden representation
```

while the output performs:

```text
Output:

hidden_state @ W.T
        ↓
vocabulary logits
```

Therefore the same learned matrix `W` is used in opposite directions:

```text
                W
Vocabulary ----------> Hidden Space

                W.T
Hidden Space ---------> Vocabulary Scores
```

The two operations are different, but they can use the same underlying vocabulary representation.

Another intuitive way to think about this is as a dictionary.

The input embedding asks:

> What vector represents the token `dog`?

The output LM head asks:

> Which token vector best matches my current hidden state?

Both questions can use the same dictionary of token representations.

---

## Tied vs Untied Weights

### Tied Weights

When:

```python
tie_word_embeddings = True
```

the input embedding and LM head share the same parameter:

```text
                    W
                  /   \
                 /     \
        embed_tokens   lm_head
```

Conceptually:

```python
model.lm_head.weight = model.embed_tokens.weight
```

Importantly, this is stronger than simply saying that the numerical values are equal.

The two modules reference the **same learned parameter**.

Conceptually:

```python
model.lm_head.weight is model.embed_tokens.weight
```

---

### Untied Weights

When:

```python
tie_word_embeddings = False
```

the model maintains two independent matrices:

```text
embed_tokens → W_embed

lm_head      → W_output
```

Therefore:

```text
W_embed and W_output
```

can learn different values during training.

The two matrices usually still have the same shape:

```text
[vocab_size, hidden_size]
```

but they are separate parameters.

---

## Untying a Previously Tied Weight

Suppose a model originally contains one shared matrix:

```text
                 W
               /   \
      embed_tokens  lm_head
```

To untie it, we can clone the weight:

```python
w_embed = model.embed_tokens.weight
w_output = w_embed.clone()
```

The result is:

```text
embed_tokens → W1

lm_head      → W2
```

Immediately after cloning:

```text
values(W1) == values(W2)
```

but:

```text
W1 and W2 are independent tensors
```

They start with identical values but can later change independently.

---

## Why Use Weight Tying?

### 1. Reduce the Number of Parameters

Both embedding and LM-head matrices can be very large.

Their parameter count is:

[
\text{vocab_size} \times \text{hidden_size}
]

For example:

```text
vocab_size  = 128,000
hidden_size = 4,096
```

One embedding matrix contains approximately:

```text
128000 × 4096
≈ 524 million parameters
```

Without weight tying:

```text
Input embedding → ~524M parameters
Output LM head  → ~524M parameters
```

With weight tying, only one matrix is needed.

Therefore weight tying can save hundreds of millions of parameters for models with large vocabularies.

---

### 2. Share the Vocabulary Representation Space

The embedding layer learns:

```text
token → representation
```

while the LM head learns:

```text
representation → token score
```

Tying the two weights encourages the model to use a consistent representation of vocabulary tokens on both the input and output sides.

For example, the vector used to represent `dog` when reading the token is also used when deciding whether the next token should be `dog`.

---

### 3. Parameter Sharing

Weight tying is a form of **parameter sharing**.

A single parameter can receive gradient information from multiple usages in the computation graph.

When the embedding and LM head are tied, the shared matrix is affected by both:

```text
Input-side usage
      ↓
shared W
      ↑
Output-side usage
```

During training, gradients from these uses contribute to updating the same shared parameter.

---

## Does Weight Tying Mean the Two Layers Are the Same?

No.

This is an important distinction.

The following is **not** true:

```text
embed_tokens == lm_head
```

The two modules perform different computations.

Instead:

```text
embed_tokens.weight == shared parameter W
lm_head.weight       == shared parameter W
```

The **operations are different**, while the **parameter is shared**.

For example:

```text
embed_tokens:
    token_id → W[token_id]

lm_head:
    hidden_state → hidden_state @ W.T
```

---

## PyTorch Example

A simplified tied language model can be implemented as:

```python
import torch
import torch.nn as nn


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        # Imagine Transformer layers here

        logits = self.lm_head(hidden_states)

        return logits
```

We can verify that they reference the same parameter:

```python
model = TinyLanguageModel(
    vocab_size=1000,
    hidden_size=128,
)

print(
    model.embed_tokens.weight
    is model.lm_head.weight
)
```

Output:

```text
True
```

---

## Hugging Face: `tie_word_embeddings`

Many Hugging Face model configurations expose:

```python
config.tie_word_embeddings
```

Conceptually:

```text
tie_word_embeddings = True
```

means:

```text
input embedding weight
        ↕
output embedding / LM-head weight
```

are shared.

While:

```text
tie_word_embeddings = False
```

means they are independent parameters.

For causal language models such as Llama-style models, this usually refers to the relationship between:

```text
model.embed_tokens.weight
```

and:

```text
lm_head.weight
```

However, whether a particular pretrained model actually uses tying is a model-specific design decision and should be determined from that model's configuration and implementation.

---

## Checkpoint Implications

Weight tying also affects how model checkpoints can be represented.

For an untied model, conceptually the checkpoint needs two independent weights:

```text
model.embed_tokens.weight → W1
lm_head.weight            → W2
```

For a tied model, both model parameters represent the same learned matrix:

```text
                    W
                  /   \
model.embed_tokens     lm_head
```

Depending on the serialization framework and checkpoint format, the shared relationship may be represented without storing two independent copies of the same tensor.

When converting or deploying checkpoints, it is therefore important to know whether the target model/runtime expects:

```text
one shared source weight
```

or:

```text
two independent source weights
```

A mismatch between the checkpoint's tying convention and the target model's expected convention may require an explicit **weight untying** conversion.

---

## Summary

Weight tying does **not** mean that the embedding layer and LM head perform the same computation.

Instead, it means:

> Two different operations share the same learned parameter matrix.

The input side uses the matrix as:

```text
token → vector
```

while the output side uses it as:

```text
vector → token scores
```

Mathematically:

[
\text{Embedding: } h = xW
]

[
\text{LM Head: } \text{logits} = hW^T
]

Therefore:

```text
             Shared W
            /        \
           /          \
token → vector      vector → token scores
   Embedding              LM Head
```

This works because both sides connect the same two spaces:

```text
Vocabulary Space ↔ Hidden Representation Space
```

Weight tying reduces parameter count and allows the input and output sides of a language model to share a common vocabulary representation.

## Reference

* Press, O. and Wolf, L. — *Using the Output Embedding to Improve Language Models*
* Inan, H., Khosravi, K., and Socher, R. — *Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling*
* PyTorch `torch.nn.Embedding`
* PyTorch `torch.nn.Linear`
* Hugging Face Transformers model configuration and weight-tying utilities

