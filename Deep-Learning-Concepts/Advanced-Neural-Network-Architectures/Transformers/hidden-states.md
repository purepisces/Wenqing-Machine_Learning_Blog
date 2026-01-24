# Hidden States in Transformers (hidden-states.md)

Hidden states are the intermediate representations produced by a neural network as it processes an input. In Transformer-based models (BERT, GPT, T5, etc.), **hidden states are the per-token vectors emitted by each layer**. They are one of the most important concepts for understanding model internals, debugging, and building downstream applications (classification, retrieval, embedding, probing, etc.).

---

## 1. What are hidden states?

Given an input token sequence, a Transformer maps each token to a vector of size `hidden_dim` (also called *hidden size*, *model dimension*, or `d_model`). The model applies multiple Transformer blocks (attention + MLP + residual connections + normalization). After each block, the model produces a new set of token vectors.

- The output of the **embedding layer** is often called the **layer-0 hidden state**.
- The output after the **first Transformer block** is the **layer-1 hidden state**.
- ...
- The output after the **last Transformer block** is the **final hidden state** (often exposed as `last_hidden_state`).

A key detail: **hidden states are not a single vector**. They contain a vector for **every token** in the sequence.

---

## 2. Shapes and dimensions

The most common shape (batch-first) is:

- `hidden_state`: `[batch_size, seq_len, hidden_dim]`

Where:
- `batch_size (BS)`: number of sequences processed in parallel
- `seq_len (SS)`: number of tokens per sequence
- `hidden_dim (H)`: embedding / hidden size (e.g., 768, 1024, 4096, ...)

If you request **all layers’ hidden states**, you typically get a tuple/list:

- `all_hidden_states`: length = `num_layers + 1`
  - includes embedding output (layer 0) + each Transformer layer output (layers 1..L)
- each element has shape `[BS, SS, H]`

---

## 3. Where do hidden states come from in a Transformer?

A simplified Transformer block (encoder or decoder, ignoring some details):

1. Input hidden state: `x`
2. Self-attention: `attn_out = SelfAttention(x)`
3. Residual + normalization: `x = LayerNorm(x + attn_out)`
4. Feed-forward / MLP: `mlp_out = MLP(x)`
5. Residual + normalization: `x = LayerNorm(x + mlp_out)`

The `x` at the end of the block is that layer’s **hidden state output**.

> Different architectures may use **Pre-LN** vs **Post-LN** ordering, which changes *where* layer norm is applied, but the concept of “layer outputs” as hidden states remains the same.

---

## 4. Hidden states vs logits vs embeddings

It’s easy to mix these up:

- **Token embeddings**: vectors produced by the embedding layer (layer 0 hidden states).
- **Hidden states**: per-layer token representations (layer 0..L).
- **Logits**: unnormalized scores over the vocabulary, usually produced by applying a linear head to the final hidden states:
  - `logits = W @ hidden_states_last + b`
- **Sentence embedding / pooled embedding**: a single vector representing the entire sequence, derived from token-level hidden states via pooling (CLS token, mean pooling, etc.).

---

## 5. How to get hidden states in Hugging Face Transformers (PyTorch)

### 5.1 Enable hidden state outputs
Most models can return hidden states by setting `output_hidden_states=True`:

```python
import torch
from transformers import AutoTokenizer, AutoModel

name = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

inputs = tok("hello world", return_tensors="pt")

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

# Final layer hidden states:
last = out.last_hidden_state               # [BS, SS, H]

# All layers hidden states:
hs = out.hidden_states                     # tuple length (L+1)
print(len(hs), hs[0].shape, hs[-1].shape)  # (L+1), [BS, SS, H], [BS, SS, H]
```

### 5.2 Getting a single vector (pooling)

**CLS pooling** (common in BERT-like models):
```
cls_vec = last[:, 0, :] # [BS, H]
```

**Mean pooling** (common for embedding/retrieval):
```
mask = inputs["attention_mask"].unsqueeze(-1) # [BS, SS, 1] mean_vec = (last * mask).sum(dim=1) / mask.sum(dim=1) # [BS, H]
```

----------

## 6. How hidden states differ in encoder vs decoder models

### 6.1 Encoder-only models (e.g., BERT)

-   Output is typically token-level representations for the full input sequence.
    
-   `last_hidden_state` is widely used for classification or feature extraction.
    

### 6.2 Decoder-only models (e.g., GPT)

-   Hidden states are still `[BS, SS, H]`, but the model is trained autoregressively.
    
-   During generation, the model often uses a **KV cache** for efficiency:
    
    -   The cache stores keys/values from attention layers, not the hidden states themselves.
        
    -   Hidden states are still computed, but caching prevents recomputing attention projections for previous tokens.
        

----------

## 7. Practical uses of hidden states

### 7.1 Feature extraction

-   Use last layer hidden states (or a weighted sum of layers) as features for downstream tasks.
    

### 7.2 Sentence embeddings

-   Pool token hidden states into a single vector for retrieval, clustering, semantic search, etc.
    

### 7.3 Probing and interpretability

-   Analyze how information changes across layers (syntax → semantics, position information, etc.).
    
-   Linear probes can test what properties are encoded at each layer.
    

### 7.4 Debugging and model verification

-   Compare hidden states across implementations/devices (CPU vs accelerator).
    
-   Identify where numerical drift begins (which layer diverges first).
    

----------

## 8. Common pitfalls

1.  **Confusing `hidden_states` with `logits`**
    
    -   Hidden states: `[BS, SS, H]`
        
    -   Logits: `[BS, SS, vocab_size]`
        
2.  **Forgetting padding masks when pooling**
    
    -   Mean pooling should ignore padded tokens via `attention_mask`.
        
3.  **Assuming the last layer is always best**
    
    -   Some tasks benefit from middle layers or a mixture of layers.
        
4.  **Pre-LN vs Post-LN differences**
    
    -   Layer outputs may have different distributions; comparisons should be consistent across architectures.
        

----------

## 9. Quick checklist

-   Do you need per-token vectors? → use `last_hidden_state`
    
-   Do you need all layers? → `output_hidden_states=True`, use `out.hidden_states`
    
-   Do you need a single vector per sentence? → apply pooling (CLS / mean / max)
    
-   Are you comparing two runs? → compare layer-by-layer hidden states to localize divergence
    

----------

## 10. References (recommended reading)

-   Vaswani et al., _Attention Is All You Need_ (Transformer original paper)
    
-   Hugging Face Transformers documentation (model outputs, hidden states, attention masks)
    
-   BERT / GPT model docs (encoder-only vs decoder-only behavior)
    

 ``If you tell me which folder you want it under (e.g., `Advanced-Neural-Network-Architectures/Transformers/` vs `Sequence-Models/Transformers/`), I can also draft the exact README snippet you should add to link this file.``
