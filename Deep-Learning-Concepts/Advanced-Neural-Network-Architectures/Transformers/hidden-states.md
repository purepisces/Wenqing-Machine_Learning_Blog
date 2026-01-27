# ⚡️temperoray using GPT generated, need further validation⚡️ Hidden States in Transformers (hidden-states.md)

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


## 一个最直观的例子：2 句话、每句 4 个 token、每个 token 用 3 维向量表示

假设：

-   **BS = 2**（这一批有 2 条句子）
    
-   **SS = 4**（每条句子我们统一 pad/截断成 4 个 token）
    
-   **H = 3**（为了好看懂，假设每个 token 的 hidden 向量只有 3 维；真实模型可能是 768/4096）
    

两条句子：

1.  “I love cats” → token: `[I, love, cats, <pad>]`
    
2.  “You like dogs” → token: `[You, like, dogs, <pad>]`
    

现在每个 token 经过 embedding（以及前面几层）都会变成一个向量，比如：

-   `I -> [0.2, -0.1, 0.9]`
    
-   `love -> [1.1, 0.0, -0.3]`
    
-   …
    

那 **hidden_states** 就是把这批所有 token 的向量装进一个三维数组：

### hidden_states 的形状：`[BS, SS, H] = [2, 4, 3]`

你可以把它想象成：

-   第 1 维（BS）：选第几句话
    
-   第 2 维（SS）：选这句话里的第几个 token
    
-   第 3 维（H）：这个 token 的向量坐标

比如：

-   `hidden_states[0, 1, :]` = 第 0 条句子的第 1 个 token（“love”）的 3 维向量
    
-   `hidden_states[1, 2, :]` = 第 1 条句子的第 2 个 token（“dogs”）的 3 维向量

---
----------

## 1）单个 token 的 hidden state

最细粒度：某一层里、某个 token 的表示向量。

-   记作：`h[b, s, :]`
    
-   形状：`[H]`
    

例子：  
`hidden_states[0, 1, :]` = 第 0 条句子里第 1 个 token（比如 “love”）在某一层的向量。

----------

## 2）一个句子的 hidden states（注意常用复数）

更常见：一句话里**所有 token 的 hidden state** 组成一个序列。

-   形状：`[SS, H]`
    

也就是把每个 token 的 `[H]` 堆起来。

----------

## 3）整个 batch 的 hidden states（更常见）

在代码里 `hidden_states` 这个变量通常就是指整个 batch：

-   形状：`[BS, SS, H]`
    

里面包含了：

-   BS 条句子
    
-   每条 SS 个 token
    
-   每个 token 一个 H 维向量
    

----------

## 4）“一个句子的 hidden state”有时指句向量（sentence embedding）

有些人确实会口头说“句子的 hidden state”，但严格讲那通常是从 token-level hidden states **汇聚**出来的句向量，比如：

-   取 `[CLS]` 的向量（BERT 类）
    
-   mean pooling：对所有 token 向量做平均
    
-   last token pooling：取最后一个 token 向量（某些 decoder-only 用法）
    

这种句向量形状也是 `[H]`，但它不是“天然就有的单个 state”，而是从 `[SS, H]` 汇聚来的。

----------

### 一句话记法

-   **token hidden state**：一个 token 的向量 `[H]`
    
-   **sentence hidden states**：一句话所有 token 的向量 `[SS, H]`
    
-   **batch hidden states**：一批句子 `[BS, SS, H]`
    
-   **sentence embedding**：从 token hidden states 聚合得到的 `[H]`（有时也被口头叫“句子的 hidden state”）

----------

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
