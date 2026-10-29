# Prefill in Autoregressive Language Models

**Prefill** is the inference phase in which a decoder-only language model processes the existing prompt and prepares the context needed for subsequent token generation. In a typical KV-cached implementation, prefill both **computes the prompt's hidden states** and **populates the key-value (KV) cache**.

This article focuses on causal Transformer language models. Other architectures and inference engines can implement the same conceptual phase differently.

> **Key idea:** Prefill processes the prompt; decode incrementally continues from it. They normally reuse the same trained model weights, even when the serving system uses different execution graphs or kernels.

## 1. Prefill in a generation timeline

Suppose the prompt is:

```text
Prompt tokens:  [The] [capital] [of] [France] [is]
                      |
                    PREFILL
                      |
            KV cache for the prompt
                      |
           Predict first new token
                      |
                    [Paris]
                      |
       DECODE: process [Paris] -> predict next token
```

The crucial distinction is that **the forward pass over the prompt usually produces logits for the first new token**. Sampling or selecting that token happens immediately after prefill. The next incremental forward pass consumes that selected token and belongs to decode.

For a prompt of length `P`, ordinary prefill processes those `P` tokens in parallel (subject to causal masking). Some systems instead process the prompt in multiple **prefill chunks**; it is still prefill.

## 2. Tensor shapes

Let:

- `B`: batch size.
- `P`: prompt length in tokens.
- `H`: model hidden size.
- `L`: number of Transformer layers.
- `h_q`: number of query heads.
- `h_kv`: number of key/value heads.
- `d`: per-head dimension.

Typical inputs and intermediates have shapes:

| Tensor | Shape |
| --- | --- |
| Input token IDs | `[B, P]` |
| Hidden states | `[B, P, H]` |
| Queries (after head reshape) | `[B, h_q, P, d]` |
| Keys and values | `[B, h_kv, P, d]` |
| Attention output (before head merge) | `[B, h_q, P, d]` |
| Logits | `[B, P, vocab_size]` (often only the final position is needed) |

In standard multi-head attention, `h_q = h_kv`. With **grouped-query attention (GQA)**, several query heads may share a key/value head, so `h_q > h_kv`.

## 3. Causal self-attention during prefill

For simplicity, consider one attention head with prompt representations `X`:

\[
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V
\]

The causal attention result is:

\[
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)V
\]

Here `M` is a causal mask that prevents a token from attending to future prompt tokens.

For four prompt positions, the *allowed* attention pattern is:

```text
                   Key positions
                   1   2   3   4
Query position 1   ✓   ·   ·   ·
Query position 2   ✓   ✓   ·   ·
Query position 3   ✓   ✓   ✓   ·
Query position 4   ✓   ✓   ✓   ✓
```

All prompt positions can be evaluated in the same model forward pass, but the mask preserves the causal dependency.

**Computational note:** Naive full attention performs approximately `O(P² d)` attention work per head. The logical attention scores have shape `[P, P]`, although memory-efficient attention kernels can avoid storing the entire score matrix.

## 4. What prefill stores: the KV cache

For every attention layer, the model retains keys and values for the processed prompt:

```text
Layer 0: K[0:P], V[0:P]
Layer 1: K[0:P], V[0:P]
...
Layer L-1: K[0:P], V[0:P]
```

A common layout is:

```text
K_cache[layer]: [B, h_kv, P, d]
V_cache[layer]: [B, h_kv, P, d]
```

The cache is **not** a copy of the prompt text and is **not** a general-purpose long-term memory. It holds intermediate key/value tensors that later attention steps reuse.

For a uniform full-attention model, an approximate KV-cache size is:

\[
\text{KV bytes} \approx 2 \times L \times B \times P
\times h_{kv} \times d \times \text{bytes per element}
\]

The factor `2` is for K and V. This estimate excludes allocator overhead, padding and other buffers. Sliding-window attention or compressed/quantized caches can have different storage behavior.

### Why cache K and V?

Without a cache, generating the next token by rerunning the full growing prefix would repeatedly recompute representations of earlier tokens. With a KV cache, incremental decode reuses the earlier keys and values and computes new ones only for the newly processed token(s).

## 5. Minimal PyTorch example

This runnable example demonstrates **causal prefill attention and cache construction for one attention layer**. It uses already projected Q, K and V, not a full language model; its random tensors do not produce meaningful text.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

B, NUM_HEADS, PROMPT_LEN, HEAD_DIM = 1, 2, 5, 4
q, k, v = (
    torch.randn(B, NUM_HEADS, PROMPT_LEN, HEAD_DIM)
    for _ in range(3)
)

# Each prompt position attends only to itself and earlier positions.
prefill_output = F.scaled_dot_product_attention(
    q, k, v, is_causal=True, dropout_p=0.0
)

# Illustrative cache for one layer; actual LLMs cache per layer.
kv_cache = {
    "key": k.clone(),
    "value": v.clone(),
}

assert prefill_output.shape == (B, NUM_HEADS, PROMPT_LEN, HEAD_DIM)
assert kv_cache["key"].shape == (B, NUM_HEADS, PROMPT_LEN, HEAD_DIM)
print("Prefill complete; cached tokens:", kv_cache["key"].shape[2])
```

Expected output:

```text
Prefill complete; cached tokens: 5
```

The example omits token embeddings, positional encodings, residual connections, feed-forward layers, the final language-model head and token sampling. The next token is selected using the final prompt position's logits in a real causal LLM.

## 6. Why prefill and decode behave differently

Prefill typically has many query positions and performs large matrix multiplications, so it can achieve high arithmetic intensity. Decode often has only one new query position per sequence and repeatedly reads model weights and the growing KV cache.

| Property | Prefill |
| --- | --- |
| Main input | Existing prompt tokens |
| Tokens processed in an ordinary forward pass | Many |
| Attention structure | Causal attention over the prompt |
| KV cache | Create or extend the prompt cache |
| Output | Prompt hidden states; next-token logits from the final position |
| Performance concern | Prompt processing latency / time to first token |

**Time to first token (TTFT)** includes more than model prefill on a deployed server: queueing, tokenization, scheduling, sampling and transport may also contribute. Improving prefill compute alone may not eliminate all TTFT.

### Chunked prefill

A long prompt can be processed in several chunks to control scheduling or memory pressure. The system must preserve the correct positional information and attention access to earlier chunks.

### Prefix caching

If multiple requests share exactly the same reusable prefix and compatible execution conditions, a serving system may reuse previously computed KV tensors. This can save redundant prefill work; it is different from performing a normal prefill for each request.

## 7. Common misconceptions

1. **"Prefill only creates the cache."** It also computes hidden states, and normally produces the logits used to choose the first output token.
2. **"Prefill means processing tokens one by one."** Causal masking encodes dependency, but implementations often process many prompt positions concurrently.
3. **"The full attention matrix must be stored."** It is a useful mathematical abstraction; optimized kernels may never materialize it.
4. **"Prefill is always one execution."** Chunked prefill can divide the prompt into multiple executions.
5. **"The cache contains the next generated token immediately after prefill."** Usually the cache initially contains the processed prompt; the first sampled token joins it when subsequently processed.

## 8. What to validate in an implementation

- The prompt positions, attention mask and positional encodings are correct.
- The KV cache length and tensor shape match the number of processed tokens.
- Chunked and unchunked prefill agree numerically within appropriate tolerances.
- Logits at the last prompt position agree with a trusted non-cached reference.
- Padding and variable prompt lengths do not leak future or padded positions into attention.
- Downstream decode consumes the cache without recomputing the entire prompt.

## 9. Takeaways

**Prefill = process the prompt, build the initial KV cache and produce the first next-token prediction.** It establishes the context that makes incremental generation efficient.

Continue with [Decode](decode.md), or read the [Prefill vs Decode comparison](prefill-vs-decode.md).

## References

- [Hugging Face Transformers: Caching](https://huggingface.co/docs/transformers/main/cache_explanation)
- [Hugging Face Transformers: Cache strategies](https://huggingface.co/docs/transformers/main/kv_cache)
- [PyTorch: scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [vLLM: Disaggregated prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
