# Decode in Autoregressive Language Models

**Decode** is the incremental generation phase of a causal language model. After the prompt has been processed during prefill, the model uses previously computed **key-value (KV) cache** tensors to generate subsequent tokens without recomputing the entire prompt.

> **Key idea:** In a typical one-token decode step, the model processes just the most recently selected token and uses the cached context to predict the *next* token.

This article explains the attention computation, tensor shapes, KV-cache updates, performance trade-offs and a reproducible PyTorch example. It assumes a decoder-only Transformer.

## 1. Decode starts after the first token is selected

Consider this simplified execution timeline:

```text
Prompt: "The capital of France is"
       |
       v
PREFILL: process prompt, populate prompt KV cache
       |
       v
Use last prompt logits to select first output token: "Paris"
       |
       v
DECODE STEP 1: process "Paris" with prompt cache
               extend cache -> predict next token
       |
       v
Select "and"
       |
       v
DECODE STEP 2: process "and" with extended cache
               extend cache -> predict next token
       |
       v
... until EOS, a token limit or another stopping condition
```

The example words are illustrative; real tokenizers may split them differently. The important distinction is **prediction vs. processing**: a token can be sampled from prefill logits *before* it has been inserted into the attention cache. A subsequent decode forward pass processes it.

## 2. Shapes in a standard decode step

Let:

- `B`: batch size.
- `T`: number of tokens already in the KV cache.
- `H`: hidden size.
- `h_q`: number of query heads.
- `h_kv`: number of KV heads.
- `d`: attention head dimension.

| Tensor | Typical shape |
| --- | --- |
| Current input token IDs | `[B, 1]` |
| Current hidden states | `[B, 1, H]` |
| Current queries | `[B, h_q, 1, d]` |
| Newly projected keys/values | `[B, h_kv, 1, d]` |
| Existing K/V cache | `[B, h_kv, T, d]` |
| Updated K/V cache | `[B, h_kv, T+1, d]` |
| Attention scores, logically | `[B, h_q, 1, T+1]` |
| Next-token logits | `[B, 1, vocab_size]` |

The table assumes a **single new token per sequence** and a full-context KV cache. Grouped-query attention can have `h_q > h_kv`. A paged, sliding-window or quantized cache may be stored differently.

Some advanced methods (e.g., speculative decoding and multi-token verification) process more than one token in an incremental pass, so `sequence_length == 1` is a common execution pattern, not a universal definition of decode.

## 3. Cached attention math

For a new token representation `x_t`, compute the current projections:

\[
q_t = x_t W_Q,\qquad k_t = x_t W_K,\qquad v_t = x_t W_V
\]

The new `k_t` and `v_t` are added to existing cached keys and values:

\[
K_{\leq t} = [K_{<t}; k_t],\qquad V_{\leq t} = [V_{<t}; v_t]
\]

Then the one-position query attends over the full accessible history:

\[
o_t = \operatorname{softmax}
\left(\frac{q_t K_{\leq t}^{\top}}{\sqrt{d}}\right)
V_{\leq t}
\]

For a single query at the **last** position, every key in that past-and-current history is causally allowed (subject to any additional attention restrictions). Unlike prefill, this is not a square `T × T` attention score matrix: the logical score vector is length `T+1` per query head.

If the model uses positional encodings such as RoPE, they must be applied with the **correct absolute or logical position** before attention. This is essential for parity with an uncached reference.

### Why no square causal mask here?

A naive instruction to always set `is_causal=True` can be wrong for unequal query and key lengths because certain libraries interpret a non-square causal mask with a different alignment. In the one-new-token example below, the single query is the newest token and may attend to every supplied key. Using `is_causal=False` is correct **for this specific configuration**, provided that all keys truly belong to its accessible history.

This is *not* a general recommendation to turn off masking in incremental attention. Packed batches, padding, sliding windows and multiple-query decode steps may require explicit masks or specialized attention kernels.

## 4. KV-cache update and storage

Before a step, each attention layer typically holds:

```text
K_cache: [B, h_kv, T, d]
V_cache: [B, h_kv, T, d]
```

After incorporating the newly processed token:

```text
K_cache: [B, h_kv, T+1, d]
V_cache: [B, h_kv, T+1, d]
```

Conceptually, appending tokens grows the cache, but production engines do not necessarily use `torch.cat` on every step. They may write into preallocated memory, use page/block tables, cap retention with sliding windows, or compress KV tensors.

**Important:** The output token sampled *from* a decode step is not yet necessarily present in the cache. It joins the cache if and when it is passed through the next model forward pass.

## 5. Minimal PyTorch experiment: cached vs. full attention

This runnable example verifies one attention head group's **final-position parity**. The query, key and value projections are represented by fixed random tensors; it does not implement tokenization, weights, RoPE, feed-forward layers, an LM head or real text generation.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

B, NUM_HEADS, HISTORY, HEAD_DIM = 1, 2, 5, 4
TOTAL = HISTORY + 1

# Projected Q/K/V for the entire sequence, used as a reference.
q_all, k_all, v_all = (
    torch.randn(B, NUM_HEADS, TOTAL, HEAD_DIM)
    for _ in range(3)
)

# Cached history from earlier positions.
cache_k = k_all[:, :, :HISTORY, :]
cache_v = v_all[:, :, :HISTORY, :]

# Q/K/V belonging to the new final-position token.
q_new = q_all[:, :, HISTORY:HISTORY + 1, :]
k_new = k_all[:, :, HISTORY:HISTORY + 1, :]
v_new = v_all[:, :, HISTORY:HISTORY + 1, :]

# Conceptual cache update; real servers often avoid repeated concatenation.
cache_k = torch.cat([cache_k, k_new], dim=2)
cache_v = torch.cat([cache_v, v_new], dim=2)

# Every supplied key is in the last token's accessible history.
cached_last = F.scaled_dot_product_attention(
    q_new, cache_k, cache_v, is_causal=False, dropout_p=0.0
)

# Reference: fully causal attention over every sequence position.
full = F.scaled_dot_product_attention(
    q_all, k_all, v_all, is_causal=True, dropout_p=0.0
)

torch.testing.assert_close(
    cached_last, full[:, :, -1:, :], rtol=1e-5, atol=1e-6
)

print("Cached attention matches the full-reference last position.")
print("Cache length:", cache_k.shape[2])
```

Expected output:

```text
Cached attention matches the full-reference last position.
Cache length: 6
```

This example checks the **attention operation**, not equivalence of two full Transformer models. Real parity testing must also validate projections, layer normalization, positional encodings, residual paths and the correct cache for every layer.

## 6. Complexity and performance

Ignoring projection and feed-forward work, a naive full-attention pass across a length-`T` sequence has quadratic `O(T²d)` attention work per head. A one-query cached step performs approximately `O(Td)` attention work per head, because the new query attends over `T` or `T+1` cached keys.

This does **not** mean that the full model's decode step costs only `O(T)`: projections, feed-forward layers, communication, cache management and weight reads still matter.

### Common performance concepts

| Metric | Meaning | Why it matters |
| --- | --- | --- |
| TTFT (time to first token) | Time until the first generated token arrives | Influenced heavily by scheduling and prefill |
| ITL / TPOT | Inter-token latency / time per output token | Tracks decode responsiveness |
| Output tokens/s | Token generation rate | Throughput within one or many requests |
| KV-cache memory | Storage required for context | Limits concurrent requests and context lengths |
| Memory bandwidth | Rate of loading weights/cache data | Can dominate low-batch decode |

Decode is often **memory-bandwidth-sensitive** at small batch sizes, since each step uses relatively little arithmetic compared with the bytes fetched. At larger batch sizes and with different kernels, its bottleneck can change; saying that decode is *always* memory-bound is too broad.

### Parallelism and scheduling

A generation step depends on the previously chosen token, so successive autoregressive steps cannot all be computed independently in the conventional algorithm. However, the model's matrix operations, heads, layers and different requests can still exploit hardware parallelism. Techniques such as continuous batching and speculative decoding improve utilization and latency without eliminating the fundamental dependency.

## 7. Common mistakes

1. **Confusing output selection with cache insertion.** The first sampled output token is often chosen after prefill and only processed during the next decode forward pass.
2. **Using the wrong causal mask for a one-query, many-key tensor.** Mask semantics for non-square attention depend on the implementation.
3. **Forgetting positional offsets.** The new token has position `T` (under zero-based indexing), not position zero.
4. **Duplicating old KV tensors.** Do not append the whole prefix again on every incremental step.
5. **Assuming caches always grow forever.** Sliding windows and bounded contexts can evict or rotate entries.
6. **Assuming decode always equals one token.** Speculative and chunked incremental variants can process several tokens.
7. **Assuming cached and full attention will be bitwise identical.** Different kernels and operation ordering can yield small floating-point differences; evaluate numerical tolerances.

## 8. How to test a decode implementation

A useful validation plan compares a cached implementation to a trusted full-sequence reference:

1. Use the same model weights, token IDs, positional configuration and attention mask.
2. Run a full causal forward pass and capture the last-position logits or hidden states.
3. Run prefill on the prefix and incremental decode on its next token, preserving cache positions.
4. Compare logits and relevant layer outputs with suitable absolute/relative error tolerances.
5. Repeat with several batch sizes, prompt lengths, padding patterns and stopping conditions.
6. If supported, add GQA/MQA, sliding-window attention, quantized KV cache and cache-reuse cases.

A real generation test should also check token-by-token behavior, cache length, EOS handling, sampled/greedy outputs and end-to-end timing.

## 9. Takeaways

**Decode = incrementally process newly selected token(s), reuse and update KV cache, and predict further tokens.** It is the repeated phase that turns a processed prompt into an autoregressively generated continuation.

Read [Prefill](prefill.md) for the initial prompt-processing phase, and [Prefill vs Decode](prefill-vs-decode.md) for the overview.

## References

- [Hugging Face Transformers: Caching](https://huggingface.co/docs/transformers/main/cache_explanation)
- [Hugging Face Transformers: Cache strategies](https://huggingface.co/docs/transformers/main/kv_cache)
- [PyTorch: scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [vLLM: PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention/)
