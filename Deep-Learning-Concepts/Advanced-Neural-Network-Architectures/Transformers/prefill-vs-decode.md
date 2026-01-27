# Prefill vs Decode in Transformer-based Language Models

Large Language Models (LLMs) such as GPT-style models generate text **autoregressively**:  
each new token is generated based on all previously generated tokens.

To make this process efficient in practice, modern Transformer implementations conceptually
split generation into two phases:

- **Prefill**
- **Decode**

Although both phases run the same Transformer layers, their **inputs, computation patterns,
tensor shapes, and system-level implications** are very different.

This document explains *why* the split exists and *how* prefill and decode differ in theory
and in real systems.

---

## 1. Why Do We Have Prefill and Decode?

Consider a prompt-based generation task:

```text
Prompt: "The capital of France is"
Target: " Paris"
```
The model must:

1.  Read and process the **entire prompt**
    
2.  Then generate **one token at a time**, feeding each new token back into the model
    

If we naïvely recomputed attention over the full sequence at every generation step,  
the cost would grow quadratically with sequence length.

To avoid this inefficiency, generation is split into two phases:

-   **Prefill**: process the full prompt once
    
-   **Decode**: generate tokens incrementally using cached intermediate results
    

----------

## 2. Prefill Phase

### 2.1 What Is Prefill?

**Prefill** is the phase where the model processes the _entire input prompt_ in one forward pass.

Example:

`Input tokens: ["The", "capital", "of", "France", "is"]` 

All tokens are fed into the Transformer at once.

----------

### 2.2 Input and Hidden State Shapes

During prefill, the typical tensor shapes are:

`input_ids:      [BS, SS]
hidden_states:  [BS, SS, H]` 

Where:

-   **BS** = batch size
    
-   **SS** = sequence length (prompt length)
    
-   **H** = hidden size
    

Each token in the prompt produces a hidden state at every Transformer layer.

----------

### 2.3 Attention Computation in Prefill

In prefill:

-   **Self-attention is computed over the full sequence**
    
-   Each token attends to all previous tokens (causal masking applies)
    

This results in:

-   High arithmetic intensity
    
-   Large attention matrices of shape `[SS, SS]` per head
    

Prefill is:

-   **Compute-heavy**
    
-   **Memory-bandwidth intensive**
    
-   Usually executed once per request
    

----------

### 2.4 KV Cache Creation

A crucial side effect of prefill is the creation of the **Key / Value (KV) cache**.

For each Transformer layer:

-   Keys and values for all prompt tokens are stored
    
-   These cached tensors are reused during decode
    

Conceptually:

`KV cache per layer:
K: [BS, SS, H]
V: [BS, SS, H]` 

This cache enables efficient token-by-token generation later.

----------

## 3. Decode Phase

### 3.1 What Is Decode?

**Decode** is the phase where the model generates **new tokens one at a time**.

At each step:

1.  The previously generated token is fed into the model
    
2.  The model predicts the next token
    
3.  The new token is appended to the sequence
    
4.  The process repeats
    

----------

### 3.2 Input and Hidden State Shapes

Unlike prefill, decode processes **a single token per step**.

Typical shapes:

`input_ids:      [BS, 1]
hidden_states:  [BS, 1, H]` 

Even though only one token is processed, the model still attends to _all previous tokens_  
via the KV cache.

----------

### 3.3 Attention Computation in Decode

In decode:

-   Query comes from the **current token**
    
-   Keys and values come from the **cached KV tensors**
    

Attention complexity per step becomes:

`O(SS) instead of O(SS²)` 

This dramatically reduces computation per token.

However:

-   Decode is executed **many times**
    
-   It is often **latency-sensitive**
    
-   Small inefficiencies are amplified over many steps
    

----------

### 3.4 KV Cache Update

At each decode step:

-   The new token’s key and value are computed
    
-   They are appended to the existing KV cache
    

Conceptually:
```
Old K: [BS, T, H]
New K: [BS, T+1, H]
```

This growing cache is what allows efficient autoregressive generation.

----------


## 4. Prefill vs Decode: Side-by-Side Comparison

| Aspect | Prefill | Decode |
|------|--------|--------|
| Tokens processed per step | Full prompt (SS tokens) | 1 token |
| Hidden states shape | `[BS, SS, H]` | `[BS, 1, H]` |
| Attention pattern | Full causal self-attention | Cached attention |
| KV cache | Created | Reused and extended |
| Compute cost | High | Low per step |
| Number of executions | Once | Many times |
| Performance bottleneck | Throughput | Latency |

---

## 5. System and Hardware Implications

### 5.1 Throughput vs Latency

- **Prefill** primarily impacts **throughput**  
  - Large matrix multiplications  
  - High arithmetic intensity  
  - Efficient on GPUs and accelerators when batched

- **Decode** primarily impacts **end-to-end latency**  
  - Executed token by token  
  - Sequential dependency between steps  
  - Sensitive to kernel launch overhead and memory access

Optimizing only prefill does not guarantee fast generation,  
and optimizing only decode does not guarantee high throughput.

---

### 5.2 Batch Size Behavior

- Prefill often benefits from **larger batch sizes**
  - Better hardware utilization
  - Amortized kernel launch cost

- Decode batching is more constrained due to:
  - Variable sequence lengths
  - Different stopping conditions (e.g., EOS tokens)
  - Real-time latency requirements in production systems

As a result, decode batching is significantly harder to scale.

---

### 5.3 Why Decode Is Often the Harder Problem

Although decode performs less computation per step, it:

- Runs sequentially and cannot be fully parallelized
- Executes many times per request
- Requires frequent reads from the KV cache
- Is highly sensitive to memory bandwidth and latency

In many real-world systems, decode dominates user-perceived latency
despite having lower per-step compute cost.

---
## 6. Common Misconceptions

### Misconception 1: “Decode is cheap”

Decode is cheap _per step_, but expensive _in total_ due to repetition and latency constraints.

----------

### Misconception 2: “Prefill doesn’t matter because it runs once”

For long prompts or large batches, prefill can dominate total compute cost.

----------

### Misconception 3: “Prefill and decode are different models”

They use the **same Transformer weights**.  
Only the _execution mode_ and _inputs_ differ.

----------

## 7. Summary

-   **Prefill** processes the full prompt and builds the KV cache
    
-   **Decode** generates tokens incrementally using cached attention
    
-   Both phases are essential for efficient autoregressive generation
    
-   Understanding their differences is critical for:
    
    -   Performance optimization
        
    -   Hardware acceleration
        
    -   Large-scale inference system design
        

