#⚡️GPT generated, need furture validation⚡️
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
答案是：

> **读“这整个过程”就是 prefill。**

不是“读完之后才叫 prefill”，  
而是：

> 从开始读，到读完整段 —— 这一整个过程，就叫 prefill。

----------

## 🌱 用小说例子讲清楚

你朋友发来一段故事：

> “从前有一座城堡……”

当你：

-   从第一个字开始读
    
-   一直读到最后一个字
    

这一整个过程，就是：

> **Prefill**

不是等你读完才叫 prefill，  
而是 **读的过程本身就是 prefill。**

----------

## 🧠 再说清楚一点

假设那段文字有 10 句话。

你：

-   读第 1 句（大脑开始建立记忆）
    
-   读第 2 句
    
-   读第 3 句
    
-   …
    
-   读第 10 句
    

整个“从第 1 句到第 10 句的处理过程”就是 prefill。

----------

## 🤖 放回模型（用最简单语言）

如果 prompt 有 10 个 token：

模型会：

-   处理 token1
    
-   处理 token2
    
-   …
    
-   处理 token10
    

这一整次处理过程，就叫：

> prefill

而在这个过程中，同时也在生成 cache。

----------

## 🎯 一句话总结

-   **Prefill 是“处理整段已有内容的整个过程”**
    
-   不是“处理完之后”
    
-   不是“结果”
    
-   而是“过程本身”
    

----------

你这个问题问得非常对，因为很多人会误以为：

> “Prefill 是一个状态”

其实不是。

它是一个阶段（一个 processing 过程）。
# Prefill 与 Cache-Gen（生活化解释版）

## 🌱 生活化例子：你要和朋友一起读一本小说

假设你们打算一起续写一个故事。

### 📖 第一步：把前面的故事读一遍

你朋友发给你：

> "从前有一座城堡，里面住着一个孤独的骑士......"

在你开始续写之前，你必须：

-   把这段文字读完\
-   记住人物是谁\
-   记住发生了什么

这个过程就是：

## ✅ Prefill

你在"填充"自己的大脑背景信息。

------------------------------------------------------------------------

## 🧠 那 Cache-Gen 是什么？

当你读这段文字时，你的大脑会：

-   记住人物\
-   记住时间线\
-   记住关系

这些"记住的东西"就像是：

## 🗂 Cache（缓存）

所以：

## ✅ Cache-Gen = 生成记忆

------------------------------------------------------------------------

## 🧩 用一句话理解

-   **Prefill = 把已有的内容一次性读完**
-   **Cache-Gen = 在读的过程中建立记忆**

它们是同一件事的两个角度。

------------------------------------------------------------------------

## 🍽 再举一个更简单的例子

你去餐厅点餐。

### Prefill

你先完整看菜单。

### Cache-Gen

你在脑子里记住：

-   哪个菜辣\
-   哪个菜贵\
-   哪个是甜的

这些"记住的东西"就是 cache。

------------------------------------------------------------------------

## 🤖 放回到模型里的直觉理解

当用户输入一整段 prompt：

    Today is a beautiful day and I want to...

模型会：

1.  一次性处理整段文本\
2.  同时建立内部"记忆"（缓存）

这一步：

> 既叫 **Prefill**，也叫 **Cache-Gen**。

------------------------------------------------------------------------

## 🔥 为什么两个词都存在？

因为关注点不同：

  术语        强调什么
  ----------- --------------------
  Prefill     把整段内容"跑进去"
  Cache-Gen   建立"记忆缓存"

但工程上通常指同一个阶段。

------------------------------------------------------------------------

## 🎯 最简单结论

**Prefill 和 Cache-Gen 是同一个阶段。**

就是：

> 把完整 prompt 一次性跑一遍，并生成内部记忆。

------------------------------------------------------------------------
好，我们只讲 **cache-gen 是什么意思**，不扯别的。

----------

# 🧠 一句话解释

**cache-gen = 生成缓存（cache generation）**

就是：

> 把已有内容跑一遍，同时把“记忆”存起来。

----------

# 🍿 生活例子 1：看电视剧

你要看第 8 集。

在看第 8 集之前，你会：

-   回忆前 7 集发生了什么
    
-   记住人物关系
    
-   记住关键剧情
    

这个“回忆并建立记忆”的过程，就像：

> cache-gen

因为你在“生成记忆缓存”。

----------

# 📚 生活例子 2：考试做阅读理解

你读一整篇文章。

读完之后，你脑子里会：

-   记住主角是谁
    
-   记住发生了什么
    
-   记住时间顺序
    

你没有开始回答问题，  
但你已经建立了“背景记忆”。

这个阶段就是：

> cache-gen

----------

# 🤖 放回到模型里（最简单版本）

假设用户输入：

`Today is a beautiful day` 

模型在生成回答之前，会：

1.  把整句话算一遍
    
2.  把每个 token 的“记忆信息”存起来
    

这些“记忆信息”就是：

> KV cache

这个“建立 KV cache 的过程”就叫：

> cache-gen

----------

# 🧩 为什么叫 cache？

因为它像“缓存”一样：

-   后面生成新内容时
    
-   不用重新算前面所有内容
    
-   直接用已经存好的“记忆”
    

就像浏览器缓存网页一样。

## 🌱 用一句生活话总结

你读一段小说：

-   从“读”的角度看 → 这是 **prefill**
    
-   从“记住内容”的角度看 → 这是 **cache-gen**
    

你没有做两件事。  
只是从不同角度给同一件事起了两个名字。
