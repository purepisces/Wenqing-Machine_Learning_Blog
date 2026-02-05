# ⚠️Need further refactor:⚠️
# MoE Block: Router → Dispatch → Expert MLP → Combine

This note explains the **Mixture-of-Experts (MoE) block** as it appears in modern large language models (e.g. Switch Transformer, Mixtral, Qwen MoE).  
The goal is to bridge the gap between **high-level MoE concepts** and **actual block-level implementations** used in training and inference.

We focus on a **top-k sparse MoE block**, which is the most common design in practice.

---

## 1. What Is an MoE Block?

An **MoE block** replaces a standard dense MLP with a sparse, conditional computation:

- Each token is routed to **k experts** (instead of all experts)
- Only selected experts are executed
- Outputs are **weighted and combined** back per token

At a high level, an MoE block performs:

```
hidden_states
   ↓
Router / Gate
   ↓
Top-k expert selection
   ↓
Dispatch tokens to experts
   ↓
Expert MLP computation
   ↓
Weighted combine
   ↓
hidden_states (same shape as input)
```

The block is shape-preserving:
```
Input : [batch_size, seq_len, hidden_dim]
Output: [batch_size, seq_len, hidden_dim]
```

---

## 2. Input and Shape Conventions

In practice, batch dimensions are often **multi-dimensional** (e.g. data parallel, micro-batch, pipeline stages).

We assume the input has shape:

```
hidden_states: [*bs_dims, seq_len, hidden_dim]
```

Where:
- `*bs_dims` : one or more batch-related dimensions
- `seq_len`  : sequence length
- `hidden_dim`: model hidden size

A common pattern is to **flatten batch dimensions**:

```python
*bs_dims, ss, hidden_dim = hidden_states.shape
bs = math.prod(bs_dims)
hidden_states = hidden_states.reshape(bs, ss, hidden_dim)
```

This simplifies routing and expert computation.

---

## 3. Router (Gate)

### 3.1 Router Logits

The **router** (also called the *gate*) is usually a linear layer:

```
hidden_dim → num_experts
```

For each token:

```python
router_logits = gate(hidden_states)
```

Shape:
```
router_logits: [bs, ss, num_experts]
```

Each value represents how strongly a token prefers a given expert.

---

### 3.2 Routing Probabilities

Routing probabilities are computed with softmax **over the expert dimension**:

```python
routing_probs = softmax(router_logits, dim=-1)
```

Common practice:
- Perform softmax in **float32** for numerical stability
- Cast back to model dtype later if needed

---

## 4. Top-k Expert Selection

Instead of sending a token to all experts, we select the **top-k experts**:

```python
topk_weights, topk_indices = topk(routing_probs, k)
```

Shapes:
```
topk_indices: [bs, ss, k]
topk_weights: [bs, ss, k]
```

Each token now has:
- `k` expert IDs
- `k` corresponding routing weights

---

### 4.1 Optional Re-normalization

Some implementations re-normalize top-k weights:

```python
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
```

This ensures the selected experts’ weights sum to 1.

---

## 5. Dispatch: Token → Expert

To compute expert outputs efficiently, tokens are often **flattened**:

```python
flat_tokens = bs * ss
```

Each token has `k` expert assignments, so we conceptually work with:

```
(flat_tokens × k) token–expert pairs
```

Expert indices are flattened:

```python
flat_expert_idx = topk_indices.reshape(-1)
```

---

## 6. Expert MLP Structure

Each expert is an independent MLP.  
A common design (e.g. SwiGLU-style) uses **three projections**:

- `gate_proj`: hidden_dim → ffn_dim
- `up_proj`  : hidden_dim → ffn_dim
- `down_proj`: ffn_dim → hidden_dim

For expert `e`:
```
y = down_proj_e( activation(gate_proj_e(x), up_proj_e(x)) )
```

---

## 7. Expert Computation (Reference Style)

For each token–expert pair:

1. **Gather expert weights** by index
2. **Replicate token hidden states** k times
3. Apply expert MLP independently

Conceptually:

```python
gate_out = Wg @ x
up_out   = Wu @ x
act      = activation(gate_out, up_out)
y        = Wd @ act
```

Where each `(Wg, Wu, Wd)` corresponds to the selected expert.

This reference-style implementation is easy to reason about and verify, but not optimized for performance.

---

## 8. Combine: Expert → Token

Each token receives `k` expert outputs.  
They are combined using routing weights:

```python
y = sum_i ( topk_weight_i * expert_output_i )
```

After combining:
```
y: [flat_tokens, hidden_dim]
```

Reshape back to the original batch structure:

```python
y = y.reshape(*bs_dims, ss, hidden_dim)
```

---

## 9. Output of an MoE Block

An MoE block typically returns:

- **MoE output** (same shape as input)
- **Router logits** (for auxiliary loss or debugging)

```
Output hidden states: [*bs_dims, ss, hidden_dim]
Router logits       : [*bs_dims, ss, num_experts]
```

---

## 10. Training vs Inference Considerations

Key differences across implementations:

- **Auxiliary load-balancing loss** (training only)
- **Capacity constraints** to limit tokens per expert
- **Dropless MoE** variants (no token dropping)
- **Noise injection** in router logits (training stability)

Inference often disables:
- Aux loss
- Router noise
- Token dropping

---

## 11. Reference vs Optimized Implementations

The reference flow described here is ideal for:
- Understanding correctness
- Debugging numerical issues
- Writing unit tests

High-performance implementations typically:
- Group tokens **by expert**
- Avoid explicit token replication
- Use fused kernels (dispatch + MLP + combine)
- Minimize scatter/gather overhead

---

## 12. Summary

An MoE block consists of four core stages:

1. **Router/Gate**: score tokens over experts
2. **Top-k Selection**: choose sparse experts per token
3. **Expert MLP**: compute expert-specific transformations
4. **Combine**: weighted sum back to token space

Understanding the MoE block at this level makes it much easier to:
- Read real LLM MoE implementations
- Debug routing or accuracy issues
- Reason about performance trade-offs

This block-level view is the foundation for scaling MoE models in practice.
——————
# MoE 里的 router / gate / top-k：一次性讲清（外卖例子版）

## 一句话最重要结论

> **router 和 gate 不是两个阶段，也不是两次计算。它们是“同一套数值”的不同叫法。**  
> - **gate 输出 = router logits**  
> - **softmax(router logits) = routing weights**  
> - **top-k 只是筛选，不会重新算一套权重**

---

## 你最容易产生的误解（也最常见）

很多人会下意识以为：

```
hidden_state
   ↓
router：算一遍权重（0.6, 0.25, …）
   ↓
gate：再算一遍权重（0.8, 0.2, …）
```

**这是不对的。**  
正确流程里 **只有一次** “打分/路由” 的计算。

---

## 正确的真实流程（按代码变量名对齐）

### Step 1：gate = router（同一件事）

```python
router_logits = self.gate(hidden_states)
```

- `self.gate` 是一个 `nn.Linear(hidden_size, num_experts)`
- 输出叫 `router_logits`
- **这一步就是 router 在“打分”**（只是 raw score / logits，还不是概率）

举个 logits 的例子（随便举，不代表真实值）：

```
router_logits =
[ 日料:  1.2
  川菜:  3.4
  汉堡: 2.6
  甜品: 0.8 ]
```

---

### Step 2：softmax（把 logits 变成概率）

```python
routing_weights = softmax(router_logits)
```

这一步只是数学变换，不是新模型。

变成：

```
routing_weights =
[ 日料:  0.10
  川菜:  0.60
  汉堡: 0.25
  甜品: 0.05 ]
```

---

### Step 3：Top-K（只筛选，不重算）

```python
routing_weights, router_indices = topk(routing_weights, k=2)
```

**关键点：**
- ✅ 只挑出最大的 K 个
- ❌ 不会重新计算权重
- ❌ 不会把 (0.60, 0.25) 变成另一套完全不同的数

结果就是：

```
选中专家：
  川菜   0.60
  汉堡   0.25
```

---

### Step 4：可选 renorm（比例缩放，让和=1）

如果你开启了：

```python
if norm_topk_prob:
    routing_weights /= routing_weights.sum()
```

那么：

```
原来：
川菜 0.60
汉堡 0.25
sum = 0.85

归一化后：
川菜 0.60 / 0.85 ≈ 0.70
汉堡 0.25 / 0.85 ≈ 0.30
```

注意：
- ✅ 排名没变
- ✅ 相对大小没变
- ✅ 只是为了加起来等于 1
- ❌ 不是重新路由 / 不是重新算 gate

---

## 最终完整 mental model（再也不会混）

```
hidden_state
   ↓
gate (Linear)        ← 这一步也叫 router（职责名）
   ↓
router_logits
   ↓
softmax
   ↓
routing_weights
   ↓
top-k（筛选，不重算）
   ↓
（可选）renorm（比例缩放）
```

---

## 为什么名字会让人误解？

名字来源不同，强调点不同：

| 名字 | 更像来自 | 强调什么 |
|---|---|---|
| gate | 神经网络术语 | 连续可学习的“门控/权重” |
| router | MoE 系统结构术语 | 决定 token 走哪几条专家路径 |

但在实现里：

> **gate 是 router 的可学习核心。**  
> router 是“职责名”，gate 是“实现名”。

---

## 用一句话永久记住

> **router 不是一个额外模块；router 就是 gate 在做的事。**

