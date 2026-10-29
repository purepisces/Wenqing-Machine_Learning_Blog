# Quantization in Large Language Models (LLMs)

## 1. Definition

**Quantization** represents model values—most commonly weights and sometimes activations or the KV cache—using fewer bits than a higher-precision reference representation.

For example, an LLM deployed with BF16 weights might instead store some weights in INT8 or packed INT4. Quantization can reduce **model size and memory bandwidth**, and may improve **inference speed** if the hardware and kernels efficiently support the chosen format. Its trade-off is quantization error and additional implementation complexity.

> **Key idea:** Fine-tuning changes what a model has learned by updating its weights; quantization changes how weights (and potentially other tensors) are represented and computed. They can be used together.

A simplified inference path:

```text
High-precision pretrained model (e.g., BF16)
                 |
        Quantization method
                 |
      Quantized model (e.g., INT4)
                 |
   Quantized / dequantizing kernels
                 |
     Inference on target hardware
```

**Important:** A quantized model is usually a *variant* of the same base model and architecture, not a new Transformer architecture. However, its deployment may require different kernels, packing layouts, scale handling, and hardware capabilities.

---

## 2. Why Quantize LLMs?

An LLM with `N` parameters stored at `b` bits per parameter needs approximately:

$$
\text{Raw weight storage (bytes)} = \frac{N\,b}{8}
$$

For a **7-billion-parameter (7B)** model:

| Representation | Bits / weight | Theoretical raw weight size |
|---|---:|---:|
| FP32 | 32 | 28 GB |
| FP16 / BF16 | 16 | 14 GB |
| INT8 | 8 | 7 GB |
| INT4 | 4 | 3.5 GB |

These are **decimal GB** and ignore scales, zero-points, packing metadata, embeddings stored at other precisions, runtime buffers, activations, and KV cache. Actual checkpoints and inference memory footprints differ.

### Potential benefits

1. **Lower model storage:** fewer bytes to store and load the weights.
2. **Lower memory traffic:** useful when token decoding is limited by reading model weights.
3. **Higher deployment capacity:** larger models or more concurrent requests may fit in available memory.
4. **Potential throughput/latency improvements:** only if the backend has efficient low-precision execution.

### Costs

- Quantization introduces numerical error and potentially reduces quality.
- Low-bit kernels, tensor packing/unpacking, and dequantization can add overhead.
- Extra per-group scales and zero-points consume memory.
- Supported compute formats are **hardware- and backend-specific**. A smaller checkpoint is **not automatically faster**.

---

## 3. How Linear (Affine) Quantization Works

A common integer quantizer maps a floating-point value `x` to an integer `q`:

$$
q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{x}{s}\right) + z,\ q_{\min},\ q_{\max}\right)
$$

To approximate the floating-point value again, we **dequantize**:

$$
\hat{x} = s(q-z)
$$

Where:

- `s > 0`: **scale**, the spacing between representable values;
- `z`: **zero-point**, the integer corresponding to real zero;
- `q_min, q_max`: integer limits (for signed INT8, usually `-128` to `127`);
- `x_hat`: reconstructed approximation of `x`.

### 3.1 Worked example

Let the three floating-point weights be:

```text
x = [0.1234, 0.5678, -0.3456]
```

For illustration, use signed INT8 with `s = 0.01`, `z = 0`:

| Floating-point `x` | `round(x / 0.01)` | Dequantized `x_hat` | Error `x_hat - x` |
|---:|---:|---:|---:|
| 0.1234 | 12 | 0.12 | -0.0034 |
| 0.5678 | 57 | 0.57 | +0.0022 |
| -0.3456 | -35 | -0.35 | -0.0044 |

These values fit in INT8, so no clipping occurs. The reconstructed numbers differ slightly from the originals: this is **quantization error**.

The chosen `0.01` scale is just an example; actual quantizers determine it from statistics or an optimization objective.

### 3.2 Symmetric quantization

A typical signed symmetric INT8 convention uses `z = 0`, the range `[-127, 127]`, and:

$$
s = \frac{\max_i |x_i|}{127}
$$

Then `q = clamp(round(x/s), -127, 127)`. A symmetric scheme is simple, but it can waste codes when the distribution is strongly asymmetric.

### 3.3 Asymmetric quantization

An affine quantizer can use a nonzero zero-point. A common min/max construction is:

$$
s = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\qquad
z = \operatorname{clamp}\left(\operatorname{round}\left(q_{\min}-\frac{x_{\min}}{s}\right),q_{\min},q_{\max}\right)
$$

This can better fit a range that is not centered on zero. Practical implementations must handle degenerate ranges, clipping, and zero preservation.

---

## 4. Weight Quantization vs. Activation Quantization

Consider a Transformer linear layer:

$$
Y = XW^{\mathsf T} + b
$$

- `W`: stored **weights** (usually static during inference).
- `X`: runtime **activations** (change with each input).

### Weight-only quantization

Example: **W4A16** means roughly **4-bit weights** with **16-bit activations**.

```text
Weights:     INT4 (packed in memory)
Activations: FP16 / BF16
Computation: hardware-dependent; often dequantize weights in a kernel
```

It is attractive for LLM decoding because weights can dominate memory traffic. However, not every W4A16 kernel performs literal INT4 matrix multiplication: some unpack/dequantize weights into a higher-precision computation path.

### Weight-and-activation quantization

Example: **W8A8** means **8-bit weights** and **8-bit activations** for the quantized operations.

```text
Weights:     INT8
Activations: INT8
Accumulation: often wider precision (e.g., INT32), depending on kernel
```

This can lower compute and activation traffic too, but managing **activation outliers** and choosing accurate scales is challenging.

### KV-cache quantization

A third possibility is quantizing the stored **Key/Value tensors** during autoregressive decoding. This is different from weight quantization: it targets memory consumption that grows with sequence length and concurrency.

---

## 5. Quantization Granularity

The quantizer's **granularity** determines which values share one scale (and potentially zero-point).

| Granularity | One scale for... | Trade-off |
|---|---|---|
| Per-tensor | An entire tensor | Simple, but less adaptive |
| Per-channel | Each output/input channel, depending on layout | Better fidelity, more metadata |
| Per-group | A group of adjacent weights | Good compromise; common in low-bit LLMs |
| Per-token / dynamic | Values computed from an input token at runtime | Can adapt to activations; adds overhead |

For a weight matrix `W` with shape `[out_features, in_features]`, *per-output-channel* quantization typically assigns one scale to each row; *group-wise* quantization divides a row into groups of size `G` and assigns one scale per group.

**Smaller groups can reduce error but increase metadata and kernel complexity.** Hardware support determines which granularities are practical.

---

## 6. Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)

### Post-Training Quantization (PTQ)

PTQ quantizes a pretrained model **without performing a full training run**. Some PTQ methods use a calibration dataset to select scales, study activation ranges, or optimize quantized weights.

```text
Pretrained checkpoint
        |
  Calibration (optional / method-dependent)
        |
   Quantized checkpoint
        |
  Accuracy and latency evaluation
```

PTQ is generally less expensive than retraining and is common for LLM deployment.

### Quantization-Aware Training (QAT)

QAT simulates quantization effects in the training forward pass (often using **fake quantization**) while retaining trainable higher-precision parameters. Gradients are approximated through nondifferentiable quantization steps using techniques such as the straight-through estimator.

This can help recover quality at low precision, but it requires training compute and a suitable optimization setup.

### Popular LLM PTQ families

- **GPTQ:** post-training weight quantization with approximate second-order, layer-wise error minimization.
- **AWQ:** activation-aware weight quantization to protect salient weight channels.
- **SmoothQuant:** moves part of activation quantization difficulty into weights using a mathematically equivalent smoothing transformation, enabling practical W8A8-style inference.

These are **algorithm families**, not universal tensor formats. Their benefits depend on bit width, group size, calibration, model, and inference backend.

---

## 7. FP32, FP16, BF16, INT8, and INT4 Are Not Interchangeable

| Format | Bits | Representation | Typical role |
|---|---:|---|---|
| FP32 | 32 | Floating point | Reference computation and high-precision operations |
| FP16 | 16 | Floating point | Training/inference on compatible hardware |
| BF16 | 16 | Floating point | Training/inference with a wide exponent range |
| INT8 | 8 | Integer | Quantized weights and/or activations |
| INT4 | 4 | Integer | Often packed weight-only quantization |

**BF16 vs. INT8 matters:** both use fewer bits than FP32, but BF16 is still a floating-point representation with an exponent, while an affine INT8 quantizer typically needs scale and zero-point metadata. Casting a tensor to BF16 is **precision reduction**, not the same process as calibrating and quantizing it to INT8.

Also, `torch.int8` stores 8 bits, while **INT4 frequently requires packing two 4-bit values into a byte**, since ordinary PyTorch tensors do not by themselves make a generic `torch.int8` tensor occupy 4 bits per element.

---

## 8. Minimal PyTorch Example: Symmetric INT8 Quantization

This example demonstrates the formulas with ordinary tensors. It is **not** a speed-optimized quantized inference kernel.

```python
import torch


def quantize_symmetric_int8(x: torch.Tensor):
    """Return signed INT8 values and a per-tensor scale."""
    x = x.to(torch.float32)
    max_abs = x.abs().max().item()
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def dequantize_symmetric_int8(q: torch.Tensor, scale: float):
    return q.to(torch.float32) * scale


weights = torch.tensor([0.1234, 0.5678, -0.3456])
q, scale = quantize_symmetric_int8(weights)
reconstructed = dequantize_symmetric_int8(q, scale)

print("original:     ", weights)
print("scale:        ", scale)
print("INT8 values:  ", q)
print("dequantized:  ", reconstructed)
print("max abs error:", (weights - reconstructed).abs().max().item())

assert q.dtype == torch.int8
assert torch.allclose(weights, reconstructed, atol=scale / 2 + 1e-6)
```

The `atol = scale / 2` assertion works here because this example uses unclipped values and rounding to the nearest representable grid point. It is **not a universal error guarantee for arbitrary clipping or nonlinear quantizers**.

### What this example does *not* do

It does not pack INT4 weights, quantize GEMM execution, measure runtime speed, or prove accuracy on a full language model. A practical quantized deployment needs backend-specific kernels and end-to-end validation.

---

## 9. Quantization in a Transformer Inference Pipeline

A conceptual quantized linear projection could look like:

```text
BF16 hidden states ───────────────┐
                                  v
Packed INT4 weights → unpack / dequantize → matrix multiplication
                                  |
                                  v
                          output activations
```

The actual implementation may fuse these operations and use different accumulation and output types.

For an LLM, quantization might apply differently to:

- Q/K/V projection matrices;
- Attention output projection;
- MLP or MoE expert weights;
- LM head;
- KV cache.

A deployment need not quantize all of them. Sensitive layers may be left at higher precision.

### Prefill vs. decode

- **Prefill:** processes many prompt tokens; often compute-intensive, especially for large input batches.
- **Decode:** often processes one new token per sequence per step; reading model weights can be a major bottleneck, making weight compression useful.

This is a tendency, **not a guarantee**: performance depends on batch size, sequence length, cache handling, parallelism, and hardware implementation.

---

## 10. What Must Be Validated on Hardware?

Quantizing a checkpoint and **supporting that quantization on an accelerator** are distinct milestones.

A useful validation ladder:

1. **Representation:** can the loader read the packed weights, scales, and zero-points correctly?
2. **Graph / compilation:** can the compiler express and compile the required quantized operators?
3. **Kernel execution:** do quantized GEMM, unpacking, dequantization, and accumulation execute on hardware?
4. **Numerical accuracy:** compare layer outputs against a reference; measure MAE, RMSE, max error, and potentially cosine similarity.
5. **Checkpoint-level validation:** run the real model through prefill and decode and compare logits, token outputs, perplexity, or downstream task quality.
6. **Performance:** measure memory, throughput, latency, and scaling on the *actual* deployment configuration.

### Example: same architecture, different deployment requirements

```text
Qwen2.5-7B (BF16)
   └── requires BF16 GEMM, attention, RoPE, normalization, KV cache, etc.

Qwen2.5-7B (INT4 weight-only)
   ├── same underlying Transformer architecture
   └── additionally requires an INT4 weight format and a compatible
       low-bit inference path (including scales / packing / kernels)
```

**Conclusion:** “RDX supports the Qwen2 architecture” does not by itself prove that an INT4 Qwen checkpoint can be compiled, executed, or validated on RDU.

---

## 11. Quantization vs. Fine-Tuning vs. Compilation

| Concept | Main question | What changes? |
|---|---|---|
| **Fine-tuning** | Can the model learn a specialized task? | Learned parameter values (and sometimes adapters) |
| **Quantization** | Can we represent/compute with fewer bits? | Numerical representation, quantization metadata, kernels |
| **Compilation** | Can this graph execute on target hardware? | Executable graph, schedule, mappings, code artifacts |

A model may be **fine-tuned first, quantized second, and compiled for its target accelerator third**. These are conceptually separate but interact in deployment.

---

## 12. Summary

- Quantization trades numerical precision for **lower memory and potentially better inference efficiency**.
- The core affine mapping uses `scale` and `zero-point`; reconstruction produces **quantization error**.
- **W4A16** typically means low-bit weights with higher-precision activations; **W8A8** quantizes both.
- **Per-tensor, per-channel, and per-group** choices affect error, memory overhead, and kernel design.
- **PTQ** converts an already-trained model; **QAT** trains while simulating quantization effects.
- Sharing a model **architecture** does not guarantee support for every **quantization format** on a given accelerator.
- Always validate **numerics and actual hardware performance**—not just the checkpoint's disk size.

## References

1. [PyTorch Quantization API Reference](https://docs.pytorch.org/docs/stable/quantization-support)
2. [PyTorch PerChannelMinMaxObserver](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html)
3. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (2022)](https://arxiv.org/abs/2210.17323)
4. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (2023)](https://arxiv.org/abs/2306.00978)
5. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models (2022)](https://arxiv.org/abs/2211.10438)