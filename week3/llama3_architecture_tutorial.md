# Understanding the Llama 3.1 Transformer Architecture

A practical guide to reading model architecture printouts, understanding every component, and mapping these concepts to other modern LLMs.

---

## 1. What is a Causal Language Model (CausalLM)?

Before diving into the architecture, let's clarify what "CausalLM" means — it's right there in the class name `LlamaForCausalLM`.

A **Causal Language Model** is a model trained to predict the **next token** given all previous tokens. The word "causal" refers to the fact that information flows in only one direction: left to right. When predicting token `t`, the model can only "see" tokens `1, 2, ..., t-1` — never future tokens. This is enforced by a **causal attention mask** (a triangular matrix that blocks attention to future positions).

```
Input:   "The cat sat on the"
Target:  "cat sat on the mat"
```

At each position, the model outputs a probability distribution over the entire vocabulary for what comes next. During training, the loss is computed against the actual next token. During inference, you sample or greedily pick from that distribution and feed it back in — this is **autoregressive generation**.

### CausalLM vs. Other Paradigms

| Paradigm | Attention Direction | Example Models | Use Case |
|---|---|---|---|
| **CausalLM (decoder-only)** | Left-to-right only | GPT, Llama, Mistral, Falcon | Text generation, chat, code |
| **Masked LM (encoder-only)** | Bidirectional | BERT, RoBERTa | Classification, NER, embeddings |
| **Seq2Seq (encoder-decoder)** | Bidirectional encoder → causal decoder | T5, BART, mBART | Translation, summarization |

Almost all modern "large language models" (ChatGPT, Claude, Gemini, Llama, Mistral) are CausalLMs.

---

## 2. The Big Picture: Top-Level Structure

```
LlamaForCausalLM                          ← Task wrapper (next-token prediction)
├── model: LlamaModel                     ← The core transformer
│   ├── embed_tokens: Embedding           ← Token → vector
│   ├── layers: 16 x LlamaDecoderLayer   ← The repeated transformer blocks
│   ├── norm: LlamaRMSNorm               ← Final normalization
│   └── rotary_emb: LlamaRotaryEmbedding ← Positional encoding
└── lm_head: Linear                       ← Vector → vocabulary logits
```

The data flows like this:

```
Tokens → [Embedding] → [Layer 0] → [Layer 1] → ... → [Layer 15] → [RMSNorm] → [lm_head] → Logits
```

Let's walk through every component.

---

## 3. Token Embedding Layer

```python
(embed_tokens): Embedding(128256, 2048)
```

This is a lookup table that converts each input token ID (an integer) into a dense vector.

- **128,256** — the vocabulary size. Llama 3.1 uses a large BPE tokenizer. Every possible token (words, subwords, special tokens) gets an integer ID from 0 to 128,255.
- **2,048** — the **hidden dimension** (`d_model`). Each token is represented as a 2048-dimensional vector. This is the "width" of the model and the most fundamental hyperparameter — it flows through every layer.

```
Token ID 4523  →  lookup row 4523  →  [0.12, -0.34, 0.56, ..., 0.78]  (length 2048)
```

This is the same concept in every transformer. GPT-2 used `d_model=768`, GPT-3 used `d_model=12288`, Mistral 7B uses `d_model=4096`.

---

## 4. The Decoder Layer (repeated 16 times)

```python
(layers): ModuleList(
  (0-15): 16 x LlamaDecoderLayer(...)
)
```

This model has **16 identical layers** stacked on top of each other. Each layer has the same structure but its own learned weights. The number of layers is often called the model's "depth."

For reference: Llama 3.1 8B has 32 layers, Llama 3.1 70B has 80 layers. The 16-layer version you're looking at is likely the **Llama 3.2 1B** or a smaller variant, quantized to 4-bit (note the `Linear4bit` layers).

Each decoder layer has two major sub-blocks:

```
Input
  │
  ├──→ [input_layernorm] → [self_attn] → + (residual connection)
  │                                       │
  └───────────────────────────────────────┘
  │
  ├──→ [post_attention_layernorm] → [mlp] → + (residual connection)
  │                                          │
  └──────────────────────────────────────────┘
  │
Output
```

This is the **Pre-Norm** residual pattern (normalize before the sub-layer, not after). Let's break down each piece.

---

## 5. RMSNorm (Root Mean Square Normalization)

```python
(input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
(post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
```

Normalization stabilizes training by keeping activations in a reasonable range. **RMSNorm** is a simpler, faster alternative to the original LayerNorm:

```
            x_i
RMSNorm(x_i) = ──────── × γ_i
            RMS(x)

where RMS(x) = sqrt(mean(x²) + eps)
```

- It only rescales by the root-mean-square (no mean subtraction like LayerNorm).
- `γ` is a learnable scale parameter (2048 values, one per dimension).
- `eps=1e-05` prevents division by zero.

There are **two** per layer (before attention and before MLP), plus **one final** norm after all layers.

### Cross-Architecture Comparison

| Model | Normalization | Placement |
|---|---|---|
| **Llama, Mistral, Gemma** | RMSNorm | Pre-norm |
| **GPT-2** | LayerNorm | Pre-norm |
| **Original Transformer** | LayerNorm | Post-norm |
| **GPT-3** | LayerNorm | Pre-norm |

---

## 6. Self-Attention with Grouped Query Attention (GQA)

```python
(self_attn): LlamaAttention(
  (q_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
  (k_proj): Linear4bit(in_features=2048, out_features=512, bias=False)
  (v_proj): Linear4bit(in_features=2048, out_features=512, bias=False)
  (o_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
)
```

This is the heart of the transformer — the mechanism that lets each token attend to all previous tokens (with the causal mask enforcing the "no peeking at the future" rule).

### Deriving the Attention Configuration

From the dimensions, we can reverse-engineer the attention setup:

- **`q_proj` output: 2048** → Queries have full dimension.
- **`k_proj` output: 512** → Keys are 4× smaller than queries.
- **`v_proj` output: 512** → Values match keys.

If we assume a standard head dimension of `d_head = 64`:

```
num_query_heads  = 2048 / 64 = 32 heads
num_kv_heads     = 512 / 64  = 8 heads
GQA group ratio  = 32 / 8    = 4 (every 4 query heads share 1 KV head)
```

This is **Grouped Query Attention (GQA)** — a middle ground between:

- **Multi-Head Attention (MHA)**: Every query head has its own KV head (expensive).
- **Multi-Query Attention (MQA)**: All query heads share a single KV head (too aggressive).
- **GQA**: Query heads are grouped, and each group shares one KV head (balanced).

GQA dramatically reduces the size of the **KV cache** during inference, which is the main memory bottleneck for long sequences.

### How Attention Works (Step by Step)

```python
# 1. Project input into Q, K, V
Q = x @ W_q    # (batch, seq_len, 2048) → (batch, seq_len, 2048)
K = x @ W_k    # (batch, seq_len, 2048) → (batch, seq_len, 512)
V = x @ W_v    # (batch, seq_len, 2048) → (batch, seq_len, 512)

# 2. Reshape into heads
Q = Q.view(batch, seq_len, 32, 64)  # 32 query heads
K = K.view(batch, seq_len, 8, 64)   # 8 KV heads
V = V.view(batch, seq_len, 8, 64)

# 3. Apply Rotary Position Embeddings (RoPE) to Q and K
Q, K = apply_rope(Q, K)

# 4. Repeat K,V to match query head count (or use efficient grouped attention)
K = K.repeat_interleave(4, dim=2)  # 8 → 32 heads
V = V.repeat_interleave(4, dim=2)

# 5. Compute attention scores
scores = (Q @ K.transpose(-2, -1)) / sqrt(64)

# 6. Apply causal mask (upper triangle = -inf)
scores = scores.masked_fill(causal_mask, -inf)

# 7. Softmax → weighted sum of values
attn_weights = softmax(scores, dim=-1)
output = attn_weights @ V               # (batch, seq_len, 32, 64)

# 8. Concatenate heads and project
output = output.view(batch, seq_len, 2048)
output = output @ W_o                    # (batch, seq_len, 2048)
```

### Attention Across Architectures

| Model | Attention Type | Query Heads | KV Heads | d_head |
|---|---|---|---|---|
| **Llama 3.1 (this)** | GQA | 32 | 8 | 64 |
| **Llama 3.1 8B** | GQA | 32 | 8 | 128 |
| **Mistral 7B** | GQA | 32 | 8 | 128 |
| **GPT-2 (124M)** | MHA | 12 | 12 | 64 |
| **Falcon 40B** | MQA | 64 | 1 | 64 |
| **Gemma 2 9B** | GQA | 16 | 8 | 256 |

---

## 7. Rotary Position Embeddings (RoPE)

```python
(rotary_emb): LlamaRotaryEmbedding()
```

Transformers have no inherent sense of token order — attention is permutation-invariant. Positional embeddings fix this. Llama uses **RoPE** (Rotary Position Embeddings), which encodes position by **rotating** the query and key vectors in 2D subspaces.

The key insight: after applying RoPE, the dot product between query at position `m` and key at position `n` depends only on their **relative distance** `m - n`, not absolute positions. This gives the model a natural sense of "how far apart are these tokens?"

```
For each pair of dimensions (2i, 2i+1) in Q and K:
  θ_i = 1 / (10000^(2i/d_head))

  [q_2i  ]     [cos(m·θ_i)  -sin(m·θ_i)] [q_2i  ]
  [q_2i+1]  =  [sin(m·θ_i)   cos(m·θ_i)] [q_2i+1]
```

### Positional Encoding Across Architectures

| Model | Position Encoding | Max Context |
|---|---|---|
| **Llama 3.1** | RoPE | 128K tokens |
| **Mistral / Mixtral** | RoPE | 32K–128K |
| **GPT-2** | Learned absolute | 1024 |
| **GPT-3/4** | Learned absolute → various | 2K–128K |
| **Original Transformer** | Sinusoidal (fixed) | Fixed |

RoPE has become the dominant choice because it generalizes to longer sequences than those seen during training (with techniques like NTK-aware scaling or YaRN).

---

## 8. The MLP (Feed-Forward Network)

```python
(mlp): LlamaMLP(
  (gate_proj): Linear4bit(in_features=2048, out_features=8192, bias=False)
  (up_proj):   Linear4bit(in_features=2048, out_features=8192, bias=False)
  (down_proj): Linear4bit(in_features=8192, out_features=2048, bias=False)
  (act_fn): SiLUActivation()
)
```

After attention mixes information between token positions, the MLP processes each token position independently. Llama uses a **SwiGLU**-variant feed-forward network:

```python
# Standard FFN:       output = W_down(activation(W_up(x)))
# SwiGLU FFN (Llama): output = W_down(SiLU(W_gate(x)) * W_up(x))
```

Step by step:

```python
gate = silu(x @ W_gate)  # (batch, seq, 2048) → (batch, seq, 8192)
up   = x @ W_up          # (batch, seq, 2048) → (batch, seq, 8192)
hidden = gate * up        # Element-wise gating
output = hidden @ W_down  # (batch, seq, 8192) → (batch, seq, 2048)
```

- **Expansion ratio**: `8192 / 2048 = 4×`. The intermediate dimension is 4× the hidden dimension.
- **SiLU (Sigmoid Linear Unit)**: `silu(x) = x * sigmoid(x)`. A smooth, non-monotonic activation.
- **Gating mechanism**: The `gate_proj` controls (gates) the flow of information from `up_proj`. This is more expressive than a simple single-projection FFN.

### MLP Across Architectures

| Model | FFN Type | Activation | Expansion |
|---|---|---|---|
| **Llama, Mistral, Gemma** | SwiGLU (3 matrices) | SiLU | ~4× (varies) |
| **GPT-2, GPT-3** | Standard (2 matrices) | GELU | 4× |
| **Original Transformer** | Standard (2 matrices) | ReLU | 4× |
| **PaLM** | SwiGLU | SiLU | ~4× |

The SwiGLU FFN has an extra matrix compared to the standard FFN, but empirically achieves better performance for the same compute budget.

---

## 9. The Language Model Head

```python
(lm_head): Linear(in_features=2048, out_features=128256, bias=False)
```

The final linear layer projects from the hidden dimension back to the vocabulary size. It produces **logits** — one raw score per vocabulary token — which are then passed through softmax to get a probability distribution.

```python
logits = hidden_states @ W_lm_head  # (batch, seq, 2048) → (batch, seq, 128256)
probs = softmax(logits, dim=-1)     # Probability of each token in vocab
next_token = sample(probs)          # Or argmax for greedy decoding
```

**Weight tying**: Many models (GPT-2, Gemma, some Llama variants) share the weights of `lm_head` with the `embed_tokens` embedding. This halves the parameter cost of these two large matrices. In this printout, they appear as separate layers, but they may share the same underlying tensor.

---

## 10. Quantization: Linear4bit

```python
(q_proj): Linear4bit(in_features=2048, out_features=2048, bias=False)
```

You'll notice `Linear4bit` instead of `Linear` — this model has been **quantized** to 4-bit precision (likely using QLoRA/bitsandbytes or GPTQ).

- **Full precision (FP16/BF16)**: Each weight is 16 bits.
- **4-bit quantized**: Each weight is ~4 bits, reducing memory by ~4×.

A model that would need ~2GB in FP16 only needs ~500MB in 4-bit. The `lm_head` remains in full precision — it's critical for output quality and relatively small compared to the total.

Quantization is an inference optimization and doesn't change the architecture — it's the same math, just with compressed weights that get dequantized on the fly during matrix multiplications.

---

## 11. Parameter Count Estimation

Let's estimate the parameters for this model:

```
Embedding:     128,256 × 2,048                    = ~263M
Per layer:
  Attention:   (2048×2048) + 2×(2048×512) + (2048×2048) = ~10.5M
  MLP:         2×(2048×8192) + (8192×2048)               = ~50.3M
  Norms:       2 × 2048                                  = ~4K
  Layer total:                                           ≈ 60.8M
All layers:    16 × 60.8M                                ≈ 973M
Final norm:    2048                                      ≈ 2K
LM head:       2048 × 128,256                            ≈ 263M

Total: ~1.5B parameters (before quantization)
```

This aligns with **Llama 3.2 1B** (the smallest Llama 3 model), quantized to 4-bit.

---

## 12. How This Maps to Other Architectures

The Llama architecture is representative of modern decoder-only transformers. Here's how other models differ:

### Mistral 7B / Mixtral 8x7B
- Nearly identical to Llama in the base transformer.
- **Mixtral** replaces the single MLP with a **Mixture of Experts (MoE)**: 8 MLP "experts" where a learned router activates only 2 per token. This gives 8× the parameters but only 2× the compute.
- Uses **Sliding Window Attention** for efficiency.

### GPT-2 / GPT-3 / GPT-4
- Uses standard Multi-Head Attention (not GQA).
- Uses LayerNorm instead of RMSNorm.
- Uses GELU activation with a standard 2-matrix FFN.
- Uses learned absolute positional embeddings.
- Same overall structure: embedding → N × (attention + FFN) → head.

### Gemma / Gemma 2 (Google)
- Very similar to Llama (RMSNorm, RoPE, GQA, SwiGLU).
- Gemma 2 adds **logit soft-capping** in attention and interleaves local/global attention layers.

### Falcon
- Uses **Multi-Query Attention** (all heads share one KV head — more aggressive than GQA).
- Otherwise structurally similar.

### Qwen 2.5
- RoPE, GQA, SwiGLU, RMSNorm — essentially the Llama recipe.
- Uses **bias terms** in some layers (Llama uses `bias=False` everywhere).

### The Common Blueprint

Despite surface-level differences, nearly all modern CausalLMs follow this template:

```
Token Embedding (+ Positional Encoding)
    │
    ▼
┌─────────────────────────────┐
│  Normalization               │
│  Multi-Head Attention        │  ×N layers
│  Normalization               │
│  Feed-Forward Network        │
│  (with residual connections) │
└─────────────────────────────┘
    │
    ▼
Final Normalization
    │
    ▼
Linear Head → Vocabulary Logits
```

The innovations between models are incremental refinements — a different norm, a different attention grouping, a different activation — not fundamental structural changes.

---

## 13. Summary Cheat Sheet

| Component | This Model | Purpose |
|---|---|---|
| `embed_tokens` | 128K vocab → 2048-dim | Convert tokens to vectors |
| `layers (×16)` | 16 decoder blocks | Core processing |
| `self_attn` | 32Q/8KV heads (GQA) | Token-to-token relationships |
| `rotary_emb` | RoPE | Position awareness |
| `mlp` | SwiGLU, 4× expansion | Per-token transformation |
| `input_layernorm` | RMSNorm | Stabilize before attention |
| `post_attention_layernorm` | RMSNorm | Stabilize before MLP |
| `norm` | Final RMSNorm | Stabilize before output |
| `lm_head` | 2048 → 128K vocab | Predict next token |
| `Linear4bit` | 4-bit quantization | Memory efficiency |

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original transformer paper
- [Llama 2 Paper](https://arxiv.org/abs/2307.09288) — Details on GQA and training
- [Llama 3 Blog Post](https://ai.meta.com/blog/meta-llama-3/) — Llama 3 architecture updates
- [RoPE Paper](https://arxiv.org/abs/2104.09864) — Rotary Position Embeddings
- [SwiGLU Paper](https://arxiv.org/abs/2002.05202) — GLU variants for transformers
- [GQA Paper](https://arxiv.org/abs/2305.13245) — Grouped Query Attention
