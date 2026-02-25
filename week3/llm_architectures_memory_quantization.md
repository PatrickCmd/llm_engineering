# LLM Architectures, Memory Representations & Quantization

## A Deep-Dive Tutorial: From Floating-Point Bits to Production Deployment

---

## Part 1: The Three Foundational Architectures

Every modern language model descends from the 2017 Transformer paper ("Attention Is All You Need"). That original design was an **encoder-decoder**, but researchers quickly discovered that using only half the architecture — either the encoder or the decoder — worked better for specific tasks. This split produced the three families that dominate NLP today.

### 1.1 Encoder-Only (Bidirectional)

```
Input:  "The [MASK] sat on the mat"
         ↓    ↓    ↓   ↓   ↓   ↓
       ┌──────────────────────────┐
       │   Bidirectional Attention │  ← Every token sees ALL other tokens
       │   (Full attention mask)   │
       └──────────────────────────┘
         ↓    ↓    ↓   ↓   ↓   ↓
Output: ctx  "cat" ctx ctx ctx ctx    ← Predict the masked token
```

**How it works:** Every token attends to every other token in both directions. The model builds a rich contextual representation of the *entire* input simultaneously. Training objective is typically Masked Language Modeling (MLM) — randomly mask 15% of tokens and predict them.

**Attention mask (fully visible):**
```
     The [MASK] sat  on  the  mat
The  [ 1    1    1    1    1    1 ]
MASK [ 1    1    1    1    1    1 ]
sat  [ 1    1    1    1    1    1 ]
on   [ 1    1    1    1    1    1 ]
the  [ 1    1    1    1    1    1 ]
mat  [ 1    1    1    1    1    1 ]
```

**Key models:** BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT, DistilBERT, XLM-RoBERTa

**Best for:** Classification, named entity recognition, semantic similarity, embeddings, retrieval — tasks where you have the *complete* input and need to understand it, not generate new text.

**Typical architecture (BERT-base):**
```
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768)       ← Token embeddings
    (position_embeddings): Embedding(512, 768)     ← Learned absolute positions
    (token_type_embeddings): Embedding(2, 768)     ← Segment A vs B
    (LayerNorm): LayerNorm(768)                    ← Post-norm architecture
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 × BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query):  Linear(768, 768, bias=True)  ← Full MHA (12 heads × 64 dim)
            (key):    Linear(768, 768, bias=True)  ← Same size as query
            (value):  Linear(768, 768, bias=True)  ← Same size as query
          )
          (output): BertSelfOutput(
            (dense): Linear(768, 768)              ← Output projection
            (LayerNorm): LayerNorm(768)            ← Post-attention norm
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(768, 3072)               ← FFN up-project (4× expansion)
          (intermediate_act_fn): GELUActivation()  ← GELU, not SiLU
        )
        (output): BertOutput(
          (dense): Linear(3072, 768)               ← FFN down-project
          (LayerNorm): LayerNorm(768)              ← Post-FFN norm
        )
      )
    )
  )
  (pooler): BertPooler(                           ← [CLS] token pooling for classification
    (dense): Linear(768, 768)
    (activation): Tanh()
  )
)
```

**Contrast with Llama:**

| Feature | BERT (Encoder) | Llama (Decoder) |
|---|---|---|
| Attention | Bidirectional (full) | Causal (triangular mask) |
| Normalization | Post-LayerNorm | Pre-RMSNorm |
| Position encoding | Learned absolute (max 512) | RoPE (extendable to 128K+) |
| FFN | 2-layer with GELU | 3-layer SwiGLU (gated) |
| Bias | Yes, everywhere | No bias |
| Attention type | Full MHA | GQA |
| Output | Contextual embeddings | Next-token logits |
| Max context | 512 tokens | 128K tokens |

---

### 1.2 Decoder-Only (Causal / Autoregressive)

```
Input:  "The cat sat on the"
         ↓    ↓    ↓   ↓   ↓
       ┌──────────────────────────┐
       │     Causal Attention      │  ← Each token sees only previous tokens
       │   (Triangular mask)       │
       └──────────────────────────┘
         ↓    ↓    ↓   ↓   ↓
Output: "cat" "sat" "on" "the" "mat" ← Predict the NEXT token at each position
```

**Attention mask (causal / lower-triangular):**
```
     The  cat  sat  on  the
The  [ 1    0    0    0    0 ]
cat  [ 1    1    0    0    0 ]
sat  [ 1    1    1    0    0 ]
on   [ 1    1    1    1    0 ]
the  [ 1    1    1    1    1 ]
```

**Key models:** GPT-2/3/4, Llama 1/2/3, Mistral, Mixtral, Falcon, Phi, Qwen, Gemma, Command-R, Yi, DeepSeek

**Best for:** Text generation, chat, code completion, reasoning, instruction-following — essentially everything a "modern AI assistant" does.

**This is the Llama architecture we dissected previously.** The defining characteristics of the modern decoder-only model are:

```
Modern Decoder-Only Recipe (2023+):
├── Token Embedding (large vocab, 100K-256K)
├── Rotary Position Embedding (RoPE)
├── N × Decoder Block:
│   ├── Pre-RMSNorm
│   ├── Grouped Query Attention (GQA)
│   │   ├── Q projection (full heads)
│   │   ├── K projection (fewer heads)
│   │   ├── V projection (fewer heads)
│   │   ├── RoPE applied to Q and K
│   │   ├── Causal mask applied
│   │   └── O projection
│   ├── Residual connection
│   ├── Pre-RMSNorm
│   ├── SwiGLU FFN (gate + up + down projections)
│   └── Residual connection
├── Final RMSNorm
└── Linear head → vocabulary logits
```

---

### 1.3 Encoder-Decoder (Sequence-to-Sequence)

```
Encoder (bidirectional):              Decoder (causal + cross-attention):
"Translate: The cat sat"              "<start> Le chat"
    ↓     ↓     ↓    ↓                    ↓     ↓
┌─────────────────────────┐          ┌─────────────────────────┐
│ Bidirectional Attention  │          │ Causal Self-Attention    │
│ (full mask on input)     │ ──────→ │ Cross-Attention to enc.  │
└─────────────────────────┘          │ FFN                      │
    ↓     ↓     ↓    ↓               └─────────────────────────┘
  [encoder hidden states]                 ↓      ↓
                                       "Le"   "chat"  "s'est"
```

**Key models:** T5, BART, mBART, mT5, FLAN-T5, NLLB, MarianMT, Whisper (for speech)

**Best for:** Translation, summarization, tasks with clear input→output mapping.

**Typical architecture (T5-base):**
```
T5ForConditionalGeneration(
  (shared): Embedding(32128, 768)              ← Shared between encoder and decoder

  (encoder): T5Stack(
    (embed_tokens): Embedding(32128, 768)
    (block): ModuleList(
      (0-11): 12 × T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(           ← Bidirectional self-attention
            (SelfAttention): T5Attention(
              (q): Linear(768, 768, bias=False)
              (k): Linear(768, 768, bias=False)
              (v): Linear(768, 768, bias=False)
              (o): Linear(768, 768, bias=False)
              (relative_attention_bias)        ← T5-style relative positions
            )
            (layer_norm): T5LayerNorm(768)     ← Pre-norm (like Llama)
          )
          (1): T5LayerFF(                      ← Standard FFN (no gating)
            (DenseReluDense):
              (wi): Linear(768, 2048)
              (wo): Linear(2048, 768)
            (layer_norm): T5LayerNorm(768)
          )
        )
      )
    )
    (final_layer_norm): T5LayerNorm(768)
  )

  (decoder): T5Stack(
    (embed_tokens): Embedding(32128, 768)
    (block): ModuleList(
      (0-11): 12 × T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(           ← CAUSAL self-attention
            (SelfAttention): T5Attention(...)
          )
          (1): T5LayerCrossAttention(          ← ★ Cross-attention to encoder
            (EncDecAttention): T5Attention(
              (q): Linear(768, 768)            ← Q from decoder states
              (k): Linear(768, 768)            ← K from encoder output
              (v): Linear(768, 768)            ← V from encoder output
              (o): Linear(768, 768)
            )
          )
          (2): T5LayerFF(...)                  ← Same FFN structure
        )
      )
    )
    (final_layer_norm): T5LayerNorm(768)
  )

  (lm_head): Linear(768, 32128, bias=False)   ← Generate output tokens
)
```

**The unique element is cross-attention** — a layer where the decoder's queries attend to the encoder's keys and values. This is how information flows from the input (encoder) to the output (decoder):

```python
# Cross-attention pseudocode
Q = decoder_hidden @ W_q     # Queries from decoder
K = encoder_output @ W_k     # Keys from encoder
V = encoder_output @ W_v     # Values from encoder

# No causal mask — decoder fully attends to all encoder positions
attn = softmax(Q @ K.T / sqrt(d)) @ V
```

---

### 1.4 Architecture Comparison Summary

```
                  Encoder-Only          Decoder-Only           Encoder-Decoder
                  ───────────           ────────────           ───────────────
Attention:        Bidirectional         Causal (left→right)    Enc: Bidirectional
                                                               Dec: Causal + Cross

Training:         Masked LM (MLM)      Next-token prediction   Seq2Seq (input→output)

Generates text?   No (embeddings)      Yes (autoregressive)    Yes (conditioned on input)

Input handling:   Sees full input       Sees only past          Encoder sees full input;
                  at once               tokens                  decoder generates output

Modern examples:  DeBERTa-v3           Llama 3.1, GPT-4,      FLAN-T5, NLLB,
                  BGE, E5              Mistral, Phi-3          Whisper

Typical params:   110M - 1.5B          1B - 405B+              250M - 13B

Primary use:      Understanding         Generation              Structured transform
                  (classify, embed,     (chat, code,            (translate, summarize,
                  retrieve, NER)        reason, create)         transcribe)
```

---

### 1.5 Mixture-of-Experts (MoE) — A Cross-Cutting Pattern

MoE is not a fourth architecture — it's a modification applied to any of the above. It replaces each FFN block with multiple "expert" FFNs and a learned router:

```
Standard (Dense):                    Mixture of Experts (Sparse):
┌──────────┐                        ┌──────────┐
│  1 FFN   │  ← All params active   │  Router  │ ← Picks top-K experts per token
└──────────┘                        ├──┬──┬──┬─┤
                                    │E1│E2│E3│E4│ ← Only K of N experts activate
                                    └──┴──┴──┴─┘

Mixtral 8x7B:  8 experts, top-2 routing
  → 46.7B total params, but only ~13B active per token
  → Inference speed of a ~13B model with quality closer to a ~40B+ model
```

**Key models:** Mixtral 8x7B, Mixtral 8x22B, DeepSeek-MoE, Grok-1, Arctic, DBRX, Switch Transformer

MoE models have more total parameters but activate fewer per forward pass, making them efficient for their quality level. The tradeoff is higher total memory (you must load *all* experts) but similar compute per token.

---

## Part 2: Number Representations in Memory

Before understanding quantization, you need to understand how numbers are stored in computers. Every weight in a neural network is a floating-point number, and the format you store it in determines both memory usage and numerical precision.

### 2.1 How Floating-Point Works

All floating-point formats follow the same structure:

```
Value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa)

┌──────┬──────────────┬─────────────────────────┐
│ Sign │   Exponent   │        Mantissa          │
│ (1b) │   (E bits)   │        (M bits)          │
└──────┴──────────────┴─────────────────────────┘
```

- **Sign bit:** 0 = positive, 1 = negative
- **Exponent:** Controls the *magnitude* (how big or small the number can be)
- **Mantissa (fraction):** Controls the *precision* (how many significant digits)

Think of it like scientific notation: `6.022 × 10²³`
- Sign: positive
- Mantissa: 6.022 (the precision)
- Exponent: 23 (the magnitude)

### 2.2 FP32 — Full Precision (32 bits)

```
┌───┬──────────┬───────────────────────────────────────┐
│ S │ Exponent │              Mantissa                  │
│ 1 │  8 bits  │             23 bits                    │
└───┴──────────┴───────────────────────────────────────┘
 Total: 32 bits = 4 bytes per parameter

Example: The number 0.15625
Binary: 0 01111100 01000000000000000000000
        ↑  ↑↑↑↑↑↑↑↑  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        +  exp=124    mantissa=0.25
        
Value = +1 × 2^(124-127) × 1.25 = 2^(-3) × 1.25 = 0.15625
```

**Properties:**
- Range: ±1.18 × 10⁻³⁸ to ±3.40 × 10³⁸
- Precision: ~7 decimal digits
- Memory per parameter: 4 bytes

**In LLM context:** FP32 is the "ground truth" — training often happens in FP32 (or with FP32 master weights). A 7B parameter model in FP32 requires:

```
7,000,000,000 × 4 bytes = 28 GB just for weights
+ optimizer states (Adam): ~3× more = ~84 GB for training
```

This is why nobody serves large models in FP32 anymore.

### 2.3 FP16 — Half Precision (16 bits)

```
┌───┬──────────┬──────────────────┐
│ S │ Exponent │    Mantissa      │
│ 1 │  5 bits  │    10 bits       │
└───┴──────────┴──────────────────┘
 Total: 16 bits = 2 bytes per parameter
```

**Properties:**
- Range: ±6.10 × 10⁻⁵ to ±65,504
- Precision: ~3.3 decimal digits
- Memory per parameter: 2 bytes
- **2× savings** over FP32

**The problem:** FP16's limited range causes **overflow** (numbers > 65,504 become infinity) and **underflow** (small gradients vanish to zero). This made FP16 training unstable for large models, which led to BF16.

### 2.4 BF16 — Brain Float 16 (16 bits)

```
┌───┬──────────┬──────────┐
│ S │ Exponent │ Mantissa │
│ 1 │  8 bits  │  7 bits  │
└───┴──────────┴──────────┘
 Total: 16 bits = 2 bytes per parameter
```

**Properties:**
- Range: Same as FP32 (±1.18 × 10⁻³⁸ to ±3.40 × 10³⁸) — the 8-bit exponent matches FP32!
- Precision: ~2.4 decimal digits (less precise than FP16)
- Memory per parameter: 2 bytes

**Why BF16 dominates LLM training:** It has the *range* of FP32 (no overflow/underflow issues) with the *memory* of FP16. The reduced precision rarely matters for neural network weights because the optimization process is inherently noisy.

```
Comparison at the bit level:

FP32:  [1 sign] [8  exponent] [23 mantissa]  ← Gold standard
BF16:  [1 sign] [8  exponent] [ 7 mantissa]  ← Same range, less precision
FP16:  [1 sign] [5  exponent] [10 mantissa]  ← More precision, much less range
```

**Llama 3.1 was trained in BF16.** Most modern LLMs are. The published model weights are in BF16 format.

### 2.5 Visual Comparison of Formats

```
Memory per parameter:

FP32:  ████████████████████████████████  32 bits (4 bytes)
BF16:  ████████████████                  16 bits (2 bytes)
FP16:  ████████████████                  16 bits (2 bytes)
INT8:  ████████                           8 bits (1 byte)
INT4:  ████                               4 bits (0.5 bytes)

Memory for a 7B parameter model (weights only):

FP32:  28.0 GB   ████████████████████████████████████████
BF16:  14.0 GB   ████████████████████
FP16:  14.0 GB   ████████████████████
INT8:   7.0 GB   ██████████
INT4:   3.5 GB   █████
```

---

## Part 3: Quantization — Compressing Model Weights

Quantization reduces the number of bits used to store each parameter. This is the key technology that allows a 70B model to run on consumer hardware.

### 3.1 The Core Idea

```
Original FP16 weight tensor:  [0.0234, -0.1562, 0.0891, -0.2451, 0.1123, ...]
                                     ↓  Quantization
INT8 representation:          [12, -80, 46, -126, 58, ...]  + scale factor 0.00194
                                     ↓  Dequantization
Reconstructed FP16:           [0.0233, -0.1552, 0.0892, -0.2444, 0.1125, ...]
                                          ↑ small errors ↑    (acceptable!)
```

The process:
1. Find the range of values in a weight tensor
2. Map those values to a smaller set of integers
3. Store the integers + a scale (and optionally zero-point) to reconstruct

### 3.2 Quantization Math

**Symmetric quantization** (simplest, used for weights):
```
scale = max(|W|) / (2^(bits-1) - 1)
W_quantized = round(W / scale)
W_dequantized = W_quantized × scale
```

**Example with INT8 (range -128 to 127):**
```
Original weights:    [-0.25, 0.10, 0.30, -0.15, 0.28]
max absolute value:  0.30
scale = 0.30 / 127 = 0.00236

Quantize:            [-106, 42, 127, -64, 119]
Dequantize:          [-0.250, 0.099, 0.300, -0.151, 0.281]
Error:               [0.000, 0.001, 0.000, 0.001, 0.001]
```

**Asymmetric quantization** (used when distributions aren't centered at zero):
```
scale = (max(W) - min(W)) / (2^bits - 1)
zero_point = round(-min(W) / scale)
W_quantized = round(W / scale) + zero_point
```

### 3.3 Block-wise Quantization (What Modern Methods Actually Do)

Real quantization doesn't use a single scale for the entire tensor. It divides weights into small **blocks** (e.g., 32, 64, or 128 values) and computes a separate scale per block. This captures local variations much better:

```
Full tensor: [block_1: 64 values | block_2: 64 values | block_3: 64 values | ...]
              ↓ scale_1           ↓ scale_2             ↓ scale_3
              
Each block gets its own scale factor, stored in FP16
→ Overhead: 1 FP16 value per 64 weights = 16/64 = 0.25 bits per weight extra
→ Total for 4-bit: 4 + 0.25 = 4.25 bits per weight effective
```

### 3.4 INT8 Quantization (8-bit)

```
┌──────────────────────────────────────┐
│ INT8: 8 bits per weight              │
│ Range: -128 to 127 (256 levels)      │
│ Memory: 1 byte per parameter         │
│ Savings: 2× vs FP16, 4× vs FP32     │
└──────────────────────────────────────┘
```

**Common INT8 methods:**

**LLM.int8() (bitsandbytes):**
- Splits weights into "normal" values (INT8) and outlier values (FP16)
- Neural networks have rare but important large-magnitude outliers that break uniform quantization
- Mixed-precision: ~99.9% of values in INT8, ~0.1% kept in FP16
- Nearly lossless for most models

```
Weight matrix decomposition:

┌──────────────────────────────┐     ┌────────────┐
│   Main weights (99.9%)       │     │  Outliers   │
│   Quantized to INT8          │  +  │  Kept in    │
│   (1 byte each)              │     │  FP16       │
└──────────────────────────────┘     └────────────┘
```

**SmoothQuant:**
- Migrates quantization difficulty from activations to weights
- Applies a mathematically equivalent transformation that smooths activation outliers
- Enables both weight AND activation quantization (important for fast inference)

**Typical quality impact:** < 0.5% perplexity degradation — essentially lossless for end users.

### 3.5 INT4 / 4-bit Quantization

```
┌──────────────────────────────────────┐
│ INT4: 4 bits per weight              │
│ Range: -8 to 7 (16 levels only!)     │
│ Memory: 0.5 bytes per parameter      │
│ Savings: 4× vs FP16, 8× vs FP32     │
└──────────────────────────────────────┘
```

With only 16 possible values, 4-bit quantization requires more sophisticated techniques:

**GPTQ (Post-Training Quantization):**
- Uses a small calibration dataset (~128 samples)
- Quantizes weights one-by-one, compensating for each weight's error by adjusting remaining weights
- Based on Optimal Brain Quantization (OBQ) framework
- Produces the quantized model offline — inference is fast

```
GPTQ process:
For each column of the weight matrix:
  1. Quantize one weight
  2. Measure the error introduced
  3. Distribute that error across unquantized weights using Hessian information
  4. Repeat → total accumulated error stays small
```

**AWQ (Activation-Aware Weight Quantization):**
- Key insight: not all weights are equally important — weights connected to frequently-activated channels matter more
- Scales important weight channels before quantization to protect them
- Often slightly better quality than GPTQ at the same bit width

```
AWQ insight:

Channel importance:  ████ ██ ████████ █ ████ ██████
                     med  low  HIGH  low med  HIGH

Strategy: Scale up important channels (giving them more quantization bins)
          then scale down during inference to compensate
```

**NF4 (NormalFloat 4-bit, used in QLoRA):**
- Observation: neural network weights follow a roughly normal distribution
- Instead of uniformly-spaced quantization levels, NF4 spaces them according to the normal distribution
- This means more levels near zero (where most weights are) and fewer at the extremes

```
Uniform INT4 levels:     |----|----|----|----|----|----|----|----| (equal spacing)
                        -8   -6   -4   -2    0    2    4    6   7

NF4 levels:              |--|--|---|-----|----|---|--|--|          (Gaussian spacing)
                        More bins near center where most weights live
```

**GGUF / llama.cpp quantization:**
- Family of mixed-precision formats: Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q8_0, etc.
- Different layers get different bit widths based on their sensitivity
- The "K" variants use k-means clustering for better level placement
- Designed for CPU inference

```
GGUF naming convention:
Q4_K_M = 4-bit, K-means quantization, Medium quality variant
Q5_K_S = 5-bit, K-means quantization, Small (more aggressive) variant
Q8_0   = 8-bit, basic quantization

Typical choices:
  Q4_K_M:  Good balance of quality and size (~4.85 bits/weight effective)
  Q5_K_M:  Better quality, slightly larger (~5.69 bits/weight effective)
  Q8_0:    Near-lossless (~8.5 bits/weight effective)
```

### 3.6 Practical Quality Impact

Here's how quantization typically affects model quality, measured in perplexity (lower = better) on a standard benchmark:

```
Example: Llama 2 7B on WikiText-2

Format      Bits/Weight   Model Size   Perplexity   Δ from FP16
─────────   ───────────   ──────────   ──────────   ───────────
FP32        32.0          28.0 GB      5.47         (reference)
BF16/FP16   16.0          14.0 GB      5.47          0.00
INT8        8.0            7.0 GB      5.48         +0.01
GPTQ-8bit   8.0            7.0 GB      5.48         +0.01
Q5_K_M      ~5.7           5.1 GB      5.52         +0.05
GPTQ-4bit   4.0            3.9 GB      5.62         +0.15
Q4_K_M      ~4.9           4.4 GB      5.56         +0.09
AWQ-4bit    4.0            3.9 GB      5.60         +0.13
NF4+DQ      4.0            3.5 GB      5.63         +0.16
Q2_K        ~2.6           2.7 GB      7.89         +2.42 ← severe degradation

Notes:
- 8-bit: Essentially no quality loss
- 4-5 bit: Small, usually acceptable degradation
- Below 3-bit: Significant quality loss, use only if memory-constrained
```

### 3.7 The Full Picture: Memory Requirements

```
                    Model Sizes at Different Precisions
                    ════════════════════════════════════
Model           FP32      FP16/BF16    INT8      INT4
─────────────   ────────  ──────────   ────────  ────────
Llama 3.1 8B    32 GB     16 GB        8 GB      4-5 GB
Llama 3.1 70B   280 GB    140 GB       70 GB     35-40 GB
Llama 3.1 405B  1620 GB   810 GB       405 GB    ~200 GB
Mistral 7B      28 GB     14 GB        7 GB      4 GB
Mixtral 8x7B    187 GB    93 GB        47 GB     24 GB
Phi-3 Mini 3.8B 15 GB     7.6 GB       3.8 GB    2 GB

Hardware fit (inference, weights only):
─────────────────────────────────────────────────────────
RTX 3060 (12GB VRAM):   7B@INT4, 3B@INT8
RTX 3090 (24GB VRAM):   13B@INT4, 7B@INT8, 3B@FP16
RTX 4090 (24GB VRAM):   13B@INT4, 7B@INT8 (faster than 3090)
A100 (80GB VRAM):       70B@INT4, 30B@INT8, 13B@FP16
2×A100 (160GB VRAM):    70B@INT8, 70B@FP16 (tight)
H100 (80GB HBM3):       70B@INT4, faster than A100
8×H100 (640GB):         405B@INT4, 405B@INT8 (tight)
Apple M2 Ultra (192GB):  70B@INT4 (CPU/unified memory)
```

But weights aren't everything! During inference, you also need memory for:

```
Total inference memory = Model weights
                       + KV Cache (grows with sequence length and batch size)
                       + Activations (intermediate computation)
                       + Framework overhead

KV Cache estimate (per token, per layer):
  = 2 × num_kv_heads × head_dim × precision_bytes
  
For Llama 3.1 8B in FP16, 128K context:
  = 2 × 8 × 128 × 2 bytes × 32 layers × 128,000 tokens
  = ~16.8 GB just for KV cache!
  
This is why GQA matters — with fewer KV heads, the cache shrinks proportionally.
```

---

## Part 4: Quantization in Practice

### 4.1 When to Use What

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION FLOWCHART                                │
│                                                                     │
│  Training a model?                                                  │
│    └─→ Use BF16 (or FP32 master weights + BF16 compute)            │
│                                                                     │
│  Fine-tuning a model?                                               │
│    ├─→ Full fine-tune: BF16 if you have the VRAM                    │
│    └─→ Limited VRAM: QLoRA (NF4 base + LoRA adapters in BF16)      │
│                                                                     │
│  Serving for inference?                                             │
│    ├─→ Quality-critical (medical, legal): FP16 or INT8              │
│    ├─→ General purpose: GPTQ-4bit or AWQ-4bit on GPU               │
│    ├─→ CPU inference: GGUF Q4_K_M or Q5_K_M                        │
│    └─→ Maximum compression: GPTQ-4bit or Q3_K_M (some quality loss)│
│                                                                     │
│  Edge / mobile deployment?                                          │
│    └─→ INT4 with small models (1-3B), or specialized formats        │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Quantization Methods Summary

| Method | Bits | Type | Key Feature | Speed | Quality |
|---|---|---|---|---|---|
| **LLM.int8()** | 8 | Weight | Outlier-aware mixed precision | Good | Excellent |
| **SmoothQuant** | 8 | Weight + Activation | Smooths activation outliers | Very good | Excellent |
| **GPTQ** | 4/3/2 | Weight (PTQ) | Hessian-based error compensation | Excellent | Good |
| **AWQ** | 4 | Weight (PTQ) | Activation-aware importance | Excellent | Very good |
| **NF4** | 4 | Weight | Normal-distribution-aware levels | Good | Good |
| **GGUF** | 2-8 | Weight | Mixed precision, CPU-friendly | Good (CPU) | Varies |
| **EXL2** | Variable | Weight | Per-layer mixed bits | Excellent | Very good |
| **HQQ** | 4/2/1 | Weight | No calibration data needed | Very fast | Good |
| **QuIP#** | 2-4 | Weight | Incoherence processing | Good | Best at 2-bit |
| **AQLM** | 2 | Weight | Additive vector quantization | Good | Good at 2-bit |

### 4.3 QLoRA — Quantization Meets Fine-Tuning

QLoRA is particularly important because it enables fine-tuning large models on consumer hardware:

```
Standard fine-tuning of Llama 3.1 8B:
  Base model in BF16: 16 GB
  Optimizer states:   32 GB (Adam: 2× model + gradients)
  Activations:        ~8 GB
  Total:              ~56 GB → needs A100 80GB

QLoRA fine-tuning of Llama 3.1 8B:
  Base model in NF4:  ~4.5 GB  (frozen, quantized)
  LoRA adapters BF16: ~0.1 GB  (only these are trained)
  Optimizer states:   ~0.3 GB  (only for LoRA params)
  Activations:        ~4 GB
  Total:              ~9 GB → fits on RTX 3060!
```

The key innovation: freeze the 4-bit quantized model and only train small low-rank adapter matrices in full precision. You get ~95% of full fine-tuning quality at ~6× less memory.

### 4.4 Speed Impact of Quantization

Quantization isn't just about memory — it also affects inference speed:

```
Relative throughput (tokens/second), single batch, Llama 7B on RTX 4090:

FP16:       ████████████████████            100%  (baseline)
INT8:       ██████████████████████████      130%  (compute-bound, fewer bytes to load)
GPTQ-4bit:  ████████████████████████████    140%  (memory-bandwidth bound → big win)
AWQ-4bit:   █████████████████████████████   145%  (slightly optimized kernels)
GGUF Q4_K:  █████████████████               85%  (CPU, depends heavily on hardware)

Why quantized models are FASTER:
- Modern GPUs are memory-bandwidth limited for LLM inference
- Smaller weights = less data to load from VRAM per token
- 4-bit loads 4× less data → up to ~2× faster (not 4× due to dequantization overhead)
- This is especially impactful for the decode phase (generating one token at a time)
```

---

## Part 5: Bringing It All Together

### The Modern LLM Stack

```
Training:                    Deployment:                    Consumer:
═════════                    ═══════════                    ═════════
BF16 compute                 FP16 or INT8 (quality)         INT4 GGUF (laptop CPU)
FP32 master weights          AWQ/GPTQ-4bit (balanced)       INT4 on phone/edge
1000s of GPUs                vLLM / TGI server              llama.cpp / Ollama
Months of training           1-8 GPUs                       Single device
$50M+ budget                 Seconds of latency             Offline capable
```

### Quick Reference Card

```
┌─────────┬───────┬────────────┬───────────────┬──────────────────────────┐
│ Format  │ Bits  │ Bytes/Param│ 7B Model Size │ Typical Use              │
├─────────┼───────┼────────────┼───────────────┼──────────────────────────┤
│ FP32    │ 32    │ 4.0        │ 28.0 GB       │ Training (master weights)│
│ BF16    │ 16    │ 2.0        │ 14.0 GB       │ Training, serving (GPU)  │
│ FP16    │ 16    │ 2.0        │ 14.0 GB       │ Serving (GPU)            │
│ INT8    │ 8     │ 1.0        │ 7.0 GB        │ Quality-critical serving │
│ NF4/INT4│ 4     │ 0.5        │ 3.5-4.5 GB    │ Consumer GPU, fine-tuning│
│ Q2_K    │ ~2.6  │ ~0.33      │ ~2.7 GB       │ Extreme compression      │
└─────────┴───────┴────────────┴───────────────┴──────────────────────────┘
```

### Key Takeaways

1. **Architecture choice determines capability.** Decoder-only (causal) models dominate generation; encoders dominate understanding; encoder-decoders excel at structured transformation. Modern AI assistants are all decoder-only.

2. **BF16 is the standard training precision.** It combines FP32's range with FP16's memory. Almost all modern LLMs train and publish weights in BF16.

3. **INT8 quantization is nearly free.** You get 2× memory savings with negligible quality loss. If you can fit INT8, always prefer it over FP16 for inference.

4. **4-bit quantization is the sweet spot for consumer deployment.** AWQ and GPTQ give you 4× compression over FP16 with small, usually acceptable quality loss. This is what lets a 7B model run on a laptop.

5. **Quantization makes models both smaller AND faster.** Because GPU inference is memory-bandwidth bound, loading fewer bytes per weight directly translates to higher throughput.

6. **Below 4 bits, quality drops steeply.** 2-3 bit quantization exists but should only be used when absolutely necessary. The degradation becomes clearly noticeable.

7. **The weights are just the beginning.** KV cache for long contexts can exceed the model weights in size. GQA (from the architecture side) and KV cache quantization (from the systems side) both address this.

8. **Match your quantization to your use case.** Medical/legal → INT8+. Creative/chat → INT4 is fine. Experimentation/local → INT4 GGUF. Training/fine-tuning → BF16 (or QLoRA with NF4).
