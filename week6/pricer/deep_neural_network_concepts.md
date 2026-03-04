# Deep Neural Network for Price Prediction — Concepts Explained

This document explains the key concepts behind `deep_neural_network.py`, a PyTorch model that predicts product prices from text descriptions.

---

## 1. Overall Architecture

```
Text input
    │
    ▼
HashingVectorizer (text → sparse binary vector, 5000 dims)
    │
    ▼
Input Layer (Linear → LayerNorm → ReLU → Dropout)
    │
    ▼
Residual Block × (num_layers - 2)
    │   ┌──────────────────────────────────┐
    │   │ Linear → LayerNorm → ReLU →      │
    │   │ Dropout → Linear → LayerNorm     │
    │   │              +                    │
    │   │         skip connection           │
    │   │              │                    │
    │   │            ReLU                   │
    │   └──────────────────────────────────┘
    │
    ▼
Output Layer (Linear → 1 neuron)
    │
    ▼
Predicted price (after denormalization)
```

The network takes a bag-of-words vector as input and regresses a single scalar: the predicted price.

---

## 2. Text Featurization — HashingVectorizer

Before text can enter a neural network, it must be converted to numbers. `HashingVectorizer` from scikit-learn does this:

```python
vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
```

**How it works:**
1. Tokenize the text into words.
2. Remove common English stop words ("the", "is", "and", …).
3. Hash each remaining word to an index in a fixed-size vector (5000 dimensions).
4. Set that index to 1 (`binary=True` — presence, not count).

**Why hashing instead of a vocabulary?**

| Feature | CountVectorizer / TfidfVectorizer | HashingVectorizer |
|---------|-----------------------------------|-------------------|
| Requires fitting a vocabulary | Yes | No |
| Memory for vocabulary | Grows with corpus | Fixed (zero) |
| Hash collisions | None | Possible but rare at 5000 dims |
| Suitable for production | Needs saved vocab | Stateless — just hash |

Hashing is preferred here because the vectorizer doesn't need to store or load a vocabulary, making deployment simpler.

---

## 3. Target Transformation — Log Normalization

Prices are typically right-skewed (many cheap items, few expensive ones). Training directly on raw prices would let the few high-priced items dominate the loss. The code applies a two-step transformation:

**Step 1 — Log transform:**

```python
y_train_log = torch.log(y_train + 1)
```

`log(price + 1)` compresses the scale so that a $1 → $10 change and a $100 → $1000 change contribute similarly. The `+1` avoids `log(0)`.

**Step 2 — Z-score normalization:**

```python
y_train_norm = (y_train_log - y_mean) / y_std
```

Centers the values around 0 with unit variance. This helps the optimizer because gradients are better scaled when the target distribution is roughly standard normal.

**At inference time the reverse is applied:**

```python
result = torch.exp(pred * y_std + y_mean) - 1
```

Undo z-score → undo log → original dollar price.

---

## 4. Residual Blocks (Skip Connections)

The core building block of this network:

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.block(x)      # Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm
        out += residual           # skip connection
        return self.relu(out)
```

**The problem residuals solve:** in deep networks (10+ layers), gradients can vanish or explode as they propagate backward through many transformations. Each layer's gradient is multiplied by the layer's weight matrix, and after many multiplications the signal can shrink to near-zero.

**The solution:** add the block's input directly to its output (`out += residual`). Now the gradient for the input is:

```
∂loss/∂x = ∂loss/∂out × (∂block(x)/∂x + 1)
```

That `+ 1` term guarantees a gradient path of magnitude at least 1, regardless of what happens inside the block. This is the same idea behind ResNets in computer vision.

**Why it enables depth:** without skip connections, stacking 8 residual blocks (the default `num_layers=10` minus the input and output layers) would be difficult to train. With them, adding more blocks can only help — in the worst case the block learns the identity function and the skip connection passes data through unchanged.

---

## 5. Layer Normalization

```python
nn.LayerNorm(hidden_size)
```

Normalizes activations across the feature dimension (not the batch dimension, unlike BatchNorm). For a single sample with hidden size *H*:

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) × γ + β
```

where `γ` and `β` are learnable scale and shift parameters.

**Why LayerNorm instead of BatchNorm?**

| | BatchNorm | LayerNorm |
|--|-----------|-----------|
| Normalizes across | batch dimension | feature dimension |
| Depends on batch size | Yes — small batches give noisy statistics | No — each sample is independent |
| Behavior at inference | Uses running averages (can drift) | Same computation as training |

LayerNorm is a safer default here because batch size is only 64, and it behaves identically during training and inference.

---

## 6. Dropout

```python
nn.Dropout(dropout_prob)   # dropout_prob=0.2
```

During training, randomly zeros out 20% of neurons in each forward pass. This forces the network to not rely on any single neuron, acting as a regularizer that reduces overfitting.

During inference (`model.eval()`), dropout is automatically disabled — all neurons are active, and outputs are scaled to compensate.

---

## 7. Loss Function — L1Loss (Mean Absolute Error)

```python
self.loss_function = nn.L1Loss()
```

```
L1Loss = (1/N) × Σ |predicted_i - target_i|
```

L1 (MAE) is more robust to outliers than L2 (MSE). Since prices can have extreme values, an MSE loss would disproportionately penalize large errors, making the model chase outliers. L1 treats every dollar of error equally.

The model trains on the **normalized log-scale** targets, so the L1 loss operates in that compressed space. At validation time, predictions are converted back to dollar scale to report a human-readable MAE.

---

## 8. Optimizer — AdamW

```python
self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
```

AdamW combines three ideas:

1. **Momentum** — smooths gradient updates using an exponential moving average, reducing oscillation.
2. **Adaptive learning rates** — each parameter gets its own effective learning rate based on the magnitude of its recent gradients. Parameters with consistently small gradients get larger steps.
3. **Decoupled weight decay** — penalizes large weights directly (multiplies weights by `1 - lr × weight_decay` each step). This is the "W" in AdamW — the original Adam mixed weight decay into the gradient, which interacted poorly with the adaptive learning rate. AdamW decouples them.

`weight_decay=0.01` acts as L2 regularization, discouraging the model from relying on any single feature too heavily.

---

## 9. Learning Rate Scheduler — CosineAnnealingLR

```python
self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
```

Instead of a fixed learning rate, the scheduler follows a cosine curve:

```
lr(t) = eta_min + 0.5 × (lr_initial - eta_min) × (1 + cos(π × t / T_max))
```

Over `T_max=10` epochs the learning rate follows this shape:

```
lr
0.001 ┤╲
      │  ╲
      │    ╲
      │      ╲
      │        ╲
      │          ╲
      │             ╲
      │                ╲
      │                    ╲
0.000 ┤                        ╲
      └─────────────────────────── epoch
      0                        10
```

**Why cosine annealing?**
- **Early epochs (high LR):** the model explores broadly, making large parameter updates to find a good region of the loss landscape.
- **Late epochs (low LR):** the model fine-tunes, making small adjustments to settle into a sharp minimum.
- The smooth decay avoids the jarring drops of step-based schedules.

---

## 10. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

After computing gradients with `loss.backward()`, but before updating weights with `optimizer.step()`, the total gradient norm is clamped to 1.0. If the norm exceeds 1.0, all gradients are proportionally scaled down.

This prevents **gradient explosion** — a single bad batch producing enormous gradients that destabilize training. It is especially important in deep networks with residual connections, where gradients from many blocks accumulate.

---

## 11. Training Loop

Each epoch follows this pattern:

```
for each batch:
    1. Forward pass:    outputs = model(batch_X)
    2. Compute loss:    loss = L1Loss(outputs, batch_y)
    3. Backward pass:   loss.backward()          ← compute gradients
    4. Clip gradients:  clip_grad_norm_(...)      ← prevent explosion
    5. Update weights:  optimizer.step()          ← apply gradients
    6. Zero gradients:  optimizer.zero_grad()     ← reset for next batch

after all batches:
    7. Validate:        model.eval() + torch.no_grad()
    8. Step scheduler:  scheduler.step()          ← reduce learning rate
```

**`model.train()` vs `model.eval()`**: switches dropout and layer norm between training and inference behavior.

**`torch.no_grad()`**: disables gradient tracking during validation, saving memory and computation.

---

## 12. DataLoader and Batching

```python
self.train_dataset = TensorDataset(self.X_train, self.y_train_norm)
self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
```

- **TensorDataset** pairs input features and targets so they can be indexed together.
- **DataLoader** handles batching (64 samples per batch), shuffling (randomize order each epoch to prevent the model from memorizing sequence patterns), and iteration.

A batch size of 64 balances GPU utilization (larger batches use hardware more efficiently) against gradient noise (smaller batches provide a regularizing effect).

---

## 13. Device Selection

```python
if torch.cuda.is_available():
    self.device = torch.device("cuda")       # NVIDIA GPU
elif torch.backends.mps.is_available():
    self.device = torch.device("mps")         # Apple Silicon GPU
else:
    self.device = torch.device("cpu")
```

The priority order is: CUDA (NVIDIA) → MPS (Apple M-series) → CPU. All tensors and the model are moved to the selected device with `.to(self.device)`. Mixing devices (e.g., model on GPU, data on CPU) causes runtime errors, so every tensor that participates in a computation must be on the same device.

---

## 14. Reproducibility — Random Seeds

```python
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

Setting seeds for NumPy, PyTorch CPU, and PyTorch CUDA ensures that weight initialization, data shuffling, and dropout masks are the same across runs. This makes experiments reproducible — the same code produces the same results.

---

## Summary

| Concept | Why it's used |
|---------|---------------|
| HashingVectorizer | Stateless text-to-vector conversion |
| Log + z-score normalization | Compress skewed price distribution for stable training |
| Residual blocks | Enable deep networks by preserving gradient flow |
| LayerNorm | Stabilize activations independent of batch size |
| Dropout | Regularize to prevent overfitting |
| L1Loss | Robust to price outliers |
| AdamW | Adaptive optimizer with proper weight decay |
| CosineAnnealingLR | Smooth learning rate decay from exploration to refinement |
| Gradient clipping | Prevent gradient explosion in deep networks |
