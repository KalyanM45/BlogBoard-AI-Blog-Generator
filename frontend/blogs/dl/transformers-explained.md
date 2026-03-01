# Transformers Explained: Attention Is All You Need (Really)

The 2017 paper *"Attention Is All You Need"* by Vaswani et al. fundamentally changed NLP and eventually all of deep learning. Today, transformers are the backbone of GPT-4, BERT, Stable Diffusion, and virtually every state-of-the-art model. Let's break down how they actually work.

---

## The Core Problem Transformers Solved

Before transformers, sequential models like RNNs and LSTMs processed tokens one by one. This meant:

- **Long-range dependencies** were hard to capture
- **Parallelism** was limited (can't process token 10 before token 9)
- **Vanishing gradients** plagued very deep sequences

Transformers sidestep all of this by using **self-attention** — every token can directly attend to every other token.

---

## Building Blocks of a Transformer

### 1. Input Embeddings

Text is tokenized and converted to dense vectors. A sentence of `T` tokens becomes a matrix of shape `(T, d_model)`.

```python
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)  # (batch, seq_len) → (batch, seq_len, d_model)
```

### 2. Positional Encoding

Since attention has no inherent notion of order, we add sinusoidal positional encodings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## Self-Attention: The Heart of Transformers

Self-attention computes **Queries (Q)**, **Keys (K)**, and **Values (V)** from the input:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- `Q` — what am I looking for?
- `K` — what do I have?
- `V` — what information do I give if selected?

The `√d_k` scale factor prevents dot products from growing too large, stabilizing gradients.

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V  # (B, H, T, d_k)
```

---

## Multi-Head Attention

Instead of one attention head, we run `h` parallel attention operations:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, T, _ = Q.shape
        Q = self.W_q(Q).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, T, self.h, self.d_k).transpose(1, 2)
        x = scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(x)
```

---

## Feed-Forward Network

After attention, each token passes through a position-wise FFN:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)
```

---

## The Full Encoder Block

```
Input → Embedding + Positional Encoding
       ↓
[Multi-Head Attention → Add & Norm] × N
       ↓
[Feed-Forward → Add & Norm] × N
       ↓
Output Representations
```

Layer normalization and residual connections (Add & Norm) are crucial for training stability.

---

## Why Transformers Dominate Everything

| Feature | RNN/LSTM | Transformer |
|---|---|---|
| Parallelism | ❌ Sequential | ✅ Fully parallel |
| Long-range context | ❌ Limited | ✅ Full attention |
| Compute scaling | Linear | Quadratic (O(T²)) |
| Pretraining | Hard | ✅ Very effective |

The quadratic scaling with sequence length is the main challenge — leading to innovations like **Longformer**, **Flash Attention**, and **Mamba** (state space models).

---

## Key Variants to Know

- **BERT** — Bidirectional encoder, masked language modeling
- **GPT** — Causal decoder, autoregressive generation  
- **T5** — Encoder-decoder, text-to-text framing
- **ViT** — Vision Transformer (image patches as tokens)
- **CLIP** — Joint vision-language model

---

*The transformer is arguably the most impactful architectural innovation in the history of deep learning. Understanding it deeply will serve you for years to come.*
