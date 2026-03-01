# Backpropagation Demystified: Chain Rule in Practice

Backpropagation is how neural networks learn. Despite sounding intimidating, it's just the chain rule from calculus applied systematically across a computation graph. By the end of this article, you'll not only understand it conceptually — you'll implement it from scratch in NumPy.

---

## The Learning Problem

A neural network makes predictions by passing data forward through layers. The gap between what it predicts and what's correct is measured by a **loss function**. Training means adjusting the network's weights to reduce that loss.

The question is: **how much should each weight change?**

The answer: compute the gradient of the loss with respect to every weight, then nudge each weight in the direction that reduces loss. This is backpropagation.

---

## The Chain Rule

If `z = f(g(x))`, then:

$$\frac{dz}{dx} = \frac{dz}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dx}$$

Neural networks are deeply nested compositions of functions. The chain rule lets us compute the gradient of the final loss with respect to any weight, by multiplying gradients all the way back.

---

## A Simple Example: 2-Layer Network

Let's build and backprop through a tiny network manually.

**Forward pass:**
```
Input:   x
Layer 1: z1 = W1 @ x + b1
         a1 = relu(z1)
Layer 2: z2 = W2 @ a1 + b2
         ŷ  = sigmoid(z2)
Loss:    L  = binary_cross_entropy(ŷ, y)
```

---

## NumPy Implementation from Scratch

```python
import numpy as np

# Activation functions and their derivatives
def relu(z):        return np.maximum(0, z)
def relu_grad(z):   return (z > 0).astype(float)

def sigmoid(z):     return 1 / (1 + np.exp(-z))
def sigmoid_grad(a): return a * (1 - a)  # a = sigmoid(z)

def bce_loss(y_hat, y):
    eps = 1e-8
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))


class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr
        # Xavier initialization
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))

    def forward(self, X):
        """Forward pass. Returns prediction and cache for backprop."""
        self.X = X
        self.z1 = self.W1 @ X + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, y):
        """Backward pass — compute gradients and update weights."""
        m = y.shape[1]  # batch size

        # --- Output layer gradient ---
        # dL/dz2 = dL/da2 * da2/dz2
        # For BCE + sigmoid, this simplifies cleanly to:
        dz2 = self.a2 - y                            # (output_dim, m)

        # Weight gradients
        dW2 = (1/m) * dz2 @ self.a1.T               # (output_dim, hidden_dim)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

        # --- Hidden layer gradient ---
        # Propagate gradient back through W2, then through relu
        da1 = self.W2.T @ dz2                         # (hidden_dim, m)
        dz1 = da1 * relu_grad(self.z1)               # element-wise

        dW1 = (1/m) * dz1 @ self.X.T                 # (hidden_dim, input_dim)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        # --- Gradient descent update ---
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
```

---

## Training Loop

```python
np.random.seed(42)

# Generate toy binary classification dataset
from sklearn.datasets import make_moons
X_data, y_data = make_moons(n_samples=500, noise=0.2)
X = X_data.T            # shape: (2, 500)
y = y_data.reshape(1, -1)  # shape: (1, 500)

model = TwoLayerNet(input_dim=2, hidden_dim=16, output_dim=1, lr=0.1)
losses = []

for epoch in range(1000):
    y_hat = model.forward(X)
    loss = bce_loss(y_hat, y)
    losses.append(loss)
    model.backward(y)

    if epoch % 100 == 0:
        predictions = (y_hat > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")
```

**Expected output:**
```
Epoch    0 | Loss: 0.7123 | Accuracy: 52.40%
Epoch  100 | Loss: 0.3821 | Accuracy: 84.60%
Epoch  500 | Loss: 0.1534 | Accuracy: 93.80%
Epoch  900 | Loss: 0.1102 | Accuracy: 95.20%
```

---

## Gradient Flow Visualization

A key insight: during backprop, gradients **flow backwards** through the computation graph.

```
Forward:  X → [W1,b1] → z1 → relu → a1 → [W2,b2] → z2 → sigmoid → ŷ → Loss
              ↑                      ↑
Backward: dW1 ←   ← dz1 ← ←  da1  ← ← dz2 ←  ←  ← ← dL
```

At each layer:
1. Receive gradient from the layer **ahead** of it (closer to loss)
2. Multiply by local derivative (chain rule)
3. Pass gradient to layer **behind** it (closer to input)
4. Also compute gradient w.r.t. weights at this layer

---

## Common Pitfalls

### Vanishing Gradients
When using sigmoid/tanh in deep networks, gradients can shrink exponentially:

```python
# sigmoid derivative at saturation is ~0
sigmoid_grad(0.99)  # → 0.0099  very small!
sigmoid_grad(0.01)  # → 0.0099  also tiny!
```

**Fix:** Use ReLU (or Leaky ReLU, GELU) for hidden layers.

### Exploding Gradients
Gradients can also grow exponentially, causing NaN losses.

**Fix:** Gradient clipping
```python
max_grad_norm = 1.0
for param in model.parameters():
    if param.grad is not None:
        torch.nn.utils.clip_grad_norm_(param, max_grad_norm)
```

---

## PyTorch Does This Automatically

In practice, PyTorch's autograd engine handles all of this:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    y_hat = model(X_tensor)
    loss = loss_fn(y_hat, y_tensor)
    loss.backward()           # ← This IS backpropagation
    optimizer.step()
```

`loss.backward()` is calling the exact same chain rule logic we implemented above — just for any arbitrary computation graph, computed automatically.

---

## Key Takeaways

| Concept | Why It Matters |
|---|---|
| Chain rule | Foundation of all gradient computation |
| Forward pass | Compute output AND cache intermediate values |
| Backward pass | Use cached values to compute parameter gradients |
| Gradient descent | Use gradients to update weights |
| ReLU activation | Prevents vanishing gradients |
| `loss.backward()` | PyTorch autograd — this is backprop |

---

*Once you've implemented backpropagation from scratch at least once, you'll have a fundamentally deeper understanding of how neural networks learn. It demystifies most of the "black box" nature of deep learning.*
