"""
opengrad.nn
============
Neural network layers built on top of Tensor. A `Module` knows its own
parameters so an optimizer can find and update them, and supports
`.zero_grad()` to reset gradients between training steps.
"""

import numpy as np
from .tensor import Tensor


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError


class Linear(Module):
    """y = x @ W + b.  x: (batch, in_features) -> y: (batch, out_features)."""

    def __init__(self, in_features, out_features, bias=True):
        scale = np.sqrt(2.0 / in_features)  # He initialization
        self.W = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def parameters(self):
        return [self.W, self.b] if self.b is not None else [self.W]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return x.leaky_relu(self.alpha)


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class MLP(Module):
    """Convenience: a stack of Linear+activation layers from a list of sizes."""

    def __init__(self, sizes, activation=ReLU, final_activation=None):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(Linear(sizes[i], sizes[i + 1]))
            is_last = (i == len(sizes) - 2)
            if not is_last:
                layers.append(activation())
            elif final_activation is not None:
                layers.append(final_activation())
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()


# --------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------- #
def mse_loss(pred, target):
    target = Tensor._ensure_tensor(target)
    diff = pred - target
    return (diff * diff).mean()


def softmax(x, axis=-1):
    shifted = x - Tensor(x.data.max(axis=axis, keepdims=True), requires_grad=False)
    e = shifted.exp()
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_loss(logits, targets):
    """
    logits: Tensor of shape (batch, num_classes)
    targets: integer numpy array of shape (batch,) with class indices
    """
    probs = softmax(logits, axis=-1)
    batch = logits.shape[0]
    correct_probs = probs[np.arange(batch), targets]
    return -(correct_probs.log()).mean()
