"""
Binary classification on the classic 'two moons' dataset (generated here
with numpy, no sklearn dependency needed). Tests softmax + cross-entropy
and a deeper MLP than the XOR demo.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opengrad import Tensor
from opengrad.nn import MLP, cross_entropy_loss
from opengrad.optim import Adam


def make_moons(n_samples=200, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    theta1 = np.linspace(0, np.pi, n)
    theta2 = np.linspace(0, np.pi, n)
    x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    x2 = np.stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5], axis=1)
    X = np.concatenate([x1, x2], axis=0)
    X += rng.randn(*X.shape) * noise
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    return X, y


X, y = make_moons(n_samples=200, noise=0.12, seed=1)

model = MLP([2, 16, 16, 2])
opt = Adam(model.parameters(), lr=0.02)

x_t = Tensor(X, requires_grad=False)

for step in range(400):
    logits = model(x_t)
    loss = cross_entropy_loss(logits, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 50 == 0 or step == 399:
        preds = np.argmax(logits.data, axis=1)
        acc = (preds == y).mean()
        print(f"step {step:3d}  loss {loss.data:.4f}  acc {acc:.3f}")
