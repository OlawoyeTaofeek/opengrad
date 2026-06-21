"""
The 'hello world' of neural nets: learn XOR.
A linear model can't solve this -- it needs a hidden layer with a
nonlinearity, which is exactly what makes this a good smoke test.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opengrad import Tensor
from opengrad.nn import MLP
from opengrad.optim import Adam

np.random.seed(42)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
Y = np.array([[0], [1], [1], [0]], dtype=np.float64)

model = MLP([2, 8, 1])
opt = Adam(model.parameters(), lr=0.05)

x = Tensor(X, requires_grad=False)
y = Tensor(Y, requires_grad=False)

for step in range(300):
    pred = model(x).sigmoid()
    loss = ((pred - y) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 50 == 0 or step == 299:
        print(f"step {step:3d}  loss {loss.data:.6f}")

print("\nfinal predictions:")
final = model(x).sigmoid()
for i in range(4):
    print(f"  {X[i]} -> {final.data[i][0]:.4f}  (target {Y[i][0]})")
