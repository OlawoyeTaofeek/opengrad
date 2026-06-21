# opengrad

A small, readable, numpy-backed automatic differentiation engine and neural
network library — built for learning, free for everyone.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd),
but built around whole **tensors** (numpy arrays) instead of individual
scalars, so it can express real matmuls, broadcasting, reductions, and train
actual networks at reasonable speed — while staying small enough to read
start to finish in an afternoon.

```python
import numpy as np
from opengrad import Tensor
from opengrad.nn import MLP
from opengrad.optim import Adam

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

model = MLP([2, 8, 1])
opt = Adam(model.parameters(), lr=0.05)

x, y = Tensor(X, requires_grad=False), Tensor(Y, requires_grad=False)

for step in range(300):
    pred = model(x).sigmoid()
    loss = ((pred - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print(model(x).sigmoid().data)  # ~[0, 1, 1, 0]
```

## Why this exists

micrograd is a brilliant teaching tool, but it's scalar-only — every number
in your network is its own Python object, which is slow and doesn't look
like how PyTorch actually works. opengrad keeps the same "build a graph as
you compute, then call `.backward()`" philosophy, but operates on numpy
arrays directly, so:

- A whole layer's matmul is *one* node in the graph, not thousands of scalar nodes.
- It can actually train networks with hundreds of parameters in milliseconds.
- The code maps much more directly onto how real frameworks work.

## What's inside

- **`opengrad/tensor.py`** — the `Tensor` class. Reverse-mode autodiff over
  numpy arrays, with broadcasting handled correctly. Ops: `+ - * / **`,
  `matmul`, `relu`, `leaky_relu`, `tanh`, `sigmoid`, `exp`, `log`, `sum`,
  `mean`, `max`, `reshape`, `transpose`, indexing.
- **`opengrad/nn.py`** — `Module`, `Linear`, activation layers, `Sequential`,
  `MLP`, `mse_loss`, `softmax`, `cross_entropy_loss`.
- **`opengrad/optim.py`** — `SGD` (with momentum) and `Adam`.
- **`opengrad/viz.py`** — `draw_graph()`, a graphviz visualizer of the
  computation graph (shows shapes, values, and gradients per node).
- **`examples/`** — XOR (sanity check) and two-moons classification (harder,
  multi-class, tests softmax + cross-entropy).
- **`tests/`** — gradient checks for every op, verified against numerical
  (finite-difference) gradients.

## Install

Locally, for development:

```bash
git clone https://github.com/YOUR_USERNAME/opengrad.git
cd opengrad
pip install -e ".[viz,dev]"
```

Once published to PyPI:

```bash
pip install opengrad
```

## Run the examples

```bash
python examples/xor_demo.py
python examples/moons_demo.py
```

## Run the tests

```bash
pytest tests/ -v
# or, without pytest:
python tests/test_tensor.py
```

## Visualizing a graph

```python
from opengrad import Tensor
from opengrad.viz import draw_graph

a = Tensor([2.0, -3.0])
b = Tensor([1.0, 4.0])
c = a * b + a.relu()
c.backward()

dot = draw_graph(c)
dot.render('my_graph')   # writes my_graph.svg
```

## Roadmap (contributions welcome!)

- [ ] Conv2d / pooling layers (im2col-based)
- [ ] Dropout, BatchNorm/LayerNorm
- [ ] Learning rate schedulers
- [ ] GPU backend (cupy as a drop-in for numpy)
- [ ] Browser-based interactive demo (Pyodide/WASM) — "try it with no install"
- [ ] More worked examples (MNIST, simple RNN)

## License

MIT — free to use, modify, and redistribute. See `LICENSE`.

## Acknowledgements

Conceptually inspired by Andrej Karpathy's micrograd and his
"Neural Networks: Zero to Hero" series — an excellent place to start if
you want to understand backpropagation from first principles before (or
after) reading this code.
