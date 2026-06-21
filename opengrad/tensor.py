"""
opengrad.tensor
================
The core of the engine. A `Tensor` wraps a numpy array and remembers how it
was produced (which other Tensors + which operation). Calling `.backward()`
walks that history in reverse and accumulates gradients everywhere, via
the chain rule -- this is reverse-mode automatic differentiation, the same
idea PyTorch/TensorFlow use under the hood.

Compared to Karpathy's micrograd (which operates on individual Python floats,
one `Value` per scalar), this version operates on whole numpy arrays at once,
so a "neuron" or a "layer" is a single Tensor op instead of thousands of
scalar nodes. That makes it both much faster and able to express real
matrix/vector operations (matmul, conv, reductions, broadcasting, indexing).
"""

import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # for debugging / graph visualization

    # ----------------------------------------------------------------- #
    # basic properties
    # ----------------------------------------------------------------- #
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def item(self):
        return self.data.item()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data=\n{self.data})"

    # ----------------------------------------------------------------- #
    # helpers
    # ----------------------------------------------------------------- #
    @staticmethod
    def _ensure_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    @staticmethod
    def _unbroadcast(grad, shape):
        """Sum-reduce `grad` down to `shape`, undoing numpy broadcasting."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape)

    # ----------------------------------------------------------------- #
    # arithmetic ops
    # ----------------------------------------------------------------- #
    def __add__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only scalar exponents supported"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def matmul(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ np.swapaxes(other.data, -1, -2)
            other.grad += np.swapaxes(self.data, -1, -2) @ out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor._ensure_tensor(other))

    def __rsub__(self, other):
        return (-self) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * Tensor._ensure_tensor(other) ** -1.0

    def __rtruediv__(self, other):
        return Tensor._ensure_tensor(other) * self ** -1.0

    # ----------------------------------------------------------------- #
    # elementwise nonlinearities
    # ----------------------------------------------------------------- #
    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0).astype(np.float64) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), (self,), 'LeakyReLU')

        def _backward():
            self.grad += np.where(self.data > 0, 1.0, alpha) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # reductions / reshaping / indexing
    # ----------------------------------------------------------------- #
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=True)
        out = Tensor(out_data if keepdims else out_data.reshape(
            [d for i, d in enumerate(self.data.shape) if i != axis] if axis is not None else ()
        ), (self,), 'max')
        mask = (self.data == out_data).astype(np.float64)
        mask /= mask.sum(axis=axis, keepdims=True)  # split gradient on ties

        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += mask * grad
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else None
        out = Tensor(np.transpose(self.data, axes), (self,), 'transpose')

        def _backward():
            if axes is None:
                self.grad += np.transpose(out.grad)
            else:
                self.grad += np.transpose(out.grad, np.argsort(axes))
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'getitem')

        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self.grad += g
        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # the main event: backward pass
    # ----------------------------------------------------------------- #
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
