"""
Gradient checking: for every op, compare the analytic gradient (computed
via backward()) against a numerical gradient (computed via finite
differences, exactly like the (f(x+h) - f(x))/h trick). If they match,
the backward pass is almost certainly correct.

Run with:  python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from opengrad.tensor import Tensor

np.random.seed(0)


def numerical_grad(f, x_data, eps=1e-6):
    """f: function mapping a numpy array -> scalar. Central-difference grad."""
    grad = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x_data[idx]
        x_data[idx] = orig + eps
        f_plus = f(x_data)
        x_data[idx] = orig - eps
        f_minus = f(x_data)
        x_data[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def check(op_name, build_fn, x_shape, atol=1e-4):
    x_data = np.random.randn(*x_shape)

    def f(data):
        x = Tensor(data.copy())
        out = build_fn(x)
        return out.data.sum() if out.data.ndim else out.data.item()

    x = Tensor(x_data.copy())
    out = build_fn(x)
    out_sum = out.sum() if out.ndim else out
    out_sum.backward()

    numgrad = numerical_grad(f, x_data.copy())
    ok = np.allclose(x.grad, numgrad, atol=atol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {op_name}  max_diff={np.max(np.abs(x.grad - numgrad)):.2e}")
    assert ok, f"{op_name} gradient mismatch"


def test_add():
    check("add", lambda x: x + 2.0, (5,))


def test_mul():
    check("mul", lambda x: x * 3.0, (5,))


def test_pow():
    check("pow", lambda x: x ** 3, (5,))


def test_relu():
    check("relu", lambda x: x.relu(), (10,))


def test_tanh():
    check("tanh", lambda x: x.tanh(), (10,))


def test_sigmoid():
    check("sigmoid", lambda x: x.sigmoid(), (10,))


def test_exp():
    check("exp", lambda x: x.exp(), (5,))


def test_log():
    check("log", lambda x: (x ** 2 + 1.0).log(), (5,))  # keep input positive


def test_matmul():
    w = Tensor(np.random.randn(4, 3))
    check("matmul", lambda x: x.matmul(w), (5, 4))


def test_broadcasting_add():
    b = Tensor(np.random.randn(3))
    check("broadcast_add", lambda x: x + b, (5, 3))


def test_sum_axis():
    check("sum_axis", lambda x: x.sum(axis=1), (4, 3))


def test_mean():
    check("mean", lambda x: x.mean(), (6,))


def test_reshape():
    check("reshape", lambda x: x.reshape(2, 6), (3, 4))


def test_transpose():
    check("transpose", lambda x: x.transpose(1, 0), (3, 4))


def test_getitem():
    check("getitem", lambda x: x[1:3], (5,))


def test_composite():
    w1 = Tensor(np.random.randn(4, 6))
    w2 = Tensor(np.random.randn(6, 1))
    check("composite_mlp", lambda x: (x.matmul(w1).relu()).matmul(w2), (3, 4))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nAll gradient checks passed.")
