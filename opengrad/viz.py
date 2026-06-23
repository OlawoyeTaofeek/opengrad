"""
opengrad.viz
=============
Draws the computation graph built up by Tensor operations, in the spirit of
micrograd's famous `draw_dot`. Each node shows the tensor's shape, op, and
(for small tensors) its actual values/gradients -- handy for teaching and
for debugging your own forward passes.
"""

from graphviz import Digraph


def _trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def _format_tensor(t, max_elems=4):
    flat = t.data.flatten()
    if flat.size <= max_elems:
        return ", ".join(f"{x:.3f}" for x in flat)
    return ", ".join(f"{x:.3f}" for x in flat[:max_elems]) + ", ..."


def draw_graph(root, show_values=True, rankdir='LR'):
    """
    Returns a graphviz Digraph you can render with `.render('name')` (writes
    a PDF/PNG to disk) or just display directly in a Jupyter notebook.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': rankdir})

    nodes, edges = _trace(root)
    for n in nodes:
        uid = str(id(n))
        label = f"shape {n.data.shape}"
        if show_values:
            label += f"| \ndata: {_format_tensor(n)}"
            label += f"| \ngrad: {_format_tensor(type(n)(n.grad))}"
        dot.node(name=uid, label=label, shape='record')
        if n._op:
            op_uid = uid + n._op
            dot.node(name=op_uid, label=n._op, shape='circle', style='filled', fillcolor='lightblue')
            dot.edge(op_uid, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
