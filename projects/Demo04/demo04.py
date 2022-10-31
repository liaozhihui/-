from torch.onnx.symbolic_registry import register_op


def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)


register_op('asinh', asinh_symbolic, '', 9)