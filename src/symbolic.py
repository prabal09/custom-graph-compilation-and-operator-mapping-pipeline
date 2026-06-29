import torch
from torch.onnx import register_custom_op_symbolic

# Define the symbolic function for ONNX graph generation
def swiglu_symbolic(g, input_tensor):
    """
    g: The ONNX computation graph object.
    input_tensor: The PyTorch Value node passed to the function.
    """
    # We map this block into a completely custom domain ('custom_ops')
    # and assign it an operator name 'CustomSwiGLU'
    return g.op("custom_ops::CustomSwiGLU", input_tensor)

def register_custom_mappings():
    # Syntax: register_custom_op_symbolic('<namespace>::<op_name>', symbolic_fn, opset_version)
    # If overriding an internal ATen/module pattern or an explicit torch namespace:
    register_custom_op_symbolic("custom_ops::swiglu", swiglu_symbolic, opset_version=15)
    print("[INFO] Successfully registered custom symbolic mapping: custom_ops::CustomSwiGLU")
