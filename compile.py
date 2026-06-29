import torch
import torch.nn as nn
from src.model import EdgeVisionTransformerBlock
from src.symbolic import swiglu_symbolic

# Let's bind our custom namespace to our module for the exporter
class ExportWrapper(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = EdgeVisionTransformerBlock(in_features)

    def forward(self, x):
        return self.block(x)

def run_pipeline():
    in_features = 128
    model = ExportWrapper(in_features=in_features)
    model.eval()

    # Create dummy tensor matching standard vision batch sequences (Batch, Sequence, Features)
    dummy_input = torch.randn(1, 16, in_features)

    # Register our custom symbolic mapping before exporting
    # For a clean torch.onnx.export workflow, we can register the custom module behavior
    from torch.onnx import register_custom_op_symbolic
    # We bind an alias to intercept the forward execution map if wrapped in a torch.ops call,
    # or let torch handle the module mapping directly.

    output_filename = "swiglu_edge_model.onnx"
    print(f"[INFO] Exporting model to {output_filename}...")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_filename,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input_tensor'],
        output_names=['output_tensor'],
        # We specify custom operator domains to tell the compiler where our node lives
        custom_opsets={"custom_ops": 1}
    )
    print("[SUCCESS] Graph compilation complete.")

if __name__ == "__main__":
    run_pipeline()
