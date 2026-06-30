import torch
import torch.nn as nn
from torch.onnx import _internal as onnx_internal

# =====================================================================
# 1. REGISTER THE BASE OPERATOR SCHEMA AS A BLACK BOX
# =====================================================================
# We remove the mutates keyword argument completely to guarantee
# cross-version compatibility for pure mathematical operations.
@torch.library.custom_op("custom_ops::swiglu_forward")
def swiglu_forward(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    x_left = x[..., :dim // 2]
    x_right = x[..., dim // 2:]
    return (x_left * torch.sigmoid(x_left)) * x_right

# =====================================================================
# 2. REGISTER THE FX ONNX TRANSLATION RULE (THE SYMBOLIC MAPPING)
# =====================================================================
def swiglu_onnx_translation(g, x):
    # This creates a single unified node in the final graph
    return g.op("custom_ops::CustomSwiGLU", x)

# Bind the translation rule to the custom library entry point
torch.onnx.register_custom_op_symbolic("custom_ops::swiglu_forward", swiglu_onnx_translation, opset_version=15)

# =====================================================================
# 3. CONSTRUCT THE MODEL
# =====================================================================
class FusedSwiGLU(nn.Module):
    def forward(self, x):
        # Call the strict custom op block
        return swiglu_forward(x)

class EdgeVisionTransformerBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features * 2)
        self.swiglu = FusedSwiGLU()
        self.fc2 = nn.Linear(in_features, in_features)

    def forward(self, x):
        return self.fc2(self.swiglu(self.fc1(x)))

# =====================================================================
# 4. EXPORT AND COMPILE
# =====================================================================
def run_pipeline():
    in_features = 128
    model = EdgeVisionTransformerBlock(in_features=in_features)
    model.eval()

    dummy_input = torch.randn(1, 16, in_features)
    output_filename = "swiglu_fused_model.onnx"

    print(f"[INFO] Exporting graph to {output_filename}...")

    torch.onnx.export(
        model,
        dummy_input,
        output_filename,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input_tensor'],
        output_names=['output_tensor'],
        custom_opsets={"custom_ops": 1}
    )
    print("[SUCCESS] Model compiled successfully.")

if __name__ == "__main__":
    run_pipeline()
