import torch
import torch.nn as nn
from minimal_transformer_llm.linear import Linear

class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device, dtype: torch.dtype):
        super(Swiglu, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_ff, d_model, device, dtype)
        self.w2 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1.forward(x)
        silu = w1x * torch.sigmoid(w1x)
        w3x = self.w3.forward(x)
        gated = silu * w3x
        output = self.w2.forward(gated)
        return output