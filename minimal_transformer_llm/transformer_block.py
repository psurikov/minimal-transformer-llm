import torch
import torch.nn as nn
from minimal_transformer_llm.swiglu import Swiglu
from minimal_transformer_llm.multihead_self_attention import MultiheadSelfAttention
from minimal_transformer_llm.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_length: int, theta: float, device: torch.device):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.multihead_self_attention = MultiheadSelfAttention(d_model, num_heads, max_seq_length, theta, device)
        self.swiglu = Swiglu(d_model, d_ff, device, torch.float)
        self.rmsnorm1 = RMSNorm(d_model, 0.00001, device, torch.float)
        self.rmsnorm2 = RMSNorm(d_model, 0.00001, device, torch.float)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        attended = x + self.multihead_self_attention.forward(self.rmsnorm1(x), token_positions)
        result = attended + self.swiglu.forward(self.rmsnorm2(attended))
        return result