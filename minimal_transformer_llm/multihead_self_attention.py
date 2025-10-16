import einops
import torch
import torch.nn as nn
from minimal_transformer_llm.linear import Linear
from minimal_transformer_llm.scaled_dot_product_attention import scaled_dot_product_attention
from minimal_transformer_llm.rope import RotaryPositionalEmbedding

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1000, theta: float = 10000.0, device: torch.device = None):
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        d_k = d_v = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device)
        self.q_proj = Linear(num_heads * d_k, d_model, device, torch.float)
        self.k_proj = Linear(num_heads * d_k, d_model, device, torch.float)
        self.v_proj = Linear(num_heads * d_v, d_model, device, torch.float)
        self.o_proj = Linear(d_model, num_heads * d_v, device, torch.float)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # projections
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        # currently the shapes are h * d_k, d_model, while we need to split them into h, d_k, d_model
        q_per_head = einops.rearrange(q, "... seq (h d_k) ->  ... h seq d_k", h = self.num_heads) 
        k_per_head = einops.rearrange(k, "... seq (h d_k) ->  ... h seq d_k", h = self.num_heads)
        v_per_head = einops.rearrange(v, "... seq (h d_k) ->  ... h seq d_k", h = self.num_heads)
        # rope
        if token_positions is not None:
            q_per_head = self.rope.forward(q_per_head, token_positions)
            k_per_head = self.rope.forward(k_per_head, token_positions)
        # causal mask
        *leading_dims, seq_len, _ = x.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q_per_head.device, dtype=torch.bool))
        causal_mask = causal_mask.view(*([1]*len(leading_dims)), 1, seq_len, seq_len)
        causal_mask = causal_mask.expand(*leading_dims, self.num_heads, seq_len, seq_len)
        # attention calculated per head due to previous reshaping
        attended = scaled_dot_product_attention(q_per_head, k_per_head, v_per_head, causal_mask)
        # reshape it back
        reshaped = einops.rearrange(attended, "... h seq d_k -> ... seq (h d_k)", h = self.num_heads)
        self_attention = self.o_proj.forward(reshaped)
        return self_attention