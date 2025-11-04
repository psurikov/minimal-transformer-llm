import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.angles = torch.empty(max_seq_len, d_k, device=device)
        i = torch.arange(max_seq_len).unsqueeze(1) # shape [max_seq_len, 1], values 0, 1, 2, 3 ...
        k = torch.arange(d_k // 2).unsqueeze(0) # shape [1, d_k//2], values 0, 1, 2, 3 ...
        angles = i / (theta ** (2 * k / d_k)) # the formula differs because k = 0, 1, 2, 3
        self.register_buffer("sin", torch.sin(angles),  persistent=False)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:
        # token_positions tensor has shape: [batch_size, seq_len]
        # or [0,1,2,3,..,len-1], [0,1,2,3,..,len-1], ...
        # self.cos has shape: [seq_len, d_k // 2]
        # by performing self.cos[token_positions], this essentially replaces each index in token_positions by respective array of angles in self.cos
        # resulting in shape [batch_size, seq_len, d_k // 2]
        # we want to broadcast (multiply element-wise) these values by x1 and x2, which are based on x
        # assume x has shape [batch_size, heads, seq_len, d_k]
        # then x1, x2 have shapes [batch_size, heads, seq_len, d_k // 2]
        # this almost matches the shape of cos, compare:
        # cos:    [batch_size, seq_len, d_k // 2]
        # x1, x2: [batch_size, heads, seq_len, d_k // 2]
        # there is an additional heads dimension here, so simply multiplying these tensors won't work
        # as a result we need to unsqueeze cos (sin) to include heads dimension, and use 1 just so the dimensions match
        # this is why .unsqueeze(-3) is used, as a result the shape [batch_size, seq_len, d_k // 2] is transformeed to [batch_size, 1, seq_len, d_k // 2]
        #
        # example:
        # batch_size = 2, seq_len = 4, d_k // 2 = 2
        # token_positions:  shape 2, 4 (batch_size, seq_len) [[0, 1, 2, 3], [0, 1, 2, 3]]
        # self.cos:         shape 5, 2 (max_seq_len, d_k // 2)    [[0.1], [0.2]], [[0.3], [0.4]], [[0.5], [0.6]], [[0.7], [0.8]], [[0.9], [1.0]]
        # cos =             shape 2, 4, 2 (batch_size, seq_len, d_k // 2)
        # [
        #   [[0.1], [0.2]], 
        #   [[0.3], [0.4]], 
        #   [[0.5], [0.6]], 
        #   [[0.7], [0.8]]
        # ], 
        # [
        #   [[0.1], [0.2]], 
        #   [[0.3], [0.4]], 
        #   [[0.5], [0.6]], 
        #   [[0.7], [0.8]]
        # ]
        cos = self.cos[token_positions].unsqueeze(-3)
        sin = self.sin[token_positions].unsqueeze(-3)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x1_rotated
        x_rotated[..., 1::2] = x2_rotated
        return x_rotated