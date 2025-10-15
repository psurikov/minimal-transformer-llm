import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        print(theta)
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
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x1_rotated
        x_rotated[..., 1::2] = x2_rotated
        return x_rotated