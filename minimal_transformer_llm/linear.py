import einops
import math
import torch
import torch.nn as nn

class Linear(nn.Module):
     def __init__(self, in_features: int, out_features: int, device: torch.device=None, dtype: torch.dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, 0.0, sigma, -3 * sigma, 3 * sigma)
     
     def forward(self, x:torch.Tensor) -> torch.Tensor:
        # return x @ self.weight.T
        return einops.einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")