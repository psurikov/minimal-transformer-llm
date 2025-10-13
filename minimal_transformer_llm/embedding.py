import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device=None, dtype: torch.dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        sigma = 1.0
        nn.init.trunc_normal_(self.weight, 0.0, sigma, - 3 * sigma, 3 * sigma)

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        return self.weight[token_ids]