import torch
import torch.nn as nn
from minimal_transformer_llm.transformer_block import TransformerBlock
from minimal_transformer_llm.embedding import Embedding
from minimal_transformer_llm.rmsnorm import RMSNorm
from minimal_transformer_llm.linear import Linear
from minimal_transformer_llm.softmax import softmax

class TransformerLm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, norm_eps: float, device: torch.device):
        super(TransformerLm, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, norm_eps, device) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, norm_eps, device, torch.float)
        self.linear = Linear(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size sequence_length
        # embeddings: batch_size sequence_length d_model
        embeddings = self.embedding(x)
        batch_size, seq_len = x.shape
        # token_positions: [0, 1, 2, 3, 4, .. seq_len - 1] * batch_size
        token_positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        transformed = embeddings
        for i in range(self.num_layers):
            transformed = self.transformer_blocks[i](transformed, token_positions)
        normalized = self.norm(transformed)
        output_embeddings = self.linear(normalized)
        return output_embeddings



