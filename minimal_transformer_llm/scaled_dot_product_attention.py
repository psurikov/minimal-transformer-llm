import einops
import torch
import torch.nn
from minimal_transformer_llm.softmax import softmax

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # q: " ... queries d_k"
    # k: " ... keys d_k"
    # v: " ... values d_v"
    # mask " ... queries keys"
    # return " ... queries d_v"
    d_k = torch.sqrt(torch.tensor(k.shape[-1], dtype=q.dtype))
    q_k = einops.einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") / d_k
    q_k_masked = q_k
    if mask is not None:
        q_k_masked = q_k.masked_fill(mask == False, float('-inf'))
    q_k_softmax = softmax(q_k_masked, dim=-1)
    q_k_v = einops.einsum(q_k_softmax, v, "... queries keys, ... keys d_v -> ... queries d_v")
    return q_k_v