import torch
import torch.nn as nn

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    largest = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - largest
    exp = torch.exp(shifted)
    result = exp / torch.sum(exp, dim, keepdim=True)
    return result