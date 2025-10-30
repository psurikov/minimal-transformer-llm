import torch
from typing import Iterable

def clip_gradient(parameters: Iterable[torch.nn.Parameter], m: float) -> None:
    e = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.sqrt(sum(g.data.norm() ** 2 for g in grads))
    if norm >= m:
        scale = m / (norm + e)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)