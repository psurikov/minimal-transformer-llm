import torch
from typing import Iterable

def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return
    norm = torch.sqrt(sum(g.norm() ** 2 for g in grads) + eps)
    if norm > max_norm:
        scale = max_norm / norm
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)