from typing import Optional, Callable
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float = 0.01, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        defaults = { "lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps }
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            a = group.get("lr")
            b1, b2 = group.get("betas")
            e = group.get("eps")
            l = group.get("weight_decay")
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                g = p.grad.data
                m = state.get("m", torch.full_like(p.data, 0.0))
                v = state.get("v", torch.full_like(p.data, 0.0))
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                t = t + 1
                at = a * (math.sqrt(1 - b2 ** t)) / (1 - b1 ** t)
                p.data -= at * m / (torch.sqrt(v) + e) + a * l * p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss