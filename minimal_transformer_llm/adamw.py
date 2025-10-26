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
            a = group.get("lr")                                          # hyperparameters
            b1, b2 = group.get("betas")
            e = group.get("eps")
            l = group.get("weight_decay")
            for p in group["params"]:                                    # iterate over each learnable parameter (e.g. weight)
                if p.grad is None:
                    continue
                state = self.state[p]                                    # get the various associated details related to current learnable parameter
                t = state.get("t", 0)                                    # iteration
                g = p.grad.data                                          # the gradient
                m = state.get("m", torch.full_like(p.data, 0.0))         # first moment has the same shape as learnable parameters (weights)
                v = state.get("v", torch.full_like(p.data, 0.0))         # second moment has the same shape as well
                m = b1 * m + (1 - b1) * g                                # compute first moment
                v = b2 * v + (1 - b2) * g ** 2                           # compute second moment
                t = t + 1                                                # increasing the iteration before, it is used in subsequent calculations
                at = a * (math.sqrt(1 - b2 ** t)) / (1 - b1 ** t)        # adjusted learning rate taking into account iteration 
                p.data -= at * m / (torch.sqrt(v) + e) + a * l * p.data  # the actual change in weights is taken from moments + weight decay (l here is the decay rate)
                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss