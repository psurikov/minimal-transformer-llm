import torch
import torch.nn as nn
from minimal_transformer_llm.sgd import SGD

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
optimizer = SGD([weights], lr = 1000)

for t in range(10):
    optimizer.zero_grad()
    loss = (weights ** 2).mean()
    print(loss.cpu().item())
    loss.backward()
    optimizer.step()