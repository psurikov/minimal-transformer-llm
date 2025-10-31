import torch
import numpy as np
from minimal_transformer_llm.data_loading import load_data, load_data_randomized

dataset = np.arange(0, 100)
context_length = 7
batch_size = 32
device = "cpu"

x1, y1 = load_data(dataset, batch_size, context_length, device)
print(x1)
print(y1)

x2, y2 = load_data_randomized(dataset, batch_size, context_length, device)
print(x2)
print(y2)