import numpy
import torch
from minimal_transformer_llm.multihead_self_attention import MultiheadSelfAttention

d_model = 4
num_heads = 2
max_seq_len = 1000
device = torch.device("cpu")

q_proj = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
k_proj = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
v_proj = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
o_proj = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
in_features = torch.Tensor([[[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]], [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]], [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]])

multiheadSelfAttention = MultiheadSelfAttention(d_model, num_heads, max_seq_len, device)
multiheadSelfAttention.q_proj.weight.data = q_proj
multiheadSelfAttention.k_proj.weight.data = k_proj
multiheadSelfAttention.v_proj.weight.data = v_proj
multiheadSelfAttention.o_proj.weight.data = o_proj
out_features = multiheadSelfAttention.forward(in_features)

# expected_output has shape (4, 12, 64)