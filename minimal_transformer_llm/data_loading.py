import torch
import numpy as np
import numpy.typing as npt

def load_data(dataset: npt.NDArray, batch_size: int, context_length: int, device_string: str) -> tuple[torch.Tensor, torch.Tensor]:
    # the inputs are in uint16 (at least this is how the tokens are serialized)
    device = torch.device(device_string)
    dtype = torch.int
    length = len(dataset) - context_length
    if length <= 0:
        raise ValueError("The dataset length is too short")
    # generate batches arrays
    input = torch.empty(batch_size, context_length, dtype=dtype, device=device)
    label = torch.empty(batch_size, context_length, dtype=dtype, device=device) 
    # assign for each batch a part of dataset
    for i in range(batch_size):
        start = np.random.randint(0, length)
        input[i] = torch.from_numpy(dataset[start : start + context_length])
        label[i] = torch.from_numpy(dataset[start + 1 : start + context_length + 1])
    return input, label

def load_data_np_arrays(dataset: npt.NDArray, batch_size: int, context_length: int, device_string: str) -> tuple[torch.Tensor, torch.Tensor]:
    # the inputs are in uint16 (at least this is how the tokens are serialized)
    device = torch.device(device_string)
    dtype = np.uint16
    length = len(dataset) - context_length
    if length <= 0:
        raise ValueError("The dataset length is too short")
    # generate batches arrays
    start_indices = np.random.randint(0, length, size=batch_size)
    input_batches = np.empty((batch_size, context_length), dtype=dtype)
    label_batches = np.empty((batch_size, context_length), dtype=dtype)
    # assign for each batch a part of dataset
    for i, start in enumerate(start_indices):
        input_batches[i] = dataset[start : start + context_length]
        label_batches[i] = dataset[start + 1 : start + context_length + 1]
    # for the actual tensors
    input_tensor = torch.from_numpy(input_batches).int().to(device)
    label_tensor = torch.from_numpy(label_batches).int().to(device)
    return input_tensor, label_tensor