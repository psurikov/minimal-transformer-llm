import torch
import torch.nn as nn
import numpy as np
from minimal_transformer_llm.tranformer_lm import TransformerLm
from minimal_transformer_llm.gradient_clipping import clip_gradient
from minimal_transformer_llm.bpe_utilities import deserialize_vocab_and_merges
from minimal_transformer_llm.data_loading import load_data
from minimal_transformer_llm.checkpointing import load_checkpoint, save_checkpoint
from minimal_transformer_llm.adamw import AdamW
from minimal_transformer_llm.softmax import softmax
from minimal_transformer_llm.learning_rate_schedule import learning_rate_schedule
from minimal_transformer_llm.cross_entropy import cross_entropy

def training_loop(vocab_size: int, batch_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, tokens_path: str, iterations: int, warmup_iterations: int, cosine_iterations: int, max_learning_rate: float, min_learning_rate: float):
    device_str = "cpu"
    rope_theta = 10000.0
    norm_eps = 0.00001
    d_ff = 4 * d_model
    load = False

    transformer = TransformerLm(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta, norm_eps=norm_eps, device=torch.device(device_str))
    optimizer = AdamW(transformer.parameters(), max_learning_rate)
    tokens = np.memmap(tokens_path, mode='r', dtype=np.uint16)
    inputs, labels = load_data(tokens, batch_size=batch_size, context_length=context_length, device_string=device_str)
    if load:
        load_checkpoint("S:/dev/cs336/training/checkpoint", transformer, optimizer)

    for t in range(iterations):
        lr = learning_rate_schedule(t, max_learning_rate, min_learning_rate, warmup_iterations, cosine_iterations)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.zero_grad()
        input_batch = inputs[t % len(inputs)]
        label_batch = labels[t % len(labels)]
        logits = transformer(input_batch)
        loss = cross_entropy(logits.view(-1, vocab_size), label_batch.view(-1).to(torch.long))
        print(loss.cpu().item())
        loss.backward()
        clip_gradient(transformer.parameters(), m=1.0)
        optimizer.step()

        if t % 10 == 0:
            print(f"Step {t}: loss = {loss.item():.4f}, lr = {lr:.2e}")
        if t % 100 == 0:
            save_checkpoint(transformer, optimizer, t, f"S:/dev/cs336/training/checkpoint-{t}")
        
    print("finish")

def train_regular():
    vocab_size = 10000
    context_length = 256
    d_model = 128
    num_layers = 6
    num_heads = 4
    batch_size = 8
    iterations = 10000
    warmup_iterations = 100
    cosine_iterations = 2000
    max_learning_rate = 2e-4
    min_learning_rate = 1e-5
    tokens_path = "S:/dev/cs336/training/train_tokens.npy"
    training_loop(vocab_size, batch_size, context_length, d_model, num_layers, num_heads, tokens_path, iterations, warmup_iterations, cosine_iterations, max_learning_rate, min_learning_rate)

train_regular()