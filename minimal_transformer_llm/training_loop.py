import os
import torch
import torch.nn as nn
import numpy as np
from minimal_transformer_llm.transformer_lm import TransformerLm
from minimal_transformer_llm.gradient_clipping import clip_gradient
from minimal_transformer_llm.bpe_utilities import deserialize_vocab_and_merges
from minimal_transformer_llm.data_loading import load_data
from minimal_transformer_llm.checkpointing import load_checkpoint, save_checkpoint
from minimal_transformer_llm.adamw import AdamW
from minimal_transformer_llm.softmax import softmax
from minimal_transformer_llm.learning_rate_schedule import learning_rate_schedule
from minimal_transformer_llm.cross_entropy import cross_entropy
from minimal_transformer_llm.training_parameters import TrainingParameters

def overfit(p: TrainingParameters):
    # create transformer based on input parameters
    transformer = TransformerLm(p.vocab_size, p.context_length, p.d_model, p.num_layers, p.num_heads, p.d_ff, p.rope_theta, p.norm_eps, device=torch.device(p.device_str))
    optimizer = AdamW(transformer.parameters(), p.max_learning_rate)
    valid_tokens = np.memmap(p.valid_tokens_file, mode='r', dtype=np.uint16)
    inputs, labels = load_data(valid_tokens, p.batch_size, 12, p.device_str)
    # iterations
    for t in range(p.iterations):
        # lr adjustments
        lr = learning_rate_schedule(t, p.max_learning_rate, p.min_learning_rate, p.warmup_iterations, p.cosine_iterations)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        # load current batch
        # do forward pass, evaluate loss
        optimizer.zero_grad()
        logits = transformer(inputs)
        loss = cross_entropy(logits.view(-1, p.vocab_size), labels.view(-1).to(torch.long))
        # do backward pass, clip the gradient if necessary
        loss.backward()
        clip_gradient(transformer.parameters(), m=1.0)
        optimizer.step()
        # show each number of iterations the current loss:
        if t % 10 == 0:
            print(f"Step {t}: loss = {loss.item():.4f}, lr = {lr:.2e}")
        #if t % 100 == 0:
        #    save_checkpoint(transformer, optimizer, t, p.save_checkpoint_file(t))
    print("finish")

def training_loop(p: TrainingParameters):
    # create transformer based on input parameters
    transformer = TransformerLm(p.vocab_size, p.context_length, p.d_model, p.num_layers, p.num_heads, p.d_ff, p.rope_theta, p.norm_eps, device=torch.device(p.device_str))
    optimizer = AdamW(transformer.parameters(), p.max_learning_rate)
    train_tokens = np.memmap(p.train_tokens_file, mode='r', dtype=np.uint16)
    valid_tokens = np.memmap(p.valid_tokens_file, mode='r', dtype=np.uint16)
    if os.path.exists(p.load_checkpoint_file):
        load_checkpoint(p.load_checkpoint_file, transformer, optimizer)
    # iterations
    for t in range(p.iterations):
        # lr adjustments
        lr = learning_rate_schedule(t, p.max_learning_rate, p.min_learning_rate, p.warmup_iterations, p.cosine_iterations)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        # load current batch
        inputs, labels = load_data(train_tokens, p.batch_size, p.context_length, p.device_str)
        # do forward pass, evaluate loss
        optimizer.zero_grad()
        logits = transformer(inputs)
        loss = cross_entropy(logits.view(-1, p.vocab_size), labels.view(-1).to(torch.long))
        # do backward pass, clip the gradient if necessary
        loss.backward()
        clip_gradient(transformer.parameters(), max_norm=1.0)
        optimizer.step()
        # show each number of iterations the current loss and divergence with validation set:
        if t % 10 == 0:
            print(f"Step {t}: loss = {loss.item():.4f}, lr = {lr:.2e}")
        if t % 100 == 0:
            save_checkpoint(transformer, optimizer, t, p.save_checkpoint_file(t))
        if t % 100 == 0:
            transformer.eval()
            with torch.no_grad():
                val_inputs, val_labels = load_data(valid_tokens, p.batch_size, p.context_length, p.device_str)
                val_logits = transformer(val_inputs)
                val_loss = cross_entropy(val_logits.view(-1, p.vocab_size), val_labels.view(-1).to(torch.long))
                print(f"[VALIDATION] Step {t}: val loss = {val_loss.item():.4f}")
            transformer.train()
    print("finish")

test_parameters = TrainingParameters(
    vocab_size=10000, 
    batch_size=4,
    context_length=16,
    d_model=64, 
    d_ff=32, 
    num_layers=3, 
    num_heads=4, 
    rope_theta=10000.0,
    norm_eps=0.00001,
    device_str="cpu",
    max_learning_rate=2e-4,
    min_learning_rate=1e-5,
    iterations=10000,
    warmup_iterations=100,
    cosine_iterations=2000,
    train_tokens_file = "S:/dev/cs336/training/train_tokens.npy",
    valid_tokens_file = "S:/dev/cs336/training/valid_tokens.npy",
    load_checkpoint_file="S:/dev/cs336/training/checkpoint",
    save_checkpoint_file=lambda t: f"S:/dev/cs336/training/checkpoint-{t}")

regular_parameters = TrainingParameters(
    vocab_size=10000, 
    batch_size=8,
    context_length=256,
    d_model=128, 
    d_ff=4*128, 
    num_layers=6, 
    num_heads=4, 
    rope_theta=10000.0,
    norm_eps=0.00001,
    device_str="cpu",
    max_learning_rate=2e-4,
    min_learning_rate=1e-5,
    iterations=10000,
    warmup_iterations=100,
    cosine_iterations=2000,
    train_tokens_file = "S:/dev/cs336/training/train_tokens.npy",
    valid_tokens_file = "S:/dev/cs336/training/valid_tokens.npy",
    load_checkpoint_file="S:/dev/cs336/training/checkpoint",
    save_checkpoint_file=lambda t: f"S:/dev/cs336/training/checkpoint-{t}")

tiny_stories_parameters = TrainingParameters(
    vocab_size=10000, 
    batch_size=8,
    context_length=256,
    d_model=512, 
    d_ff=1344, 
    num_layers=4, 
    num_heads=16, 
    rope_theta=10000.0,
    norm_eps=0.00001,
    device_str="cpu",
    max_learning_rate=3e-4,
    min_learning_rate=1e-5,
    iterations=10000,
    warmup_iterations=100,
    cosine_iterations=8000,
    train_tokens_file = "S:/dev/cs336/training/train_tokens.npy",
    valid_tokens_file = "S:/dev/cs336/training/valid_tokens.npy",
    load_checkpoint_file="S:/dev/cs336/training/checkpoint",
    save_checkpoint_file=lambda t: f"S:/dev/cs336/training/checkpoint-{t}")

#overfit(tiny_stories_parameters)
training_loop(tiny_stories_parameters)