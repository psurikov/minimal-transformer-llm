import torch
import numpy as np
from minimal_transformer_llm.softmax import softmax
from minimal_transformer_llm.bpe_utilities import deserialize_vocab_and_merges
from minimal_transformer_llm.checkpointing import load_checkpoint
from minimal_transformer_llm.transformer_lm import TransformerLm
from minimal_transformer_llm.bpe_tokenizer import BpeTokenizer
from minimal_transformer_llm.training_parameters import TrainingParameters

def select_token(logits, top_p=0.9, temperature=1.0):
    # apply temperature
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    # sort by descending probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # remove tokens with cumulative probability above the threshold
    cutoff = cumulative_probs > top_p
    if torch.any(cutoff):
        cutoff_index = torch.nonzero(cutoff, as_tuple=False)[0].item()
        sorted_probs = sorted_probs[:cutoff_index + 1]
        sorted_indices = sorted_indices[:cutoff_index + 1]
    # renormalize and sample
    sorted_probs = sorted_probs / sorted_probs.sum()
    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_indices[next_token]

def generate_text2(transformer, prompt_tokens, max_new_tokens, tokenizer: BpeTokenizer, parameters: TrainingParameters, temperature=0.8, top_p=0.9):
    transformer.eval()
    generated = torch.tensor(prompt_tokens, dtype=torch.long, device=parameters.device_str).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_seq = generated[:, -parameters.context_length:]
            logits = transformer(input_seq)
            next_token_logits = logits[0, -1, :]
            next_token = select_token(next_token_logits, top_p=top_p, temperature=temperature)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if tokenizer.decode([next_token.item()]) == "<|endoftext|>":
                break
    tokens = generated[0].cpu().tolist()
    decoded_text = tokenizer.decode(tokens)
    return decoded_text

def generate_text(transformer, prompt_tokens, max_new_tokens, tokenizer: BpeTokenizer, parameters: TrainingParameters, temperature=1.0):
    # prompt_tokens: list[int] or 1D torch.Tensor of token IDs
    # max_new_tokens: int, how many tokens to generate
    # vocab: dict[int, str] mapping token IDs to text (used for decoding)
    # temperature: float, controls randomness (0 = deterministic, >1 = more random)
    # device_str: "cpu" or "cuda"
    transformer.eval()
    generated = torch.tensor(prompt_tokens, dtype=torch.long, device=parameters.device_str).unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_seq = generated[:, -parameters.context_length:]                # make sure the token sequence doens't exceed context_length
            logits = transformer(input_seq)
            next_token_logits = logits[0, -1, :] / temperature
            probs = softmax(next_token_logits, -1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if tokenizer.vocab.get(next_token.item()) == "<|endoftext|>":
                break
    tokens = generated[0].cpu().tolist()
    decoded_text = tokenizer.decode(tokens)
    return decoded_text

parameters = TrainingParameters(
    vocab_size=10000, 
    batch_size=32,
    context_length=256,
    d_model=512, 
    d_ff=1344, 
    num_layers=4, 
    num_heads=16, 
    rope_theta=10000.0,
    norm_eps=0.00001,
    device_str="cpu",
    max_learning_rate=1e-4,
    min_learning_rate=1e-5,
    iterations=10000,
    warmup_iterations=100,
    cosine_iterations=8000,
    train_tokens_file = "S:/dev/cs336/training/train_tokens.npy",
    valid_tokens_file = "S:/dev/cs336/training/valid_tokens.npy",
    load_checkpoint_file="S:/dev/cs336/training/checkpoint",
    save_checkpoint_file=lambda t: f"S:/dev/cs336/training/checkpoint-{t}")

merges_file = "S:/dev/cs336/training/merges.txt"
vocab_file = "S:/dev/cs336/training/vocab.json"
checkpoint_file = "S:/dev/cs336/training/weights"

vocab, merges = deserialize_vocab_and_merges(vocab_file, merges_file)
checkpoint = torch.load(checkpoint_file)
transformer = TransformerLm(
    parameters.vocab_size, 
    parameters.context_length, 
    parameters.d_model, 
    parameters.num_layers, 
    parameters.num_heads, 
    parameters.d_ff, 
    parameters.rope_theta, 
    parameters.norm_eps, 
    torch.device(parameters.device_str))
transformer.load_state_dict(checkpoint["model_state"])
transformer.eval()
tokenizer = BpeTokenizer(vocab, merges, ["<|endoftext|>"])
prompt = tokenizer.encode("Not all things are as they look.")
text = generate_text2(transformer, prompt, max_new_tokens=300, tokenizer=tokenizer, parameters=parameters, temperature=0.8)
print(text)