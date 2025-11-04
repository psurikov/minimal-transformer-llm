import torch
import numpy as np
from minimal_transformer_llm.softmax import softmax
from minimal_transformer_llm.bpe_utilities import deserialize_vocab_and_merges
from minimal_transformer_llm.transformer_lm import TransformerLm

def generate_text(transformer, prompt_tokens, max_new_tokens, vocab, temperature=1.0, device_str="cpu"):
    # prompt_tokens: list[int] or 1D torch.Tensor of token IDs
    # max_new_tokens: int, how many tokens to generate
    # vocab: dict[int, str] mapping token IDs to text (used for decoding)
    # temperature: float, controls randomness (0 = deterministic, >1 = more random)
    # device_str: "cpu" or "cuda"
    transformer.eval()
    generated = torch.tensor(prompt_tokens, dtype=torch.long, device=device_str).unsqueeze(0)  # (1, seq_len)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_seq = generated[:, -transformer.context_length:]                # make sure the token sequence doens't exceed context_length
            logits = transformer(input_seq)
            next_token_logits = logits[0, -1, :] / temperature
            probs = softmax(next_token_logits)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if vocab.get(next_token.item()) == "<|endoftext|>":
                break
    tokens = generated[0].cpu().tolist()
    decoded_text = "".join(vocab[t] for t in tokens if t in vocab)
    return decoded_text

d_model = 512
context_length = 1024
num_layers = 4
num_heads = 16
d_ff = 4 * d_model
rope_theta = 10000.0
norm_eps = 0.00001
device = torch.device("cpu")
vocab, merges = deserialize_vocab_and_merges("S:/dev/cs336/training/vocab.json", "S:/dev/cs336/training/merges.txt")
transformer = TransformerLm(len(vocab), context_length, d_model, num_layers, num_heads, d_ff, rope_theta, norm_eps, device)
transformer.eval()
prompt = [vocab.get("Hello", 0)]
text = generate_text(transformer, prompt, max_new_tokens=50, vocab=vocab, temperature=0.8)
print(text)