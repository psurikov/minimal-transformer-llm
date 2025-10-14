import json
from functools import lru_cache
from minimal_transformer_llm.bpe_utilities import gpt2_bytes_to_unicode

def longest_words_from_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return []
    max_len = max(len(key) for key in data.keys())
    longest_keys = [key for key in data.keys() if len(key) == max_len]
    return longest_keys

file_path = "S:/dev/cs336/minimal-transformer-llm-experiments/output_owt_train_vocab.json"
longest = longest_words_from_json(file_path)
encoder = gpt2_bytes_to_unicode()
decoder = {v: k for k, v in encoder.items()}
decoded = []
for token in longest:
    # Convert each unicode character in token back to its original byte
    byte_values = [decoder[ch] for ch in token]
    # Convert bytes to a string (the true original token text)
    decoded_token = bytes(byte_values).decode("utf-8", errors="replace")
    decoded.append(decoded_token)
print("Longest word(s):", longest)
print("Longest decoded word(s):", decoded)