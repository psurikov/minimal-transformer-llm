import numpy as np
from minimal_transformer_llm.bpe_tokenizer import BpeTokenizer

def tokenize_file_to_uint16(tokenizer: BpeTokenizer, input_path: str, output_path: str):
    print(f"Tokenizing {input_path}")
    ids = []
    with open(input_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
    arr = np.array(ids, dtype=np.uint16)
    np.save(output_path, arr)
    print(f"Saved {len(arr):,} tokens to {output_path} ({arr.nbytes/1e6:.2f} MB)")

openwebtext_input = "dataset_open_web_valid.txt"
openwebtext_output = "dataset_open_web_valid_tokens.npy"
openwebtext_vocab = "output_owt_train_vocab.json"
openwebtext_merge = "output_owt_train_merges.txt"
openwebtext_tokenizer = BpeTokenizer.from_files(openwebtext_vocab, openwebtext_merge, ["<|endoftext|>"])

tinystories_input = "dataset_tiny_stories_train.txt"
tinystories_output = "dataset_tiny_stories_train_tokens.npy"
tinystories_vocab = "output_tinystoriesv2-GPT4-train_vocab-v1.json"
tinystories_merge = "output_tinystoriesv2-GPT4-train_merges-v1.txt"
tinystories_tokenizer = BpeTokenizer.from_files(tinystories_vocab, tinystories_merge, ["<|endoftext|>"])

tokenize_file_to_uint16(openwebtext_tokenizer, openwebtext_input, openwebtext_output)
tokenize_file_to_uint16(tinystories_tokenizer, tinystories_input, tinystories_output)