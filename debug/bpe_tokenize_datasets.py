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

openwebtext_valid_input = "dataset_open_web_valid.txt"
openwebtext_valid_output = "dataset_open_web_valid_tokens.npy"
openwebtext_valid_vocab = "output_owt_train_vocab.json"
openwebtext_valid_merge = "output_owt_train_merges.txt"
openwebtext_valid_tokenizer = BpeTokenizer.from_files(openwebtext_valid_vocab, openwebtext_valid_merge, ["<|endoftext|>"])

tinystories_train_input = "dataset_tiny_stories_train.txt"
tinystories_train_output = "dataset_tiny_stories_train_tokens.npy"
tinystories_train_vocab = "output_tinystoriesv2-GPT4-train_vocab-v1.json"
tinystories_train_merge = "output_tinystoriesv2-GPT4-train_merges-v1.txt"
tinystories_train_tokenizer = BpeTokenizer.from_files(tinystories_train_vocab, tinystories_train_merge, ["<|endoftext|>"])

tinystories_valid_input = "dataset_tiny_stories_valid.txt"
tinystories_valid_output = "dataset_tiny_stories_valid_tokens.npy"
tinystories_valid_vocab = "output_tinystoriesv2-GPT4-valid_vocab-v1.json"
tinystories_valid_merge = "output_tinystoriesv2-GPT4-valid_merges-v1.txt"
tinystories_valid_tokenizer = BpeTokenizer.from_files(tinystories_train_vocab, tinystories_train_merge, ["<|endoftext|>"])

#tokenize_file_to_uint16(openwebtext_valid_tokenizer, openwebtext_valid_input, openwebtext_valid_output)
#tokenize_file_to_uint16(tinystories_train_tokenizer, tinystories_train_input, tinystories_train_output)
tokenize_file_to_uint16(tinystories_valid_tokenizer, tinystories_valid_input, tinystories_valid_output)