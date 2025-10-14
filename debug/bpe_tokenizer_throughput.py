from minimal_transformer_llm.bpe_utilities import deserialize_vocab_and_merges
from minimal_transformer_llm.bpe_tokenizer import BpeTokenizer
import time

def tokenizer_throughput(tokenizer: BpeTokenizer, tokenizer_name: str, file_path: str) -> float:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        byte_len = len(text.encode("utf-8"))
        start = time.perf_counter()
        tokenizer.encode(text)
        end = time.perf_counter()
        elapsed = end - start
        throughput = byte_len / elapsed if elapsed > 0 else float("inf")
        print(f"Tokenizer throughput for {tokenizer_name} on {file_path}: {throughput:,.0f} bytes/sec")
        print(f"Elapsed time: {elapsed:.4f} s")
        print(f"File size: {byte_len:,} bytes")
    return throughput

tinystories_input = "output_tinystories_throughput.txt"
tinystories_vocab = "output_tinystoriesv2-GPT4-train_vocab-v1.json"
tinystories_merge = "output_tinystoriesv2-GPT4-train_merges-v1.txt"
tinystories_tokenizer = BpeTokenizer.from_files(tinystories_vocab, tinystories_merge, ["<|endoftext|>"])

openwebtext_input = "output_owt_throughput.txt"
openwebtext_vocab = "output_owt_train_vocab.json"
openwebtext_merge = "output_owt_train_merges.txt"
openwebtext_tokenizer = BpeTokenizer.from_files(openwebtext_vocab, openwebtext_merge, ["<|endoftext|>"])

tokenizer_throughput(tinystories_tokenizer, "tiny stories tokenizer", tinystories_input)
tokenizer_throughput(openwebtext_tokenizer, "open web text tokenizer", openwebtext_input)