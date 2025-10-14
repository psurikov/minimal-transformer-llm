from minimal_transformer_llm.bpe_utilities import deserialize_vocab_and_merges
from minimal_transformer_llm.bpe_tokenizer import BpeTokenizer

def compression_ratio(tokenizer: BpeTokenizer, tokenizer_name: str, file_path: str) -> float:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        tokens = tokenizer.encode(text)
        token_len = len(tokens)
        byte_len = len(text.encode("utf-8"))
        ratio = byte_len / token_len
        print(f"Compression ratio for {tokenizer_name} on {file_path}: {ratio:.2f} bytes/token")
        print(f"total file length: {byte_len}")
        print(f"total token length: {token_len}")
    return ratio

tinystories_input = "output_tinystories_10_documents.txt"
tinystories_vocab = "output_tinystoriesv2-GPT4-train_vocab-v1.json"
tinystories_merge = "output_tinystoriesv2-GPT4-train_merges-v1.txt"
tinystories_tokenizer = BpeTokenizer.from_files(tinystories_vocab, tinystories_merge, ["<|endoftext|>"])

openwebtext_input = "output_owt_10_documents.txt"
openwebtext_vocab = "output_owt_train_vocab.json"
openwebtext_merge = "output_owt_train_merges.txt"
openwebtext_tokenizer = BpeTokenizer.from_files(openwebtext_vocab, openwebtext_merge, ["<|endoftext|>"])

compression_ratio(tinystories_tokenizer, "tiny stories tokenizer", tinystories_input)
compression_ratio(openwebtext_tokenizer, "open web text tokenizer", openwebtext_input)
compression_ratio(tinystories_tokenizer, "tiny stories tokenizer", openwebtext_input)

