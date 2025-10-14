from minimal_transformer_llm.bpe_tokenizer import BpeTokenizer

vocab_path = r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\gpt2_vocab.json"""
merges_path = r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\gpt2_merges.txt"""
special_tokens = ["<|endoftext|>"]
tokenizer = BpeTokenizer.from_files(vocab_path, merges_path, special_tokens)
test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
encoded_ids = tokenizer.encode(test_string)
tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
# Ensure the special <|endoftext|> token is preserved
assert tokenized_string.count("<|endoftext|>") == 3

decoded_string = tokenizer.decode(encoded_ids)
assert test_string == decoded_string

tiny_stories_sample_5m = r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\tinystories_sample_5M.txt"""
tokenizer1 = BpeTokenizer.from_files(vocab_path, merges_path, special_tokens)
with open(tiny_stories_sample_5m, encoding="utf-8") as f:
    ids = []
    for _id in tokenizer1.encode_iterable(f):
        ids.append(_id)