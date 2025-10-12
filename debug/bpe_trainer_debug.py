import pickle
import json
import time
from minimal_transformer_llm.bpe_tokenizer import BpeTokenizer
from minimal_transformer_llm.bpe_trainer import BpeTrainer
from minimal_transformer_llm.bpe_utilities import gpt2_bytes_to_unicode
from functools import lru_cache
from pathlib import Path

# -----------------------------------------------------------------------------------------------------------------------
def test1():
    start_time = time.time()
    trainer = BpeTrainer()
    #trainer.train(r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\tinystories_sample.txt""", 500, ["<|endoftext|>"])
    trainer.train(r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\corpus.en""", 500, ["<|endoftext|>"])
    vocab = trainer.vocab
    merges = trainer.merges

    reference_vocab_path = r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\train-bpe-reference-vocab.json""" 
    reference_merges_path = r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\train-bpe-reference-merges.txt"""

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())
    end_time = time.time()
    span_time = end_time - start_time
    print(span_time)
#-----------------------------------------------------------------------------------------------------------------------
def test2():
    trainer1 = BpeTrainer()
    trainer1.train(r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\tinystories_sample_5M.txt""", 1000, ["<|endoftext|>"])
    snapshot_file = Path(r"S:\dev\cs336\minimal-transformer-llm\tests\_snapshots\test_train_bpe_special_tokens.pkl")
    with snapshot_file.open("rb") as f:
        saved_snapshot = pickle.load(f)
    vocab1 = trainer1.vocab
    merges1 = trainer1.merges
    vocabs_without_specials = [word for word in vocab1.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes
    # Your current test data
    current_data = {
        "vocab_keys": set(vocab1.keys()),
        "vocab_values": set(vocab1.values()),
        "merges": merges1,
    }

    # Compare manually
    assert current_data == saved_snapshot, "Snapshot does not match!"

if __name__ == "__main__":
    test1()
    test1()
    test1()
    test2()