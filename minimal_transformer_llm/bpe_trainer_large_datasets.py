import pathlib
import json
import time
from minimal_transformer_llm.bpe_trainer_sequential import BpeTrainerSequential
from minimal_transformer_llm.bpe_trainer_multithreaded import BpeTrainerMultiThreaded
from minimal_transformer_llm.bpe_trainer_multiprocess import BpeTrainerMultiProcess
from functools import lru_cache

def gpt2_bytes_to_unicode1():
    """
    Returns a dict mapping from byte values (0–255) to unique unicode strings.
    Used in GPT-2 byte-level BPE encoding.
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
# -----------------------------------------------------------------------------------------------------------------------
def test(input_path: str, vocab_size: int, special_tokens: list[str], vocab_path: str, merges_path: str):
    start_time = time.time()
    trainer = BpeTrainerMultiProcess()
    trainer.train(input_path, vocab_size, special_tokens)
    vocab = trainer.vocab
    merges = trainer.merges

    encoder = gpt2_bytes_to_unicode()
    decoder = {v: k for k, v in encoder.items()}
    vocab_serializable = { "".join(encoder[byte] for byte in token): idx for idx, token in vocab.items() }
    merges_serializable = [f"{''.join(encoder[byte] for byte in a)} {''.join(encoder[byte] for byte in b)}" for a, b in merges]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("\n".join(merges_serializable))

    end_time = time.time()
    span_time = end_time - start_time
    print(f"{input_path}: {span_time:.2f} seconds")

if __name__ == "__main__":
    #test(r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\corpus.en""", 500, ["<|endoftext|>"], "output_corpus.en_vocab.json", "output_corpus.en_merges.txt")
    #test(r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\tinystories_sample.txt""", 500, ["<|endoftext|>"], "output_tinystories_sample_vocab.json", "output_tinystories_sample_merges.txt")
    #test(r"""S:\dev\cs336\minimal-transformer-llm\tests\fixtures\tinystories_sample_5M.txt""", 1000, ["<|endoftext|>"], "output_tinystories_5M_sample_vocab.json", "output_tinystories_5M_sample_merges.txt")
    #test(r"""S:\dev\cs336\datasets\TinyStoriesV2-GPT4-valid.txt""", 10000, ["<|endoftext|>"], "output_tinystoriesv2-GPT4-valid_vocab.json", "output_tinystoriesv2-GPT4-valid_merges.txt")
    test(r"""/app/tiny-stories-train.txt""", 10000, ["<|endoftext|>"], "output_tinystoriesv2-GPT4-train_vocab.json", "output_tinystoriesv2-GPT4-train_merges.txt")
    test(r"""/app/tiny-stories-train.txt""", 32000, ["<|endoftext|>"], "output_tinystoriesv2-GPT4-train_vocab.json", "output_tinystoriesv2-GPT4-train_merges.txt")