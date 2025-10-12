import json
import mmap
import regex

# returns a dict mapping from byte values (0-255) to unique unicode strings. Used in GPT-2 byte-level BPE encoding
def gpt2_bytes_to_unicode():
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

# serializes vocab and merges into .json and .txt files
def serialize_vocab_and_merges(vocab_path: str, merges_path: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
    encoder = gpt2_bytes_to_unicode()
    vocab_serializable = { "".join(encoder[byte] for byte in token): idx for idx, token in vocab.items() }
    merges_serializable = [f"{''.join(encoder[byte] for byte in a)} {''.join(encoder[byte] for byte in b)}" for a, b in merges]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("\n".join(merges_serializable))

# deserialize vocab and merges from .json and .txt files
def deserialize_vocab_and_merges(vocab_path: str, merges_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    encoder = gpt2_bytes_to_unicode()
    decoder = {v: k for k, v in encoder.items()}
    # deserializing vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_serializable = json.load(f)
    vocab = {}
    for token_str, idx in vocab_serializable.items():
        token_bytes = bytes([decoder[ch] for ch in token_str])
        vocab[idx] = token_bytes
    # deserializing merges
    with open(merges_path, "r", encoding="utf-8") as f:
        merges_serializable = [line.strip() for line in f if line.strip()]
    merges = []
    for line in merges_serializable:
        a_str, b_str = line.split(" ")
        a_bytes = bytes([decoder[ch] for ch in a_str])
        b_bytes = bytes([decoder[ch] for ch in b_str])
        merges.append((a_bytes, b_bytes))
    return vocab, merges

# returns a regex based on gpt2 pretoken regex with special tokens
def gpt2_pretoken_regex(special_tokens: list[str]) -> str:
    pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    regex_pattern = "|".join([regex.escape(special_token) for special_token in special_tokens]) + "|" + pretoken_regex
    return regex_pattern

# splits file into chunks based on special_token edge and a desired size
def split_into_chunks(mm: mmap.mmap, start: int, end: int, desired_size: int, special_tokens: list[bytes]) -> list[tuple[int, int]]:
    chunks = []
    total_size = end - start
    chunk_size = min(desired_size, total_size)
    chunk_start = start
    chunk_end = chunk_start + desired_size
    while chunk_start < end:
        chunk_end = _adjust_chunk_end(mm, chunk_end, end, special_tokens)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
        chunk_end = min(chunk_start + chunk_size, end)
    return chunks

# adjusts the chunk to end on a special token
def _adjust_chunk_end(mm: mmap.mmap, chunk_end: int, end: int, special_tokens: list[bytes]) -> int:
    if chunk_end >= end:
        return end
    while chunk_end < end:
        read_count = min(1024, end - chunk_end)
        read_ahead = mm[chunk_end:chunk_end + read_count]
        found_token = False
        for special_token in special_tokens:
            pos = read_ahead.find(special_token)
            if pos != -1:
                chunk_end += pos
                found_token = True
                break
        if found_token:
            break
        else:
            chunk_end += read_count
    return chunk_end