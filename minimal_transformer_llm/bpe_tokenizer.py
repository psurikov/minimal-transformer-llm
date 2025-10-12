from typing import Iterable, Iterator
from minimal_transformer_llm.bpe_utilities import deserialize_vocab_and_merges
from minimal_transformer_llm.bpe_pretokenizer import BpePretokenizer

class BpeTokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.pretokenizer = BpePretokenizer(special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab, merges = deserialize_vocab_and_merges(vocab_filepath, merges_filepath)
        tokenizer = cls(vocab, merges, special_tokens)
        return tokenizer

    def encode(self, text: str) -> list[int]:
        merged = []
        cache = {}
        for special_token in self.special_tokens:
            cache[special_token] = [self.reverse_vocab[special_token.encode("utf-8")]]
        for token in self.pretokenizer.next_token(text):
            if token in cache:
                merged += cache[token]
            else:
                token_bytes = token.encode("utf-8")
                token_bytes_parts = [bytes([a]) for a in token_bytes]
                for merge in self.merges:
                    left, right = merge
                    merged_parts = []
                    i = 0
                    while i < len(token_bytes_parts):
                        if i + 1 < len(token_bytes_parts) and token_bytes_parts[i] == left and token_bytes_parts[i + 1] == right:
                            merged_parts.append(left + right)
                            i += 2
                        else:
                            merged_parts.append(token_bytes_parts[i])
                            i += 1
                    token_bytes_parts = merged_parts
                ids = ([self.reverse_vocab[a] for a in token_bytes_parts])
                cache[token] = ids
                merged += ids
        return merged
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        merged_bytes = b"".join([self.vocab.get(id, b"") for id in ids])
        decoded_str = merged_bytes.decode("utf-8", errors="replace")
        return decoded_str
