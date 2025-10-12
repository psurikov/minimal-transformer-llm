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
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab, merges = deserialize_vocab_and_merges(vocab_filepath, merges_filepath)
        tokenizer = cls(vocab, merges, special_tokens)
        return tokenizer

    def encode(self, text: str) -> list[int]:
        merged = []
        for special_token in self.special_tokens:
            self.cache[special_token] = [self.reverse_vocab[special_token.encode("utf-8")]]
        for token in self.pretokenizer.next_token(text):
            if token in self.cache:
                merged += self.cache[token]
            else:
                token_bytes = token.encode("utf-8")
                token_bytes_parts = [bytes([a]) for a in token_bytes]
                count = len(token_bytes_parts)
                for merge in self.merges:
                    left, right = merge
                    i = 0
                    into = 0
                    while i < count:
                        if i + 1 < count and token_bytes_parts[i] == left and token_bytes_parts[i + 1] == right:
                            token_bytes_parts[into] = left + right
                            i += 2
                            into += 1
                        else:
                            token_bytes_parts[into] = token_bytes_parts[i]
                            i += 1
                            into += 1
                    count = into
                ids = ([self.reverse_vocab[a] for a in token_bytes_parts[:count]])
                self.cache[token] = ids
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
