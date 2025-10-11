from typing import Iterable, Iterator

class BpeTokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        byte_parts = [bytes([byte_part]) for byte_part in text_bytes]
        for merge in self.merges:
            bytes1, bytes2 = merge
            mergedparts = []
            i = 0
            while i < len(byte_parts):
                if byte_parts[i] == bytes1 and i < len(byte_parts) - 1 and byte_parts[i + 1] == bytes2:
                    mergedparts.append(bytes1 + bytes2)
                    i += 2
                else:
                    mergedparts.append(byte_parts[i])
                    i += 1
            byte_parts = mergedparts
        return [self.reverse_vocab[byte_part] for byte_part in byte_parts]
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterator[list[int]]:
        for text in texts:
            yield self.encode(text)

    def decode(self, ids: list[int]) -> str:
        merged_bytes = b"".join([self.vocab[id] for id in ids])
        decoded_str = merged_bytes.decode("utf-8", errors="replace")
        return decoded_str
