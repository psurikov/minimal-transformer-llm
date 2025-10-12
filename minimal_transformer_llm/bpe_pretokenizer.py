import regex
from typing import Iterator

normal_token_const = 0
special_token_const = 1

class BpePretokenizer:
    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens or []
        self._compiled_regex = self._build_regex()

    def _build_regex(self) -> regex.Pattern:
        pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        return regex.compile(pattern)

    def _split(self, text: str) -> list[tuple[int, int, str]]:
        spans = []
        i = 0
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        while i < len(text):
            span_start = i
            span_end = len(text)
            next_special_token = None
            for special_token in sorted_special_tokens:
                current = text.find(special_token, i)
                if current != -1 and current < span_end:
                    span_end = current
                    next_special_token = special_token
            if span_end > span_start:
                spans.append((span_start, span_end, normal_token_const))
            if next_special_token:                
                spans.append((span_end, span_end + len(next_special_token), special_token_const))
                span_end = span_end + len(next_special_token)
            i = span_end
        return spans

    def next_token(self, text: str) -> Iterator[str]:
        chunks = self._split(text)
        for start, end, kind in chunks:
            chunk_text = text[start:end]
            if kind == special_token_const:
                yield chunk_text
            else:
                for match in self._compiled_regex.finditer(chunk_text):
                    yield match.group(0)