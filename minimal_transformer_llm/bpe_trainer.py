from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import BinaryIO
import os
import regex

class BpeTrainer:
    def __init__(self):
        self.pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.chunk_min_size = 1024 #65536
        self.buffer_size = 256
        self.whitespace_search_size = 1024
        self.special_token_search_size = 50
        
    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        chunks = []
        process_count = cpu_count()
        regex_pattern = "|".join([regex.escape(special_token) for special_token in special_tokens]) + "|" + self.pretoken_regex
        encoded_special_tokens = [token.encode('utf-8') for token in special_tokens]
        iterations = vocab_size - 256 - len(special_tokens)
        with open(input_path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file_size_bytes = file.tell()
            desired_chunk_size = min(max(file_size_bytes // process_count, self.chunk_min_size), file_size_bytes)
            chunks = self._splitByWhitespace(file, 0, file_size_bytes, desired_chunk_size, encoded_special_tokens)
            vocab, merges = self._train_chunks(file, chunks, regex_pattern, special_tokens, iterations)
            return vocab, merges

    def _train_chunks(self, file: BinaryIO, chunks: list[tuple[int, int]], regex_pattern: str, special_tokens: list[str], iterations: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        # extend with:
        #with Pool(process_count) as pool:
        #    pretokens = pool.map(self.train_chunk, chunks)

        pretokens = Counter()
        for chunk in chunks:
            pretokens += self._pretokenize(file, chunk, regex_pattern, special_tokens)
        for special_token in special_tokens:
            if special_token in pretokens:
                del pretokens[special_token]

        # count all pairs
        pretoken_ids_list = [None] * len(pretokens)
        pretoken_ids_index = 0
        pair_counts = {}
        for pretoken, count in pretokens.items():
            pretoken_ids = list(pretoken.encode("utf-8"))
            pretoken_ids_list[pretoken_ids_index] = pretoken_ids
            pretoken_ids_index += 1
            for pair in zip(pretoken_ids, pretoken_ids[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        
        # generate vocab and merges
        merges = []
        vocab = {i: bytes([i]) for i in range(256)}
        replace_id = 256
        for _ in range(iterations):
            max_pair = self._maxPair(pair_counts)
            del pair_counts[max_pair]
            for i in range(len(pretoken_ids_list)):
                pretoken_ids_list[i] = self._merge(pretoken_ids_list[i], max_pair, replace_id, pair_counts)
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
            vocab[replace_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            replace_id += 1

        # appending special tokens
        for special_token in special_tokens:
            vocab[replace_id] = special_token.encode('utf-8')

        return vocab, merges

    def _maxPair(self, pair_counts: dict[tuple[int, int], int]) -> tuple[int, int]:
        # lexicographically largest pair
        max_pair = (-1, -1)
        max_count = 0
        for pair, count in pair_counts.items():
            if count > max_count:
                max_pair = pair
                max_count = count
            elif count == max_count:
                if pair[0] > max_pair[0] or pair[0] == max_pair[0] and pair[1] > max_pair[1]:
                    max_pair = pair
                    max_count = count
        return max_pair

    def _merge(self, ids: list[int], pair: tuple[int, int], replace_id: int, pair_counts: dict[tuple[int, int], int]):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(replace_id)
                if i + 2 < len(ids):
                    right = (ids[i + 1], ids[i + 2])
                    if right in pair_counts:
                        pair_counts[right] -= 1
                if i > 0:
                    left = (ids[i - 1], ids[i])
                    if left in pair_counts:
                        pair_counts[left] -= 1
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def _pretokenize(self, file: BinaryIO, chunk: tuple[int, int], regex_pattern: str, special_tokens: list[bytes]) -> Counter[str]:
        regex_instance = regex.compile(regex_pattern)
        read_size = self.buffer_size
        read_parts = self._splitByWhitespace(file, chunk[0], chunk[1], read_size, special_tokens)
        file.seek(chunk[0])
        counter = Counter()
        for read_part in read_parts:
            read_bytes = file.read(read_part[1] - read_part[0])
            text = read_bytes.decode("utf-8", errors="ignore")
            for match in regex_instance.finditer(text):
                token = match.group(0)
                counter[token] += 1
        return counter

    def _splitByWhitespace(self, file: BinaryIO, start: int, end: int, desired_size: int, special_tokens: list[bytes]) -> list[tuple[int, int]]:
        chunks = []
        file.seek(start)
        total_size = end - start
        chunk_size = min(desired_size, total_size)
        chunk_start = start
        chunk_end = chunk_start + desired_size
        while chunk_start < end:
            file.seek(chunk_end)
            read_count = min(self.whitespace_search_size, end - chunk_end)
            read_ahead = file.read(read_count)
            if read_ahead != b'':
                for i in range(1, len(read_ahead)):
                    previous = chr(read_ahead[i - 1])
                    current = chr(read_ahead[i])
                    if not previous.isspace() and current.isspace():
                        chunk_end += i + 1
                        break
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
            chunk_end = min(chunk_start + chunk_size, end)
        return chunks

    # splits the file into a number of chunks of desired size
    # returns a list of chunks
    def _splitBySpecialToken(self, file: BinaryIO, start: int, end: int, desired_size: int, special_tokens: list[bytes]) -> list[tuple[int, int]]:
        chunks = []
        file.seek(start)
        total_size = end - start
        chunk_size = min(desired_size, total_size)
        chunk_start = start
        chunk_end = chunk_start + desired_size
        while chunk_start < end:
            file.seek(chunk_end)
            read_count = min(self.special_token_search_size * 2, end - chunk_end)
            read_ahead = file.read(read_count)
            if read_ahead != b'':
                for special_token in special_tokens:
                    special_token_position = read_ahead.find(special_token)
                    if special_token_position != -1:
                        chunk_end += special_token_position
                        break
                else:
                    chunk_end += min(self.special_token_search_size, len(read_ahead))
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
            chunk_end = min(chunk_start + chunk_size, total_size)
        return chunks
