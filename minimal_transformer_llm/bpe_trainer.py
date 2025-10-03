from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import BinaryIO
import os
import regex

class BpeTrainer:
    pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunk_min_size = 1024
    buffer_size = 256
    whitespace_search_size = 1024
    special_token_search_size = 1024

    def __init__(self):
        self.vocab = []
        self.merges = []
        
    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        chunks = []
        process_count = cpu_count()
        regex_pattern = "|".join([regex.escape(special_token) for special_token in special_tokens]) + "|" + BpeTrainer.pretoken_regex
        encoded_special_tokens = [token.encode('utf-8') for token in special_tokens]
        iterations = vocab_size - 256 - len(special_tokens)
        with open(input_path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file_size_bytes = file.tell()
            desired_chunk_size = min(max(file_size_bytes // process_count, BpeTrainer.chunk_min_size), file_size_bytes)
            chunks = self._split(file, 0, file_size_bytes, desired_chunk_size, encoded_special_tokens)
            vocab, merges = self._train_chunks(file, chunks, regex_pattern, encoded_special_tokens, iterations)
            self.vocab = vocab
            self.merges = merges

    def _train_chunks(self, file: BinaryIO, chunks: list[tuple[int, int]], regex_pattern: str, special_tokens: list[bytes], iterations: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        # extend with:
        #with Pool(process_count) as pool:
        #    pretokens = pool.map(self.train_chunk, chunks)

        pretokens = Counter()
        for chunk in chunks:
            pretokens += self._pretokenize(file, chunk, regex_pattern, special_tokens)
        for special_token in special_tokens:
            special_token_str = special_token.decode('utf-8')
            if special_token_str in pretokens:
                del pretokens[special_token_str]

        # count pairs
        pretoken_counts = [None] * len(pretokens)
        pretoken_ids_list = [None] * len(pretokens)
        pretoken_ids_index = 0
        for pretoken, count in pretokens.items():
            pretoken_counts[pretoken_ids_index] = count
            pretoken_ids = list(pretoken.encode("utf-8"))
            pretoken_ids_list[pretoken_ids_index] = pretoken_ids
            pretoken_ids_index += 1
        
        # generate vocab and merges
        merges = []
        vocab = {i: bytes([i]) for i in range(256)}
        replace_id = 256
        pair_counts = self._countPairs(pretoken_ids_list, pretoken_counts)
        for _ in range(iterations):
            # pair_counts = self._countPairs(pretoken_ids_list, pretoken_counts)
            max_pair = self._maxPair(pair_counts, vocab)
            del pair_counts[max_pair]
            for i in range(len(pretoken_ids_list)):
                pretoken_ids_list[i] = self._merge(pretoken_ids_list[i], pretoken_counts[i], max_pair, replace_id, pair_counts)
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
            vocab[replace_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            replace_id += 1

        # appending special tokens
        for special_token in special_tokens:
            vocab[replace_id] = special_token
            replace_id += 1

        return vocab, merges
    
    # calculates the frequencies of pairs
    def _countPairs(self, pretoken_ids_list: list[list[int]], pretoken_counts: list[int]) -> dict[tuple[int, int], int]:
        pair_counts = {}
        for i in range(len(pretoken_ids_list)):
            pretoken_ids = pretoken_ids_list[i]
            pretoken_count = pretoken_counts[i]
            for pair in zip(pretoken_ids, pretoken_ids[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + pretoken_count
        return pair_counts

    # finds the most frequent pair, in case of a tie returns lexicographically largest pair
    def _maxPair(self, pair_counts: dict[tuple[int, int], int], vocab: dict[int, bytes]) -> tuple[int, int]:
        max_pair = (-1, -1)
        max_count = 0
        for pair, count in pair_counts.items():
            if count > max_count:
                max_pair = pair
                max_count = count
            elif count == max_count:
                pair_a = vocab[pair[0]]
                pair_b = vocab[pair[1]]
                max_pair_a = vocab[max_pair[0]]
                max_pair_b = vocab[max_pair[1]]
                if pair_a > max_pair_a or (pair_a == max_pair_a and pair_b > max_pair_b):
                    max_pair = pair
        return max_pair

    def _merge(self, ids: list[int], ids_count: int, pair: tuple[int, int], replace_id: int, pair_counts: dict[tuple[int, int], int]):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(replace_id)
                if i + 2 < len(ids):
                    # take the new pair that is formed to the right, and add it to the counts
                    right_replaced = (replace_id, ids[i + 2])
                    pair_counts[right_replaced] = pair_counts.get(right_replaced, 0) + ids_count
                    # the old pair can now be sutracted from the counts
                    right_deleted = (ids[i + 1], ids[i + 2])
                    if right_deleted in pair_counts:
                        pair_counts[right_deleted] -= ids_count
                        if pair_counts[right_deleted] == 0:
                            del pair_counts[right_deleted]
                if i > 0:
                    # take the new pair that is formed to the left, and add it to the counts
                    left_replaced = (ids[i - 1], replace_id)
                    pair_counts[left_replaced] = pair_counts.get(left_replaced, 0) + ids_count
                    # the old pair toe the left can be subtracted from the counts
                    left_deleted = (ids[i - 1], ids[i])
                    if left_deleted in pair_counts:
                        pair_counts[left_deleted] -= ids_count
                        if pair_counts[left_deleted] == 0:
                            del pair_counts[left_deleted]
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def _pretokenize(self, file: BinaryIO, chunk: tuple[int, int], regex_pattern: str, special_tokens: list[bytes]) -> Counter[str]:
        regex_instance = regex.compile(regex_pattern)
        read_size = BpeTrainer.buffer_size
        read_parts = self._split(file, chunk[0], chunk[1], read_size, special_tokens)
        file.seek(chunk[0])
        counter = Counter()
        for read_part in read_parts:
            read_bytes = file.read(read_part[1] - read_part[0])
            text = read_bytes.decode("utf-8", errors="replace")
            for match in regex_instance.finditer(text):
                token = match.group(0)
                counter[token] += 1
        return counter

    # splits the file into a number of chunks of desired size, adjusting the edges for special_tokens
    def _split(self, file: BinaryIO, start: int, end: int, desired_size: int, special_tokens: list[bytes]) -> list[tuple[int, int]]:
        chunks = []
        file.seek(start)
        total_size = end - start
        chunk_size = min(desired_size, total_size)
        chunk_start = start
        chunk_end = chunk_start + desired_size
        while chunk_start < end:
            chunk_end = self._adjust_chunk_end(file, chunk_end, end, special_tokens)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
            chunk_end = min(chunk_start + chunk_size, end)
        return chunks
    
    def _adjust_chunk_end(self, file: BinaryIO, chunk_end: int, end: int, special_tokens: list[bytes]) -> int:
        if chunk_end == end:
            return chunk_end
        file.seek(chunk_end)
        while chunk_end < end:
            read_count = max(min(BpeTrainer.special_token_search_size, end - chunk_end), 0)
            read_ahead = file.read(read_count)
            found_token = False
            found_at = 0
            for special_token in special_tokens:
                found_at = read_ahead.find(special_token)
                if found_at != -1:
                    found_token = True
                    break
            if found_token:
                chunk_end += found_at
                break
            else:
                chunk_end += read_count
        return chunk_end
    
    def _adjust_chunk_end1(self, file: BinaryIO, chunk_end: int, end: int, special_tokens: list[bytes]) -> int:
        if chunk_end == end:
            return chunk_end
        file.seek(chunk_end)
        while chunk_end < end:
            read_count = max(min(BpeTrainer.special_token_search_size, end - chunk_end), 0)
            read_ahead = file.read(read_count)
            found_token = False
            found_at = 0
            for special_token in special_tokens:
                found_at = read_ahead.find(special_token)
                if found_at != -1:
                    found_token = True
                    break
                found_at = read_ahead.find(b'\n')
                if found_at != -1:
                    found_token = True
                    break
            if found_token:
                chunk_end += found_at
                break
            else:
                chunk_end += read_count
        return chunk_end

    def _split2(self, file: BinaryIO, start: int, end: int, desired_size: int, special_tokens: list[bytes]) -> list[tuple[int, int]]:
        chunks = []
        file.seek(start)
        total_size = end - start
        chunk_size = min(desired_size, total_size)
        chunk_start = start
        chunk_end = chunk_start + desired_size
        while chunk_start < end:
            file.seek(chunk_end)
            read_count = max(min(BpeTrainer.whitespace_search_size, end - chunk_end), 0)
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