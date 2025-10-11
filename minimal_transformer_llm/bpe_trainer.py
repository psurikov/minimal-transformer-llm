from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import BinaryIO
import os
import time
import regex
import mmap
import logging

class BpeTrainer:
    pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunk_min_size = 1024
    buffer_size = 256
    whitespace_search_size = 1024
    special_token_search_size = 1024

    def __init__(self):
        self.vocab = []
        self.merges = []
        self._setup_logging()
        
    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # pretokenize
        self._log_message("Started training for %s with vocab size %d and %d special tokens", input_path, vocab_size, len(special_tokens))
        self._log_message("Started collecting pretokens")
        pretokens = self._pretokenize(input_path, special_tokens)
        self._log_message("Finished collecting pretokens, collected %d pretokens", len(pretokens))

        # count pairs
        self._log_message("Started counting pairs")
        pretoken_counts = [None] * len(pretokens)
        pretoken_ids_list = [None] * len(pretokens)
        pretoken_ids_index = 0
        for pretoken, count in pretokens.items():
            pretoken_counts[pretoken_ids_index] = count
            pretoken_ids = list(pretoken.encode("utf-8"))
            pretoken_ids_list[pretoken_ids_index] = pretoken_ids
            pretoken_ids_index += 1
        self._log_message("Finished counting pairs")
        
        # generate vocab and merges
        self._log_message("Started merging pairs")
        merges = []
        vocab = {i: bytes([i]) for i in range(256)}
        replace_id = 256
        count = len(pretoken_ids_list)
        pair_counts = self._count_pairs(pretoken_ids_list, pretoken_counts, count)
        start_time = time.time()
        iterations = vocab_size - 256 - len(special_tokens)
        self._log_message("Iterations count %d", iterations)
        for iteration in range(iterations):
            if not pair_counts:
                break
            self._log_message("Iteration %d", iteration)
            # find the max pair and delete, since we no longer need it
            max_pair = self._max_pair(pair_counts, vocab)
            del pair_counts[max_pair]
            # merge the max pair
            i = 0
            while i < count:
                write = self._merge(pretoken_ids_list[i], pretoken_counts[i], max_pair, replace_id, pair_counts)
                if write <= 1:
                    count -= 1
                    pretoken_ids_list[i] = pretoken_ids_list[count]
                    pretoken_counts[i] = pretoken_counts[count]
                else:
                    i += 1
            # append the pair to the list of merges
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
            vocab[replace_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
            replace_id += 1

        end_time = time.time()
        span_time = end_time - start_time
        print(f"iterations: {span_time} seconds")

        # appending special tokens
        encoded_special_tokens = [token.encode('utf-8') for token in special_tokens]
        for encoded_special_token in encoded_special_tokens:
            vocab[replace_id] = encoded_special_token
            replace_id += 1

        self._log_message("Finished merging pairs")
        self._log_message("Finished training for %s", input_path)
        self.vocab = vocab
        self.merges = merges
        return vocab, merges
    
    # splits the text into pretokens
    def _pretokenize(self, input_path: str, special_tokens: list[str]) -> Counter[str]:
        chunks = []
        process_count = cpu_count()
        encoded_special_tokens = [token.encode('utf-8') for token in special_tokens]
        with open(input_path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file_size_bytes = file.tell()
            desired_chunk_size = min(max(file_size_bytes // process_count, BpeTrainer.chunk_min_size), file_size_bytes)
            mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            chunks = self._split(mm, 0, file_size_bytes, desired_chunk_size, encoded_special_tokens)
            mm.close()
            pretokens = BpeTrainer._pretokenize_parallel(input_path, chunks, special_tokens)
            for special_token in special_tokens:
                if special_token in pretokens:
                    del pretokens[special_token]
        return pretokens

    # splits the chunks into pretokens, and returns their statistics
    @staticmethod
    def _pretokenize_parallel(file_path, chunks, special_tokens):
        regex_pattern = "|".join([regex.escape(special_token) for special_token in special_tokens]) + "|" + BpeTrainer.pretoken_regex
        encoded_special_tokens = [token.encode('utf-8') for token in special_tokens]
        args_list = [(start, end, regex_pattern, encoded_special_tokens, file_path) for start, end in chunks]
        with Pool(len(chunks)) as pool:
            results = pool.map(BpeTrainer._pretokenize_worker, args_list)
        pretokens = Counter()
        for c in results:
            pretokens.update(c)
        return pretokens

    @staticmethod
    def _pretokenize_worker(args) -> Counter[str]:
        start, end, regex_pattern, special_tokens, input_path = args
        counter = Counter()
        regex_instance = regex.compile(regex_pattern)
        with open(input_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            read_parts = BpeTrainer._split(mm, start, end, BpeTrainer.buffer_size, special_tokens)
            for read_start, read_end in read_parts:
                read_bytes = mm[read_start:read_end]
                text = read_bytes.decode("utf-8", errors="replace")
                for match in regex_instance.finditer(text):
                    token = match.group(0)
                    counter[token] += 1
            mm.close()
        return counter

    # calculates the frequencies of pairs
    @staticmethod
    def _count_pairs(pretoken_ids_list: list[list[int]], pretoken_counts: list[int], count: int) -> dict[tuple[int, int], int]:
        pair_counts = {}
        for i in range(count):
            pretoken_ids = pretoken_ids_list[i]
            pretoken_count = pretoken_counts[i]
            for pair in zip(pretoken_ids, pretoken_ids[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + pretoken_count
        return pair_counts

    # finds the most frequent pair, in case of a tie returns lexicographically largest pair
    @staticmethod
    def _max_pair(pair_counts: dict[tuple[int, int], int], vocab: dict[int, bytes]) -> tuple[int, int]:
        max_pair = (-1, -1)
        max_count = 0
        items = pair_counts.items()
        for pair, count in items:
            if count > max_count:
                max_pair = pair
                max_count = count
            elif count == max_count:
                a0 = vocab[pair[0]]
                b0 = vocab[pair[1]]
                am = vocab[max_pair[0]]
                bm = vocab[max_pair[1]]
                if a0 > am or (a0 == am and b0 > bm):
                    max_pair = pair
        return max_pair

    @staticmethod
    def _merge(ids: list[int],  ids_count: int, pair: tuple[int, int], replace_id: int, pair_counts: dict[tuple[int, int], int]) -> int:
        sentinel = -1
        n = len(ids)
        if n <= 1:
            return 0
        write = 0
        i = 0
        a, b = pair
        while i < n and ids[i] != sentinel:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                ids[write] = replace_id
                # right neighbor
                if i + 2 < n and ids[i + 2] != sentinel:
                    right_new = (replace_id, ids[i + 2])
                    pair_counts[right_new] = pair_counts.get(right_new, 0) + ids_count
                    right_old = (b, ids[i + 2])
                    count = pair_counts.get(right_old)
                    if count is not None:
                        if count == ids_count:
                            del pair_counts[right_old]
                        else:
                            pair_counts[right_old] = count - ids_count
                # left neighbor
                if write > 0:
                    left_new = (ids[write - 1], replace_id)
                    pair_counts[left_new] = pair_counts.get(left_new, 0) + ids_count
                    left_old = (ids[write - 1], a)
                    count = pair_counts.get(left_old)
                    if count is not None:
                        if count == ids_count:
                            del pair_counts[left_old]
                        else:
                            pair_counts[left_old] = count - ids_count
                write += 1
                i += 2
            else:
                ids[write] = ids[i]
                write += 1
                i += 1
        # Mark the end with sentinel
        if write < n:
            ids[write] = sentinel
        return write

    @staticmethod
    def _split(mm: mmap.mmap, start: int, end: int, desired_size: int, special_tokens: list[bytes]) -> list[tuple[int, int]]:
        chunks = []
        total_size = end - start
        chunk_size = min(desired_size, total_size)
        chunk_start = start
        chunk_end = chunk_start + desired_size
        while chunk_start < end:
            chunk_end = BpeTrainer._adjust_chunk_end(mm, chunk_end, end, special_tokens)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
            chunk_end = min(chunk_start + chunk_size, end)
        return chunks

    @staticmethod
    def _adjust_chunk_end(mm: mmap.mmap, chunk_end: int, end: int, special_tokens: list[bytes]) -> int:
        if chunk_end >= end:
            return end
        while chunk_end < end:
            read_count = min(BpeTrainer.special_token_search_size, end - chunk_end)
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
    
    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.logger.hasHandlers():
            return
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _log_message(self, msg: str, *args):
        self.logger.info(msg, *args)