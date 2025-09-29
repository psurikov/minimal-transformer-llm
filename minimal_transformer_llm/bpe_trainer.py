from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import BinaryIO
import os
import regex

class BpeTrainer:
    def __init__(self):
        self.pretoken_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.min_chunk_size_bytes = 1024 #65536
        self.max_special_token_size_bytes = 50
        
    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        process_count = cpu_count()
        chunks = []
        regex_pattern = "|".join([regex.escape(special_token) for special_token in special_tokens]) + "|" + self.pretoken_regex
        with open(input_path, "rb") as file:
            chunks = self.split(file, process_count, special_tokens)
            vocab, merges = self.parallelize(file, chunks, regex_pattern, special_tokens)
            return vocab, merges

    def parallelize(self, file: BinaryIO, chunks: list[tuple[int, int]], regex_pattern: str, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        pretokens = Counter()
        for chunk in chunks:
            pretokens += self.pretokenize(file, chunk, regex_pattern)
        merges = []
        vocab = []
        return vocab, merges
        #with Pool(process_count) as pool:
        #    pretokens = pool.map(self.train_chunk, chunks)

    def pretokenize(self, file: BinaryIO, chunk: tuple[int, int], regex_pattern: str) -> Counter[str]:
        file.seek(chunk[0])
        chunk_bytes = file.read(chunk[1] - chunk[0])
        text = chunk_bytes.decode("utf-8", errors="ignore")
        regex_instance = regex.compile(regex_pattern)
        counter = Counter()
        for match in regex_instance.finditer(text):
            token = match.group(0)
            counter[token] += 1
        return counter

    # 1
    # splits the file into a number chunks 
    # returns a list of chunks with starting and ending positions
    def split(self, file: BinaryIO, count: int, special_tokens: list[bytes]) -> list[tuple[int, int]]:
        chunks = []
        # total file size
        file.seek(0, os.SEEK_END)
        file_size_bytes = file.tell()
        # try to split the whole size by count, if the chunk size is lower than a buffer size, use buffer size
        chunk_size = min(max(file_size_bytes // count, self.min_chunk_size_bytes), file_size_bytes)
        chunk_start = 0
        chunk_end = chunk_start + chunk_size
        encoded_special_tokens = [token.encode('utf-8') for token in special_tokens]
        # for each chunk, find out if there is a special token division
        while chunk_start < file_size_bytes:
            file.seek(chunk_end)
            read_ahead = file.read(self.max_special_token_size_bytes * 2)
            if read_ahead != b'':
                for encoded_special_token in encoded_special_tokens:
                    # if token found - append it, else - append the safe part where there is no special token
                    special_token_position = read_ahead.find(encoded_special_token)
                    if special_token_position != -1:
                        chunk_end += special_token_position
                        break
                else:
                    chunk_end += min(self.max_special_token_size_bytes, len(read_ahead))
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
            chunk_end = min(chunk_start + chunk_size, file_size_bytes)
        return chunks
    
    # splits the file into a number of chunks of desired size
    # returns a list of chunks
    def split(self, file: BinaryIO, start: int, end: int, desired_size: int, special_tokens: list[bytes]) -> list[tuple[int, int]]:
        chunks = []
        total_size = end - start
        chunk_size = min(desired_size, total_size)
        chunk_start = start
        chunk_end = chunk_start + desired_size
        while chunk_start < end:
            file.seek(chunk_end)
            read_count = min(self.max_special_token_size_bytes * 2, end - chunk_end)
            read_ahead = file.read(read_count)
            if read_ahead != b'':
                for special_token in special_tokens:
                    special_token_position = read_ahead.find(special_token)
                    if special_token_position != -1:
                        chunk_end += special_token_position
                        break
                else:
                    chunk_end += min(self.max_special_token_size_bytes, len(read_ahead))
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
            chunk_end = min(chunk_start + chunk_size, total_size)
        return chunks

    def merge(self):
        pass
