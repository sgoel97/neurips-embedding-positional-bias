import hashlib
import random
import tiktoken
import numpy as np
from functools import lru_cache


tokenizer = tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=1024)  # Adjust size as needed
def cached_encode(text: str):
    return tokenizer.encode(text)


def word_count(text: str):
    """Get the number of words in a string"""
    return len(text.split(" "))


def hash_string(text: str):
    """Get the SHA256 hash of a string"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def num_tokens(text: str | list[str]) -> int | list[int]:
    """Get the number of tokens in a string"""
    if type(text) == str:
        return len(cached_encode(text))

    return [num_tokens(t) for t in text]


def truncate(
    text: str | list[str], max_length: int = 8192, short_ok: bool = True
) -> str | list[str]:
    """Truncate text to max_length tokens"""
    assert max_length > 0

    if type(text) == str:
        encoding = cached_encode(text)

        if not short_ok and len(encoding) < max_length:
            print(f"WARNING: text is too short to be truncated")

        if len(encoding) < max_length:
            return text

        truncated_encoding = encoding[:max_length]
        return tokenizer.decode(truncated_encoding)

    return [truncate(t, max_length) for t in text]


def sample_from_text(text: str | list[str], pct_size: float) -> str | list:
    if type(text) == str:
        upper = 1 - pct_size
        start = random.uniform(0, upper)
        start = int(num_tokens(text) * start)

        size = int(num_tokens(text) * pct_size)
        sample = tokenizer.decode(cached_encode(text)[start : start + size])
        return sample
    return [sample_from_text(t, pct_size) for t in text]
