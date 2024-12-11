from utils.string_utils import truncate, num_tokens, tokenizer


def add_needle_single(text: str, corpus: str, size: float, posn: float) -> str:
    assert corpus is not None, "Corpora is required for needle addition."
    if not 0 <= posn <= 1:
        raise ValueError("Percent location must be between 0 and 1 (inclusive).")

    n_example_tokens = num_tokens(text)
    assert isinstance(n_example_tokens, int)
    needle_token_length = max(int(size * n_example_tokens), 1)
    needle = truncate(corpus, needle_token_length)
    assert isinstance(needle, str)
    loc = int(posn * len(text))
    return text[:loc] + needle + text[loc:]


def add_removal_single(text: str, corpus: str, size: float, posn: float) -> str:
    del corpus

    assert size <= 1, "Needle size must be less than or equal to 1."
    if posn == 0 and size == 1:
        raise ValueError("Cannot remove the entire text.")
    if not 0 <= posn <= 1:
        raise ValueError("Percent location must be between 0 and 1 (inclusive).")

    encoding = tokenizer.encode(text)
    adjusted_posn = posn - size * posn
    loc = int(num_tokens(text) * adjusted_posn)
    length = int(num_tokens(text) * size)
    return tokenizer.decode(encoding[:loc] + encoding[loc + length :])
