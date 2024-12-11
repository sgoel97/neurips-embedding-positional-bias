import torch
import tiktoken
import numpy as np
import pandas as pd
from typing import Union
from rank_bm25 import BM25Plus
from rouge_score import rouge_scorer
from Levenshtein import ratio, distance

tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
tokenizer.tokenize = tokenizer.encode

rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2"], tokenizer=tokenizer)

metrics = ["cosine", "jaccard", "levenshtein", "bm25", "rouge"]


def cosine_similarity(x, y) -> np.ndarray | float:
    """
    Calculates cosine similarity between arrays or matrices
    """
    if type(x) == pd.Series:
        el = x.iloc[0]
    else:
        el = x[0]
    if type(el) == list or type(el) == np.ndarray or type(el) == torch.Tensor:
        x = torch.tensor(x.tolist())
        y = torch.tensor(y.tolist())
        return torch.cosine_similarity(x, y).numpy()

    else:
        x = torch.tensor(x).unsqueeze(0)
        y = torch.tensor(y).unsqueeze(0)
        return torch.cosine_similarity(x, y).item()


def levenshtein_distance(x, y) -> np.ndarray | float:
    """
    Calculates Levenshtein distance between two strings
    """
    if type(x) == str:
        return distance(x, y)
    if type(x) == pd.Series:
        x = x.values
        y = y.values
    return np.array([distance(x[i], y[i]) for i in range(len(x))])


def levenshtein_ratio(x, y) -> np.ndarray | float:
    """
    Calculates Levenshtein distance ratio between two strings
    """
    if type(x) == str:
        return ratio(x, y)
    if type(x) == pd.Series:
        x = x.values
        y = y.values
    return np.array([ratio(x[i], y[i]) for i in range(len(x))])


def rouge_score(x, y) -> np.ndarray | float:
    """
    Calculates ROUGE score between two strings
    """

    def tokenized_rouge_score(reference, target):
        rouge_scores = rouge.score(reference, target)
        r1_score = rouge_scores["rouge1"].fmeasure
        r2_score = rouge_scores["rouge2"].fmeasure
        return (r1_score + r2_score) / 2

    if type(x) == str:
        return tokenized_rouge_score(x, y)
    if type(x) == pd.Series:
        x = x.values
        y = y.values
    return np.array([tokenized_rouge_score(x[i], y[i]) for i in range(len(x))])


def jaccard_similarity(x, y) -> np.ndarray | float:
    """
    Calculates Jaccard similarity between two strings
    """

    def tokenized_jaccard_similarity(reference, target):
        reference_tokens = set(reference)
        target_tokens = set(target)
        intersection = reference_tokens.intersection(target_tokens)
        union = reference_tokens.union(target_tokens)
        return len(intersection) / len(union)

    if type(x) == str:
        return tokenized_jaccard_similarity(tokenizer.encode(x), tokenizer.encode(y))
    if type(x) == pd.Series:
        x = x.values
        y = y.values
    return np.array(
        [
            tokenized_jaccard_similarity(tokenizer.encode(x[i]), tokenizer.encode(y[i]))
            for i in range(len(x))
        ]
    )


def bm25_score(x, y) -> np.ndarray | float:
    """
    Calculates BM25 score between two strings
    """

    def tokenized_bm25_score(reference, target):
        bm25 = BM25Plus([reference])
        doc_scores = bm25.get_scores(target)
        return doc_scores[0]

    if type(x) == str:
        return tokenized_bm25_score(tokenizer.encode(x), tokenizer.encode(y))
    if type(x) == pd.Series:
        x = x.values
        y = y.values
    out = np.array(
        [
            tokenized_bm25_score(tokenizer.encode(x[i]), tokenizer.encode(y[i]))
            for i in range(len(x))
        ]
    )
    return (out - np.min(out)) / (np.max(out) - np.min(out))
