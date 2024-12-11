import pandas as pd
import numpy as np
from utils.string_utils import *

np.random.seed(42)


def circular_rotate(series: list, n: int) -> list:
    n = n % len(series)  # Handle cases where n is larger than the length of the series
    return series[n:] + series[:n]


def get_positives(df: pd.DataFrame, pct_size: float) -> pd.DataFrame:
    sample1 = sample_from_text(df["original"].tolist(), pct_size)
    sample2 = sample_from_text(df["original"].tolist(), pct_size)
    pos = [1] * len(sample1)

    return pd.DataFrame(
        {
            "sample1": sample1,
            "sample2": sample2,
            "pos": pos,
        }
    )


def get_negatives(df: pd.DataFrame, pct_size: float) -> pd.DataFrame:
    sample1 = sample_from_text(df["original"].tolist(), pct_size)
    sample2 = circular_rotate(sample_from_text(df["original"].tolist(), pct_size), 1)
    pos = [0] * len(sample1)
    return pd.DataFrame(
        {
            "sample1": sample1,
            "sample2": sample2,
            "pos": pos,
        }
    )


def get_truncated_df(df: pd.DataFrame, pct_size: float) -> pd.DataFrame:
    assert "original" in df.columns
    assert 0 < pct_size < 1

    positives_df = get_positives(df, pct_size)
    negatives_df = get_negatives(df, pct_size)
    return pd.concat((positives_df, negatives_df))
