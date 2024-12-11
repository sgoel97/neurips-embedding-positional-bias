import numpy as np
import pandas as pd


def prepare_dataset(name: str, dataset: pd.DataFrame) -> pd.DataFrame:
    if "scientific_papers" in name:
        return pd.DataFrame({"original": dataset["abstract"]})
    if "paul_graham" in name:
        return pd.DataFrame({"original": dataset["text"]})
    if "amazon_polarity" in name:
        return pd.DataFrame({"original": dataset["content"]})
    if "arxiv-clustering-p2p" in name:
        return pd.DataFrame({"original": np.concatenate(dataset["sentences"])})
    if "arguana" in name:
        return pd.DataFrame({"original": dataset["text"]})
    if "sts22" in name:
        return pd.DataFrame({"original": dataset[dataset["lang"] == "en"]["sentence1"]})
    if "reddit" in name:
        return pd.DataFrame({"original": np.concatenate(dataset["sentences"])})
    return dataset
