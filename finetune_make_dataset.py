import pandas as pd
from pathlib import Path
import numpy as np

from _datasets.config import DatasetConfig
from utils.finetune_make_dataset_utils import get_truncated_df

data_path = "project/data/truncate_data_combined.pkl"

datasets = [
    "paul_graham",
    "scientific_papers",
    "amazon_polarity",
    "arguana",
    "reddit",
]

np.random.seed(42)

Path(data_path).parent.mkdir(parents=True, exist_ok=True)

print("Loading dataframes...")
datasets = [DatasetConfig(name=dataset) for dataset in datasets]
dataframes = [dataset.get_dataset(truncate=True, max_length=8192) for dataset in datasets]

print("Truncating dataframes...")
dataframes = [get_truncated_df(df, 0.5) for df in dataframes]

print("Concatenating dataframes...")
data = pd.concat(dataframes)
data.to_pickle(data_path)
print(f"Saved {len(data)} datapoints to {data_path}")
