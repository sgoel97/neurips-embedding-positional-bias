from typing import Optional
from datasets import load_dataset
from dataclasses import dataclass
import pandas as pd

from utils.string_utils import truncate as truncate_fn
from _datasets.utils import prepare_dataset
from _datasets.datasets import hf_datasets


@dataclass
class DatasetConfig:
    name: str
    num_examples: Optional[int] = None
    split: Optional[str] = None

    def get_dataset(self, truncate: bool = False, max_length: int = 0) -> pd.DataFrame:
        if self.name in hf_datasets:
            kwargs = hf_datasets[self.name].copy()
            split = self.split or kwargs.pop("split", None) or "train"

            ds = load_dataset(**kwargs, split=split).to_pandas()
            ds = prepare_dataset(self.name, ds)
        else:
            raise ValueError(f"Unrecognized Dataset {self.name}")

        if self.num_examples:
            ds = ds[: self.num_examples]
        if truncate:
            assert max_length != 0, f"Specify the max length {max_length}"
            ds["original"] = ds["original"].apply(lambda x: truncate_fn(x, max_length))
        return ds
