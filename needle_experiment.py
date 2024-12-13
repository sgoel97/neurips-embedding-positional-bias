import torch
import inspect
from typing import Literal
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd

from needle_config import NeedleConfig
from _models.model import get_embedding_func_batched
from _datasets.config import DatasetConfig
from utils.transform_utils import *
from utils.string_utils import *
from utils.needle_utils import *


class NeedleExperiment:
    def __init__(
        self,
        mode: Literal["insert", "remove"],
        dataset_name: str,
        needle_configs: list[NeedleConfig],
        num_examples: int | None = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        max_length: int = 8192,  # TODO: automate getting this number based on the model name
    ):
        assert mode in ["insert", "remove"], "Mode must be 'insert' or 'remove'."

        self.mode = mode
        self.model_name = model_name
        self.needle_configs = needle_configs
        assert len(set([config.mode for config in self.needle_configs])) == 1

        self.dataset_config = DatasetConfig(dataset_name, num_examples)
        self.df = self.dataset_config.get_dataset(True, max_length)
        assert "original" in self.df.columns
        original_text = self.df["original"].tolist()
        sentences = pd.Series(get_sentences(original_text))
        self.df["sentences"] = sentences
        self.df["sentence_positions"] = sentences.apply(get_sentence_positions)
        self.df["sentence_proportions"] = sentences.apply(get_sentence_proportions)
        self.df["num_sentences"] = sentences.apply(len)
        print(f"Dataset {dataset_name} loaded in mode '{mode}'.")

        # Truncate dataset examples to max_text_size
        if mode == "insert":
            max_needle_size = max(self.needle_configs, key=lambda x: x.size).size
            max_text_size = int(max_length // (1 + max_needle_size))
            self.df["original"] = truncate(self.df["original"].tolist(), max_text_size)
            print(f"Filtered dataset to {len(self.df)} examples.")

        self.embedding_func = get_embedding_func_batched(model_name)

        # Create directory for model data if it doesn't exist, replacing '/' with '_' to avoid subdirectories
        self.model_data_path = Path("data") / self.model_name.replace("/", "_")
        self.model_data_path.mkdir(parents=True, exist_ok=True)

    def run(self, use_gpu: bool = False):
        generation_kwargs = {"model_name": self.model_name, "use_gpu": use_gpu}

        print("Generating text with needles...")
        self.generate_text_with_needles()

        print("Generating embeddings...")
        self.generate_text_embeddings(embedding_func=self.embedding_func, **generation_kwargs)

        print("Generating sentence embeddings...")
        self.generate_sentence_embeddings(embedding_func=self.embedding_func, **generation_kwargs)

        print("Calculating similarities...")
        self.calculate_similarities()

        # Save the similarity data to a CSV file in the model-specific directory
        file_path = f"{self.model_data_path}/{self.dataset_config.name}_{self.mode}.pkl"
        self.df.to_pickle(file_path)
        print(f"Saved data to {file_path}.")

    def generate_text_with_needles(self):
        for config in self.needle_configs:
            ablation_method = add_needle_single if config.mode == "insert" else add_removal_single
            original_text = self.df["original"].to_list()
            needle_text = [
                ablation_method(text=t, corpus=config.corpus, size=config.size, posn=config.posn)
                for t in original_text
            ]
            self.df[f"text_{config.name}"] = needle_text

    def generate_text_embeddings(self, embedding_func, **kwargs):
        # For models that are not from huggingface
        source_code = inspect.getsource(embedding_func)
        if not "huggingface" in source_code:
            kwargs["model"] = kwargs["model_name"]
            del kwargs["model_name"]
            del kwargs["use_gpu"]

        original_text = self.df["original"].dropna().tolist()
        self.df["embeddings_original"] = embedding_func(prompts=original_text, **kwargs)

        shuffled_text = shuffle_text(original_text, spacing=False)
        self.df["original_shuffled"] = shuffled_text
        self.df["embeddings_shuffled"] = embedding_func(shuffled_text, **kwargs)

        sentences = pd.Series(get_sentences(shuffled_text))
        self.df["sentences_shuffled"] = sentences
        self.df["sentence_positions_shuffled"] = sentences.apply(get_sentence_positions)
        self.df["sentence_proportions_shuffled"] = sentences.apply(get_sentence_proportions)

        # Generate embeddings for each modified text column.
        for config in tqdm(self.needle_configs, desc="embedding needle text"):
            needle_text = self.df[f"text_{config.name}"].dropna().tolist()
            self.df[f"embeddings_{config.name}"] = embedding_func(prompts=needle_text, **kwargs)

    def generate_sentence_embeddings(self, embedding_func, **kwargs):
        # For models that are not from huggingface
        source_code = inspect.getsource(embedding_func)
        if not "huggingface" in source_code:
            kwargs["model"] = kwargs["model_name"]
            del kwargs["model_name"]
            del kwargs["use_gpu"]

        sentence_embeddings = []
        for i in tqdm(range(len(self.df)), desc="embedding sentences"):
            sentences = self.df["sentences"][i]
            sentence_embedding = embedding_func(prompts=sentences, **kwargs)
            sentence_embeddings.append(sentence_embedding)
        self.df["sentence_embeddings"] = sentence_embeddings

        sentence_embeddings_shuffled = []
        for i in tqdm(range(len(self.df)), desc="embedding shuffled sentences"):
            sentences_shuffled = self.df["sentences_shuffled"][i]
            sentence_embedding_shuffled = embedding_func(prompts=sentences_shuffled, **kwargs)
            sentence_embeddings_shuffled.append(sentence_embedding_shuffled)
        self.df["sentence_embeddings_shuffled"] = sentence_embeddings_shuffled

    def calculate_similarities(self):
        original_text_embeddings = torch.tensor(self.df["embeddings_original"].tolist()).to("cpu")

        for config in self.needle_configs:
            embeddings_column = f"embeddings_{config.name}"
            needle_text_embeddings = torch.tensor(self.df[embeddings_column].tolist()).to("cpu")
            self.df[f"cosine_similarity_{config.name}"] = torch.cosine_similarity(
                original_text_embeddings, needle_text_embeddings, dim=1
            )
