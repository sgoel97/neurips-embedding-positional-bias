from pathlib import Path

from needle_config import NeedleConfig
from needle_experiment import NeedleExperiment


"""
Workflow:
1. Get the embeddings and ablation similarities here by running ``python needle_queue_file.py`` with the proper changes.
    - If it's a new model, add it into ``_models/constants.py``
2. Run ``python needle_haystack_similarity_plotting.py`` with the proper changes. 
     - If you have a huggingface model e.g. `"junnyu/roformer_chinese_base", switch out the `/`
"""


def main():
    modes = [
        "insert",
        "remove",
    ]
    models = [
        # Huggingface Models
        "BAAI/bge-small-en-v1.5",
        "junnyu/roformer_chinese_base",
        "google-bert/bert-base-uncased",
        "intfloat/e5-mistral-7b-instruct",
        "BAAI/bge-m3",
        "nomic-ai/nomic-embed-text-v1.5",
        "jinaai/jina-embeddings-v2-base-en",
        "dwzhu/e5rope-base",
        "mosaicml/mosaic-bert-base-seqlen-1024",
        "intfloat/e5-large-v2",
        "dwzhu/e5-base-4k",
        # Cohere Models
        "embed-english-v3.0",
        # Openai Models
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]
    datasets = [
        "scientific_papers",
        "paul_graham",
        "amazon_polarity",
        "arguana",
        "reddit",
    ]

    needle_configs: list[NeedleConfig] = []
    corpus_path = Path("_datasets/needles/lorem.txt")
    needle_posns = [0, 0.25, 0.5, 0.75, 1]
    for mode in modes:
        if mode == "insert":
            needle_sizes = [0.05, 0.1, 0.2, 0.5, 1]
        else:
            needle_sizes = [0.05, 0.1, 0.2, 0.5]

        for needle_size in needle_sizes:
            for posn in needle_posns:
                config = NeedleConfig(
                    corpus_path=corpus_path, mode=mode, size=needle_size, posn=posn
                )
                needle_configs.append(config)

    experiments: list[NeedleExperiment] = []
    for mode in modes:
        for model in models:
            for dataset in datasets:
                experiment = NeedleExperiment(
                    mode=mode,
                    dataset_name=dataset,
                    needle_configs=[config for config in needle_configs if config.mode == mode],
                    model_name=model,
                    max_length=2048,
                )
                experiments.append(experiment)

    for experiment in experiments:
        experiment.run()


if __name__ == "__main__":
    main()
