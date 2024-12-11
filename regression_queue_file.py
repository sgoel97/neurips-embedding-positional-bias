from pathlib import Path
from regression_experiment import RegressionExperiment


def main():
    model_groups = {
        "ALIBI (Jina)": [
            "jinaai_jina-embeddings-v2-base-en",
        ],
        "APE": [
            "intfloat_e5-large-v2",
            "BAAI_bge-m3",
        ],
        "ROPE": [
            "nomic-ai_nomic-embed-text-v1.5",
            "dwzhu_e5rope-base",
        ],
        "ALIBI (Mosaic)": [
            "mosaicml_mosaic-bert-base-seqlen-1024",
        ],
    }
    modes = [
        "insert",
        "remove",
    ]
    datasets = [
        "scientific_papers",
        "paul_graham",
        "amazon_polarity",
        "arguana",
        "reddit",
    ]

    for model_group in model_groups:

        dataset_paths = []
        for model in model_groups[model_group]:
            for dataset in datasets:
                for mode in modes:
                    file_path = f"./data/{model}/{dataset}_{mode}.pkl"
                    assert Path(file_path).exists(), f"{file_path} does not exist"
                    dataset_paths.append(f"./data/{model}/{dataset}_{mode}.pkl")

        experiment = RegressionExperiment(
            exp_name=model_group,
            dataset_paths=dataset_paths,
            min_sentences=5,
            max_sentences=65,
            sentence_bucket_size=10,
        )

        experiment.run()
        experiment.plot(plot_save_path="./plots")


if __name__ == "__main__":
    main()
