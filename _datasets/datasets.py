hf_datasets: dict[str, dict] = {
    "scientific_papers": {
        "path": "scientific_papers",
        "name": "pubmed",
        "trust_remote_code": True,
    },
    "wikipedia": {
        "path": "wikipedia",
        "name": "20220301.en",
    },
    "paul_graham": {
        "path": "sgoel9/paul_graham_essays",
    },
    "amazon_polarity": {
        "path": "amazon_polarity",
    },
    "arxiv-clustering-p2p": {
        "path": "mteb/arxiv-clustering-p2p",
        "split": "test",
    },
    "arguana": {
        "path": "BeIR/arguana",
        "name": "corpus",
        "split": "corpus",
    },
    "sts22": {
        "path": "mteb/sts22-crosslingual-sts",
    },
    "reddit": {
        "path": "mteb/reddit-clustering-p2p",
        "split": "test",
    },
}
