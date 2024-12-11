import os
import math
import cohere
import numpy as np
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def get_cohere_embedding(
    prompt: str, model: str = "embed-english-light-v3.0", input_type: str = "search_document"
) -> list[float]:
    response = co.embed(texts=[prompt], model=model, input_type=input_type)

    return response.embeddings[0]


def get_cohere_embeddings_batched(
    prompts: list[str],
    batch_size: int = 96,
    model: str = "embed-english-light-v3.0",
    input_type: str = "search_document",
    pbar: bool = False,
) -> list[list[float]]:
    assert batch_size <= 96, "cohere limits batch size at 96"

    num_batches = math.ceil(len(prompts) / batch_size)
    embeddings = []
    for i in tqdm(range(num_batches), disable=not pbar):
        inputs = prompts[i * batch_size : (i + 1) * batch_size]
        response = co.embed(texts=inputs, model=model, input_type=input_type, truncate="END")
        embedding = response.embeddings
        embeddings.append(embedding)
    embeddings = np.concatenate(embeddings)
    embeddings = embeddings.tolist()
    return embeddings
