import math
import voyageai
import numpy as np
from tqdm.auto import tqdm


def get_voyage_embedding(
    prompt: str, model: str = "voyage-2", input_type: bool = None
) -> list[float]:
    vo = voyageai.Client()

    response = vo.embed([prompt], model=model, input_type=input_type)
    return response.embeddings[0]


def get_voyage_embeddings_batched(
    prompts: list[str],
    batch_size: int = 128,
    model: str = "voyage-2",
    input_type: bool = None,
    pbar: bool = False,
) -> list[list[float]]:
    assert batch_size <= 128, "voyage limits batch size at 128"
    vo = voyageai.Client()

    num_batches = math.ceil(len(prompts) / batch_size)
    embeddings = []
    for i in tqdm(range(num_batches), disable=not pbar):
        inputs = prompts[i * batch_size : (i + 1) * batch_size]
        response = vo.embed(texts=inputs, model=model, input_type=input_type)
        embedding = response.embeddings
        embeddings.append(embedding)
    embeddings = np.hstack(embeddings)
    embeddings = embeddings.tolist()
    return embeddings
