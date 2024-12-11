from functools import partial
from typing import Callable

from _models.utils import *
from _models.constants import *

from _models.openai.openai import *
from _models.claude.claude import *
from _models.cohere.cohere import *
from _models.voyage.voyage import *
from _models.huggingface.huggingface import *


def get_embedding(prompt: str, model: str = "text-embedding-3-small") -> list[float]:
    if model in ollama_encoder_models:
        return get_ollama_embedding(prompt, model)
    elif model in huggingface_encoder_models:
        return get_huggingface_embedding(prompt, model)
    elif model in openai_encoder_models:
        return get_openai_embedding(prompt, model)
    elif model in cohere_models:
        return get_cohere_embedding(prompt, model)
    elif model in voyage_models:
        return get_voyage_embedding(prompt, model)
    else:
        raise ValueError(f"Encoder Model {model} not supported")


def get_embeddings_batched(
    prompts: list[str],
    batch_size: int = 100,
    model: str = "text-embedding-3-small",
    **kwargs,
) -> list[list[float]]:
    if model in openai_encoder_models:
        return batch_openai_embeddings(prompts, batch_size, model, **kwargs)
    elif model in huggingface_decoder_models:
        return get_huggingface_embeddings_batched(prompts, batch_size, model, **kwargs)
    elif model in cohere_models:
        return get_cohere_embeddings_batched(prompts, batch_size, model, **kwargs)
    elif model in voyage_models:
        return get_voyage_embeddings_batched(prompts, batch_size, model, **kwargs)
    else:
        output = []
        for i in tqdm(range(len(prompts)), disable=not kwargs.get("pbar", True)):
            prompt = prompts[i]
            output.append(get_embedding(prompt, model))
        return output


def get_response(prompt: str, system_prompt: str | None = None, model: str = "mistral") -> str:
    if model in ollama_models:
        return get_ollama_response(prompt, system_prompt, model)
    elif model in huggingface_decoder_models:
        return get_huggingface_response(prompt, system_prompt, model)
    elif model in openai_models:
        return get_openai_response(prompt, system_prompt, model)
    elif model in anthropic_models:
        return get_anthropic_response(prompt, system_prompt, model)
    elif model in groq_models:
        return get_groq_response(prompt, system_prompt, model)
    else:
        raise ValueError(f"Model {model} not supported")


def get_response_batched(
    prompts: list[str],
    system_prompt: str | None = None,
    model: str = "mistral",
    **kwargs,
) -> list[str]:
    if model in openai_models:
        return get_openai_response_batched(prompts, system_prompt, model)
    elif model in huggingface_decoder_models:
        return get_huggingface_response_batched(prompts, system_prompt, model=model, **kwargs)
    else:
        return [get_response(prompt, system_prompt, model) for prompt in prompts]


def get_embedding_func_batched(model: str = "text-embedding-3-small") -> Callable:
    if model in openai_encoder_models:
        return get_openai_embeddings_batched
    elif model in huggingface_models:
        return get_huggingface_embeddings_batched
    elif model in cohere_models:
        return get_cohere_embeddings_batched
    elif model in voyage_models:
        return get_voyage_embeddings_batched
    else:
        return partial(get_embeddings_batched, model=model)
