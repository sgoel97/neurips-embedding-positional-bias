import os
import requests
import transformers
from groq import Groq


def get_model_and_tokenizer_names(model_name: str) -> tuple[str, str]:
    """
    Returns (tokenizer_name, model_name)
    Most should be the same, with just a few having exceptions
    Predefined list of model names with their corresponding tokenizer names
    """
    model_mapping: dict[str, str] = {
        "mosaicml/mosaic-bert-base-seqlen-1024": "bert-base-uncased",
    }

    # Check if the model name is in the list and return the corresponding names
    if model_name in model_mapping:
        return (model_mapping[model_name], model_name)

    # If the model name is not found, return the name itself
    return (model_name, model_name)


def get_config(model_name: str) -> transformers.PretrainedConfig | None:
    mapping: dict[str, transformers.PretrainedConfig] = {
        "mosaicml/mosaic-bert-base-seqlen-1024": transformers.BertConfig.from_pretrained(
            "mosaicml/mosaic-bert-base-seqlen-1024"
        )
    }
    if model_name in mapping:
        return mapping[model_name]
    return None


def get_groq_response(
    prompt: str, system_prompt: str | None = None, model: str = "llama3-8b-8192"
) -> str:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    messages = [{"role": "user", "content": prompt}]

    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})

    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model=model,
    )

    return chat_completion.choices[0].message.content


def get_ollama_response(
    prompt: str, system_prompt: str | None = None, model: str = "mistral"
) -> str:
    content = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    if system_prompt is not None:
        content["system_prompt"] = system_prompt

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=content,
    )

    return response.json()["response"]


def get_ollama_embedding(prompt: str, model: str = "all-minilm") -> list[float]:
    content = {
        "model": model,
        "prompt": prompt,
    }

    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json=content,
    )

    return response.json()["embedding"]
