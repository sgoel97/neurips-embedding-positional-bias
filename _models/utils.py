import os
import requests
from groq import Groq
import numpy as np


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
