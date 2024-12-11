import os
import math
import torch
import requests
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from _models.constants import *

hf_token = os.environ.get("HF_TOKEN")
curr_tokenizer = None
curr_model = None


def get_device(use_gpu=True):
    if not use_gpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def load_model(model="BAAI/bge-small-en-v1.5", device=None):
    global curr_model, curr_tokenizer  # Assuming globals are still necessary

    tokenizer_name, model_name = get_model_and_tokenizer_names(model)

    if curr_model is None or curr_model.name_or_path != model:
        if model in huggingface_encoder_local_models:
            print(f"Loading model from local files: {model}")
            curr_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=False)
            curr_model = AutoModel.from_pretrained(model_name, local_files_only=False)
        else:
            curr_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

            config = get_config(model_name)

            if config:
                curr_model = AutoModel.from_pretrained(
                    model_name, config=config, trust_remote_code=True
                )
            else:
                curr_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        curr_model.eval()  # Set the model to evaluation mode
        print(f"Setting tokenizer to {tokenizer_name}. ")
        print(f"Setting model to {model_name}. ")

    if device is not None:
        curr_model = curr_model.to(device)  # Move model to device only if specified
    return curr_model, curr_tokenizer


def get_huggingface_embeddings_batched(
    prompts: list[str],
    batch_size: int = 4,
    model_name: str = "BAAI/bge-small-en-v1.5",
    pbar: bool = False,
    use_gpu: bool = False,
) -> list[list[float]]:
    device = get_device(use_gpu=use_gpu)
    model, tokenizer = load_model(model_name, device)

    num_batches = math.ceil(len(prompts) / batch_size)
    embeddings = []
    if model_name == "nomic-ai/nomic-embed-text-v1.5":
        prompts = ["search_document: " + prompt for prompt in prompts]

    for i in tqdm(range(num_batches), disable=not pbar):
        batch = prompts[i * batch_size : (i + 1) * batch_size]

        encoded_input = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.append(sentence_embeddings.cpu())  # Move embeddings to CPU after computation

        # Optional: Clear the CUDA cache after processing each batch
        torch.cuda.empty_cache()

    return torch.vstack(embeddings).detach().tolist()


def get_huggingface_embedding(prompt: str, model: str = "BAAI/bge-small-en-v1.5") -> list[float]:
    device = get_device()
    model, tokenizer = load_model(model)
    model = model.to(device)

    encoded_input = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings[0]
    return sentence_embeddings.detach().cpu().tolist()


def get_huggingface_embeddings_api(prompts: list[str], model: str = "BAAI/bge-small-en-v1.5"):
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": prompts, "options": {"wait_for_model": True}},
    )

    if response.status_code == 200:
        embeddings_matrix = response.json()
        # Convert the list of embeddings to a torch tensor for normalization
        embeddings_tensor = torch.tensor(embeddings_matrix)
        # Normalize the embeddings
        normalized_embeddings = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
        # Convert to numpy array and return
        print("generated embeddings")
        return normalized_embeddings.numpy()

    else:
        response.raise_for_status()


def get_huggingface_response(
    prompt: str, system_prompt: str | None = None, model: str = "facebook/opt-125m"
) -> str:
    device = get_device()
    model, tokenizer = load_model(model)
    model = model.to(device)

    encoded_input = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model.generate(**encoded_input)
    response = tokenizer.batch_decode(model_output, skip_special_tokens=True)
    return response


def get_huggingface_response_batched(
    prompts: list[str],
    system_prompt: str | None = None,
    batch_size: int = 128,
    model: str = "facebook/opt-125m",
    pbar: bool = False,
) -> list[str]:
    device = get_device()
    model, tokenizer = load_model(model)
    model = model.to(device)

    num_batches = math.ceil(len(prompts) / batch_size)
    responses = []
    for i in tqdm(range(num_batches), disable=not pbar):
        batch = prompts[i * batch_size : (i + 1) * batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            model_output = model.generate(**encoded_input)
        response = tokenizer.batch_decode(model_output, skip_special_tokens=True)
        responses.extend(response)
    return responses
