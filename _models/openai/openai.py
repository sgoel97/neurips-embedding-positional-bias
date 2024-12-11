import json
import math
import subprocess
from pathlib import Path
from openai import OpenAI
from tqdm.auto import tqdm

from utils.file_utils import read_data, write_data

DATASET_DIR = Path("_models/openai")


def openai_message_template(prompt: str, system_prompt: str | None = None):
    messages = [{"role": "user", "content": prompt}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def get_openai_embedding(prompt: str, model: str = "text-embedding-3-small") -> list[float]:
    client = OpenAI()
    response = client.embeddings.create(model=model, input=prompt)
    return response.data[0].embedding


def get_openai_embeddings_batched(
    prompts: list[str],
    batch_size: int = 100,
    model: str = "text-embedding-3-small",
    pbar: bool = False,
    verbose: bool = False,
) -> list[list[float]]:
    num_batches = math.ceil(len(prompts) / batch_size)

    embeddings = []
    for i in tqdm(range(num_batches), disable=not pbar):
        inputs = prompts[i * batch_size : (i + 1) * batch_size]
        embedding = batch_openai_embeddings(inputs, model=model, verbose=verbose)
        embeddings.extend(embedding)

    return embeddings


def batch_openai_embeddings(prompts, model="text-embedding-3-small", verbose=True):
    """
    Parallelize calls to OpenAI API using `/utils/openai-parallel-processing.py` script

    Source: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
    """
    requests = [
        {
            "model": model,
            "input": prompts[i],
            "metadata": {"id": i},
        }
        for i in range(len(prompts))
    ]
    jsonl_data = "\n".join(json.dumps(d) for d in requests)

    input_filepath = DATASET_DIR / "data" / "embedding_requests.jsonl"
    output_filepath = DATASET_DIR / "data" / "embedding_responses.jsonl"
    write_data(jsonl_data, input_filepath)

    if output_filepath.exists():
        output_filepath.unlink()

    command = [
        "python3",
        f"{str(DATASET_DIR)}/openai_parallel_processing.py",
        "--requests_filepath",
        input_filepath,
        "--save_filepath",
        output_filepath,
        "--request_url",
        "https://api.openai.com/v1/embeddings",
        "--max_requests_per_minute",
        "500",
        "--max_tokens_per_minute",
        "60000",
        "--token_encoding_name",
        "cl100k_base",
        "--max_attempts",
        "5",
        "--logging_level",
        "20",
    ]

    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Uncomment if errors are encountered
    if verbose:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    output_data = read_data(output_filepath)
    output_data = sorted(output_data, key=lambda x: x[2]["id"])
    output_data = list(map(lambda x: x[1]["data"][0]["embedding"], output_data))

    return output_data


def get_openai_response(
    prompt: str, system_prompt: str | None = None, model: str = "gpt-3.5-turbo"
) -> str:
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=openai_message_template(prompt, system_prompt),
    )
    return response.choices[0].message.content


def get_openai_response_batched(
    prompts: list[str],
    system_prompt: str | None = None,
    model: str = "gpt-3.5-turbo",
    verbose: bool = False,
) -> list[str]:
    """
    Parallelize calls to OpenAI API using `/utils/openai-parallel-processing.py` script

    Source: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
    """
    requests = [
        {
            "model": model,
            "messages": openai_message_template(prompts[i], system_prompt),
            "metadata": {"id": i},
        }
        for i in range(len(prompts))
    ]
    jsonl_data = "\n".join(json.dumps(d) for d in requests)

    input_filepath = DATASET_DIR / "data" / "gpt3_requests.jsonl"
    output_filepath = DATASET_DIR / "data" / "gpt3_responses.jsonl"
    write_data(jsonl_data, input_filepath)

    if output_filepath.exists():
        output_filepath.unlink()

    command = [
        "python3",
        f"{str(DATASET_DIR)}/openai_parallel_processing.py",
        "--requests_filepath",
        input_filepath,
        "--save_filepath",
        output_filepath,
        "--request_url",
        "https://api.openai.com/v1/chat/completions",
        "--max_requests_per_minute",
        "500",
        "--max_tokens_per_minute",
        "60000",
        "--token_encoding_name",
        "cl100k_base",
        "--max_attempts",
        "5",
        "--logging_level",
        "20",
    ]

    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Uncomment if errors are encountered
    if verbose:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    output_data = read_data(output_filepath)
    output_data = sorted(output_data, key=lambda x: x[2]["id"])
    output_data = list(map(lambda x: x[1]["choices"][0]["message"]["content"], output_data))

    return output_data
