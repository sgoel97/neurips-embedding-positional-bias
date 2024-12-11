import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from finetune_make_dataset import data_path
from _models.huggingface.huggingface import get_device


def train(model, tokenizer, data, criterion, optimizer, device, batch_size=4):
    data = data.sample(frac=1, random_state=42)
    model.train()
    total_loss = 0.0
    num_batches = max(int(len(data) / batch_size + 0.99), 1)

    for i in tqdm(range(num_batches), disable=not True):
        batch = data[i * batch_size : (i + 1) * batch_size]
        encoded_input1 = tokenizer(
            list(batch["sample1"]), padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        encoded_input2 = tokenizer(
            list(batch["sample2"]), padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        labels = torch.tensor(batch["pos"].values).to(device)

        optimizer.zero_grad()

        model_output1 = model(**encoded_input1)
        model_output2 = model(**encoded_input2)

        embeddings1 = model_output1[0][:, 0]
        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)

        embeddings2 = model_output2[0][:, 0]
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        loss = criterion(embeddings1, embeddings2, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Move tensors to CPU to free up GPU memory
        del (
            encoded_input1,
            encoded_input2,
            labels,
            model_output1,
            model_output2,
            embeddings1,
            embeddings2,
        )
        torch.cuda.empty_cache()

    return total_loss / len(data)


def main():
    model_path = "project/models/bge-m3-finetune-combined"
    Path(model_path).mkdir(parents=True, exist_ok=True)

    device = get_device(use_gpu=True)

    data = pd.read_pickle(data_path)

    model_name = "BAAI/bge-m3"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    num_epochs = 3
    for epoch in tqdm(range(num_epochs)):
        loss = train(model, tokenizer, data, criterion, optimizer, device, batch_size=1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    main()
