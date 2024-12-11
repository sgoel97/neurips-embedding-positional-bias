import transformers

# CoHere
cohere_models = ["embed-english-light-v3.0", "embed-english-v3.0", "embed-english-v2.0"]

# Voyage AI
voyage_models = ["voyage-2", "voyage-law-2", "voyage-large-2-instruct"]

# OpenAI
openai_decoder_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-0125-preview", "gpt-4-turbo"]
openai_encoder_models = ["text-embedding-3-small", "text-embedding-3-large"]
openai_models = openai_decoder_models + openai_encoder_models


# Anthropic
anthropic_models = ["claude-3-haiku-20240307", "claude-3-opus-20240229"]

# Groq
groq_models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]

# Ollama
ollama_decoder_models = ["mistral", "phi", "llama2", "llama3"]
ollama_encoder_models = ["nomic-embed-text", "all-minilm"]
ollama_models = ollama_decoder_models + ollama_encoder_models

# Huggingface
huggingface_decoder_models = ["facebook/opt-125m"]
huggingface_encoder_remote_models = [
    "BAAI/bge-m3",
    "nomic-ai/nomic-embed-text-v1.5",
    "jinaai/jina-embeddings-v2-base-en",
    "BAAI/bge-small-en-v1.5",
    "Salesforce/SFR-Embedding-Mistral",
    "BAAI/bge-large-en-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "junnyu/roformer_chinese_base",
    "google-bert/bert-base-uncased",
    "dwzhu/e5rope-base",
    "intfloat/e5-mistral-7b-instruct",
    "mosaicml/mosaic-bert-base-seqlen-1024",
    "intfloat/e5-large-v2",
    "dwzhu/e5-base-4k",
    # Custom on huggingface
    "reaganjlee/baai-truncate-finetune",
]
huggingface_encoder_local_models = [
    # Custom finetuned models based on HF models
    "_models/baai-tuned",
]
huggingface_encoder_models = huggingface_encoder_remote_models + huggingface_encoder_local_models
huggingface_models = huggingface_decoder_models + huggingface_encoder_models
