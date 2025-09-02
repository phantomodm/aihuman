import os
from pathlib import Path

# Dataset name (base for all output file paths)
DATASET_NAME = os.getenv("NOVABRAIN_DATASET", "novabrain_inventor_dataset")

# Vector storage backend: faiss (local), pinecone, or weaviate
VECTOR_BACKEND = os.getenv("NOVABRAIN_VECTOR_BACKEND", "faiss").lower()

# File paths based on dataset name
OUTPUT_JSONL = Path(f"{DATASET_NAME}.jsonl")
OUTPUT_CSV = Path(f"{DATASET_NAME}.csv")
INDEX_PATH = Path(f"{DATASET_NAME}_index.faiss")
TEXTS_PATH = Path(f"{DATASET_NAME}_texts.npy")
EMBEDDINGS_PATH = Path(f"{DATASET_NAME}_embeddings.npy")
METADATA_PATH = Path(f"{DATASET_NAME}_metadata.json")


# ===============================
# Pinecone Configuration (if used)
# ===============================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME = f"{DATASET_NAME}-index"

# ===============================
# Weaviate Configuration (if used)
# ===============================
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_CLASS_NAME = f"{DATASET_NAME.replace('-', '_').capitalize()}Compound"
