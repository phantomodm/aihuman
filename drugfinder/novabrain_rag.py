import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Optional
from novabrain_config import *

# Optional vector DB imports
try:
    import pinecone
except ImportError:
    pinecone = None

try:
    import weaviate
except ImportError:
    weaviate = None

# Embedding model
embed_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')


# -------------------- Helpers --------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split a long string into overlapping chunks for embedding."""
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks


# -------------------- Index Builder --------------------
def build_index(input_file: Optional[Path] = None, output_dir: Path = None):
    """
    Build a vector index from a given dataset file (JSONL).
    Supports FAISS (local), Pinecone, or Weaviate backends.
    """
    dataset_path = Path(input_file) if input_file else OUTPUT_JSONL
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    texts = []
    metadata = []

    print(f"📥 Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            base_info = {
                "drug_name": rec.get("drug_name"),
                "designation": rec.get("designation"),
                "rd_class": rec.get("rd_class"),
            }
            combined_text = json.dumps(rec, ensure_ascii=False)
            for chunk in chunk_text(combined_text):
                texts.append(chunk)
                metadata.append(base_info)

    print(f"🔍 Creating embeddings for {len(texts)} chunks...")
    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # ----------- FAISS Backend -----------
    if VECTOR_BACKEND == "faiss":
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        if output_dir:
            index_path = output_dir / "index.faiss"
            texts_path = output_dir / "texts.npy"
            embeds_path = output_dir / "embeddings.npy"
            meta_path = output_dir / "metadata.json"
        else:
            index_path, texts_path, embeds_path, meta_path = INDEX_PATH, TEXTS_PATH, EMBEDDINGS_PATH, METADATA_PATH
        

        faiss.write_index(index, str(INDEX_PATH))
        np.save(TEXTS_PATH, np.array(texts, dtype=object))
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved FAISS index locally at: {INDEX_PATH}")

    # ----------- Pinecone Backend -----------
    elif VECTOR_BACKEND == "pinecone" and pinecone:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        if PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(PINECONE_INDEX_NAME, dimension=embeddings.shape[1], metric="cosine")
        index = pinecone.Index(PINECONE_INDEX_NAME)
        vectors = [(str(i), emb.tolist(), meta) for i, (emb, meta) in enumerate(zip(embeddings, metadata))]
        index.upsert(vectors)
        print(f"✅ Upserted {len(vectors)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'")

    # ----------- Weaviate Backend -----------
    elif VECTOR_BACKEND == "weaviate" and weaviate:
        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
        )
        # Ensure class exists
        if not client.schema.contains({"classes": [{"class": WEAVIATE_CLASS_NAME}]}):
            schema = {
                "classes": [{
                    "class": WEAVIATE_CLASS_NAME,
                    "properties": [
                        {"name": "text", "dataType": ["text"]},
                        {"name": "drug_name", "dataType": ["string"]},
                        {"name": "designation", "dataType": ["string"]},
                        {"name": "rd_class", "dataType": ["string"]},
                    ]
                }]
            }
            client.schema.create(schema)
        for text, vector, meta in zip(texts, embeddings, metadata):
            client.data_object.create(
                {"text": text, **meta},
                class_name=WEAVIATE_CLASS_NAME,
                vector=vector.tolist()
            )
        print(f"✅ Uploaded {len(texts)} objects to Weaviate class '{WEAVIATE_CLASS_NAME}'")

    else:
        raise ValueError(f"Unknown or unsupported VECTOR_BACKEND: {VECTOR_BACKEND}")


# -------------------- Local FAISS Loader --------------------
def load_index() -> Tuple[List[str], faiss.IndexFlatIP, List[dict]]:
    """Load FAISS index and related data (local backend only)."""
    if VECTOR_BACKEND != "faiss":
        print(f"⚠ load_index() is only implemented for FAISS backend right now.")
        return [], None, []
    texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return texts, index, metadata
