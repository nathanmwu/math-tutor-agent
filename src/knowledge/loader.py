import json
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

COLLECTION_NAME = "math_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_or_create_collection(chroma_dir: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL, device="cpu")
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def load_knowledge_base(kb_dir: Path, chroma_dir: Path) -> int:
    collection = get_or_create_collection(chroma_dir)
    newly_added = 0

    for json_file in sorted(kb_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            chunks: list[dict] = json.load(f)

        all_ids = [chunk["id"] for chunk in chunks]
        existing = collection.get(ids=all_ids)
        existing_ids = set(existing["ids"])

        new_chunks = [chunk for chunk in chunks if chunk["id"] not in existing_ids]
        if not new_chunks:
            continue

        def _clean_metadata(meta: dict) -> dict:
            return {k: v for k, v in meta.items() if v is not None}

        collection.add(
            ids=[chunk["id"] for chunk in new_chunks],
            documents=[chunk["text"] for chunk in new_chunks],
            metadatas=[_clean_metadata(chunk["metadata"]) for chunk in new_chunks],
        )
        newly_added += len(new_chunks)

    return newly_added
