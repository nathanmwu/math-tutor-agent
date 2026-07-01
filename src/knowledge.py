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


def retrieve_content(
    topic: str,
    subtopic: str | None,
    difficulty: int,
    chroma_dir: Path,
    error_category: str | None = None,
    n_results: int = 3,
    query_text: str | None = None,
) -> list[str]:
    collection = get_or_create_collection(chroma_dir)
    total_docs = collection.count()
    if total_docs == 0:
        return []

    # Use the problem text as the semantic query when available — more specific than topic/subtopic alone
    if not query_text:
        query_text = f"{topic} {subtopic}" if subtopic else topic

    def build_where(extra: dict | None = None) -> dict:
        conditions: list[dict] = [{"topic": {"$eq": topic}}]
        if subtopic:
            conditions.append({"subtopic": {"$eq": subtopic}})
        if extra:
            for key, value in extra.items():
                conditions.append({key: {"$eq": value}})
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    if error_category:
        misconception_where = build_where({"misconception_tag": error_category})
        miscon_n = min(n_results, total_docs)
        miscon_results = collection.query(
            query_texts=[query_text],
            n_results=miscon_n,
            where=misconception_where,
        )
        found_docs: list[str] = miscon_results["documents"][0]
        found_ids: set[str] = set(miscon_results["ids"][0])

        if len(found_docs) < n_results:
            remaining = n_results - len(found_docs)
            general_where = build_where()
            general_n = min(remaining + len(found_docs), total_docs)
            general_results = collection.query(
                query_texts=[query_text],
                n_results=general_n,
                where=general_where,
            )
            for doc_id, doc in zip(general_results["ids"][0], general_results["documents"][0]):
                if doc_id not in found_ids and len(found_docs) < n_results:
                    found_docs.append(doc)
                    found_ids.add(doc_id)

        return found_docs

    general_where = build_where()
    safe_n = min(n_results, total_docs)
    results = collection.query(
        query_texts=[query_text],
        n_results=safe_n,
        where=general_where,
    )
    return results["documents"][0]
