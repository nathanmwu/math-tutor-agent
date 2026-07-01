"""One-time script to populate ChromaDB from data/knowledge_base/*.json"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os

load_dotenv()

from src.knowledge import load_knowledge_base

KB_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base"))
CHROMA_DIR = Path(os.getenv("CHROMADB_PATH", "data/chromadb"))

if __name__ == "__main__":
    print(f"Loading knowledge base from {KB_DIR} into {CHROMA_DIR} ...")
    added = load_knowledge_base(KB_DIR, CHROMA_DIR)
    print(f"Done. {added} new chunks added.")
