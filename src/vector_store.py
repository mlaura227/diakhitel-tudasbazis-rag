"""Versioned, batch-processed Chroma ingestion; keep old collections intact."""
from __future__ import annotations
from collections import Counter
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_BASE_URL
from src.document_loader import load_documents, split_documents


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PATH),
    )


def create_vector_store(batch_size: int = 12) -> Chroma:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks = split_documents(load_documents())
    ids = [f"{d.metadata['doc_id']}::{d.metadata['section_id']}" for d in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs detected before vector database mutation")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    existing = [c.name if hasattr(c, "name") else str(c) for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Replacing the v2 collection only: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
    store = Chroma(client=client, collection_name=COLLECTION_NAME,
                   embedding_function=get_embeddings())
    print("Source sections:", dict(Counter(d.metadata["doc_id"] for d in chunks)))
    print("Embedding model:", EMBEDDING_MODEL)
    print("Collection:", COLLECTION_NAME)
    for start in range(0, len(chunks), batch_size):
        end = min(len(chunks), start + batch_size)
        store.add_documents(documents=chunks[start:end], ids=ids[start:end])
        print(f"Indexed {end}/{len(chunks)} sections", flush=True)
    print("Final Chroma count:", store._collection.count())
    return store
