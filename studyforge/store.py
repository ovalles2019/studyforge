from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from studyforge.chunk import Chunk
from studyforge.config import Settings, get_settings


def _embedding_fn(settings: Settings) -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)


def get_collection(settings: Settings | None = None) -> Collection:
    settings = settings or get_settings()
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=_embedding_fn(settings),
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: list[Chunk], settings: Settings | None = None) -> int:
    if not chunks:
        return 0
    collection = get_collection(settings)
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata() for c in chunks],
    )
    return len(chunks)


def retrieve(query: str, k: int | None = None, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    collection = get_collection(settings)
    n = collection.count()
    if n == 0:
        return []
    result = collection.query(
        query_texts=[query],
        n_results=min(k or settings.retrieve_k, n),
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": doc,
                "page_start": int(meta.get("page_start", 0)),
                "page_end": int(meta.get("page_end", 0)),
                "source": str(meta.get("source", "")),
                "distance": float(dist),
            }
        )
    return hits


def reset_collection(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    get_collection(settings)
