"""Qdrant vector store — adds get_chunks_for_thread for the chunks viewer."""
from __future__ import annotations

import os, time
from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIM
from app.rag.embeddings import get_embeddings

_THREAD_ID_KEY = "metadata.thread_id"

print(f"[vector_store] Connecting to Qdrant at {QDRANT_URL}...")
_client = QdrantClient(url=QDRANT_URL)

if not _client.collection_exists(QDRANT_COLLECTION):
    print(f"[vector_store] Creating collection '{QDRANT_COLLECTION}'...")
    _client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM, distance=models.Distance.COSINE,
        ),
    )

_store = QdrantVectorStore(
    client=_client,
    collection_name=QDRANT_COLLECTION,
    embedding=get_embeddings(),
)
print("[vector_store] ✅ Vector store ready.")


def _thread_filter(thread_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(
        key=_THREAD_ID_KEY,
        match=models.MatchValue(value=str(thread_id)),
    )])


def add_chunks(chunks: List[Document], thread_id: str) -> int:
    for chunk in chunks:
        chunk.metadata["thread_id"] = str(thread_id)
    _store.add_documents(chunks)
    return len(chunks)


def has_chunks(thread_id: str) -> bool:
    hits, _ = _client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=_thread_filter(thread_id),
        limit=1,
    )
    return len(hits) > 0


def retrieve(query: str, thread_id: str, k: int = 5) -> List[Document]:
    return _store.similarity_search(
        query, k=k, filter=_thread_filter(thread_id),
    )


def delete_thread(thread_id: str) -> None:
    _client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=_thread_filter(thread_id)),
    )


def get_chunks_for_thread(thread_id: str) -> list[dict]:
    """Return ALL chunks for a thread — used by the Streamlit chunks viewer."""
    all_chunks = []
    offset = None
    idx = 1

    while True:
        results, next_offset = _client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=_thread_filter(thread_id),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            payload  = point.payload or {}
            metadata = payload.get("metadata", {})
            # langchain-qdrant stores text in payload["page_content"]
            text     = payload.get("page_content", "")
            all_chunks.append({
                "id":          idx,
                "text":        text,
                "source_file": metadata.get("source_file", "unknown"),
                "page":        metadata.get("page"),
            })
            idx += 1

        if next_offset is None:
            break
        offset = next_offset

    # Sort by source_file then page
    all_chunks.sort(key=lambda c: (c["source_file"], c["page"] or 0, c["id"]))
    return all_chunks