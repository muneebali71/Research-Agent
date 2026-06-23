"""Shared embedding model — loads from local folder at import time."""
from __future__ import annotations

import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# Walk up from this file (app/rag/embeddings.py) to project root, then hf_cache
_HERE       = os.path.abspath(__file__)            # app/rag/embeddings.py
_RAG_DIR    = os.path.dirname(_HERE)               # app/rag/
_APP_DIR    = os.path.dirname(_RAG_DIR)            # app/
_ROOT       = os.path.dirname(_APP_DIR)            # project root  ← E:\Research Agent
_MODEL_DIR  = os.path.join(_ROOT, "hf_cache", "sentence-transformers_all-MiniLM-L6-v2")


class _LocalEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


if not os.path.exists(_MODEL_DIR):
    raise RuntimeError(
        f"Model folder not found: {_MODEL_DIR}\n"
        "Run:  uv run python test_embed.py"
    )

print(f"[embeddings] Loading model from {_MODEL_DIR} ...")
_t = time.time()
_model = SentenceTransformer(_MODEL_DIR)
_embeddings_instance = _LocalEmbeddings(_model)
print(f"[embeddings] Ready in {time.time() - _t:.1f}s")


def get_embeddings() -> _LocalEmbeddings:
    return _embeddings_instance