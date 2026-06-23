
"""PDF ingestion + RAG retrieval, backed by Qdrant.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool

from app.rag.vector_store import add_chunks, retrieve


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """Load an uploaded PDF, split it, and index the chunks in Qdrant."""
    if not file_bytes:
        raise ValueError("No PDF bytes received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        resolved_name = filename or os.path.basename(temp_path)
        for chunk in chunks:
            chunk.metadata["source_file"] = resolved_name

        added = add_chunks(chunks, thread_id=thread_id)
        print(f"[ingest_pdf] Indexed {added} chunks from '{resolved_name}' for thread {thread_id}")

        return {
            "filename":  resolved_name,
            "documents": len(docs),
            "chunks":    added,
            "thread_id": str(thread_id),
            "status":    "success",
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """Retrieve relevant information from the PDF(s) uploaded in this thread."""
    docs = retrieve(query, thread_id=thread_id, k=5)

    if not docs:
        return {
            "error": "No relevant content found. Upload a PDF first, or rephrase the query.",
            "query": query,
            "thread_id": str(thread_id),
        }

    return {
        "query":       query,
        "context":     [doc.page_content for doc in docs],
        "metadata":    [doc.metadata for doc in docs],
        "source_file": docs[0].metadata.get("source_file"),
    }