"""RAG node — retrieves chunks and stores their metadata for source citations."""
from __future__ import annotations

import asyncio
from app.rag.vector_store import retrieve
from app.state.schema import ResearchState


async def rag_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Node: rag_node — retrieving from uploaded PDF...")
    print("=" * 50)

    query     = state["topic"]
    thread_id = state.get("thread_id", "default")

    docs = await asyncio.to_thread(retrieve, query, thread_id, 5)

    if not docs:
        print("[rag_node] No chunks found.")
        return {"pdf_context": "", "pdf_chunks_meta": []}

    pdf_context      = "\n\n".join(doc.page_content for doc in docs)
    pdf_chunks_meta  = [doc.metadata for doc in docs]   # contains source_file, page, etc.

    # Print sources found
    sources_found = set()
    for meta in pdf_chunks_meta:
        fname = meta.get("source_file", "unknown")
        page  = meta.get("page")
        sources_found.add(f"{fname} p.{page+1}" if page is not None else fname)
    print(f"[rag_node] Retrieved {len(docs)} chunks from: {', '.join(sources_found)}")

    return {"pdf_context": pdf_context, "pdf_chunks_meta": pdf_chunks_meta}