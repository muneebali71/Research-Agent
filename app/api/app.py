"""FastAPI application factory + lifespan.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from config import DATABASE_URL_SYNC
from app.db.database import init_db, engine
from app.graph.graph import build_research_graph
from app.api.routes import router

# ── These imports run their module-level code RIGHT NOW, synchronously ────────
# embeddings.py  → imports sentence_transformers (15s) + loads model (0.5s)
# vector_store.py → connects Qdrant + ensures collection exists
print("[startup] Loading embedding model and connecting Qdrant...")
_t = time.time()
import app.rag.embeddings    # noqa: F401  loads HF model
import app.rag.vector_store  # noqa: F401  connects Qdrant
print(f"[startup] ✅ RAG ready in {time.time() - _t:.1f}s")
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # RAG is already loaded above — nothing to do for embeddings/Qdrant here.
    await init_db()

    pool = AsyncConnectionPool(
        conninfo=DATABASE_URL_SYNC,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    app.state.graph = build_research_graph(checkpointer=checkpointer)

    print("\n✅ Research Agent ready — http://localhost:8020\n")
    try:
        yield
    finally:
        await pool.close()
        await engine.dispose()


def create_app() -> FastAPI:
    application = FastAPI(title="Research Agent Chatbot", lifespan=lifespan)
    application.include_router(router)

    @application.get("/")
    async def root():
        return {"status": "ok", "service": "research-agent-chatbot"}

    return application


app = create_app()