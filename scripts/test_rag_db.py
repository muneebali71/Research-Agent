from __future__ import annotations

import asyncio
import sys

from langchain_core.documents import Document
from sqlalchemy import select

from app.db.database import async_session, engine, init_db
from app.db.model import Message, Session as ChatSession, Document as DocRow
from app.db.checkpointer import get_checkpointer
from app.rag.vector_store import add_chunks, retrieve, delete_thread


async def test_db() -> str:
    print("\n[1] Creating tables ...")
    await init_db()
    print("    tables ready.")

    print("[2] Writing a session + messages + document ...")
    async with async_session() as db:
        session = ChatSession(title="Test chat")
        db.add(session)
        await db.flush()  # populate session.id
        sid = session.id

        db.add_all([
            Message(session_id=sid, role="user", content="What is in the PDF?"),
            Message(session_id=sid, role="assistant", content="It is about Paris."),
        ])
        db.add(DocRow(session_id=sid, filename="paris.pdf", chunks=2))
        await db.commit()

    print("[3] Reading the session back ...")
    async with async_session() as db:
        msgs = (await db.execute(
            select(Message).where(Message.session_id == sid)
        )).scalars().all()
        print(f"    session {sid} has {len(msgs)} messages:")
        for m in msgs:
            print(f"      - {m.role}: {m.content}")

    return sid


async def test_checkpointer() -> None:
    print("[4] Setting up LangGraph checkpointer tables ...")
    async with get_checkpointer() as cp:
        await cp.setup()
    print("    checkpointer ready.")


def test_rag(thread_id: str) -> None:
    print("[5] Qdrant ingest + thread-scoped retrieval ...")
    add_chunks([
        Document(page_content="The Eiffel Tower is located in Paris, France."),
        Document(page_content="Photosynthesis converts sunlight into energy."),
    ], thread_id=thread_id)

    hits = retrieve("Where is the Eiffel Tower?", thread_id=thread_id, k=1)
    print(f"    top hit: {hits[0].page_content!r}")
    assert "Eiffel" in hits[0].page_content

    other = retrieve("Eiffel Tower", thread_id="some-other-thread", k=3)
    assert other == [], "thread filter leaked across threads!"
    print("    thread filtering OK (other thread returns nothing).")

    delete_thread(thread_id)
    print("    cleaned up Qdrant chunks.")


async def main() -> None:
    sid = await test_db()
    await test_checkpointer()
    test_rag(sid)
    await engine.dispose()
    print("\n[OK] RAG + DB layer verified end-to-end.")


if __name__ == "__main__":
    # psycopg's async driver cannot use Windows' default ProactorEventLoop,
    # so any async entry point that touches the Postgres checkpointer must
    # select the SelectorEventLoop policy first.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
