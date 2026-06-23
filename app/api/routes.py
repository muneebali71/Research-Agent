"""FastAPI routes — includes new /sessions/{sid}/chunks endpoint."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session
from app.db.model import Document as DocRow, Message, Session as ChatSession
from app.Agents.pdf_reader import ingest_pdf
from app.state.schema import ResearchState
from app.rag.vector_store import has_chunks, get_chunks_for_thread
from langsmith import traceable
from app.state.schema import SessionOut, ChatRequest, ChatResponse, MessageOut

router = APIRouter()


async def get_db() -> AsyncSession:
    async with async_session() as db:
        yield db


class DocumentOut(BaseModel):
    filename: str
    chunks: int


class ChunkOut(BaseModel):
    id: int
    text: str
    source_file: str
    page: int | None


async def _get_session_or_404(db: AsyncSession, sid: str) -> ChatSession:
    session = await db.get(ChatSession, sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@traceable(name="create session")
@router.post("/sessions", response_model=SessionOut)
async def create_session(db: AsyncSession = Depends(get_db)):
    session = ChatSession()
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionOut(id=session.id, title=session.title)


@traceable(name="list sessions")
@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(ChatSession))).scalars().all()
    return [SessionOut(id=s.id, title=s.title) for s in rows]


@router.get("/sessions/{sid}/documents", response_model=list[DocumentOut])
async def get_documents(sid: str, db: AsyncSession = Depends(get_db)):
    await _get_session_or_404(db, sid)
    rows = (await db.execute(
        select(DocRow).where(DocRow.session_id == sid).order_by(DocRow.created_at)
    )).scalars().all()
    return [DocumentOut(filename=d.filename, chunks=d.chunks) for d in rows]


# NEW: return all chunks stored in Qdrant for this session
@router.get("/sessions/{sid}/chunks", response_model=list[ChunkOut])
async def get_chunks(sid: str, db: AsyncSession = Depends(get_db)):
    await _get_session_or_404(db, sid)
    chunks = await asyncio.to_thread(get_chunks_for_thread, sid)
    return chunks


@traceable(name="session/upload")
@router.post("/sessions/{sid}/upload")
async def upload_pdf(
    sid: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    await _get_session_or_404(db, sid)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_bytes = await file.read()
    result = await asyncio.to_thread(ingest_pdf, file_bytes, sid, file.filename)
    stored = await asyncio.to_thread(has_chunks, sid)
    if not stored:
        raise HTTPException(status_code=500, detail="PDF ingestion failed.")
    db.add(DocRow(session_id=sid, filename=result["filename"], chunks=result["chunks"]))
    await db.commit()
    return result


@traceable(name="session/chat")
@router.post("/sessions/{sid}/chat", response_model=ChatResponse)
async def chat(
    sid: str,
    body: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    await _get_session_or_404(db, sid)
    db.add(Message(session_id=sid, role="user", content=body.query))
    await db.commit()
    graph = request.app.state.graph
    initial_state: ResearchState = {
        "topic": body.query, "thread_id": sid,
        "revision_count": 0, "revision_history": [],
    }
    final_state = await graph.ainvoke(
        initial_state, config={"configurable": {"thread_id": sid}},
    )
    answer = final_state.get("report", "")
    route  = final_state.get("route")
    score  = final_state.get("final_score") if route != "chat" else None
    db.add(Message(session_id=sid, role="assistant", content=answer,
                   route=route, final_score=score))
    await db.commit()
    return ChatResponse(session_id=sid, route=route, answer=answer, score=score)


@traceable(name="sessions/messages")
@router.get("/sessions/{sid}/messages", response_model=list[MessageOut])
async def get_messages(sid: str, db: AsyncSession = Depends(get_db)):
    await _get_session_or_404(db, sid)
    rows = (await db.execute(
        select(Message).where(Message.session_id == sid).order_by(Message.created_at)
    )).scalars().all()
    return [
        MessageOut(role=m.role, content=m.content, route=m.route, score=m.final_score)
        for m in rows
    ]












