"""Supervisor: routes queries to pdf / web / both / chat.
"""
from __future__ import annotations

import asyncio
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm_model.llm import get_llm
from app.rag.vector_store import has_chunks, get_chunks_for_thread


class RouteDecision(BaseModel):
    route: Literal["pdf", "web", "both", "chat"] = Field(
        description=(
            "chat = greetings, small-talk, thanks — no research needed; "
            "pdf  = the query is about the SAME topic as the uploaded PDF; "
            "web  = the query is about a DIFFERENT topic than the PDF, or needs current/external info; "
            "both = the query needs PDF content AND fresh web info on the SAME topic as the PDF."
        )
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")


_PROMPT_WITH_PDF = ChatPromptTemplate.from_messages([
    ("system",
     "You are a routing supervisor. The user has uploaded a PDF.\n\n"
     "PDF Topic Summary (first few chunks): {pdf_topic_hint}\n\n"
     "ROUTING RULES — follow in order:\n"
     "1. Greeting/small-talk/thanks → 'chat'\n"
     "2. Compare the user's query topic against the PDF Topic Summary above.\n"
     "   - If the query is about the SAME subject as the PDF → 'pdf' (or 'both' if it also needs current web info)\n"
     "   - If the query is about a COMPLETELY DIFFERENT subject than the PDF → 'web'\n"
     "3. Only use 'both' when the query is on-topic with the PDF AND explicitly needs fresh/current information\n\n"
     "CRITICAL: Do NOT default to 'pdf' or 'both' just because a PDF exists. "
     "If the query topic has nothing to do with the PDF's subject matter, use 'web' only. "
     "Pulling in an irrelevant PDF as a 'source' is worse than not using it at all."),
    ("human", "User message:\n{query}"),
])

_PROMPT_NO_PDF = ChatPromptTemplate.from_messages([
    ("system",
     "You are a routing supervisor. No PDF has been uploaded.\n\n"
     "1. Greeting/small-talk/thanks → 'chat'\n"
     "2. Otherwise → 'web'"),
    ("human", "User message:\n{query}"),
])


def _classifier(has_pdf: bool):
    prompt = _PROMPT_WITH_PDF if has_pdf else _PROMPT_NO_PDF
    return prompt | get_llm().with_structured_output(RouteDecision)


def _get_pdf_topic_hint(thread_id: str) -> str:
    """Grab a short hint of what the uploaded PDF is about, from its first chunks."""
    try:
        chunks = get_chunks_for_thread(thread_id)
        if not chunks:
            return "No PDF content available."
        # Use first 2 chunks (usually title/abstract/intro) as a topic hint.
        preview = " ".join(c["text"][:300] for c in chunks[:2])
        return preview[:600]
    except Exception:
        return "Unable to determine PDF topic."


async def supervisor_node(state: dict) -> dict:
    query     = state["topic"]
    thread_id = state.get("thread_id", "default")

    has_pdf = await asyncio.to_thread(has_chunks, thread_id)

    if has_pdf:
        pdf_topic_hint = await asyncio.to_thread(_get_pdf_topic_hint, thread_id)
        decision: RouteDecision = await _classifier(has_pdf).ainvoke({
            "query": query,
            "pdf_topic_hint": pdf_topic_hint,
        })
    else:
        decision: RouteDecision = await _classifier(has_pdf).ainvoke({"query": query})

    print(f"\n[supervisor] has_pdf={has_pdf}  route='{decision.route}'  ({decision.reasoning})")
    return {"has_pdf": has_pdf, "route": decision.route}


def route_after_supervisor(state: dict) -> str:
    route = state["route"]
    if route == "chat":
        return "chat_node"
    if route in ("pdf", "both"):
        return "rag_node"
    return "search_node"


def route_after_rag(state: dict) -> str:
    return "search_node" if state["route"] == "both" else "writer_node"