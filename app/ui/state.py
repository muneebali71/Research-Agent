
"""Streamlit st.session_state initialization + helpers."""
from __future__ import annotations

import streamlit as st

from app.ui.api_client import DEFAULT_BASE_URL


def init_state() -> None:
    defaults = {
        "api_base_url":       DEFAULT_BASE_URL,
        "sessions":           [],
        "current_session_id": None,
        "session_messages":   {},   # {session_id: [msg, ...] | None}
        "session_docs":       {},   # {session_id: [{"filename":..,"chunks":..}, ...] | None}
        "sessions_loaded":    False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_session_title() -> str | None:
    sid = st.session_state.current_session_id
    for s in st.session_state.sessions:
        if s["id"] == sid:
            return s.get("title")
    return None


# ── Messages ──────────────────────────────────────────────────────────────────
def get_current_messages() -> list[dict]:
    sid = st.session_state.current_session_id
    if sid is None:
        return []
    return st.session_state.session_messages.get(sid) or []


def set_current_messages(messages: list[dict]) -> None:
    sid = st.session_state.current_session_id
    if sid is not None:
        st.session_state.session_messages[sid] = messages


def append_message(message: dict) -> None:
    sid = st.session_state.current_session_id
    if sid is not None:
        st.session_state.session_messages.setdefault(sid, []).append(message)


def messages_loaded(session_id: str) -> bool:
    val = st.session_state.session_messages.get(session_id)
    return val is not None


# ── Documents ─────────────────────────────────────────────────────────────────
def get_current_docs() -> list[dict]:
    sid = st.session_state.current_session_id
    if sid is None:
        return []
    return st.session_state.session_docs.get(sid) or []


def set_current_docs(docs: list[dict]) -> None:
    sid = st.session_state.current_session_id
    if sid is not None:
        st.session_state.session_docs[sid] = docs


def docs_loaded(session_id: str) -> bool:
    return st.session_state.session_docs.get(session_id) is not None


def append_doc(doc: dict) -> None:
    """Add a freshly uploaded doc to the in-memory cache (avoids a round-trip)."""
    sid = st.session_state.current_session_id
    if sid is not None:
        st.session_state.session_docs.setdefault(sid, []).append(doc)


# ── Session switching ─────────────────────────────────────────────────────────
def set_current_session(session_id: str) -> None:
    st.session_state.current_session_id = session_id
    if session_id not in st.session_state.session_messages:
        st.session_state.session_messages[session_id] = None
    if session_id not in st.session_state.session_docs:
        st.session_state.session_docs[session_id] = None