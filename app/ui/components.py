"""Reusable rendering helpers for the chat UI."""
from __future__ import annotations

import streamlit as st

ROUTE_LABELS = {
    "pdf":  ("📄 PDF",       "#6366f1"),
    "web":  ("🌐 Web",       "#0ea5e9"),
    "both": ("🧩 PDF + Web", "#8b5cf6"),
    "chat": ("💬 Chat",      "#10b981"),  # NEW: conversational route
}


def render_badges(route: str | None, score: float | None) -> None:
    """Small pill badges showing which route the supervisor took and the
    critic's score, rendered under an assistant message.
    Chat-route messages have no score badge (no critic runs for chat)."""
    if not route and score is None:
        return

    parts: list[str] = []

    if route:
        label, color = ROUTE_LABELS.get(route, (route, "#6b7280"))
        parts.append(
            f"<span style='background:{color}1a;color:{color};"
            f"padding:2px 10px;border-radius:999px;font-size:0.75rem;"
            f"font-weight:600;margin-right:6px;display:inline-block;"
            f"margin-top:4px;'>{label}</span>"
        )

    # Only show score badge when a score actually exists (not for chat route).
    if score is not None:
        if score >= 8:
            score_color = "#16a34a"
        elif score >= 5:
            score_color = "#d97706"
        else:
            score_color = "#dc2626"
        parts.append(
            f"<span style='background:{score_color}1a;color:{score_color};"
            f"padding:2px 10px;border-radius:999px;font-size:0.75rem;"
            f"font-weight:600;display:inline-block;margin-top:4px;'>"
            f"★ {score:.1f}/10</span>"
        )

    if parts:
        st.markdown("".join(parts), unsafe_allow_html=True)


def render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message.get("content", ""))
        if role == "assistant":
            render_badges(message.get("route"), message.get("score"))   