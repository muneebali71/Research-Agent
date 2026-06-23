"""LangGraph PostgreSQL checkpointer = automatic conversation memory.
"""
from __future__ import annotations

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config import DATABASE_URL_SYNC


def get_checkpointer():
    """Return an async context manager yielding an AsyncPostgresSaver.

    Call ``await checkpointer.setup()`` once to create the checkpoint tables.
    """
    return AsyncPostgresSaver.from_conn_string(DATABASE_URL_SYNC)
