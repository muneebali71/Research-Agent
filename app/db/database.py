"""Async SQLAlchemy engine, session factory, and table creation.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import DATABASE_URL


class Base(DeclarativeBase):
    """Base class for all ORM models."""


engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables if they do not exist.

    Fine for development. Switch to Alembic migrations once the schema starts
    evolving in production.
    """
    # Import models so they register on Base.metadata before create_all.
    from app.db import model  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
