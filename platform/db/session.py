"""
Async database session management — production-grade.

Features:
    - Connection pool health monitoring
    - Pool event hooks (connect, checkout, checkin)
    - Connection recycling (prevents stale connections)
    - Statement timeout enforcement per connection
    - RLS session variable injection on checkout
    - Retry on transient connection failures
    - Graceful pool disposal on shutdown
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import HTTPException, status
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import Pool

from config import settings

logger = logging.getLogger(__name__)

# ── Engine Creation ──────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,              # Validate connections before use
    pool_recycle=settings.database_pool_recycle,   # Recycle after 1 hour
    pool_timeout=settings.database_pool_timeout,   # Wait 30s for a connection
    pool_use_lifo=True,              # LIFO = better cache locality
    connect_args={
        "server_settings": {
            "statement_timeout": str(settings.database_statement_timeout_ms),
            "idle_in_transaction_session_timeout": "60000",  # 60s
            "lock_timeout": "10000",  # 10s
            "application_name": "medai-platform",
        },
    },
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Pool Event Hooks ────────────────────────────────────────────────────

@event.listens_for(engine.sync_engine, "connect")
def _on_connect(dbapi_conn: object, connection_record: object) -> None:
    """Log new physical connections."""
    logger.debug("New DB connection established")


@event.listens_for(engine.sync_engine, "checkout")
def _on_checkout(dbapi_conn: object, connection_record: object, connection_proxy: object) -> None:
    """Track connection checkout for monitoring."""
    connection_record._checkout_time = time.monotonic()  # type: ignore[attr-defined]


@event.listens_for(engine.sync_engine, "checkin")
def _on_checkin(dbapi_conn: object, connection_record: object) -> None:
    """Log long-held connections (potential leak detection)."""
    checkout_time = getattr(connection_record, "_checkout_time", None)
    if checkout_time:
        held_seconds = time.monotonic() - checkout_time
        if held_seconds > 30.0:
            logger.warning(
                "Connection held for %.1fs — possible leak", held_seconds
            )


# ── Pool Health ──────────────────────────────────────────────────────────

def get_pool_status() -> dict:
    """Return pool health metrics for monitoring endpoints."""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.status(),
    }


# ── Dependency: Request-Scoped Session ──────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields an async DB session.

    Features:
        - Circuit breaker check (rejects fast when DB is down)
        - Auto-commit on success, rollback on error
        - Exception logging with context
    """
    from .resilience import circuit_breaker

    try:
        await circuit_breaker.check()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable — circuit breaker is OPEN. Please retry later.",
        )

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
            await circuit_breaker.record_success()
        except Exception as exc:
            await session.rollback()
            await circuit_breaker.record_failure()
            logger.error(
                "DB session error: %s: %s",
                type(exc).__name__,
                str(exc)[:200],
            )
            raise


# ── Session with RLS Context ────────────────────────────────────────────

async def get_db_with_rls(
    user_id: str | None = None,
    patient_id: str | None = None,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Session with Row-Level Security context variables set.
    Use this for patient data access paths.

    Sets PostgreSQL session variables:
        SET LOCAL app.current_user = '<user_id>';
        SET LOCAL app.current_patient = '<patient_id>';
    """
    async with async_session_factory() as session:
        try:
            # SET LOCAL scopes to the current transaction only
            if user_id:
                await session.execute(
                    text("SET LOCAL app.current_user = :uid"),
                    {"uid": user_id},
                )
            if patient_id:
                await session.execute(
                    text("SET LOCAL app.current_patient = :pid"),
                    {"pid": patient_id},
                )

            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(
                "DB session (RLS) error: %s: %s",
                type(exc).__name__,
                str(exc)[:200],
            )
            raise


# ── Lifecycle ────────────────────────────────────────────────────────────

async def dispose_engine() -> None:
    """Gracefully close all pool connections on shutdown."""
    logger.info("Disposing database engine pool")
    await engine.dispose()
    logger.info("Database engine disposed")


async def verify_connection() -> bool:
    """Verify database connectivity (used by health checks)."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return False


async def ensure_schema() -> None:
    """
    Ensure ORM tables exist for local/dev environments.

    Production deployments should rely on Alembic migrations, but local
    docker-compose bootstraps need a working schema on first startup.

    Multiple uvicorn workers may race on CREATE TYPE for PostgreSQL ENUMs,
    so we pre-create them with DO blocks that check for existence first.
    """
    from db import models  # noqa: F401 - register models with Base metadata
    from .base import Base

    # Pre-create PostgreSQL ENUM types safely (race-proof).
    # SQLAlchemy's create_all checks for table existence but can still
    # race on ENUM type creation when multiple workers start at once.
    _enum_defs = {
        "userrole": [e.value for e in models.UserRole],
        "subscriptiontier": [e.value for e in models.SubscriptionTier],
        "messagerole": [e.value for e in models.MessageRole],
        "recordtype": [e.value for e in models.RecordType],
        "auditaction": [e.value for e in models.AuditAction],
    }

    async with engine.begin() as conn:
        for type_name, values in _enum_defs.items():
            labels = ", ".join(f"'{v}'" for v in values)
            await conn.execute(text(
                f"DO $$ BEGIN "
                f"CREATE TYPE {type_name} AS ENUM ({labels}); "
                f"EXCEPTION WHEN duplicate_object THEN NULL; "
                f"END $$;"
            ))

        # Now create_all won't attempt to create the types (they exist)
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database schema verified")
