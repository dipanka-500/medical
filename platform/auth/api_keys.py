"""
API Key management service — create, validate, rotate, and revoke API keys.

Security design:
    - Keys are generated as ``mdk_`` + secrets.token_urlsafe(32)
    - Only the bcrypt hash is stored; the plaintext is returned ONCE at creation
    - The first 8 characters (``key_prefix``) allow lookup without scanning all rows
    - Rotation keeps the old key alive for a configurable grace period
    - Scopes restrict which endpoints a key may access
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import APIKey

logger = logging.getLogger(__name__)


class APIKeyService:
    """Stateless service for API key lifecycle management."""

    # ── Create ────────────────────────────────────────────────────────────

    @staticmethod
    async def create_key(
        db: AsyncSession,
        user_id: UUID,
        name: str,
        scopes: list[str],
        expires_in_days: int = 365,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key for *user_id*.

        Returns:
            (plaintext_key, api_key_model) — the plaintext is shown **once**.
        """
        raw_key = "mdk_" + secrets.token_urlsafe(32)
        key_prefix = raw_key[:8]
        key_hash = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()

        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            scopes=scopes,
            expires_at=expires_at,
            version=1,
        )
        db.add(api_key)
        await db.flush()

        logger.info("API key created: prefix=%s user=%s name=%s", key_prefix, user_id, name)
        return raw_key, api_key

    # ── Validate ──────────────────────────────────────────────────────────

    @staticmethod
    async def validate_key(
        db: AsyncSession,
        raw_key: str,
        client_ip: str | None = None,
    ) -> APIKey | None:
        """
        Look up and verify *raw_key*.

        Returns the ``APIKey`` model if the key is valid, active, and not
        expired; otherwise ``None``.
        """
        if not raw_key.startswith("mdk_"):
            return None

        key_prefix = raw_key[:8]

        result = await db.execute(
            select(APIKey).where(
                APIKey.key_prefix == key_prefix,
                APIKey.is_active.is_(True),
            )
        )
        candidates = result.scalars().all()

        for candidate in candidates:
            if bcrypt.checkpw(raw_key.encode(), candidate.key_hash.encode()):
                # Check expiry
                if candidate.expires_at and candidate.expires_at < datetime.now(timezone.utc):
                    logger.info("API key expired: prefix=%s", key_prefix)
                    return None

                # Update last-used metadata
                candidate.last_used_at = datetime.now(timezone.utc)
                if client_ip:
                    candidate.last_used_ip = client_ip
                await db.flush()

                return candidate

        return None

    # ── Rotate ────────────────────────────────────────────────────────────

    @staticmethod
    async def rotate_key(
        db: AsyncSession,
        old_key_id: UUID,
        user_id: UUID,
        grace_period_hours: int = 24,
    ) -> tuple[str, APIKey]:
        """
        Rotate an existing key: create a replacement and schedule the old
        key for expiry after *grace_period_hours*.

        Returns:
            (new_plaintext_key, new_api_key_model)

        Raises:
            ValueError: if the old key is not found or does not belong to *user_id*.
        """
        result = await db.execute(
            select(APIKey).where(
                APIKey.id == old_key_id,
                APIKey.user_id == user_id,
            )
        )
        old_key = result.scalar_one_or_none()

        if not old_key:
            raise ValueError("API key not found")

        if not old_key.is_active:
            raise ValueError("Cannot rotate a revoked key")

        # Schedule old key to expire after grace period
        old_key.expires_at = datetime.now(timezone.utc) + timedelta(hours=grace_period_hours)

        # Create new key inheriting name, scopes, and incremented version
        new_version = old_key.version + 1
        raw_key, new_api_key = await APIKeyService.create_key(
            db,
            user_id=user_id,
            name=old_key.name,
            scopes=old_key.scopes or [],
        )
        new_api_key.version = new_version
        await db.flush()

        logger.info(
            "API key rotated: old_prefix=%s new_prefix=%s version=%d grace_hours=%d",
            old_key.key_prefix, new_api_key.key_prefix, new_version, grace_period_hours,
        )
        return raw_key, new_api_key

    # ── Revoke ────────────────────────────────────────────────────────────

    @staticmethod
    async def revoke_key(
        db: AsyncSession,
        key_id: UUID,
        user_id: UUID,
    ) -> bool:
        """
        Revoke (deactivate) an API key.

        Returns ``True`` if the key was found and revoked, ``False`` otherwise.
        """
        result = await db.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.user_id == user_id,
            )
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            return False

        api_key.is_active = False
        await db.flush()

        logger.info("API key revoked: prefix=%s user=%s", api_key.key_prefix, user_id)
        return True

    # ── List ──────────────────────────────────────────────────────────────

    @staticmethod
    async def list_keys(
        db: AsyncSession,
        user_id: UUID,
    ) -> list[dict]:
        """
        Return metadata for all API keys belonging to *user_id*.

        The key hash is **never** included in the response.
        """
        result = await db.execute(
            select(APIKey).where(APIKey.user_id == user_id).order_by(APIKey.created_at.desc())
        )
        keys = result.scalars().all()

        return [
            {
                "id": str(k.id),
                "key_prefix": k.key_prefix,
                "name": k.name,
                "scopes": k.scopes,
                "is_active": k.is_active,
                "version": k.version,
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "updated_at": k.updated_at.isoformat() if k.updated_at else None,
                "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                "last_used_ip": k.last_used_ip,
            }
            for k in keys
        ]
