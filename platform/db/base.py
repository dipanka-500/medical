"""
SQLAlchemy declarative base with production-grade mixins.

Mixins available:
    - UUIDPrimaryKeyMixin : UUID v4 primary key
    - TimestampMixin      : created_at / updated_at
    - SoftDeleteMixin     : deleted_at + is_deleted (preserves audit trail)
    - VersionedMixin      : version column for optimistic locking
    - TenantMixin         : tenant_id for multi-hospital isolation

All models inherit `Base` which provides:
    - Auto __repr__ for debugging
    - Naming convention for constraints (PostgreSQL best practice)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, MetaData, String, event, func, inspect
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column


# ── Naming Convention (PostgreSQL best practice) ─────────────────────────
# Ensures all constraints have predictable, deterministic names.
# This is CRITICAL for Alembic auto-generated migrations to work correctly.

_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """
    Base class for all database models.

    Features:
        - Consistent naming convention for all constraints
        - Auto __repr__ for debugging (shows PK + first 3 columns)
        - Registry for model discovery
    """

    metadata = MetaData(naming_convention=_NAMING_CONVENTION)

    # Columns that must NEVER appear in __repr__ (PHI / secrets)
    _repr_exclude: set[str] = {
        "email", "full_name", "phone", "phone_number", "address",
        "date_of_birth", "dob", "ssn", "password_hash", "totp_secret",
        "two_factor_secret", "token_hash", "blood_type", "emergency_contact",
        "medical_history", "allergies", "medications", "diagnosis",
    }

    def __repr__(self) -> str:
        """Auto-generate repr from PK columns + safe non-PK attributes."""
        cls_name = type(self).__name__
        mapper = inspect(type(self))
        pk_cols = [col.key for col in mapper.primary_key]

        parts = []
        for key in pk_cols:
            val = getattr(self, key, None)
            parts.append(f"{key}={val!r}")

        # Add a few non-PK, non-sensitive columns for context
        non_pk = [
            col.key for col in mapper.column_attrs
            if col.key not in pk_cols and col.key not in self._repr_exclude
        ][:3]
        for key in non_pk:
            val = getattr(self, key, None)
            # Truncate long strings
            if isinstance(val, str) and len(val) > 50:
                val = val[:47] + "..."
            parts.append(f"{key}={val!r}")

        return f"<{cls_name}({', '.join(parts)})>"


# ── Mixins ───────────────────────────────────────────────────────────────

class UUIDPrimaryKeyMixin:
    """UUID v4 primary key — globally unique, no sequence contention."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """Adds created_at and updated_at columns with automatic management."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """
    Soft-delete support — marks records as deleted instead of removing them.

    CRITICAL for medical data: records must NEVER be truly deleted
    for regulatory compliance (HIPAA, GDPR audit trail).

    Usage in queries:
        select(Model).where(Model.is_deleted == False)  # noqa
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        index=True,
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        index=True,
    )
    deleted_by: Mapped[str | None] = mapped_column(
        String(64),
        default=None,
    )

    def soft_delete(self, deleted_by: str | None = None) -> None:
        """Mark this record as deleted."""
        self.is_deleted = True  # type: ignore[assignment]
        self.deleted_at = datetime.now(timezone.utc)  # type: ignore[assignment]
        self.deleted_by = deleted_by  # type: ignore[assignment]

    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.is_deleted = False  # type: ignore[assignment]
        self.deleted_at = None  # type: ignore[assignment]
        self.deleted_by = None  # type: ignore[assignment]


class VersionedMixin:
    """
    Optimistic locking via version column.

    Prevents race conditions when two users update the same record.
    SQLAlchemy will raise StaleDataError on version mismatch.

    Usage:
        __mapper_args__ = {"version_id_col": version}
    """

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        server_default="1",
    )


class TenantMixin:
    """
    Multi-tenant isolation via tenant_id.

    When scaling to multi-hospital SaaS, this column enables
    data isolation at the application layer. Combined with RLS,
    it provides defense-in-depth.
    """

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        index=True,
        default=None,
        comment="Hospital/organization ID for multi-tenant isolation",
    )
