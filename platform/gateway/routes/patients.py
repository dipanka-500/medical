"""
Patient endpoints — production-grade with data isolation.

Features:
    - Pydantic response models
    - Patient data isolation (only own data)
    - Health timeline with validated events
    - Encrypted field support
    - Audit logging for PHI access
    - Pagination with limit bounds
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, cast, select, update
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user, require_role
from config import settings
from db.models import (
    AuditAction, PatientProfile, PatientRecord, User, UserRole,
)
from db.session import get_db
from security.audit import AuditService

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Response Models ──────────────────────────────────────────────────────

class PatientProfileResponse(BaseModel):
    id: str
    user_id: str
    date_of_birth: str | None = None
    gender: str | None = None
    blood_type: str | None = None
    allergies: list[str] | None = None
    chronic_conditions: list[str] | None = None
    emergency_contact: dict[str, str] | None = None
    created_at: str | None = None

class PatientRecordResponse(BaseModel):
    id: str
    record_type: str | None = None
    title: str | None = None
    summary: str | None = None
    created_at: str | None = None

class TimelineEvent(BaseModel):
    """Validated timeline event schema."""
    event_type: str = Field(..., max_length=50, pattern=r"^[a-z_]+$")
    description: str = Field(..., max_length=1000)
    severity: str = Field(default="info", pattern=r"^(info|warning|critical)$")
    metadata: dict[str, Any] | None = None


# ── Helpers ──────────────────────────────────────────────────────────────

async def _get_patient_profile(
    db: AsyncSession, user_id: UUID,
) -> PatientProfile:
    """Get patient profile or 404 — enforces data isolation."""
    result = await db.execute(
        select(PatientProfile).where(PatientProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient profile not found",
        )
    return profile


def _safe_uuid(value: str) -> UUID:
    """Parse UUID or raise 400."""
    try:
        return UUID(value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format",
        )


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get("/me/profile", response_model=PatientProfileResponse)
async def get_my_profile(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get the current patient's profile."""
    profile = await _get_patient_profile(db, user.id)

    # Audit PHI access
    await AuditService.log_phi_access(
        db, user.id, str(user.id),
        action=AuditAction.VIEW_RECORD,
        resource_type="patient_profile",
        resource_id=str(profile.id),
        ip_address=request.client.host if request.client else None,
    )

    return _profile_to_response(profile)


class UpdateProfileRequest(BaseModel):
    """Request body for profile updates — keeps PHI out of query params / URLs."""
    date_of_birth: str | None = None
    gender: str | None = None
    blood_type: str | None = None
    allergies: list[str] | None = None
    emergency_contact: dict[str, str] | None = None


@router.put("/me/profile")
async def update_my_profile(
    body: UpdateProfileRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update the current patient's profile."""
    profile = await _get_patient_profile(db, user.id)

    update_values: dict[str, Any] = {}
    if body.date_of_birth is not None:
        update_values["date_of_birth"] = body.date_of_birth
    if body.gender is not None:
        update_values["gender"] = body.gender
    if body.blood_type is not None:
        update_values["blood_type"] = body.blood_type
    if body.allergies is not None:
        update_values["allergies"] = body.allergies
    if body.emergency_contact is not None:
        update_values["emergency_contact"] = body.emergency_contact

    if update_values:
        await db.execute(
            update(PatientProfile).where(
                PatientProfile.id == profile.id
            ).values(**update_values)
        )

    # Audit
    await AuditService.log_phi_access(
        db, user.id, str(user.id),
        action=AuditAction.UPDATE_RECORD,
        resource_type="patient_profile",
        resource_id=str(profile.id),
        ip_address=request.client.host if request.client else None,
    )

    return {"message": "Profile updated"}


@router.get("/me/records")
async def get_my_records(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List the current patient's medical records with pagination."""
    profile = await _get_patient_profile(db, user.id)

    result = await db.execute(
        select(PatientRecord).where(
            PatientRecord.patient_id == profile.id
        ).order_by(PatientRecord.created_at.desc())
        .offset(offset).limit(limit)
    )
    records = result.scalars().all()

    # Audit
    await AuditService.log_phi_access(
        db, user.id, str(user.id),
        action=AuditAction.VIEW_RECORD,
        resource_type="patient_records",
        ip_address=request.client.host if request.client else None,
    )

    return {
        "records": [
            {
                "id": str(r.id),
                "record_type": getattr(r, "record_type", None),
                "title": getattr(r, "title", None),
                "summary": getattr(r, "summary", None),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
        "limit": limit,
        "offset": offset,
    }


@router.post("/me/timeline")
async def add_timeline_event(
    event: TimelineEvent,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Add a validated event to the patient's health timeline."""
    profile = await _get_patient_profile(db, user.id)

    # Atomic JSONB append using PostgreSQL
    new_entry = {
        "type": event.event_type,
        "description": event.description,
        "severity": event.severity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": event.metadata or {},
    }

    # Atomic JSONB append — avoids race condition from concurrent read-modify-write
    from sqlalchemy import func as sqlfunc
    await db.execute(
        update(PatientProfile).where(
            PatientProfile.id == profile.id
        ).values(
            health_timeline=sqlfunc.coalesce(
                PatientProfile.health_timeline, cast([], PG_JSONB)
            ) + cast([new_entry], PG_JSONB)
        )
    )

    # Audit
    await AuditService.log_phi_access(
        db, user.id, str(user.id),
        action=AuditAction.UPDATE_RECORD,
        resource_type="health_timeline",
        resource_id=str(profile.id),
        ip_address=request.client.host if request.client else None,
    )

    return {"message": "Timeline event added", "event": new_entry}


# ── Helpers ──────────────────────────────────────────────────────────────

def _profile_to_response(profile: PatientProfile) -> PatientProfileResponse:
    """Convert SQLAlchemy model to Pydantic response."""
    return PatientProfileResponse(
        id=str(profile.id),
        user_id=str(profile.user_id),
        date_of_birth=str(profile.date_of_birth) if getattr(profile, "date_of_birth", None) else None,
        gender=getattr(profile, "gender", None),
        blood_type=getattr(profile, "blood_type", None),
        allergies=getattr(profile, "allergies", None),
        chronic_conditions=getattr(profile, "chronic_conditions", None),
        emergency_contact=getattr(profile, "emergency_contact", None),
        created_at=profile.created_at.isoformat() if profile.created_at else None,
    )
