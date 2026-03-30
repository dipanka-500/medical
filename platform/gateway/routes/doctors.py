"""
Doctor endpoints — production-grade with patient link management.

Features:
    - Patient search with wildcard sanitization
    - Patient linking with audit trail
    - Dashboard with optimized queries
    - Data access verification
    - Pydantic response models
    - PHI access audit logging
"""

from __future__ import annotations

import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user, require_role
from config import settings
from db.models import (
    AuditAction, DoctorPatientLink, DoctorProfile, PatientProfile,
    PatientRecord, User, UserRole,
)
from db.session import get_db
from security.audit import AuditService
from security.input_validator import sanitize_search_query

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Response Models ──────────────────────────────────────────────────────

class DoctorProfileResponse(BaseModel):
    id: str
    user_id: str
    specialization: str | None = None
    license_number: str | None = None
    hospital_id: str | None = None
    is_verified: bool = False
    patient_count: int = 0

class PatientSummary(BaseModel):
    patient_id: str
    full_name: str | None = None
    last_visit: str | None = None
    record_count: int = 0

class DashboardResponse(BaseModel):
    total_patients: int = 0
    total_records: int = 0
    recent_patients: list[PatientSummary] = []
    record_type_distribution: dict[str, int] = {}
    recent_records: list[dict[str, Any]] = []


class LinkPatientRequest(BaseModel):
    patient_email: str = Field(..., min_length=3, max_length=320)
    notes: str | None = Field(default=None, max_length=2000)


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format",
        )


async def _get_doctor_profile(
    db: AsyncSession, user_id: UUID,
) -> DoctorProfile:
    """Get doctor profile or 404."""
    result = await db.execute(
        select(DoctorProfile).where(DoctorProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Doctor profile not found",
        )
    return profile


async def _verify_patient_access(
    db: AsyncSession, doctor_profile_id: UUID, patient_profile_id: UUID,
) -> bool:
    """Verify doctor has an active link to this patient."""
    result = await db.execute(
        select(DoctorPatientLink).where(
            and_(
                DoctorPatientLink.doctor_id == doctor_profile_id,
                DoctorPatientLink.patient_id == patient_profile_id,
                DoctorPatientLink.is_active == True,  # noqa: E712
            )
        )
    )
    return result.scalar_one_or_none() is not None


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get(
    "/me/profile",
    response_model=DoctorProfileResponse,
    dependencies=[Depends(require_role(UserRole.DOCTOR, UserRole.ADMIN))],
)
async def get_my_doctor_profile(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get the current doctor's profile."""
    profile = await _get_doctor_profile(db, user.id)

    # Count linked patients
    count_result = await db.execute(
        select(func.count(DoctorPatientLink.id)).where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.is_active == True,  # noqa: E712
            )
        )
    )
    patient_count = count_result.scalar() or 0

    return DoctorProfileResponse(
        id=str(profile.id),
        user_id=str(profile.user_id),
        specialization=getattr(profile, "specialization", None),
        license_number=getattr(profile, "license_number", None),
        hospital_id=str(profile.hospital_id) if getattr(profile, "hospital_id", None) else None,
        is_verified=getattr(profile, "is_verified", False),
        patient_count=patient_count,
    )


@router.get(
    "/me/patients",
    dependencies=[Depends(require_role(UserRole.DOCTOR, UserRole.ADMIN))],
)
async def list_my_patients(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    search: str | None = Query(default=None, max_length=100),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List patients linked to this doctor with optional search."""
    profile = await _get_doctor_profile(db, user.id)

    # Base query: linked patients
    query = (
        select(PatientProfile, User)
        .join(DoctorPatientLink, DoctorPatientLink.patient_id == PatientProfile.id)
        .join(User, User.id == PatientProfile.user_id)
        .where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.is_active == True,  # noqa: E712
            )
        )
    )

    # Search filter with wildcard sanitization
    if search:
        clean_search = sanitize_search_query(search)
        query = query.where(
            or_(
                User.full_name.ilike(f"%{clean_search}%"),
                User.email.ilike(f"%{clean_search}%"),
            )
        )

    # Execute with pagination
    result = await db.execute(
        query.order_by(User.full_name.asc())
        .offset(offset).limit(limit)
    )
    rows = result.all()

    return {
        "patients": [
            {
                "patient_id": str(pp.id),
                "user_id": str(u.id),
                "full_name": u.full_name,
                "email": u.email,
                "gender": getattr(pp, "gender", None),
                "blood_type": getattr(pp, "blood_type", None),
            }
            for pp, u in rows
        ],
        "limit": limit,
        "offset": offset,
    }


@router.post(
    "/me/patients/link",
    dependencies=[Depends(require_role(UserRole.DOCTOR, UserRole.ADMIN))],
)
async def link_patient_by_email(
    body: LinkPatientRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Link a patient to this doctor using the patient's account email.

    This matches the frontend dashboard flow and keeps patient identifiers out
    of manually typed URLs/forms.
    """
    profile = await _get_doctor_profile(db, user.id)

    patient_result = await db.execute(
        select(PatientProfile, User)
        .join(User, User.id == PatientProfile.user_id)
        .where(func.lower(User.email) == body.patient_email.lower())
    )
    row = patient_result.first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )

    patient, patient_user = row

    existing = await db.execute(
        select(DoctorPatientLink).where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.patient_id == patient.id,
            )
        )
    )
    link = existing.scalar_one_or_none()

    if link:
        if link.is_active:
            return {
                "message": "Patient already linked",
                "patient_id": str(patient.id),
                "patient_email": patient_user.email,
            }
        link.is_active = True
        if body.notes is not None:
            link.notes = body.notes
    else:
        db.add(
            DoctorPatientLink(
                doctor_id=profile.id,
                patient_id=patient.id,
                notes=body.notes,
                is_active=True,
            )
        )

    await AuditService.log_phi_access(
        db, user.id, str(patient.id),
        action=AuditAction.LINK_PATIENT,
        resource_type="doctor_patient_link",
        resource_id=str(profile.id),
        ip_address=request.client.host if request.client else None,
        clinical_justification="Doctor linked patient by account email",
    )

    return {
        "message": "Patient linked successfully",
        "patient_id": str(patient.id),
        "patient_email": patient_user.email,
    }


@router.post(
    "/me/patients/{patient_id}/link",
    dependencies=[Depends(require_role(UserRole.DOCTOR, UserRole.ADMIN))],
)
async def link_patient(
    patient_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Link a patient to this doctor's care."""
    pid = _safe_uuid(patient_id)
    profile = await _get_doctor_profile(db, user.id)

    # Verify patient exists
    patient_result = await db.execute(
        select(PatientProfile).where(PatientProfile.id == pid)
    )
    patient = patient_result.scalar_one_or_none()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )

    # Check existing link
    existing = await db.execute(
        select(DoctorPatientLink).where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.patient_id == pid,
            )
        )
    )
    link = existing.scalar_one_or_none()

    if link:
        if link.is_active:
            return {"message": "Patient already linked"}
        # Reactivate
        link.is_active = True
    else:
        new_link = DoctorPatientLink(
            doctor_id=profile.id,
            patient_id=pid,
            is_active=True,
        )
        db.add(new_link)

    # Audit
    await AuditService.log_phi_access(
        db, user.id, str(pid),
        action=AuditAction.LINK_PATIENT,
        resource_type="doctor_patient_link",
        resource_id=str(profile.id),
        ip_address=request.client.host if request.client else None,
    )

    return {"message": "Patient linked successfully"}


@router.get(
    "/me/patients/{patient_id}/records",
    dependencies=[Depends(require_role(UserRole.DOCTOR, UserRole.ADMIN))],
)
async def get_patient_records(
    patient_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """View a linked patient's medical records."""
    pid = _safe_uuid(patient_id)
    profile = await _get_doctor_profile(db, user.id)

    # Verify access
    if not await _verify_patient_access(db, profile.id, pid):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No active link to this patient",
        )

    result = await db.execute(
        select(PatientRecord).where(
            PatientRecord.patient_id == pid
        ).order_by(PatientRecord.created_at.desc())
        .offset(offset).limit(limit)
    )
    records = result.scalars().all()

    # Audit PHI access
    await AuditService.log_phi_access(
        db, user.id, str(pid),
        action=AuditAction.VIEW_RECORD,
        resource_type="patient_records",
        ip_address=request.client.host if request.client else None,
        clinical_justification="Doctor viewing linked patient records",
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


@router.get(
    "/me/dashboard",
    response_model=DashboardResponse,
    dependencies=[Depends(require_role(UserRole.DOCTOR, UserRole.ADMIN))],
)
async def get_dashboard(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Doctor dashboard — summary statistics."""
    profile = await _get_doctor_profile(db, user.id)

    # Total active patients
    patient_count_result = await db.execute(
        select(func.count(DoctorPatientLink.id)).where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.is_active == True,  # noqa: E712
            )
        )
    )
    total_patients = patient_count_result.scalar() or 0

    # Total records across all linked patients
    record_count_result = await db.execute(
        select(func.count(PatientRecord.id))
        .join(DoctorPatientLink, DoctorPatientLink.patient_id == PatientRecord.patient_id)
        .where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.is_active == True,  # noqa: E712
            )
        )
    )
    total_records = record_count_result.scalar() or 0

    # Record type distribution (single query, no N+1)
    type_dist_result = await db.execute(
        select(PatientRecord.record_type, func.count(PatientRecord.id))
        .join(DoctorPatientLink, DoctorPatientLink.patient_id == PatientRecord.patient_id)
        .where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.is_active == True,  # noqa: E712
            )
        )
        .group_by(PatientRecord.record_type)
    )
    record_type_distribution = {
        str(row[0] or "unknown"): row[1] for row in type_dist_result.all()
    }

    # Recent patients (single join query, no N+1)
    recent_patients_result = await db.execute(
        select(PatientProfile, User)
        .join(DoctorPatientLink, DoctorPatientLink.patient_id == PatientProfile.id)
        .join(User, User.id == PatientProfile.user_id)
        .where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.is_active == True,  # noqa: E712
            )
        )
        .order_by(DoctorPatientLink.created_at.desc())
        .limit(5)
    )
    recent_patients = [
        PatientSummary(
            patient_id=str(pp.id),
            full_name=u.full_name,
        )
        for pp, u in recent_patients_result.all()
    ]

    # Recent records across all linked patients
    recent_records_result = await db.execute(
        select(PatientRecord)
        .join(DoctorPatientLink, DoctorPatientLink.patient_id == PatientRecord.patient_id)
        .where(
            and_(
                DoctorPatientLink.doctor_id == profile.id,
                DoctorPatientLink.is_active == True,  # noqa: E712
            )
        )
        .order_by(PatientRecord.created_at.desc())
        .limit(10)
    )
    recent_records = [
        {
            "id": str(r.id),
            "title": getattr(r, "title", None) or "Untitled",
            "record_type": getattr(r, "record_type", None),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent_records_result.scalars().all()
    ]

    return DashboardResponse(
        total_patients=total_patients,
        total_records=total_records,
        recent_patients=recent_patients,
        record_type_distribution=record_type_distribution,
        recent_records=recent_records,
    )
