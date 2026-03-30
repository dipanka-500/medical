"""
Database models for the MedAI Platform.

Entities:
    - User           (base account for all roles)
    - PatientProfile (patient-specific data)
    - DoctorProfile  (doctor workspace + multi-patient)
    - Hospital       (organisation / team mode)
    - HospitalMember (doctor/staff ↔ hospital link)
    - PatientRecord  (isolated medical records per patient)
    - ChatSession    (conversation history)
    - ChatMessage    (individual messages)
    - FileUpload     (uploaded documents / images)
    - AuditLog       (who accessed what, when)
    - Subscription   (free / paid tier tracking)
    - RefreshToken   (JWT refresh token storage)
    - DoctorPatientLink (doctor ↔ patient link)
    - APIKey          (API key for programmatic access)
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship, synonym

from .base import Base, TimestampMixin, UUIDPrimaryKeyMixin


# ── Enums ────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    HOSPITAL_ADMIN = "hospital_admin"
    STAFF = "staff"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


class SubscriptionTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class RecordType(str, enum.Enum):
    REPORT = "report"
    XRAY = "xray"
    MRI = "mri"
    CT_SCAN = "ct_scan"
    LAB_RESULT = "lab_result"
    PRESCRIPTION = "prescription"
    VIDEO = "video"
    OTHER = "other"


class AuditAction(str, enum.Enum):
    # Auth actions
    LOGIN = "login"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    REGISTER = "register"
    PASSWORD_CHANGE = "password_change"

    # Patient/record actions
    VIEW_PATIENT = "view_patient"
    VIEW_RECORD = "view_record"
    CREATE_RECORD = "create_record"
    UPDATE_RECORD = "update_record"
    DELETE_RECORD = "delete_record"

    # Doctor-patient actions
    LINK_PATIENT = "link_patient"
    UNLINK_PATIENT = "unlink_patient"

    # AI actions
    AI_QUERY = "ai_query"
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"
    EXPORT_DATA = "export_data"

    # Settings/admin actions
    SETTINGS_CHANGE = "settings_change"
    UPDATE_SETTINGS = "update_settings"
    CREATE_HOSPITAL = "create_hospital"
    DELETE_HOSPITAL = "delete_hospital"


# ── User ─────────────────────────────────────────────────────────────────────

class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    full_name: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False, default=UserRole.PATIENT)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 2FA
    totp_secret: Mapped[str | None] = mapped_column(String(64))
    two_factor_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # Alias for auth service compatibility (maps to same DB column)
    two_factor_secret = synonym("totp_secret")

    # Account lockout (referenced by auth/service.py LockoutService)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failed_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Password history (list of old hashes for reuse prevention)
    password_history: Mapped[list | None] = mapped_column(JSONB, default=list)

    # Subscription
    subscription_tier: Mapped[SubscriptionTier] = mapped_column(
        Enum(SubscriptionTier), default=SubscriptionTier.FREE
    )
    daily_queries_used: Mapped[int] = mapped_column(Integer, default=0)
    daily_queries_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    patient_profile: Mapped[PatientProfile | None] = relationship(back_populates="user", uselist=False)
    doctor_profile: Mapped[DoctorProfile | None] = relationship(back_populates="user", uselist=False)
    hospital_memberships: Mapped[list[HospitalMember]] = relationship(back_populates="user")
    chat_sessions: Mapped[list[ChatSession]] = relationship(back_populates="user")
    file_uploads: Mapped[list[FileUpload]] = relationship(back_populates="uploaded_by_user")
    audit_logs: Mapped[list[AuditLog]] = relationship(back_populates="user")
    refresh_tokens: Mapped[list[RefreshToken]] = relationship(back_populates="user")
    api_keys: Mapped[list[APIKey]] = relationship(back_populates="user")


# ── Patient ──────────────────────────────────────────────────────────────────

class PatientProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "patient_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    date_of_birth: Mapped[datetime | None] = mapped_column(DateTime)
    gender: Mapped[str | None] = mapped_column(String(20))
    blood_type: Mapped[str | None] = mapped_column(String(5))
    # FIX: JSONB type annotations match actual content (lists, not dicts)
    allergies: Mapped[list | None] = mapped_column(JSONB, default=list)
    chronic_conditions: Mapped[list | None] = mapped_column(JSONB, default=list)
    emergency_contact: Mapped[dict | None] = mapped_column(JSONB)
    health_timeline: Mapped[list | None] = mapped_column(JSONB, default=list)

    # Relationships
    user: Mapped[User] = relationship(back_populates="patient_profile")
    records: Mapped[list[PatientRecord]] = relationship(
        back_populates="patient", foreign_keys="PatientRecord.patient_id"
    )
    doctor_links: Mapped[list[DoctorPatientLink]] = relationship(back_populates="patient")


# ── Doctor ───────────────────────────────────────────────────────────────────

class DoctorProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "doctor_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    license_number: Mapped[str] = mapped_column(String(100), nullable=False)
    specialization: Mapped[str] = mapped_column(String(200), nullable=False)
    qualifications: Mapped[list | None] = mapped_column(JSONB, default=list)
    practice_name: Mapped[str | None] = mapped_column(String(256))
    hospital_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hospitals.id"),
    )
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verified = synonym("is_verified")

    # Relationships
    user: Mapped[User] = relationship(back_populates="doctor_profile")
    patient_links: Mapped[list[DoctorPatientLink]] = relationship(back_populates="doctor")


class DoctorPatientLink(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Links a doctor to a patient — enables multi-patient workspace."""
    __tablename__ = "doctor_patient_links"
    __table_args__ = (
        UniqueConstraint("doctor_id", "patient_id", name="uq_doctor_patient"),
        Index("ix_dpl_doctor_active", "doctor_id", "is_active"),
        Index("ix_dpl_patient_active", "patient_id", "is_active"),
    )

    doctor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctor_profiles.id", ondelete="CASCADE"), nullable=False
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False
    )
    notes: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    doctor: Mapped[DoctorProfile] = relationship(back_populates="patient_links")
    patient: Mapped[PatientProfile] = relationship(back_populates="doctor_links")


# ── Hospital ─────────────────────────────────────────────────────────────────

class Hospital(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "hospitals"

    name: Mapped[str] = mapped_column(String(512), nullable=False)
    # Both fields for compatibility: registration_number is the canonical DB column,
    # 'code' is an alias used by some route handlers
    registration_number: Mapped[str | None] = mapped_column(String(100), unique=True)
    code: Mapped[str | None] = mapped_column(String(50), unique=True)
    address: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100))
    phone: Mapped[str | None] = mapped_column(String(20))
    settings: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    members: Mapped[list[HospitalMember]] = relationship(back_populates="hospital")


class HospitalMember(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Links users (doctors/staff) to hospitals."""
    __tablename__ = "hospital_members"
    __table_args__ = (
        UniqueConstraint("hospital_id", "user_id", name="uq_hospital_user"),
    )

    hospital_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hospitals.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    permissions: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    hospital: Mapped[Hospital] = relationship(back_populates="members")
    user: Mapped[User] = relationship(back_populates="hospital_memberships")


# ── Patient Records (Isolated per patient) ───────────────────────────────────

class PatientRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "patient_records"
    __table_args__ = (
        Index("ix_patient_records_patient_id", "patient_id"),
        Index("ix_patient_records_type", "record_type"),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False
    )
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    record_type: Mapped[RecordType] = mapped_column(Enum(RecordType), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    ai_analysis: Mapped[dict | None] = mapped_column(JSONB)
    record_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)
    file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("file_uploads.id")
    )
    is_encrypted: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    patient: Mapped[PatientProfile] = relationship(back_populates="records")
    file: Mapped[FileUpload | None] = relationship()


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "chat_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(512), default="New Chat")
    patient_context_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patient_profiles.id")
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    session_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)

    # Relationships
    user: Mapped[User] = relationship(back_populates="chat_sessions")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_id", "session_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    attachments: Mapped[dict | None] = mapped_column(JSONB)
    ai_metadata: Mapped[dict | None] = mapped_column(JSONB)
    tokens_used: Mapped[int | None] = mapped_column(Integer)

    # Relationships
    session: Mapped[ChatSession] = relationship(back_populates="messages")


# ── File Uploads ─────────────────────────────────────────────────────────────

class FileUpload(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "file_uploads"

    uploaded_by_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    is_encrypted: Mapped[bool] = mapped_column(Boolean, default=True)
    scan_status: Mapped[str] = mapped_column(String(20), default="pending")

    # Relationships
    uploaded_by_user: Mapped[User] = relationship(back_populates="file_uploads")


# ── Audit Log ────────────────────────────────────────────────────────────────

class AuditLog(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_timestamp", "timestamp"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    action: Mapped[AuditAction] = mapped_column(Enum(AuditAction), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(100))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict | None] = mapped_column(JSONB)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(512))
    integrity_hash: Mapped[str | None] = mapped_column(String(64))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped[User] = relationship(back_populates="audit_logs")


# ── Refresh Token ────────────────────────────────────────────────────────────

class RefreshToken(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "refresh_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    # Alias for auth service compatibility (uses 'revoked')
    revoked = synonym("is_revoked")
    device_info: Mapped[str | None] = mapped_column(String(512))

    # Relationships
    user: Mapped[User] = relationship(back_populates="refresh_tokens")


# ── API Key ─────────────────────────────────────────────────────────────

class APIKey(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """API keys for programmatic access — supports rotation and scoped permissions."""
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_user_id", "user_id"),
        Index("ix_api_keys_key_prefix", "key_prefix"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    scopes: Mapped[list | None] = mapped_column(JSONB, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_ip: Mapped[str | None] = mapped_column(String(45))
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Relationships
    user: Mapped[User] = relationship(back_populates="api_keys")
