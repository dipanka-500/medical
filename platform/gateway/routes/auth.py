"""Authentication endpoints — register, login, refresh, 2FA, logout, password change, password reset, email verification."""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auth.api_keys import APIKeyService
from auth.dependencies import get_current_user
from auth.service import AuthService, PasswordService, TokenService
from config import settings
from db.models import AuditAction, DoctorProfile, PatientProfile, User, UserRole
from db.session import get_db
from security.audit import AuditService

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response Schemas ───────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=10, max_length=128)
    full_name: str = Field(..., min_length=2, max_length=100)
    role: str = Field(default="patient", pattern=r"^(patient|doctor)$")

class RegisterResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    message: str = "Registration successful."

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)
    device_info: str | None = None

class LoginResponse(BaseModel):
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int | None = None
    requires_2fa: bool = False
    user_id: str | None = None
    user: dict[str, Any] | None = None
    message: str | None = None

class TokenRefreshRequest(BaseModel):
    refresh_token: str

class Verify2FARequest(BaseModel):
    user_id: str
    totp_code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")
    device_info: str | None = Field(default=None, max_length=512)

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=10, max_length=128)

class LogoutRequest(BaseModel):
    refresh_token: str | None = None
    all_devices: bool = False


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    scopes: list[str] = Field(default_factory=list)
    expires_in_days: int = Field(default=365, ge=1, le=3650)


class RotateAPIKeyRequest(BaseModel):
    grace_period_hours: int = Field(default=24, ge=0, le=720)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=10, max_length=128)


class VerifyEmailRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=256)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


# ── Token Store (Redis-backed with in-memory fallback) ────────────────────

# Redis keys: medai:reset:<token> = email, TTL = expiry
#             medai:verify:<token> = email, TTL = expiry
# Falls back to in-memory dicts when Redis is unavailable.
_reset_tokens: dict[str, tuple[str, datetime]] = {}
_verify_tokens: dict[str, tuple[str, datetime]] = {}
_last_token_cleanup: datetime = datetime.min

_RESET_PREFIX = "medai:reset:"
_VERIFY_PREFIX = "medai:verify:"


async def _store_token(request: Request, prefix: str, token: str, email: str, ttl_seconds: int) -> None:
    """Store a token in Redis (preferred) or in-memory fallback."""
    redis = getattr(request.app.state, "redis", None)
    if redis:
        try:
            await redis.setex(f"{prefix}{token}", ttl_seconds, email)
            return
        except Exception:
            logger.debug("Redis token store failed, using in-memory fallback")
    # Fallback
    store = _reset_tokens if prefix == _RESET_PREFIX else _verify_tokens
    store[token] = (email, datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds))
    _cleanup_expired_tokens()


async def _get_token(request: Request, prefix: str, token: str) -> str | None:
    """Retrieve the email for a token from Redis or in-memory fallback."""
    redis = getattr(request.app.state, "redis", None)
    if redis:
        try:
            email = await redis.get(f"{prefix}{token}")
            if email is not None:
                return email if isinstance(email, str) else email.decode()
        except Exception:
            pass
    # Fallback
    store = _reset_tokens if prefix == _RESET_PREFIX else _verify_tokens
    entry = store.get(token)
    if entry:
        email, expires_at = entry
        if datetime.now(timezone.utc) <= expires_at:
            return email
        store.pop(token, None)
    return None


async def _delete_token(request: Request, prefix: str, token: str) -> None:
    """Remove a consumed token."""
    redis = getattr(request.app.state, "redis", None)
    if redis:
        try:
            await redis.delete(f"{prefix}{token}")
        except Exception:
            pass
    store = _reset_tokens if prefix == _RESET_PREFIX else _verify_tokens
    store.pop(token, None)


def _cleanup_expired_tokens() -> None:
    """Sweep expired tokens from in-memory fallback stores."""
    global _last_token_cleanup
    now = datetime.now(timezone.utc)
    if (now - _last_token_cleanup).total_seconds() < 300:
        return
    _last_token_cleanup = now
    for store in (_reset_tokens, _verify_tokens):
        expired_keys = [k for k, (_, exp) in store.items() if exp < now]
        for k in expired_keys:
            store.pop(k, None)

def _get_ip(request: Request) -> str:
    """Extract client IP — prefer X-Forwarded-For (set by middleware/proxy)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ips = [ip.strip() for ip in forwarded.split(",") if ip.strip()]
        if ips:
            return ips[-1]  # Rightmost = proxy-appended, most trusted
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Register a new user account."""
    # Check password strength
    issues = PasswordService.check_password_strength(body.password)
    if issues:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password too weak: {'; '.join(issues)}",
        )

    # Check if email already exists
    existing = await db.execute(
        select(User).where(func.lower(User.email) == body.email.lower())
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create user — only allow self-registration as patient or doctor
    hashed = PasswordService.hash_password(body.password)
    _ALLOWED_REGISTRATION_ROLES = {UserRole.PATIENT, UserRole.DOCTOR}
    try:
        role = UserRole(body.role)
    except ValueError:
        role = UserRole.PATIENT
    if role not in _ALLOWED_REGISTRATION_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-registration is only allowed for patient and doctor roles.",
        )

    user = User(
        email=body.email.lower(),
        hashed_password=hashed,
        full_name=body.full_name,
        role=role,
    )
    db.add(user)
    await db.flush()

    # Create role-specific profile so downstream endpoints work immediately
    if role == UserRole.PATIENT:
        db.add(PatientProfile(user_id=user.id))
    elif role == UserRole.DOCTOR:
        db.add(DoctorProfile(
            user_id=user.id,
            license_number="PENDING",
            specialization="General",
        ))
        await db.flush()

    # Audit
    await AuditService.log(
        db, user.id, AuditAction.REGISTER,
        details={"role": role.value},
        ip_address=_get_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    return RegisterResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        message=(
            "Registration successful. Please verify your email."
            if settings.require_email_verification
            else "Registration successful."
        ),
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Authenticate and receive JWT tokens."""
    try:
        result = await AuthService.login(
            db,
            email=body.email,
            password=body.password,
            ip_address=_get_ip(request),
            user_agent=request.headers.get("user-agent"),
            device_info=body.device_info,
        )
        return LoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/2fa/verify", response_model=LoginResponse)
async def verify_2fa(
    body: Verify2FARequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Complete login with 2FA verification."""
    from uuid import UUID
    try:
        user_id = UUID(body.user_id)
        result = await AuthService.verify_2fa(
            db, user_id, body.totp_code,
            ip_address=_get_ip(request),
            device_info=body.device_info,
            user_agent=request.headers.get("user-agent"),
        )
        return LoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    body: TokenRefreshRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Rotate refresh token for a new access token."""
    try:
        result = await AuthService.refresh_tokens(
            db, body.refresh_token,
            ip_address=_get_ip(request),
        )
        return LoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    body: LogoutRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Logout — revoke current or all sessions."""
    redis_client = getattr(request.app.state, "redis", None)
    result = await AuthService.logout(
        db,
        user_id=user.id,
        refresh_token=body.refresh_token,
        all_devices=body.all_devices,
        ip_address=_get_ip(request),
        redis=redis_client,
    )
    return result


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    body: ChangePasswordRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Change password — requires current password, revokes all sessions."""
    try:
        redis_client = getattr(request.app.state, "redis", None)
        result = await AuthService.change_password(
            db,
            user_id=user.id,
            current_password=body.current_password,
            new_password=body.new_password,
            ip_address=_get_ip(request),
            redis=redis_client,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/me")
async def get_current_profile(
    user: Annotated[User, Depends(get_current_user)],
):
    """Get the current authenticated user's profile."""
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role.value,
        "is_verified": user.is_verified,
        "two_factor_enabled": user.two_factor_enabled,
        "subscription_tier": user.subscription_tier.value if hasattr(user.subscription_tier, "value") else str(user.subscription_tier),
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }


# ── Password Reset ──────────────────────────────────────────────────────

@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    body: ForgotPasswordRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Request a password reset link.

    Always returns 200 to prevent email enumeration.
    In production, sends an email with a reset link.
    """
    result = await db.execute(
        select(User).where(func.lower(User.email) == body.email.lower())
    )
    user = result.scalar_one_or_none()

    if user:
        token = secrets.token_urlsafe(48)
        await _store_token(request, _RESET_PREFIX, token, user.email.lower(), ttl_seconds=3600)

        if settings.environment == "development":
            logger.info("Password reset token for %s: %s", user.email, token)

        await AuditService.log(
            db, user.id, AuditAction.PASSWORD_CHANGE,
            resource_type="password_reset_request",
            ip_address=_get_ip(request),
            user_agent=request.headers.get("user-agent"),
        )

    return {"message": "If that email is registered, a password reset link has been sent."}


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    body: ResetPasswordRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Reset password using a valid reset token."""
    email = await _get_token(request, _RESET_PREFIX, body.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Validate password strength
    issues = PasswordService.check_password_strength(body.new_password)
    if issues:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password too weak: {'; '.join(issues)}",
        )

    # Find user and reset password
    result = await db.execute(
        select(User).where(func.lower(User.email) == email)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token",
        )

    new_hash = PasswordService.hash_password(body.new_password)
    history = getattr(user, "password_history", None) or []
    history.append(user.hashed_password)
    history = history[-settings.password_history_count:]

    await db.execute(
        update(User).where(User.id == user.id).values(
            hashed_password=new_hash,
            password_history=history,
        )
    )

    # Invalidate the token (one-time use)
    await _delete_token(request, _RESET_PREFIX, body.token)

    await AuditService.log(
        db, user.id, AuditAction.PASSWORD_CHANGE,
        resource_type="password_reset_complete",
        ip_address=_get_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    return {"message": "Password has been reset. Please log in with your new password."}


# ── Email Verification ──────────────────────────────────────────────────

@router.post("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(
    body: VerifyEmailRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Verify email address using a verification token."""
    email = await _get_token(request, _VERIFY_PREFIX, body.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token",
        )

    result = await db.execute(
        select(User).where(func.lower(User.email) == email)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token",
        )

    await db.execute(
        update(User).where(User.id == user.id).values(
            is_verified=True,
            email_verified_at=datetime.now(timezone.utc),
        )
    )

    await _delete_token(request, _VERIFY_PREFIX, body.token)

    await AuditService.log(
        db, user.id, AuditAction.REGISTER,
        resource_type="email_verified",
        ip_address=_get_ip(request),
    )

    return {"message": "Email verified successfully. You can now log in."}


@router.post("/resend-verification", status_code=status.HTTP_200_OK)
async def resend_verification(
    body: ResendVerificationRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Resend email verification link. Always returns 200 to prevent enumeration."""
    result = await db.execute(
        select(User).where(func.lower(User.email) == body.email.lower())
    )
    user = result.scalar_one_or_none()

    if user and not user.is_verified:
        token = secrets.token_urlsafe(48)
        await _store_token(request, _VERIFY_PREFIX, token, user.email.lower(), ttl_seconds=86400)

        if settings.environment == "development":
            logger.info("Email verification token for %s: %s", user.email, token)

    return {"message": "If that email is registered and unverified, a verification link has been sent."}


# ── API Key Management ───────────────────────────────────────────────────

@router.post("/api-keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: CreateAPIKeyRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Create a new API key.

    The plaintext key is returned **once** in the response.
    Store it securely — it cannot be retrieved again.
    """
    raw_key, api_key = await APIKeyService.create_key(
        db,
        user_id=user.id,
        name=body.name,
        scopes=body.scopes,
        expires_in_days=body.expires_in_days,
    )
    return {
        "key": raw_key,
        "id": str(api_key.id),
        "key_prefix": api_key.key_prefix,
        "name": api_key.name,
        "scopes": api_key.scopes,
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
        "message": "Store this key securely. It will not be shown again.",
    }


@router.get("/api-keys")
async def list_api_keys(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List all API keys for the current user (metadata only, no hashes)."""
    keys = await APIKeyService.list_keys(db, user.id)
    return {"api_keys": keys}


@router.post("/api-keys/{key_id}/rotate")
async def rotate_api_key(
    key_id: str,
    body: RotateAPIKeyRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Rotate an API key — creates a new key and schedules the old one for expiry.

    The old key remains valid for the specified grace period (default 24 hours).
    The new plaintext key is returned **once**.
    """
    from uuid import UUID as _UUID

    try:
        parsed_key_id = _UUID(key_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid key ID format",
        )

    try:
        raw_key, new_api_key = await APIKeyService.rotate_key(
            db,
            old_key_id=parsed_key_id,
            user_id=user.id,
            grace_period_hours=body.grace_period_hours,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return {
        "key": raw_key,
        "id": str(new_api_key.id),
        "key_prefix": new_api_key.key_prefix,
        "name": new_api_key.name,
        "scopes": new_api_key.scopes,
        "version": new_api_key.version,
        "expires_at": new_api_key.expires_at.isoformat() if new_api_key.expires_at else None,
        "message": "Store this key securely. It will not be shown again. "
                   f"The old key will expire in {body.grace_period_hours} hours.",
    }


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_200_OK)
async def revoke_api_key(
    key_id: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Revoke (deactivate) an API key immediately."""
    from uuid import UUID as _UUID

    try:
        parsed_key_id = _UUID(key_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid key ID format",
        )

    revoked = await APIKeyService.revoke_key(db, parsed_key_id, user.id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return {"message": "API key revoked successfully"}


# ── 2FA Setup / Disable ──────────────────────────────────────────────────

class Enable2FARequest(BaseModel):
    totp_code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


@router.post("/2fa/setup", status_code=status.HTTP_200_OK)
async def setup_2fa(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Generate a TOTP secret for 2FA setup.

    Returns the secret and a provisioning URI for QR code display.
    The user must confirm with a valid code via POST /2fa/enable.
    """
    import pyotp

    if user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled",
        )

    secret = pyotp.random_base32()
    # Store the provisional secret — it becomes active only after confirmation
    await db.execute(
        update(User).where(User.id == user.id).values(totp_secret=secret)
    )

    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=user.email,
        issuer_name=settings.app_name,
    )

    return {
        "secret": secret,
        "provisioning_uri": provisioning_uri,
        "message": "Scan the QR code with your authenticator app, then confirm with POST /auth/2fa/enable",
    }


@router.post("/2fa/enable", status_code=status.HTTP_200_OK)
async def enable_2fa(
    body: Enable2FARequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Confirm 2FA setup by verifying a TOTP code."""
    import pyotp

    totp_secret = getattr(user, "totp_secret", None) or getattr(user, "two_factor_secret", None)
    if not totp_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Call POST /auth/2fa/setup first to generate a secret",
        )

    if user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled",
        )

    totp = pyotp.TOTP(totp_secret)
    if not totp.verify(body.totp_code, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid TOTP code. Try again.",
        )

    await db.execute(
        update(User).where(User.id == user.id).values(two_factor_enabled=True)
    )

    await AuditService.log(
        db, user.id, AuditAction.UPDATE_SETTINGS,
        resource_type="2fa_enabled",
        ip_address=_get_ip(request),
    )

    return {"message": "2FA enabled successfully"}


@router.post("/2fa/disable", status_code=status.HTTP_200_OK)
async def disable_2fa(
    body: Enable2FARequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Disable 2FA. Requires a valid TOTP code as confirmation."""
    import pyotp

    if not user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is not enabled",
        )

    totp_secret = getattr(user, "totp_secret", None) or getattr(user, "two_factor_secret", None)
    if not totp_secret:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No 2FA secret found")

    totp = pyotp.TOTP(totp_secret)
    if not totp.verify(body.totp_code, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid TOTP code",
        )

    await db.execute(
        update(User).where(User.id == user.id).values(
            two_factor_enabled=False,
            totp_secret=None,
        )
    )

    await AuditService.log(
        db, user.id, AuditAction.UPDATE_SETTINGS,
        resource_type="2fa_disabled",
        ip_address=_get_ip(request),
    )

    return {"message": "2FA disabled successfully"}


# ── Subscription Tier ─────────────────────────────────────────────────────

class UpdateSubscriptionRequest(BaseModel):
    tier: str = Field(..., pattern=r"^(free|pro|enterprise)$")


@router.put("/subscription", status_code=status.HTTP_200_OK)
async def update_subscription(
    body: UpdateSubscriptionRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Update user subscription tier.

    In production, this would integrate with a payment provider (Stripe).
    For now, it directly updates the tier for development/testing.
    """
    from db.models import SubscriptionTier

    try:
        new_tier = SubscriptionTier(body.tier)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid subscription tier",
        )

    await db.execute(
        update(User).where(User.id == user.id).values(subscription_tier=new_tier)
    )

    await AuditService.log(
        db, user.id, AuditAction.UPDATE_SETTINGS,
        resource_type="subscription",
        details={"tier": body.tier},
        ip_address=_get_ip(request),
    )

    return {"message": f"Subscription updated to {body.tier}", "tier": body.tier}
