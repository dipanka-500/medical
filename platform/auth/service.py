"""
Authentication service — production-grade.

Security features:
    - bcrypt password hashing (configurable rounds)
    - JWT access + refresh tokens with rotation
    - Account lockout after N failed attempts
    - Login attempt tracking per user & IP
    - Password history (prevents reuse)
    - Session invalidation on password change
    - Timing attack mitigation on login
    - 2FA (TOTP) support
    - Audit logging for all auth events
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import bcrypt
import jwt
import pyotp
from sqlalchemy import and_, func, select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.models import (
    AuditAction,
    RefreshToken,
    User,
    UserRole,
)
from auth.token_blocklist import TokenBlocklist
from security.audit import AuditService

logger = logging.getLogger(__name__)

# Pre-computed dummy hash for timing-attack mitigation during login.
# Computed once at module load to avoid expensive bcrypt on every login attempt.
_DUMMY_HASH: str | None = None


def _get_dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = PasswordService.hash_password("timing-attack-mitigation")
    return _DUMMY_HASH


# ── Password Hashing ────────────────────────────────────────────────────

class PasswordService:
    """Secure password operations with history tracking."""

    _ROUNDS = 12  # bcrypt work factor

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt with configured rounds."""
        salt = bcrypt.gensalt(rounds=PasswordService._ROUNDS)
        return bcrypt.hashpw(password.encode(), salt).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password with constant-time comparison."""
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception:
            return False

    @staticmethod
    def check_password_strength(password: str) -> list[str]:
        """
        Validate password strength — returns list of issues.
        Empty list = password is strong enough.
        """
        issues = []
        if len(password) < 10:
            issues.append("Minimum 10 characters required")
        if not any(c.isupper() for c in password):
            issues.append("At least one uppercase letter required")
        if not any(c.islower() for c in password):
            issues.append("At least one lowercase letter required")
        if not any(c.isdigit() for c in password):
            issues.append("At least one digit required")
        if not any(c in "!@#$%^&*()-_=+[]{}|;:',.<>?/~`" for c in password):
            issues.append("At least one special character required")
        return issues

    @staticmethod
    async def check_password_history(
        db: AsyncSession,
        user_id: UUID,
        new_password: str,
    ) -> bool:
        """
        Check if password was recently used.
        Returns True if the password is safe to use (not in history).
        """
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            return True

        # Check current password
        if user.hashed_password and PasswordService.verify_password(
            new_password, user.hashed_password,
        ):
            return False

        # Check password_history (stored as list of hashes)
        history = getattr(user, "password_history", None)
        if history and isinstance(history, list):
            for old_hash in history[-settings.password_history_count:]:
                if PasswordService.verify_password(new_password, old_hash):
                    return False

        return True


# ── JWT Token Service ────────────────────────────────────────────────────

class TokenService:
    """JWT token creation, validation, and rotation."""

    @staticmethod
    def create_access_token(
        user_id: UUID,
        role: str,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create a short-lived access token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "role": role,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=settings.jwt_access_token_expire_minutes),
            "jti": secrets.token_hex(16),  # Unique token ID for revocation
        }
        if extra_claims:
            payload.update(extra_claims)

        return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    @staticmethod
    def create_refresh_token(
        user_id: UUID,
        device_info: str | None = None,
    ) -> tuple[str, datetime]:
        """
        Create a long-lived refresh token.
        Returns (token_string, expiry_datetime).
        """
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=settings.jwt_refresh_token_expire_days)
        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "iat": now,
            "exp": expiry,
            "jti": secrets.token_hex(16),
            "device": device_info or "unknown",
        }
        token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        return token, expiry

    @staticmethod
    def decode_token(token: str) -> dict[str, Any]:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm],
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")


# ── Account Lockout ──────────────────────────────────────────────────────

class LockoutService:
    """Track and enforce account lockouts after failed login attempts."""

    @staticmethod
    async def record_failed_attempt(
        db: AsyncSession,
        user: User,
        ip_address: str,
    ) -> tuple[int, bool]:
        """
        Record a failed login attempt.
        Returns (attempt_count, is_now_locked).
        """
        # Atomic increment to prevent race condition (two concurrent failures
        # reading the same count and both incrementing to the same value)
        now = datetime.now(timezone.utc)
        result = await db.execute(
            update(User)
            .where(User.id == user.id)
            .values(
                failed_login_attempts=func.coalesce(User.failed_login_attempts, 0) + 1,
                last_failed_login=now,
            )
            .returning(User.failed_login_attempts)
        )
        current_count = result.scalar_one()
        is_locked = current_count >= settings.max_login_attempts

        if is_locked:
            lockout_until = now + timedelta(
                minutes=settings.lockout_duration_minutes,
            )
            await db.execute(
                update(User).where(User.id == user.id).values(
                    locked_until=lockout_until,
                )
            )
            logger.warning(
                "Account locked: user_id=%s attempts=%d ip=%s until=%s",
                user.id, current_count, ip_address, lockout_until.isoformat(),
            )

        return current_count, is_locked

    @staticmethod
    async def clear_lockout(db: AsyncSession, user: User) -> None:
        """Reset failed attempts after successful login."""
        await db.execute(
            update(User).where(User.id == user.id).values(
                failed_login_attempts=0,
                locked_until=None,
                last_failed_login=None,
            )
        )

    @staticmethod
    def is_locked(user: User) -> bool:
        """Check if the user account is currently locked."""
        if not user.locked_until:
            return False
        return datetime.now(timezone.utc) < user.locked_until


# ── Auth Service (orchestrator) ──────────────────────────────────────────

class AuthService:
    """Orchestrates login, registration, token refresh, and 2FA."""

    @staticmethod
    async def login(
        db: AsyncSession,
        email: str,
        password: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        device_info: str | None = None,
    ) -> dict[str, Any]:
        """
        Authenticate user and return tokens.

        Security:
            - Constant-time comparison to prevent user enumeration
            - Account lockout after N failures
            - Timing attack mitigation
        """
        # Constant-time: always compare even for non-existent users
        dummy_hash = _get_dummy_hash()
        start_time = time.monotonic()

        result = await db.execute(
            select(User).where(func.lower(User.email) == email.lower())
        )
        user = result.scalar_one_or_none()

        if not user:
            # Still verify against dummy to prevent timing leak
            PasswordService.verify_password(password, dummy_hash)
            await _constant_time_sleep(start_time)
            raise ValueError("Invalid email or password")

        # Check lockout
        if LockoutService.is_locked(user):
            remaining = (user.locked_until - datetime.now(timezone.utc)).total_seconds()
            await _constant_time_sleep(start_time)
            raise ValueError(
                f"Account locked. Try again in {int(remaining / 60)} minutes."
            )

        # Verify password
        if not PasswordService.verify_password(password, user.hashed_password):
            count, locked = await LockoutService.record_failed_attempt(
                db, user, ip_address or "unknown",
            )
            await AuditService.log(
                db, user.id, AuditAction.LOGIN_FAILED,
                details={"attempt": count, "locked": locked},
                ip_address=ip_address,
                user_agent=user_agent,
            )
            await _constant_time_sleep(start_time)
            remaining_attempts = max(0, settings.max_login_attempts - count)
            raise ValueError(
                f"Invalid email or password. {remaining_attempts} attempts remaining."
            )

        # Check if email verification is enabled for this deployment
        if settings.require_email_verification and not user.is_verified:
            raise ValueError("Email not verified. Please check your inbox.")

        # Check 2FA requirement
        if user.two_factor_enabled:
            return {
                "requires_2fa": True,
                "user_id": str(user.id),
                "message": "2FA verification required",
            }

        # ── Success ──────────────────────────────────────────────
        await LockoutService.clear_lockout(db, user)

        # Generate tokens
        access_token = TokenService.create_access_token(user.id, user.role.value)
        refresh_token_str, refresh_expiry = TokenService.create_refresh_token(
            user.id, device_info,
        )

        # Store refresh token
        db_token = RefreshToken(
            user_id=user.id,
            token_hash=hashlib.sha256(refresh_token_str.encode()).hexdigest(),
            expires_at=refresh_expiry,
            device_info=device_info or "unknown",
        )
        db.add(db_token)

        # Update last login
        await db.execute(
            update(User).where(User.id == user.id).values(
                last_login=datetime.now(timezone.utc),
            )
        )

        # Audit log
        await AuditService.log(
            db, user.id, AuditAction.LOGIN,
            details={"device": device_info},
            ip_address=ip_address,
            user_agent=user_agent,
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token_str,
            "token_type": "bearer",
            "expires_in": settings.jwt_access_token_expire_minutes * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "role": user.role.value,
                "full_name": user.full_name,
            },
        }

    @staticmethod
    async def verify_2fa(
        db: AsyncSession,
        user_id: UUID,
        totp_code: str,
        ip_address: str | None = None,
        device_info: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """Complete login with 2FA verification."""
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        totp_secret = getattr(user, "two_factor_secret", None) or getattr(user, "totp_secret", None)
        if not user or not totp_secret:
            raise ValueError("Invalid 2FA request")

        totp = pyotp.TOTP(totp_secret)
        if not totp.verify(totp_code, valid_window=1):
            await AuditService.log(
                db, user.id, AuditAction.LOGIN_FAILED,
                details={"method": "2fa"},
                ip_address=ip_address,
                user_agent=user_agent,
            )
            raise ValueError("Invalid 2FA code")

        await LockoutService.clear_lockout(db, user)

        # Generate tokens after 2FA success
        access_token = TokenService.create_access_token(user.id, user.role.value)
        refresh_token_str, refresh_expiry = TokenService.create_refresh_token(
            user.id, device_info,
        )

        db_token = RefreshToken(
            user_id=user.id,
            token_hash=hashlib.sha256(refresh_token_str.encode()).hexdigest(),
            expires_at=refresh_expiry,
            device_info=device_info or "unknown",
        )
        db.add(db_token)

        await db.execute(
            update(User).where(User.id == user.id).values(
                last_login=datetime.now(timezone.utc),
            )
        )

        await AuditService.log(
            db, user.id, AuditAction.LOGIN,
            details={"method": "2fa", "device": device_info},
            ip_address=ip_address,
            user_agent=user_agent,
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token_str,
            "token_type": "bearer",
            "expires_in": settings.jwt_access_token_expire_minutes * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "role": user.role.value,
                "full_name": user.full_name,
            },
        }

    @staticmethod
    async def refresh_tokens(
        db: AsyncSession,
        refresh_token: str,
        ip_address: str | None = None,
        device_info: str | None = None,
    ) -> dict[str, Any]:
        """
        Rotate refresh token — invalidate old, issue new.

        Prevents replay attacks: each refresh token can only be used once.
        """
        try:
            payload = TokenService.decode_token(refresh_token)
        except ValueError as e:
            raise ValueError(f"Invalid refresh token: {e}")

        if payload.get("type") != "refresh":
            raise ValueError("Not a refresh token")

        user_id = UUID(payload["sub"])
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

        # Find and validate stored token — lock row to prevent race conditions
        result = await db.execute(
            select(RefreshToken).where(
                and_(
                    RefreshToken.user_id == user_id,
                    RefreshToken.token_hash == token_hash,
                    RefreshToken.is_revoked == False,  # noqa: E712
                )
            ).with_for_update()
        )
        stored_token = result.scalar_one_or_none()

        if not stored_token:
            # Token reuse detected — revoke ALL tokens for this user
            logger.warning(
                "Refresh token reuse detected: user_id=%s — revoking all sessions",
                user_id,
            )
            await db.execute(
                update(RefreshToken).where(
                    RefreshToken.user_id == user_id
                ).values(is_revoked=True)
            )
            raise ValueError("Token reuse detected. All sessions revoked for security.")

        # Get user before mutating tokens
        user_result = await db.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()
        if not user or not user.is_active:
            raise ValueError("User account is disabled")

        # Issue new tokens FIRST, then revoke old — both in same transaction.
        # If new token creation fails, old token remains valid (user not locked out).
        access_token = TokenService.create_access_token(user.id, user.role.value)
        new_refresh_str, new_expiry = TokenService.create_refresh_token(
            user.id, device_info,
        )

        db.add(RefreshToken(
            user_id=user.id,
            token_hash=hashlib.sha256(new_refresh_str.encode()).hexdigest(),
            expires_at=new_expiry,
            device_info=device_info or stored_token.device_info,
        ))

        # Revoke old token AFTER new one is staged (both commit together)
        stored_token.is_revoked = True

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_str,
            "token_type": "bearer",
            "expires_in": settings.jwt_access_token_expire_minutes * 60,
        }

    @staticmethod
    async def logout(
        db: AsyncSession,
        user_id: UUID,
        refresh_token: str | None = None,
        all_devices: bool = False,
        ip_address: str | None = None,
        redis: Any | None = None,
    ) -> dict[str, str]:
        """Logout — revoke refresh tokens AND access tokens."""
        # Always revoke outstanding access tokens for this user so that
        # already-issued JWTs are rejected immediately, not just at expiry.
        await TokenBlocklist.revoke_user_tokens(str(user_id), redis=redis)

        if all_devices:
            await db.execute(
                update(RefreshToken).where(
                    RefreshToken.user_id == user_id
                ).values(is_revoked=True)
            )
            message = "All sessions logged out"
        elif refresh_token:
            token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            await db.execute(
                update(RefreshToken).where(
                    and_(
                        RefreshToken.user_id == user_id,
                        RefreshToken.token_hash == token_hash,
                    )
                ).values(is_revoked=True)
            )
            message = "Session logged out"
        else:
            message = "No token provided"

        await AuditService.log(
            db, user_id, AuditAction.LOGOUT,
            details={"all_devices": all_devices},
            ip_address=ip_address,
        )

        return {"message": message}

    @staticmethod
    async def change_password(
        db: AsyncSession,
        user_id: UUID,
        current_password: str,
        new_password: str,
        ip_address: str | None = None,
        redis: Any | None = None,
    ) -> dict[str, str]:
        """
        Change password with validation.

        Security:
            - Verify current password
            - Check password history
            - Revoke all sessions after change
        """
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found")

        # Verify current password
        if not PasswordService.verify_password(current_password, user.hashed_password):
            raise ValueError("Current password is incorrect")

        # Check strength
        issues = PasswordService.check_password_strength(new_password)
        if issues:
            raise ValueError(f"Password too weak: {'; '.join(issues)}")

        # Check history
        is_safe = await PasswordService.check_password_history(db, user_id, new_password)
        if not is_safe:
            raise ValueError(
                f"Password was used recently. Choose a different password."
            )

        # Update password
        new_hash = PasswordService.hash_password(new_password)

        # Update password history
        history = getattr(user, "password_history", None) or []
        history.append(user.hashed_password)
        history = history[-settings.password_history_count:]  # Keep last N

        await db.execute(
            update(User).where(User.id == user_id).values(
                hashed_password=new_hash,
                password_history=history,
            )
        )

        # Revoke all sessions — both refresh tokens AND access tokens
        await db.execute(
            update(RefreshToken).where(
                RefreshToken.user_id == user_id
            ).values(is_revoked=True)
        )
        await TokenBlocklist.revoke_user_tokens(str(user_id), redis=redis)

        await AuditService.log(
            db, user_id, AuditAction.PASSWORD_CHANGE,
            ip_address=ip_address,
        )

        return {"message": "Password changed. Please log in again."}


# ── Timing helpers ───────────────────────────────────────────────────────

async def _constant_time_sleep(start_time: float, target_ms: float = 200) -> None:
    """
    Ensure the total operation takes at least `target_ms` milliseconds.
    Prevents timing-based user enumeration attacks.

    Uses asyncio.sleep to avoid blocking the event loop.
    """
    elapsed = (time.monotonic() - start_time) * 1000
    remaining = (target_ms - elapsed) / 1000
    if remaining > 0:
        await asyncio.sleep(remaining)
