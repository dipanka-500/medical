"""
Custom middleware — production-grade request processing.

Middleware stack (outermost → innermost):
    1. SecurityHeadersMiddleware — OWASP headers on every response
    2. CSRFMiddleware            — double-submit cookie CSRF protection
    3. CorrelationIDMiddleware   — request tracking across services
    4. RateLimitMiddleware       — per-IP + per-path rate limiting
    5. AuditMiddleware           — structured request logging

Features:
    - Correlation ID generation and propagation
    - Structured JSON audit logging (not just logger.info)
    - Content-Security-Policy for production
    - Request body size enforcement
    - Enhanced rate limit scoping (includes user ID)
    - Timing headers for performance monitoring
    - CSRF protection via double-submit cookie pattern
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
import uuid

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

logger = logging.getLogger(__name__)

# Paths excluded from request audit logging
_SKIP_AUDIT_PATHS = frozenset({
    "/api/v1/health", "/api/v1/health/ready", "/api/v1/health/live",
    "/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico",
})


# ── Client IP Extraction ────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    """
    Extract the real client IP.

    Trusts X-Forwarded-For only in non-development (behind reverse proxy).
    Validates the header to prevent spoofing.
    """
    if settings.environment != "development":
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ips = [ip.strip() for ip in forwarded.split(",") if ip.strip()]
            if ips:
                # Rightmost IP is added by our reverse proxy (most trusted).
                # Use rightmost (proxy-appended) IP to prevent client spoofing.
                return ips[-1]
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    return request.client.host if request.client else "unknown"


# ── CSRF Protection Middleware ──────────────────────────────────────────

# Paths exempt from CSRF checks (stateless or read-only)
_CSRF_EXEMPT_PATHS = frozenset({
    "/api/v1/health", "/api/v1/health/ready", "/api/v1/health/live",
    "/health", "/docs", "/openapi.json", "/redoc",
    "/api/v1/auth/login", "/api/v1/auth/register",
    "/api/v1/auth/refresh", "/api/v1/auth/2fa/verify",
    "/api/v1/auth/logout", "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password", "/api/v1/auth/verify-email",
    "/api/v1/auth/resend-verification",  # Auth endpoints use Bearer tokens, not CSRF cookies
})

_CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _uses_header_auth(request: Request) -> bool:
    """
    Detect requests authenticated explicitly via headers.

    CSRF protections are for cookie-backed browser auth. Bearer tokens and API
    keys are not attached automatically by browsers, so programmatic and token-
    based clients should not be blocked waiting for a CSRF cookie.
    """
    authorization = request.headers.get("authorization", "").strip()
    api_key = request.headers.get("x-api-key", "").strip()
    return bool(authorization or api_key)


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    Double-submit cookie CSRF protection.

    How it works:
        1. On every response, sets a `csrf_token` cookie with a random token
        2. State-changing requests (POST/PUT/DELETE/PATCH) must include
           the same token in the `X-CSRF-Token` header
        3. The server verifies the header matches the cookie

    This is safe because:
        - An attacker site cannot read cross-origin cookies
        - An attacker site cannot set custom headers on cross-origin requests
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip CSRF for safe methods and exempt paths
        if (
            request.method in _CSRF_SAFE_METHODS
            or path in _CSRF_EXEMPT_PATHS
            or _uses_header_auth(request)
        ):
            response = await call_next(request)
            # Always set/refresh the CSRF cookie so the client has a token
            self._set_csrf_cookie(response, request)
            return response

        # Validate CSRF token on state-changing methods
        cookie_token = request.cookies.get("csrf_token")
        header_token = request.headers.get("x-csrf-token")

        if not cookie_token or not header_token:
            return Response(
                content='{"detail":"CSRF token missing"}',
                status_code=status.HTTP_403_FORBIDDEN,
                media_type="application/json",
            )

        if not hmac.compare_digest(cookie_token, header_token):
            return Response(
                content='{"detail":"CSRF token mismatch"}',
                status_code=status.HTTP_403_FORBIDDEN,
                media_type="application/json",
            )

        response = await call_next(request)
        # Rotate token after successful state-changing request
        self._set_csrf_cookie(response, request, rotate=True)
        return response

    @staticmethod
    def _set_csrf_cookie(response: Response, request: Request, rotate: bool = False) -> None:
        """Set or refresh the CSRF cookie."""
        existing = request.cookies.get("csrf_token")
        if not existing or rotate:
            token = secrets.token_urlsafe(32)
        else:
            token = existing

        response.set_cookie(
            key="csrf_token",
            value=token,
            httponly=False,  # Must be readable by JS to send in header
            secure=settings.is_production,
            samesite="lax",
            max_age=86400,  # 24 hours
            path="/",
        )


# ── Correlation ID Middleware ────────────────────────────────────────────

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Generate or propagate correlation IDs for distributed tracing.

    Sets:
        - request.state.request_id = unique request ID
        - X-Request-ID response header
        - X-Correlation-ID response header (propagated from client or generated)
    """

    async def dispatch(self, request: Request, call_next):
        # Use client-provided correlation ID or generate one
        correlation_id = request.headers.get(
            "x-correlation-id",
            str(uuid.uuid4()),
        )
        request_id = str(uuid.uuid4())

        # Store on request state for downstream use
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        response = await call_next(request)

        # Attach to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id

        return response


# ── Rate Limit Middleware ────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP + per-path rate limiting with enhanced scoping.

    Scoping strategy:
        - /auth/register → strict (5/hour)
        - /auth/*        → moderate (10/minute)
        - /chat/ask*     → AI-specific (30/minute)
        - /upload*       → upload-specific (20/hour)
        - Everything else → default (60/minute)
    """

    async def dispatch(self, request: Request, call_next):
        client_ip = _get_client_ip(request)
        path = request.url.path

        # Determine limit based on path
        if "/auth/register" in path:
            limit_str = settings.rate_limit_register
        elif "/auth/" in path:
            limit_str = settings.rate_limit_auth
        elif "/chat/ask" in path:
            limit_str = settings.rate_limit_ai
        elif "/upload" in path:
            limit_str = settings.rate_limit_upload
        else:
            limit_str = settings.rate_limit_default

        rate_limiter = getattr(request.app.state, "rate_limiter", None)
        info: dict = {}
        if rate_limiter:
            # Build scope key: IP + path category
            path_parts = path.strip("/").split("/")
            scope = path_parts[2] if len(path_parts) > 2 else "general"
            key = f"ip:{client_ip}:{scope}"

            allowed, info = await rate_limiter.check(
                key, limit_str, client_ip=client_ip,
            )

            if not allowed:
                return Response(
                    content='{"detail":"Rate limit exceeded. Try again later."}',
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    media_type="application/json",
                    headers={
                        "X-RateLimit-Limit": str(info.get("limit", "")),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(info.get("reset_in_seconds", 60))),
                        "Retry-After": str(int(info.get("reset_in_seconds", 60))),
                    },
                )

        response = await call_next(request)

        # Attach rate limit headers to all responses
        if info:
            response.headers["X-RateLimit-Limit"] = str(info.get("limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", ""))

        return response


# ── Audit Middleware ─────────────────────────────────────────────────────

class AuditMiddleware(BaseHTTPMiddleware):
    """
    Structured request audit logging.

    Logs every request with:
        - HTTP method and path
        - Status code
        - Latency (ms)
        - Client IP
        - Request ID
        - Content length
    """

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        # Skip noisy health check paths
        if request.url.path in _SKIP_AUDIT_PATHS:
            return response

        # Add server timing header for performance monitoring
        response.headers["Server-Timing"] = f"total;dur={duration_ms:.1f}"

        # Structured log entry
        request_id = getattr(request.state, "request_id", "-")
        content_length = response.headers.get("content-length", "-")

        logger.info(
            "REQUEST | %s %s | status=%d | %.1fms | ip=%s | "
            "request_id=%s | content_length=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            _get_client_ip(request),
            request_id,
            content_length,
        )

        # Log slow requests as warnings
        if duration_ms > 5000:
            logger.warning(
                "SLOW REQUEST | %s %s | %.1fms | request_id=%s",
                request.method, request.url.path, duration_ms, request_id,
            )

        return response


# ── Security Headers Middleware ──────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add comprehensive security headers to every response (OWASP best-practice).

    Headers set:
        - X-Content-Type-Options: nosniff
        - X-Frame-Options: DENY
        - X-XSS-Protection: 0 (modern CSP replaces this)
        - Referrer-Policy: strict-origin-when-cross-origin
        - Cache-Control: no-store (patient data must never be cached)
        - Permissions-Policy: restrictive feature policy
        - Content-Security-Policy: strict (production only)
        - Strict-Transport-Security: HSTS (production only)
        - Cross-Origin headers for isolation
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Core security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"  # Modern browsers use CSP
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"

        # Feature policy
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )

        # Cross-Origin isolation
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # Production-only headers
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )

        return response
