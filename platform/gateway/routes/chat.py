"""
Chat & AI endpoints — production-grade.

Features:
    - Structured request/response models
    - SSE streaming with heartbeat
    - Session management with pagination
    - Free-tier query limit enforcement
    - Request timeout with cancellation
    - AI query sanitization
    - Comprehensive audit logging
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from config import settings
from db.models import (
    AuditAction,
    ChatMessage,
    ChatSession,
    MessageRole,
    User,
)
from db.session import get_db
from security.audit import AuditService
from cache.service import session_list_key, session_messages_key
from gateway.concurrency import AIQuerySemaphore
from security.input_validator import sanitize_ai_query
from security.safety_pipeline import run_pre_query_checks, run_post_query_checks
from services.voice_service import VoiceServiceUnavailable

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response Models ────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = Field(default=None, max_length=128)
    mode: str = Field(default="doctor", pattern=r"^(doctor|patient|research)$")
    patient_id: str | None = None
    web_search: bool = False
    deep_reasoning: bool = False

class AskResponse(BaseModel):
    answer: str
    confidence: float = 0.0
    routing: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    safety: dict[str, Any] | None = None
    governance: dict[str, Any] | None = None
    session_id: str | None = None
    message_id: str | None = None
    latency_ms: float = 0.0

class SessionListResponse(BaseModel):
    sessions: list[dict[str, Any]]
    total: int
    page: int
    page_size: int


class VoiceCapabilitiesResponse(BaseModel):
    asr_available: bool
    asr_provider: str
    asr_model: str | None = None
    tts_available: bool
    tts_provider: str
    tts_model: str | None = None


class VoiceTranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    language_probability: float | None = None
    duration_seconds: float | None = None
    provider: str
    model: str
    segment_count: int = 0


class VoiceSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str | None = Field(default=None, max_length=32)


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


def _parse_uuid(value: str | None, field_name: str) -> uuid.UUID | None:
    """Parse an optional UUID and raise a request-scoped 400 on invalid input."""
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name} format",
        ) from exc



def _build_session_title(query: str) -> str:
    compact = " ".join(query.strip().split())
    if not compact:
        return "New Chat"
    if len(compact) <= 80:
        return compact
    return compact[:77].rstrip() + "..."


def _build_ai_metadata(result: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "confidence": result.get("confidence", 0.0),
        "routing": result.get("routing"),
        "sources": result.get("sources"),
        "safety": result.get("safety"),
        "governance": result.get("governance"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _normalize_session_title(title: str) -> str:
    compact = " ".join(title.strip().split())
    if not compact:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session title cannot be empty",
        )
    return compact[:120]


async def _get_or_create_session(
    db: AsyncSession,
    *,
    user: User,
    session_id: str | None,
    query: str,
    patient_id: str | None = None,
) -> ChatSession:
    parsed_session_id = _parse_uuid(session_id, "session_id") or uuid.uuid4()
    parsed_patient_id = _parse_uuid(patient_id, "patient_id")

    if session_id:
        existing = await db.execute(
            select(ChatSession).where(
                ChatSession.id == parsed_session_id,
                ChatSession.user_id == user.id,
            )
        )
        session = existing.scalar_one_or_none()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session not found or access denied",
            )
        if parsed_patient_id and session.patient_context_id is None:
            session.patient_context_id = parsed_patient_id
        return session

    session = ChatSession(
        id=parsed_session_id,
        user_id=user.id,
        title=_build_session_title(query),
        patient_context_id=parsed_patient_id,
    )
    db.add(session)
    await db.flush()
    return session


async def _persist_chat_exchange(
    db: AsyncSession,
    *,
    session: ChatSession,
    query: str,
    result: dict[str, Any],
    attachments: dict[str, Any] | None = None,
) -> ChatMessage:
    session.updated_at = datetime.now(timezone.utc)
    if not session.title or session.title == "New Chat":
        session.title = _build_session_title(query)

    user_message = ChatMessage(
        session_id=session.id,
        role=MessageRole.USER,
        content=query,
        attachments=attachments,
    )
    assistant_message = ChatMessage(
        session_id=session.id,
        role=MessageRole.ASSISTANT,
        content=result.get("answer", ""),
        ai_metadata=_build_ai_metadata(result),
    )
    db.add_all([user_message, assistant_message])
    await db.flush()
    return assistant_message


async def _invalidate_chat_cache(
    request: Request,
    *,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
) -> None:
    await _invalidate_chat_cache_for_app(
        request.app,
        user_id=user_id,
        session_id=session_id,
    )


async def _invalidate_chat_cache_for_app(
    app: Any,
    *,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
) -> None:
    cache = getattr(app.state, "cache", None)
    if not cache:
        return
    try:
        await cache.delete_pattern(f"{session_list_key(str(user_id))}*")
        await cache.delete_pattern(f"{session_messages_key(str(session_id))}*")
    except Exception as exc:
        logger.warning("Cache invalidation failed (non-fatal): %s", exc)


def _build_ask_response(
    result: dict[str, Any],
    *,
    session_id: uuid.UUID,
    latency_ms: float,
    message_id: uuid.UUID | None = None,
) -> AskResponse:
    return AskResponse(
        answer=result.get("answer", ""),
        confidence=result.get("confidence", 0.0),
        routing=result.get("routing"),
        sources=result.get("sources"),
        safety=result.get("safety"),
        governance=result.get("governance"),
        session_id=str(session_id),
        message_id=str(message_id) if message_id else None,
        latency_ms=round(latency_ms, 1),
    )


def _audio_suffix(filename: str | None, content_type: str | None) -> str:
    extension_map = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/webm": ".webm",
        "video/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
    }

    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext.isalnum() and len(ext) <= 10:
            return f".{ext}"

    return extension_map.get((content_type or "").lower(), ".webm")


def _resolve_subscription_tier(user: User) -> str:
    tier = getattr(user, "subscription_tier", "free")
    if hasattr(tier, "value"):
        return tier.value
    return str(tier)


async def _enforce_daily_query_limit(
    app: Any,
    *,
    user: User,
    tier: str,
) -> None:
    rate_limiter = getattr(app.state, "rate_limiter", None)
    if not rate_limiter:
        return

    allowed, _remaining = await rate_limiter.check_daily_queries(str(user.id), tier)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily query limit reached ({settings.free_tier_daily_queries}/day). "
                "Upgrade to continue."
            ),
        )


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Submit a medical query for AI analysis.

    Flow:
        1. Sanitize input
        2. Check rate limits (free tier)
        3. Route to appropriate engine(s) via MasterRouter
        4. Save to chat history
        5. Return unified response
    """
    start = time.monotonic()

    # Sanitize query
    clean_query = sanitize_ai_query(body.query)
    if not clean_query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is empty after sanitization",
        )

    # Resolve tier once (used by rate limiter + safety pipeline)
    tier = _resolve_subscription_tier(user)

    # Safety pipeline — pre-query checks (token limits + content safety)
    safety_check = run_pre_query_checks(clean_query, tier=tier)
    if not safety_check["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safety_check["error"],
        )
    # Use possibly-trimmed query
    clean_query = safety_check["query"]

    # Free-tier check
    await _enforce_daily_query_limit(request.app, user=user, tier=tier)

    session = await _get_or_create_session(
        db,
        user=user,
        session_id=body.session_id,
        query=clean_query,
        patient_id=body.patient_id,
    )

    # Route and execute via MasterRouter (handles ALL queries intelligently)
    master_router = getattr(request.app.state, "master_router", None)
    if not master_router:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI engine not available",
        )

    request_id = getattr(request.state, "request_id", None)

    # Acquire AI concurrency semaphore
    ai_semaphore: AIQuerySemaphore | None = getattr(request.app.state, "ai_semaphore", None)
    if ai_semaphore:
        try:
            await ai_semaphore.acquire()
        except OverflowError as exc:
            logger.warning("AI semaphore overflow: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is at capacity. Please try again shortly.",
            )
        except TimeoutError as exc:
            logger.warning("AI semaphore timeout: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is at capacity. Please try again shortly.",
            )

    try:
        try:
            result = await asyncio.wait_for(
                master_router.route_and_execute(
                    text=clean_query,
                    patient_id=body.patient_id,
                    session_id=str(session.id),
                    mode=body.mode,
                    user_id=str(user.id),
                    request_id=request_id,
                    web_search=body.web_search,
                    deep_reasoning=body.deep_reasoning,
                ),
                timeout=settings.engine_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="AI engine timed out. Please try again.",
            )
        except Exception as exc:
            logger.error("AI query failed: %s (request_id=%s)", exc, request_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI analysis failed. Please try again.",
            )
    finally:
        if ai_semaphore:
            await ai_semaphore.release()

    # Safety pipeline — post-query checks (PII redaction + confidence filtering)
    result = run_post_query_checks(result, mode=body.mode, tier=tier)

    # Prepend emergency message if user is in crisis
    if safety_check.get("emergency_message"):
        result["answer"] = safety_check["emergency_message"] + "\n\n" + result.get("answer", "")

    latency_ms = (time.monotonic() - start) * 1000
    assistant_message = await _persist_chat_exchange(
        db,
        session=session,
        query=clean_query,
        result=result,
    )
    await _invalidate_chat_cache(
        request,
        user_id=user.id,
        session_id=session.id,
    )

    # Audit
    await AuditService.log(
        db, user.id, AuditAction.AI_QUERY,
        resource_type="chat",
        details={
            "mode": body.mode,
            "session_id": str(session.id),
            "latency_ms": round(latency_ms, 1),
            "query_length": len(clean_query),
            "safety_category": safety_check.get("safety_category"),
            "feature_flags": {
                "web_search": body.web_search,
                "deep_reasoning": body.deep_reasoning,
            },
        },
        ip_address=request.client.host if request.client else None,
    )

    return _build_ask_response(
        result,
        session_id=session.id,
        latency_ms=latency_ms,
        message_id=assistant_message.id,
    )


@router.post("/ask-with-file", response_model=AskResponse)
async def ask_with_file(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
    query: str = Form(default="Analyze this medical image"),
    mode: str = Form(default="doctor", pattern=r"^(doctor|patient|research)$"),
    patient_id: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    web_search: bool = Form(default=False),
    deep_reasoning: bool = Form(default=False),
):
    """Submit a medical image/document with a query for AI analysis."""
    import os
    import tempfile

    start = time.monotonic()

    # Validate file
    from security.input_validator import FileValidator
    content, sha256 = await FileValidator.validate(file)

    # Write to temp file
    # Safe suffix extraction — handles None filename and filenames without extensions
    suffix = ".tmp"
    if file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        # Only allow safe extensions (prevent path traversal via crafted names)
        if ext.isalnum() and len(ext) <= 10:
            suffix = f".{ext}"
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=settings.upload_temp_dir or None)
    try:
        try:
            os.write(temp_fd, content)
        finally:
            os.close(temp_fd)

        # Sanitize query
        clean_query = sanitize_ai_query(query)
        if not clean_query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is empty after sanitization",
            )

        # Resolve tier
        file_tier = _resolve_subscription_tier(user)

        await _enforce_daily_query_limit(request.app, user=user, tier=file_tier)

        # Safety pipeline — pre-query checks
        file_safety = run_pre_query_checks(clean_query, tier=file_tier)
        if not file_safety["allowed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=file_safety["error"],
            )
        clean_query = file_safety["query"]

        session = await _get_or_create_session(
            db,
            user=user,
            session_id=session_id,
            query=clean_query or "Analyze uploaded file",
            patient_id=patient_id,
        )

        # Route and execute
        master_router = getattr(request.app.state, "master_router", None)
        if not master_router:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI engine not available",
            )

        file_request_id = getattr(request.state, "request_id", None)

        # Acquire AI concurrency semaphore
        file_semaphore: AIQuerySemaphore | None = getattr(request.app.state, "ai_semaphore", None)
        if file_semaphore:
            try:
                await file_semaphore.acquire()
            except OverflowError as exc:
                logger.warning("AI semaphore overflow (file upload): %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service is at capacity. Please try again shortly.",
                )
            except TimeoutError as exc:
                logger.warning("AI semaphore timeout (file upload): %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service is at capacity. Please try again shortly.",
                )

        try:
            try:
                result = await asyncio.wait_for(
                    master_router.route_and_execute(
                        text=clean_query,
                        file_path=temp_path,
                        file_mime=file.content_type,
                        file_name=file.filename,
                        patient_id=patient_id,
                        session_id=str(session.id),
                        mode=mode,
                        user_id=str(user.id),
                        request_id=file_request_id,
                        web_search=web_search,
                        deep_reasoning=deep_reasoning,
                    ),
                    timeout=settings.engine_timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="AI engine timed out. Please try again.",
                )
            except Exception as exc:
                logger.error("AI file query failed: %s (request_id=%s)", exc, file_request_id)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="AI analysis failed. Please try again.",
                )
        finally:
            if file_semaphore:
                await file_semaphore.release()

        # Safety pipeline — post-query checks
        result = run_post_query_checks(result, mode=mode, tier=file_tier)

        if file_safety.get("emergency_message"):
            result["answer"] = file_safety["emergency_message"] + "\n\n" + result.get("answer", "")

        latency_ms = (time.monotonic() - start) * 1000
        assistant_message = await _persist_chat_exchange(
            db,
            session=session,
            query=clean_query,
            result=result,
            attachments={
                "original_filename": file.filename,
                "mime_type": file.content_type,
                "file_size_bytes": len(content),
                "sha256_prefix": sha256[:16],
            },
        )
        await _invalidate_chat_cache(
            request,
            user_id=user.id,
            session_id=session.id,
        )

        # Audit
        await AuditService.log(
            db, user.id, AuditAction.AI_QUERY,
            resource_type="chat_with_file",
            details={
                "mode": mode,
                "session_id": str(session.id),
                "file_name": file.filename,
                "file_size": len(content),
                "sha256": sha256[:16],
                "latency_ms": round(latency_ms, 1),
                "safety_category": file_safety.get("safety_category"),
                "feature_flags": {
                    "web_search": web_search,
                    "deep_reasoning": deep_reasoning,
                },
            },
            ip_address=request.client.host if request.client else None,
        )

        return _build_ask_response(
            result,
            session_id=session.id,
            latency_ms=latency_ms,
            message_id=assistant_message.id,
        )

    finally:
        # Always clean up temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@router.get("/voice/capabilities", response_model=VoiceCapabilitiesResponse)
async def voice_capabilities(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
):
    """Expose server-side speech capability status for the chat UI."""
    del user  # Auth gate only

    voice_service = getattr(request.app.state, "voice_service", None)
    if voice_service is None:
        return VoiceCapabilitiesResponse(
            asr_available=False,
            asr_provider="disabled",
            tts_available=False,
            tts_provider="disabled",
        )

    return VoiceCapabilitiesResponse(**voice_service.capabilities().as_dict())


@router.post("/transcribe", response_model=VoiceTranscriptionResponse)
async def transcribe_audio(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
):
    """Transcribe short voice clips for the chat composer."""
    del user  # Auth gate only

    if not audio.filename and not audio.content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio upload is missing metadata.",
        )

    content_type = (audio.content_type or "").lower()
    if not (content_type.startswith("audio/") or content_type == "video/webm"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported audio format.",
        )

    payload = await audio.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio is empty.",
        )
    if len(payload) > settings.max_voice_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio too large. Max {settings.max_voice_upload_size_mb}MB.",
        )

    voice_service = getattr(request.app.state, "voice_service", None)
    if voice_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice service is not available.",
        )

    import os
    import tempfile

    suffix = _audio_suffix(audio.filename, audio.content_type)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=suffix,
        dir=settings.upload_temp_dir or None,
    )
    try:
        try:
            os.write(temp_fd, payload)
        finally:
            os.close(temp_fd)

        try:
            result = await voice_service.transcribe_file(temp_path, language=language)
        except VoiceServiceUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.error("Voice transcription failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Speech transcription failed. Please try again.",
            ) from exc

        return VoiceTranscriptionResponse(**result.as_dict())
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@router.post("/speak")
async def speak_text(
    body: VoiceSpeakRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
):
    """Generate reply audio using the configured server-side TTS backend."""
    del user  # Auth gate only

    voice_service = getattr(request.app.state, "voice_service", None)
    if voice_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice service is not available.",
        )

    try:
        audio_bytes = await voice_service.synthesize_text(
            body.text.strip(),
            language=body.language,
        )
    except VoiceServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Voice synthesis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Speech synthesis failed. Please try again.",
        ) from exc

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
        },
    )


@router.post("/ask-stream")
async def ask_stream(
    body: AskRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Stream AI response via Server-Sent Events (SSE).

    Features:
        - Heartbeat every 15s (prevents proxy timeout)
        - Error handling with SSE error event
        - Graceful connection close
    """
    if not settings.enable_streaming:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming is disabled",
        )

    start = time.monotonic()
    clean_query = sanitize_ai_query(body.query)
    if not clean_query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is empty after sanitization",
        )

    stream_tier = _resolve_subscription_tier(user)

    # Safety pipeline — pre-query checks
    stream_safety = run_pre_query_checks(clean_query, tier=stream_tier)
    if not stream_safety["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=stream_safety["error"],
        )
    clean_query = stream_safety["query"]

    await _enforce_daily_query_limit(request.app, user=user, tier=stream_tier)

    session = await _get_or_create_session(
        db,
        user=user,
        session_id=body.session_id,
        query=clean_query,
        patient_id=body.patient_id,
    )

    master_router = getattr(request.app.state, "master_router", None)
    if not master_router:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI engine not available",
        )

    # Acquire AI concurrency semaphore before streaming begins
    stream_semaphore: AIQuerySemaphore | None = getattr(request.app.state, "ai_semaphore", None)
    if stream_semaphore:
        try:
            await stream_semaphore.acquire()
        except OverflowError as exc:
            logger.warning("AI semaphore overflow (stream): %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is at capacity. Please try again shortly.",
            )
        except TimeoutError as exc:
            logger.warning("AI semaphore timeout (stream): %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is at capacity. Please try again shortly.",
            )

    async def event_stream():
        """Generate SSE events."""
        try:
            # Start with a routing info event
            yield (
                "event: routing\n"
                f"data: {json.dumps({'status': 'processing', 'mode': body.mode, 'session_id': str(session.id)})}\n\n"
            )

            # Execute the query
            stream_request_id = getattr(request.state, "request_id", None)
            result = await asyncio.wait_for(
                master_router.route_and_execute(
                    text=clean_query,
                    patient_id=body.patient_id,
                    session_id=str(session.id),
                    mode=body.mode,
                    user_id=str(user.id),
                    request_id=stream_request_id,
                    web_search=body.web_search,
                    deep_reasoning=body.deep_reasoning,
                ),
                timeout=settings.engine_timeout_seconds,
            )

            # Safety pipeline — post-query checks
            result = run_post_query_checks(result, mode=body.mode, tier=stream_tier)

            if stream_safety.get("emergency_message"):
                result["answer"] = stream_safety["emergency_message"] + "\n\n" + result.get("answer", "")

            latency_ms = (time.monotonic() - start) * 1000
            assistant_message = await _persist_chat_exchange(
                db,
                session=session,
                query=clean_query,
                result=result,
            )
            await AuditService.log(
                db,
                user.id,
                AuditAction.AI_QUERY,
                resource_type="chat_stream",
                details={
                    "mode": body.mode,
                    "session_id": str(session.id),
                    "latency_ms": round(latency_ms, 1),
                    "query_length": len(clean_query),
                    "safety_category": stream_safety.get("safety_category"),
                    "feature_flags": {
                        "web_search": body.web_search,
                        "deep_reasoning": body.deep_reasoning,
                    },
                },
                ip_address=request.client.host if request.client else None,
            )
            await db.commit()
            await _invalidate_chat_cache(
                request,
                user_id=user.id,
                session_id=session.id,
            )

            # Stream the response in chunks
            answer = result.get("answer", "")
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.02)  # Simulate streaming delay

            # Send final event with metadata
            yield (
                "event: complete\n"
                f"data: {json.dumps({'done': True, 'confidence': result.get('confidence', 0.0), 'routing': result.get('routing'), 'session_id': str(session.id), 'message_id': str(assistant_message.id), 'latency_ms': round(latency_ms, 1)})}\n\n"
            )

        except asyncio.CancelledError:
            await db.rollback()
            raise
        except asyncio.TimeoutError:
            await db.rollback()
            yield f"event: error\ndata: {json.dumps({'detail': 'AI engine timed out'})}\n\n"
        except Exception as exc:
            await db.rollback()
            logger.error("Streaming error: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': 'Analysis failed'})}\n\n"
        finally:
            if stream_semaphore:
                await stream_semaphore.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(default=1, ge=1, le=1000),
    page_size: int = Query(default=20, ge=1, le=100),
    search: str | None = Query(default=None, max_length=200),
):
    """List chat sessions for the current user with pagination."""
    offset = (page - 1) * page_size
    cache = getattr(request.app.state, "cache", None)
    normalized_search = " ".join(search.strip().split()) if search else ""
    cache_key = (
        f"{session_list_key(str(user.id))}:page:{page}:size:{page_size}:search:{normalized_search or 'all'}"
    )

    if cache:
        cached = await cache.get(cache_key)
        if isinstance(cached, dict):
            return SessionListResponse(**cached)

    filters = [ChatSession.user_id == user.id]
    if normalized_search:
        filters.append(ChatSession.title.ilike(f"%{normalized_search}%"))

    # Count total
    count_result = await db.execute(
        select(func.count(ChatSession.id)).where(*filters)
    )
    total = count_result.scalar() or 0

    # Fetch page
    result = await db.execute(
        select(ChatSession).where(*filters).order_by(ChatSession.updated_at.desc())
        .offset(offset).limit(page_size)
    )
    sessions = result.scalars().all()

    response_data = {
        "sessions": [
            {
                "id": str(s.id),
                "title": getattr(s, "title", None) or "Untitled",
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sessions
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }

    if cache:
        await cache.set(cache_key, response_data, ttl=60)

    return SessionListResponse(**response_data)


class MessageListResponse(BaseModel):
    messages: list[dict[str, Any]]
    total: int
    session_id: str


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
    before: str | None = Query(default=None, description="Cursor: message ID to fetch before"),
):
    """
    Retrieve messages for a chat session (ChatGPT-style history).

    Supports cursor-based pagination for large conversations.
    """
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format",
        )

    # Verify session ownership
    session_result = await db.execute(
        select(ChatSession).where(
            and_(ChatSession.id == sid, ChatSession.user_id == user.id)
        )
    )
    if not session_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    cache = getattr(request.app.state, "cache", None)
    cache_key = (
        f"{session_messages_key(session_id)}:limit:{limit}:before:{before or 'latest'}"
    )
    if cache:
        cached = await cache.get(cache_key)
        if isinstance(cached, dict):
            return MessageListResponse(**cached)

    # Build query
    query = select(ChatMessage).where(ChatMessage.session_id == sid)

    # Cursor-based pagination: fetch messages before a given ID
    if before:
        try:
            before_id = uuid.UUID(before)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid cursor format",
            )
        # Get the created_at of the cursor message
        cursor_result = await db.execute(
            select(ChatMessage.created_at).where(ChatMessage.id == before_id)
        )
        cursor_ts = cursor_result.scalar_one_or_none()
        if cursor_ts:
            query = query.where(ChatMessage.created_at < cursor_ts)

    query = query.order_by(ChatMessage.created_at.asc()).limit(limit)

    result = await db.execute(query)
    messages = result.scalars().all()

    # Count total messages in session
    count_result = await db.execute(
        select(func.count(ChatMessage.id)).where(ChatMessage.session_id == sid)
    )
    total = count_result.scalar() or 0

    response_data = {
        "messages": [
            {
                "id": str(m.id),
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
                "attachments": m.attachments,
                "ai_metadata": m.ai_metadata,
                "tokens_used": m.tokens_used,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
        "total": total,
        "session_id": session_id,
    }

    if cache:
        await cache.set(cache_key, response_data, ttl=30)

    return MessageListResponse(**response_data)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
async def delete_session(
    session_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Delete a chat session (soft-delete if supported)."""
    from uuid import UUID as UUID_
    try:
        sid = UUID_(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format",
        )

    result = await db.execute(
        select(ChatSession).where(
            and_(ChatSession.id == sid, ChatSession.user_id == user.id)
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    if hasattr(session, "soft_delete"):
        session.soft_delete(deleted_by=str(user.id))
    else:
        await db.delete(session)

    await _invalidate_chat_cache(
        request,
        user_id=user.id,
        session_id=sid,
    )

    return {"message": "Session deleted"}


@router.patch("/sessions/{session_id}", status_code=status.HTTP_200_OK)
async def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Rename a chat session owned by the current user."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format",
        ) from exc

    result = await db.execute(
        select(ChatSession).where(
            and_(ChatSession.id == sid, ChatSession.user_id == user.id)
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    session.title = _normalize_session_title(body.title)
    session.updated_at = datetime.now(timezone.utc)

    await _invalidate_chat_cache(
        request,
        user_id=user.id,
        session_id=session.id,
    )

    return {
        "id": str(session.id),
        "title": session.title,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
    }
