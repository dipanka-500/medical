"""
WebSocket endpoints for real-time chat and notifications.

Features:
    - ConnectionManager with per-user connection limits
    - JWT authentication via first-message protocol (token never in URL)
    - Real-time AI query streaming over WebSocket
    - Notification push channel
    - Heartbeat / keepalive (30s ping)
    - Graceful error handling and cleanup
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select

from auth.service import TokenService
from config import settings
from db.models import AuditAction, User
from db.session import async_session_factory
from gateway.concurrency import AIQuerySemaphore
from security.audit import AuditService
from security.input_validator import sanitize_ai_query
from security.safety_pipeline import run_pre_query_checks, run_post_query_checks
from .chat import (
    _enforce_daily_query_limit,
    _get_or_create_session,
    _invalidate_chat_cache_for_app,
    _persist_chat_exchange,
    _resolve_subscription_tier,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_CONNECTIONS_PER_USER = 3


# ── Connection Manager ──────────────────────────────────────────────────


class ConnectionManager:
    """
    Manages active WebSocket connections keyed by user_id.

    Each user may have up to MAX_CONNECTIONS_PER_USER concurrent connections
    (e.g. multiple browser tabs). Provides targeted and broadcast messaging.
    """

    def __init__(self) -> None:
        # user_id -> list of WebSocket connections
        self._connections: dict[str, list[WebSocket]] = {}
        self._connection_count: int = 0

    @property
    def connection_count(self) -> int:
        return self._connection_count

    @property
    def user_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket, user_id: str) -> bool:
        """
        Accept and register a WebSocket connection.

        Returns False if the user has reached the max connection limit.
        """
        existing = self._connections.get(user_id, [])
        if len(existing) >= MAX_CONNECTIONS_PER_USER:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=f"Maximum {MAX_CONNECTIONS_PER_USER} connections per user",
            )
            logger.warning(
                "Connection rejected for user %s: max connections (%d) reached",
                user_id, MAX_CONNECTIONS_PER_USER,
            )
            return False

        await websocket.accept()
        self._connections.setdefault(user_id, []).append(websocket)
        self._connection_count += 1
        logger.info(
            "WebSocket connected: user=%s total_connections=%d",
            user_id, self._connection_count,
        )
        return True

    def disconnect(self, user_id: str, websocket: WebSocket | None = None) -> None:
        """Remove a specific connection (or all connections) for a user."""
        conns = self._connections.get(user_id)
        if not conns:
            return

        if websocket is not None:
            try:
                conns.remove(websocket)
                self._connection_count -= 1
            except ValueError:
                pass
            if not conns:
                del self._connections[user_id]
        else:
            self._connection_count -= len(conns)
            del self._connections[user_id]

        logger.info(
            "WebSocket disconnected: user=%s total_connections=%d",
            user_id, self._connection_count,
        )

    async def send_to_user(self, user_id: str, message: dict[str, Any]) -> None:
        """Send a JSON message to all connections of a specific user."""
        conns = self._connections.get(user_id, [])
        stale: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        # Clean up broken connections
        for ws in stale:
            self.disconnect(user_id, ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to all connected users (system announcements)."""
        for user_id in list(self._connections.keys()):
            await self.send_to_user(user_id, message)


# Singleton managers
chat_manager = ConnectionManager()
notification_manager = ConnectionManager()


# ── JWT Authentication Helper ────────────────────────────────────────────


def _authenticate_token(token: str | None) -> str:
    """
    Validate a JWT token and return the user_id (sub claim).

    Raises ValueError if authentication fails.
    """
    if not token:
        raise ValueError("Missing authentication token")

    try:
        payload = TokenService.decode_token(token)
    except ValueError:
        raise ValueError("Invalid or expired token")

    if payload.get("type") != "access":
        raise ValueError("Invalid token type — use an access token")

    user_id = payload.get("sub")
    if not user_id:
        raise ValueError("Invalid token payload: missing subject")

    return user_id


# ── Heartbeat Task ───────────────────────────────────────────────────────


async def _heartbeat(websocket: WebSocket, interval: float = 30.0) -> None:
    """Send periodic pings to keep the WebSocket connection alive."""
    try:
        while True:
            await asyncio.sleep(interval)
            await websocket.send_json({"type": "ping", "ts": time.time()})
    except Exception:
        # Connection closed — heartbeat task exits silently
        pass


# ── First-message authentication helper ─────────────────────────────────


async def _authenticate_first_message(websocket: WebSocket, timeout: float = 10.0) -> str:
    """
    Accept the WebSocket, then wait for the first message to carry the JWT.

    Expected first message: ``{"type": "auth", "token": "<jwt>"}``

    This avoids putting the JWT in the URL query string, which can leak
    through browser history, proxy access logs, and server-side tracing.

    Returns the authenticated user_id, or closes the socket and raises.
    """
    await websocket.accept()

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
    except asyncio.TimeoutError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Auth timeout")
        raise ValueError("Auth timeout")
    except WebSocketDisconnect:
        raise ValueError("Disconnected before auth")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid JSON")
        raise ValueError("Invalid JSON")

    if data.get("type") != "auth" or not data.get("token"):
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="First message must be {\"type\": \"auth\", \"token\": \"...\"}",
        )
        raise ValueError("Missing auth message")

    try:
        user_id = _authenticate_token(data["token"])
    except ValueError as exc:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc))
        raise

    return user_id


# ── /ws/chat Endpoint ────────────────────────────────────────────────────


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Real-time chat WebSocket endpoint.

    Authentication: first message must be ``{"type": "auth", "token": "<jwt>"}``.
    The JWT is never placed in the URL, avoiding leaks via browser history
    and proxy logs.

    Message protocol (client -> server):
        {"type": "auth", "token": "<jwt>"}           (MUST be first message)
        {"type": "query", "query": "...", "session_id": "...", "mode": "doctor",
         "patient_id": "...", "web_search": false, "deep_reasoning": false}

    Response protocol (server -> client):
        {"type": "auth_ok"}
        {"type": "status", "status": "processing"}
        {"type": "chunk", "text": "...", "done": false}
        {"type": "complete", "done": true, "confidence": 0.95, "session_id": "..."}
        {"type": "error", "detail": "..."}
        {"type": "ping", "ts": 1234567890.0}
    """
    # ── Authenticate via first message ────────────────────────────
    try:
        user_id = await _authenticate_first_message(websocket)
    except ValueError:
        return  # Socket already closed inside helper
    try:
        parsed_user_id = uuid.UUID(user_id)
    except ValueError:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid token subject",
        )
        return

    # Register connection (socket is already accepted by _authenticate_first_message)
    existing = chat_manager._connections.get(user_id, [])
    if len(existing) >= MAX_CONNECTIONS_PER_USER:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=f"Maximum {MAX_CONNECTIONS_PER_USER} connections per user",
        )
        return
    chat_manager._connections.setdefault(user_id, []).append(websocket)
    chat_manager._connection_count += 1

    # Confirm auth to client
    await websocket.send_json({"type": "auth_ok"})

    # Start heartbeat
    heartbeat_task = asyncio.create_task(_heartbeat(websocket))

    try:
        while True:
            # Receive message
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "detail": "Invalid JSON",
                })
                continue

            msg_type = data.get("type")

            # Handle pong (client responding to our ping)
            if msg_type == "pong":
                continue

            if msg_type != "query":
                await websocket.send_json({
                    "type": "error",
                    "detail": f"Unknown message type: {msg_type}",
                })
                continue

            # ── Process Query ────────────────────────────────────
            start = time.monotonic()
            query = data.get("query", "")
            requested_session_id = data.get("session_id")
            mode = data.get("mode", "doctor")
            patient_id = data.get("patient_id")
            web_search = bool(data.get("web_search", False))
            deep_reasoning = bool(data.get("deep_reasoning", False))

            # Validate mode
            if mode not in ("doctor", "patient", "research"):
                await websocket.send_json({
                    "type": "error",
                    "detail": "Invalid mode. Must be one of: doctor, patient, research",
                })
                continue

            # Sanitize
            try:
                clean_query = sanitize_ai_query(query)
            except Exception:
                await websocket.send_json({
                    "type": "error",
                    "detail": "Query failed sanitization",
                })
                continue

            if not clean_query.strip():
                await websocket.send_json({
                    "type": "error",
                    "detail": "Query is empty after sanitization",
                })
                continue

            master_router = getattr(websocket.app.state, "master_router", None)
            if not master_router:
                await websocket.send_json({
                    "type": "error",
                    "detail": "AI engine not available",
                })
                continue

            async with async_session_factory() as db:
                try:
                    user_result = await db.execute(
                        select(User).where(User.id == parsed_user_id)
                    )
                    user = user_result.scalar_one_or_none()
                    if user is None or not user.is_active:
                        await websocket.send_json({
                            "type": "error",
                            "detail": "User account is unavailable",
                        })
                        await db.rollback()
                        continue

                    ws_tier = _resolve_subscription_tier(user)
                    ws_safety = run_pre_query_checks(clean_query, tier=ws_tier)
                    if not ws_safety["allowed"]:
                        await websocket.send_json({
                            "type": "error",
                            "detail": ws_safety["error"],
                        })
                        await db.rollback()
                        continue
                    clean_query = ws_safety["query"]

                    await _enforce_daily_query_limit(websocket.app, user=user, tier=ws_tier)
                    session = await _get_or_create_session(
                        db,
                        user=user,
                        session_id=requested_session_id,
                        query=clean_query,
                        patient_id=patient_id,
                    )

                    await websocket.send_json({
                        "type": "status",
                        "status": "processing",
                        "session_id": str(session.id),
                    })

                    ws_semaphore: AIQuerySemaphore | None = getattr(
                        websocket.app.state,
                        "ai_semaphore",
                        None,
                    )
                    if ws_semaphore:
                        try:
                            await ws_semaphore.acquire()
                        except (OverflowError, TimeoutError):
                            await websocket.send_json({
                                "type": "error",
                                "detail": "Service is at capacity. Please try again shortly.",
                            })
                            await db.rollback()
                            continue

                    try:
                        result = await asyncio.wait_for(
                            master_router.route_and_execute(
                                text=clean_query,
                                patient_id=patient_id,
                                session_id=str(session.id),
                                mode=mode,
                                user_id=user_id,
                                request_id=f"ws-{uuid.uuid4()}",
                                web_search=web_search,
                                deep_reasoning=deep_reasoning,
                            ),
                            timeout=settings.engine_timeout_seconds,
                        )
                    finally:
                        if ws_semaphore:
                            await ws_semaphore.release()

                    result = run_post_query_checks(result, mode=mode, tier=ws_tier)

                    if ws_safety.get("emergency_message"):
                        result["answer"] = (
                            ws_safety["emergency_message"]
                            + "\n\n"
                            + result.get("answer", "")
                        )

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
                        resource_type="chat_ws",
                        details={
                            "mode": mode,
                            "session_id": str(session.id),
                            "latency_ms": round(latency_ms, 1),
                            "query_length": len(clean_query),
                            "safety_category": ws_safety.get("safety_category"),
                            "feature_flags": {
                                "web_search": web_search,
                                "deep_reasoning": deep_reasoning,
                            },
                        },
                        ip_address=websocket.client.host if websocket.client else None,
                    )
                    await db.commit()
                    await _invalidate_chat_cache_for_app(
                        websocket.app,
                        user_id=user.id,
                        session_id=session.id,
                    )

                    answer = result.get("answer", "")
                    chunk_size = 50
                    for i in range(0, len(answer), chunk_size):
                        chunk = answer[i:i + chunk_size]
                        await websocket.send_json({
                            "type": "chunk",
                            "text": chunk,
                            "done": False,
                        })
                        await asyncio.sleep(0.02)

                    await websocket.send_json({
                        "type": "complete",
                        "done": True,
                        "confidence": result.get("confidence", 0.0),
                        "routing": result.get("routing"),
                        "session_id": str(session.id),
                        "message_id": str(assistant_message.id),
                        "latency_ms": round(latency_ms, 1),
                    })
                except HTTPException as exc:
                    await db.rollback()
                    await websocket.send_json({
                        "type": "error",
                        "detail": exc.detail,
                    })
                except asyncio.TimeoutError:
                    await db.rollback()
                    await websocket.send_json({
                        "type": "error",
                        "detail": "AI engine timed out. Please try again.",
                    })
                except Exception as exc:
                    await db.rollback()
                    logger.error(
                        "WebSocket AI query failed: user=%s error=%s",
                        user_id, str(exc)[:500],
                    )
                    await websocket.send_json({
                        "type": "error",
                        "detail": "AI analysis failed. Please try again.",
                    })

    except WebSocketDisconnect:
        logger.info("WebSocket chat client disconnected: user=%s", user_id)
    except Exception as exc:
        logger.error(
            "WebSocket chat error: user=%s error=%s",
            user_id, str(exc)[:500],
        )
        try:
            await websocket.send_json({
                "type": "error",
                "detail": "Internal server error",
            })
        except Exception:
            pass
    finally:
        heartbeat_task.cancel()
        chat_manager.disconnect(user_id, websocket)


# ── /ws/notifications Endpoint ───────────────────────────────────────────


@router.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    """
    Read-only notification WebSocket endpoint.

    Authentication: first message must be ``{"type": "auth", "token": "<jwt>"}``.

    Server pushes notifications to connected clients:
        {"type": "notification", "event": "analysis_complete", "data": {...}}
        {"type": "notification", "event": "system_alert", "message": "..."}
        {"type": "ping", "ts": 1234567890.0}

    Clients do not need to send further messages (read-only channel).
    """
    # ── Authenticate via first message ────────────────────────────
    try:
        user_id = await _authenticate_first_message(websocket)
    except ValueError:
        return

    # Register connection (socket already accepted by _authenticate_first_message)
    existing = notification_manager._connections.get(user_id, [])
    if len(existing) >= MAX_CONNECTIONS_PER_USER:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=f"Maximum {MAX_CONNECTIONS_PER_USER} connections per user",
        )
        return
    notification_manager._connections.setdefault(user_id, []).append(websocket)
    notification_manager._connection_count += 1

    await websocket.send_json({"type": "auth_ok"})

    # Start heartbeat
    heartbeat_task = asyncio.create_task(_heartbeat(websocket))

    try:
        # Keep the connection open; this endpoint is server-push only.
        # We still read from the socket to detect disconnects.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("WebSocket notifications client disconnected: user=%s", user_id)
    except Exception as exc:
        logger.error(
            "WebSocket notifications error: user=%s error=%s",
            user_id, str(exc)[:500],
        )
    finally:
        heartbeat_task.cancel()
        notification_manager.disconnect(user_id, websocket)
