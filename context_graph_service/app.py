"""
Context Graph Service — FastAPI Application.

Production-ready Neo4j-backed longitudinal patient memory:
  - Three-memory architecture (short-term, long-term, reasoning)
  - POLE+O entity knowledge graph
  - Care timeline with temporal ordering
  - Decision trace provenance
  - Graph-based multi-hop queries
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from context_graph_service.config import load_config
from context_graph_service.graph_client import (
    CareTimeline,
    ConversationMessage,
    DecisionTrace,
    Entity,
    Neo4jGraphClient,
    Relationship,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


class _State:
    config = None
    graph: Optional[Neo4jGraphClient] = None


_state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.config = load_config()
    _state.graph = Neo4jGraphClient(_state.config)
    try:
        _state.graph.connect()
        logger.info("Context Graph service started")
    except Exception as e:
        logger.error("Failed to connect to Neo4j: %s", e)
    yield
    if _state.graph:
        _state.graph.close()


app = FastAPI(
    title="MedAI Context Graph Service",
    version="0.1.0",
    description="Neo4j-backed longitudinal patient memory with POLE+O knowledge graph",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Request Models ────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    session_id: str
    patient_id: str
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityRequest(BaseModel):
    entity_type: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_doc_id: str = ""
    patient_id: Optional[str] = None
    relation: str = "RELATED_TO"


class RelationshipRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_doc_id: str = ""


class CareEventRequest(BaseModel):
    patient_id: str
    event_type: str
    event_date: str
    description: str
    provider: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecisionTraceRequest(BaseModel):
    patient_id: str
    query: str
    thought_chain: List[str]
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    conclusion: str
    confidence: float = 0.0
    sources: List[str] = Field(default_factory=list)


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if _state.graph is None:
        return {"status": "not_initialized"}
    return _state.graph.health()


# ── Short-Term Memory ────────────────────────────────────────────────────

@app.post("/memory/short-term/message")
def store_message(req: MessageRequest):
    """Store a conversation message."""
    msg = ConversationMessage(
        message_id=hashlib.sha256(
            f"{req.session_id}::{req.content[:100]}::{time.time()}".encode()
        ).hexdigest()[:32],
        session_id=req.session_id,
        patient_id=req.patient_id,
        role=req.role,
        content=req.content,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata=req.metadata,
    )
    _state.graph.store_message(msg)
    return {"message_id": msg.message_id, "stored": True}


@app.get("/memory/short-term/{session_id}")
def get_history(session_id: str, limit: int = 50):
    """Get conversation history for a session."""
    return {"messages": _state.graph.get_conversation_history(session_id, limit)}


# ── Long-Term Memory (Entity Graph) ──────────────────────────────────────

@app.post("/memory/long-term/entity")
def upsert_entity(req: EntityRequest):
    """Create or update a POLE+O entity."""
    entity_id = hashlib.sha256(
        f"{req.entity_type}::{req.name}".encode()
    ).hexdigest()[:32]

    entity = Entity(
        entity_id=entity_id,
        entity_type=req.entity_type,
        name=req.name,
        properties=req.properties,
        source_doc_id=req.source_doc_id,
    )
    _state.graph.upsert_entity(entity)

    # Link to patient if provided
    if req.patient_id:
        _state.graph.link_entity_to_patient(req.patient_id, entity_id, req.relation)

    return {"entity_id": entity_id, "stored": True}


@app.post("/memory/long-term/relationship")
def create_relationship(req: RelationshipRequest):
    """Create a relationship between two entities."""
    rel = Relationship(
        source_id=req.source_id,
        target_id=req.target_id,
        relation_type=req.relation_type,
        properties=req.properties,
        source_doc_id=req.source_doc_id,
    )
    _state.graph.create_relationship(rel)
    return {"created": True}


@app.get("/memory/long-term/patient/{patient_id}/entities")
def get_patient_entities(patient_id: str):
    """Get all entities related to a patient."""
    return {"entities": _state.graph.get_patient_entities(patient_id)}


# ── Reasoning Memory (Decision Traces) ───────────────────────────────────

@app.post("/memory/reasoning/trace")
def store_trace(req: DecisionTraceRequest):
    """Store a decision trace with provenance."""
    trace = DecisionTrace(
        trace_id=hashlib.sha256(
            f"{req.patient_id}::{req.query[:100]}::{time.time()}".encode()
        ).hexdigest()[:32],
        patient_id=req.patient_id,
        query=req.query,
        thought_chain=req.thought_chain,
        tool_calls=req.tool_calls,
        conclusion=req.conclusion,
        confidence=req.confidence,
        sources=req.sources,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _state.graph.store_decision_trace(trace)
    return {"trace_id": trace.trace_id, "stored": True}


@app.get("/memory/reasoning/{patient_id}/traces")
def get_traces(patient_id: str, limit: int = 20):
    """Get decision traces for a patient."""
    return {"traces": _state.graph.get_decision_traces(patient_id, limit)}


# ── Care Timeline ────────────────────────────────────────────────────────

@app.post("/timeline/event")
def add_care_event(req: CareEventRequest):
    """Add an event to a patient's care timeline."""
    event = CareTimeline(
        event_id=hashlib.sha256(
            f"{req.patient_id}::{req.event_type}::{req.event_date}::{time.time()}".encode()
        ).hexdigest()[:32],
        patient_id=req.patient_id,
        event_type=req.event_type,
        event_date=req.event_date,
        description=req.description,
        provider=req.provider,
        metadata=req.metadata,
    )
    _state.graph.add_care_event(event)
    return {"event_id": event.event_id, "stored": True}


@app.get("/timeline/{patient_id}")
def get_timeline(patient_id: str, event_type: Optional[str] = None, limit: int = 50):
    """Get a patient's care timeline."""
    return {"events": _state.graph.get_care_timeline(patient_id, event_type, limit)}


# ── Patient Summary ──────────────────────────────────────────────────────

@app.get("/patient/{patient_id}/summary")
def get_patient_summary(patient_id: str):
    """Get comprehensive patient summary from the graph."""
    return _state.graph.get_patient_summary(patient_id)


@app.get("/patient/{patient_id}/related")
def get_related_patients(patient_id: str):
    """Find patients with shared entities."""
    return {"related": _state.graph.find_related_patients(patient_id)}


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(app, host=config.host, port=config.port)
