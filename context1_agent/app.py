"""
Chroma Context-1 Agent Service — FastAPI Application.

STATUS: WATCH MODE — using stub harness until official release.

Provides:
  - /query — Multi-hop agentic retrieval
  - /health — Service status + harness version
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from context1_agent.agent_harness import Context1AgentHarness, HARNESS_VERSION, MODEL_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


class _State:
    harness: Optional[Context1AgentHarness] = None


_state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.harness = Context1AgentHarness(
        openrag_url=os.getenv("CONTEXT1_OPENRAG_URL", "http://openrag-service:8006"),
        llm_url=os.getenv("CONTEXT1_LLM_URL", "http://general-llm:8004"),
        llm_model=os.getenv("CONTEXT1_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        max_turns=int(os.getenv("CONTEXT1_MAX_TURNS", "4")),
        max_tool_calls_per_turn=int(os.getenv("CONTEXT1_MAX_TOOL_CALLS", "3")),
        token_budget=int(os.getenv("CONTEXT1_TOKEN_BUDGET", "8000")),
        timeout=float(os.getenv("CONTEXT1_TIMEOUT", "60")),
    )
    logger.info(
        "Context-1 Agent started (harness=%s, model=%s)",
        HARNESS_VERSION, MODEL_ID,
    )
    yield
    if _state.harness:
        await _state.harness.close()


app = FastAPI(
    title="MedAI Context-1 Agent",
    version="0.1.0",
    description=(
        "Chroma Context-1 multi-hop retrieval agent. "
        "STATUS: Stub harness — official harness not yet released."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "context1-agent",
        "model_id": MODEL_ID,
        "harness_version": HARNESS_VERSION,
        "harness_status": "stub — official harness not yet public",
        "note": "Monitor https://huggingface.co/chromadb/context-1 for official release",
    }


@app.post("/query")
async def query(req: QueryRequest):
    """Execute a multi-hop retrieval query via Context-1 agent."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = await _state.harness.query(req.question)
    return asdict(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CONTEXT1_PORT", "8008")))
