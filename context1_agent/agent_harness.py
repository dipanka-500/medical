"""
Chroma Context-1 — Agent Harness Stub.

STATUS: WATCH MODE — The official Context-1 agent harness is NOT YET PUBLIC.
This module provides the harness interface so the rest of the system can
integrate now and switch to the real harness when Chroma releases it.

Context-1 is a 20B MoE model trained for agentic multi-hop retrieval:
  - Query decomposition into targeted sub-queries
  - Parallel tool calling (avg 2.56 calls/turn)
  - Self-editing context (0.94 prune accuracy)
  - Cross-domain generalization

Until the official harness is released, this stub:
  1. Accepts multi-hop queries
  2. Decomposes them using the existing general-llm backbone
  3. Executes retrieval against the OpenRAG pipeline
  4. Returns results in the Context-1 expected format

When the official harness drops, replace this file's internals while
keeping the same external API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Context-1 Agent Configuration ─────────────────────────────────────────

MODEL_ID = "chromadb/context-1"
HARNESS_VERSION = "stub-0.1.0"  # Will become "official-X.Y.Z" when released


@dataclass
class ToolCall:
    """A single tool call made by the agent."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None
    latency_ms: float = 0.0


@dataclass
class AgentTurn:
    """A single turn in the agent's search process."""
    turn_number: int
    thought: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    context_before: int = 0  # docs in context before pruning
    context_after: int = 0   # docs after pruning
    pruned_doc_ids: List[str] = field(default_factory=list)


@dataclass
class Context1Result:
    """Result from a Context-1 agent query."""
    query: str
    status: str  # "completed", "partial", "failed", "harness_unavailable"
    supporting_documents: List[Dict[str, Any]] = field(default_factory=list)
    turns: List[AgentTurn] = field(default_factory=list)
    total_tool_calls: int = 0
    total_turns: int = 0
    total_latency_ms: float = 0.0
    harness_version: str = HARNESS_VERSION
    model_id: str = MODEL_ID


class Context1AgentHarness:
    """Agent harness managing tool execution, token budgets, and context pruning.

    This is the STUB implementation. The real harness (when released by Chroma)
    will handle:
      - Tool execution loop
      - Token budget management
      - Context deduplication
      - Self-editing context pruning

    The stub delegates to the OpenRAG pipeline for retrieval.
    """

    def __init__(
        self,
        openrag_url: str = "http://openrag-service:8006",
        llm_url: str = "http://general-llm:8004",
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_turns: int = 4,
        max_tool_calls_per_turn: int = 3,
        token_budget: int = 8000,
        timeout: float = 60.0,
    ) -> None:
        self._openrag_url = openrag_url
        self._llm_url = llm_url
        self._llm_model = llm_model
        self._max_turns = max_turns
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._token_budget = token_budget
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _llm_call(self, prompt: str, max_tokens: int = 1024) -> str:
        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self._llm_url}/v1/chat/completions",
                json={
                    "model": self._llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Context-1 LLM call failed: %s", e)
            return ""

    async def _search_openrag(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search via the OpenRAG service."""
        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self._openrag_url}/search",
                data={"query": query, "top_k": top_k, "use_reranker": True},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as e:
            logger.warning("OpenRAG search failed: %s", e)
            return []

    async def _decompose_query(self, query: str) -> List[str]:
        """Simulate Context-1's query decomposition."""
        prompt = (
            "You are a multi-hop retrieval agent. Break this complex question "
            "into 1-4 targeted sub-queries for parallel search.\n\n"
            f"Question: {query}\n\n"
            "Return ONLY a JSON array of strings."
        )
        result = await self._llm_call(prompt, max_tokens=512)
        try:
            sub_queries = json.loads(result)
            if isinstance(sub_queries, list):
                return sub_queries[:4]
        except (json.JSONDecodeError, TypeError):
            pass
        return [query]

    async def _should_prune(self, query: str, doc: Dict[str, Any]) -> bool:
        """Simulate Context-1's self-editing context pruning (0.94 accuracy)."""
        # In the stub, we use a simple relevance threshold
        score = doc.get("score", 0) or doc.get("rerank_score", 0)
        return score < 0.2

    async def query(self, question: str) -> Context1Result:
        """Execute a full Context-1 agent query.

        Simulates the agent loop:
          Turn 1: Decompose query → parallel search
          Turn 2: Prune irrelevant docs (self-editing)
          Turn 3+: If gaps found, do targeted follow-up searches
        """
        start = time.monotonic()
        turns: List[AgentTurn] = []
        all_docs: List[Dict[str, Any]] = []
        total_tool_calls = 0

        # ── Turn 1: Decompose and parallel search ──────────────────────
        sub_queries = await self._decompose_query(question)
        tool_calls_t1 = []

        search_tasks = [self._search_openrag(sq) for sq in sub_queries]
        results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

        for sq, results in zip(sub_queries, results_list):
            tc = ToolCall(
                tool_name="search",
                arguments={"query": sq},
                result=len(results) if not isinstance(results, Exception) else 0,
                latency_ms=0,
            )
            tool_calls_t1.append(tc)
            if not isinstance(results, Exception):
                for r in results:
                    r["source_query"] = sq
                    all_docs.append(r)

        total_tool_calls += len(tool_calls_t1)
        turns.append(AgentTurn(
            turn_number=1,
            thought=f"Decomposed into {len(sub_queries)} sub-queries, retrieved {len(all_docs)} docs",
            tool_calls=tool_calls_t1,
            context_before=len(all_docs),
            context_after=len(all_docs),
        ))

        # ── Turn 2: Self-editing context pruning ───────────────────────
        if all_docs:
            pruned_ids = []
            kept = []
            for doc in all_docs:
                if await self._should_prune(question, doc):
                    pruned_ids.append(doc.get("doc_id", "unknown")[:16])
                else:
                    kept.append(doc)

            turns.append(AgentTurn(
                turn_number=2,
                thought=f"Self-editing: pruned {len(pruned_ids)} irrelevant documents",
                context_before=len(all_docs),
                context_after=len(kept),
                pruned_doc_ids=pruned_ids,
            ))
            all_docs = kept

        # ── Deduplicate ────────────────────────────────────────────────
        seen = set()
        deduped = []
        for doc in all_docs:
            key = hashlib.md5(doc.get("text", "")[:200].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                deduped.append(doc)
        all_docs = deduped

        total_latency = (time.monotonic() - start) * 1000

        return Context1Result(
            query=question,
            status="completed" if all_docs else "partial",
            supporting_documents=all_docs[:20],
            turns=turns,
            total_tool_calls=total_tool_calls,
            total_turns=len(turns),
            total_latency_ms=round(total_latency, 2),
        )

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
