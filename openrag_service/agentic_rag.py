"""
Agentic RAG — Multi-hop query decomposition and iterative retrieval.

Implements OpenRAG-style agentic workflows:
  1. Query decomposition: Break complex queries into sub-queries
  2. Parallel tool calling: Execute multiple retrievals in parallel
  3. Self-editing context: Prune irrelevant results mid-search
  4. Iterative refinement: Multi-turn retrieval until answer is complete

Works with the existing general-llm backbone for reasoning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from openrag_service.config import OpenRAGConfig
from openrag_service.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    """A single turn in the agentic retrieval process."""
    turn_number: int
    thought: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    pruned_results: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class AgenticRAGResult:
    """Complete result from an agentic RAG query."""
    query: str
    answer: str
    turns: List[AgentTurn] = field(default_factory=list)
    total_results_retrieved: int = 0
    total_results_kept: int = 0
    total_latency_ms: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)


DECOMPOSITION_PROMPT = """You are a medical query decomposition agent.
Given a complex medical question, break it into 1-4 focused sub-queries that can be
independently searched in a medical knowledge base.

Rules:
- Each sub-query should target ONE specific piece of information
- Sub-queries should be self-contained (not reference other sub-queries)
- Focus on medical facts, not opinions
- If the question is simple, return just one sub-query

Input question: {question}

Respond ONLY with a JSON array of strings. Example:
["What is the standard dosage of metformin for type 2 diabetes?", "What are the contraindications of metformin?"]"""


SYNTHESIS_PROMPT = """You are a medical AI assistant synthesizing search results.

Original question: {question}

Retrieved evidence:
{evidence}

Instructions:
- Synthesize a clear, evidence-based answer
- Cite specific sources using [Source N] notation
- Flag any contradictions between sources
- If evidence is insufficient, say so clearly
- Include confidence level: HIGH / MEDIUM / LOW
- Never fabricate medical information

Answer:"""


CONTEXT_PRUNE_PROMPT = """Given the query and retrieved documents below, identify which documents
are NOT relevant to answering the query. Return ONLY a JSON array of document indices (0-based)
that should be REMOVED.

Query: {query}

Documents:
{documents}

Return JSON array of indices to remove (empty array if all are relevant):"""


class AgenticRAGEngine:
    """Multi-hop agentic retrieval engine."""

    def __init__(self, config: OpenRAGConfig, pipeline: IngestionPipeline) -> None:
        self.config = config
        self.pipeline = pipeline
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.config.agent_timeout_seconds)
        return self._http_client

    async def _llm_call(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call the general-llm backbone."""
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.config.llm_url}/v1/chat/completions",
                json={
                    "model": self.config.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return ""

    async def _decompose_query(self, question: str) -> List[str]:
        """Decompose a complex query into sub-queries."""
        prompt = DECOMPOSITION_PROMPT.format(question=question)
        result = await self._llm_call(prompt, max_tokens=512)

        try:
            sub_queries = json.loads(result)
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                return sub_queries[:4]  # cap at 4
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: use original question
        return [question]

    async def _parallel_search(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Execute multiple searches in parallel."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None, self.pipeline.search, query, top_k, True
            )
            for query in queries
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and deduplicate
        seen_texts = set()
        all_results = []
        for i, results in enumerate(results_list):
            if isinstance(results, Exception):
                logger.warning("Search failed for sub-query %d: %s", i, results)
                continue
            for r in results:
                text_hash = hash(r["text"][:200])
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    r["source_query"] = queries[i]
                    all_results.append(r)

        return all_results

    async def _prune_context(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Self-editing: prune irrelevant results."""
        if len(results) <= 3:
            return results

        docs_text = "\n".join(
            f"[{i}] {r['text'][:300]}" for i, r in enumerate(results)
        )
        prompt = CONTEXT_PRUNE_PROMPT.format(query=query, documents=docs_text)
        response = await self._llm_call(prompt, max_tokens=256)

        try:
            indices_to_remove = set(json.loads(response))
            return [r for i, r in enumerate(results) if i not in indices_to_remove]
        except (json.JSONDecodeError, TypeError):
            return results

    async def _synthesize(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        """Synthesize final answer from evidence."""
        evidence_text = "\n\n".join(
            f"[Source {i+1}] (score: {r.get('score', 0):.3f}, "
            f"doc: {r.get('metadata', {}).get('filename', 'unknown')}):\n{r['text']}"
            for i, r in enumerate(evidence[:10])
        )
        prompt = SYNTHESIS_PROMPT.format(question=question, evidence=evidence_text)
        return await self._llm_call(prompt, max_tokens=self.config.llm_max_tokens)

    async def query(self, question: str, top_k: int = 10) -> AgenticRAGResult:
        """Execute a full agentic RAG query."""
        start = time.monotonic()
        turns = []
        all_results = []

        # Turn 1: Decompose and search
        sub_queries = await self._decompose_query(question)
        turn1_start = time.monotonic()
        results = await self._parallel_search(sub_queries, top_k=top_k)
        all_results.extend(results)

        turns.append(AgentTurn(
            turn_number=1,
            thought=f"Decomposed into {len(sub_queries)} sub-queries",
            tool_calls=[{"type": "search", "query": q} for q in sub_queries],
            results=[{"text": r["text"][:200], "score": r.get("score", 0)} for r in results],
            latency_ms=(time.monotonic() - turn1_start) * 1000,
        ))

        # Turn 2: Prune irrelevant context
        if self.config.enable_multi_agent and len(results) > 3:
            turn2_start = time.monotonic()
            pruned = await self._prune_context(question, results)
            pruned_ids = [r["text"][:50] for r in results if r not in pruned]

            turns.append(AgentTurn(
                turn_number=2,
                thought=f"Pruned {len(results) - len(pruned)} irrelevant results",
                pruned_results=pruned_ids,
                latency_ms=(time.monotonic() - turn2_start) * 1000,
            ))
            results = pruned

        # Turn 3: Synthesize answer
        turn3_start = time.monotonic()
        answer = await self._synthesize(question, results)

        turns.append(AgentTurn(
            turn_number=len(turns) + 1,
            thought="Synthesized final answer from evidence",
            latency_ms=(time.monotonic() - turn3_start) * 1000,
        ))

        total_latency = (time.monotonic() - start) * 1000

        sources = [
            {
                "text": r["text"][:500],
                "score": r.get("score", 0),
                "doc_id": r.get("doc_id", ""),
                "filename": r.get("metadata", {}).get("filename", ""),
            }
            for r in results[:10]
        ]

        return AgenticRAGResult(
            query=question,
            answer=answer,
            turns=turns,
            total_results_retrieved=len(all_results),
            total_results_kept=len(results),
            total_latency_ms=round(total_latency, 2),
            sources=sources,
        )

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
