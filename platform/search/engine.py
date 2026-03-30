"""
Unified Search Engine — browsing-style medical retrieval orchestration.

Architecture:
    query analyzer -> search orchestration -> extraction/rerank ->
    smart context builder -> citations payload

When the medical-llm service exposes the advanced retrieval pipeline, this
service delegates to it. If that path is unavailable, it falls back to its
own source connectors.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus, urlparse

import httpx

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    content: str
    url: str = ""
    source: str = ""
    relevance_score: float = 0.0
    authority_score: float = 0.0
    freshness_score: float = 0.0
    citation_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "confidence": round(self.citation_confidence or self.relevance_score, 3),
            **({"domain": self.metadata.get("domain")} if self.metadata.get("domain") else {}),
        }


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    total: int = 0
    sources_queried: list[str] = field(default_factory=list)
    cached: bool = False
    query_analysis: dict[str, Any] = field(default_factory=dict)
    context: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class _SearchPlan:
    query: str
    sanitized_query: str
    redacted_query: str
    intent: str
    needs_freshness: bool
    sources: list[str]
    rewritten_queries: list[str]
    removed_phi: list[str] = field(default_factory=list)


class _SearchCache:
    """LRU cache with TTL for search results."""

    def __init__(self, max_size: int = 500, ttl_seconds: float = 600.0) -> None:
        self._cache: OrderedDict[str, tuple[SearchResponse, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, key: str) -> SearchResponse | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        response, timestamp = entry
        if (time.monotonic() - timestamp) > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return response

    def set(self, key: str, response: SearchResponse) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (response, time.monotonic())
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


class SearchEngine:
    """
    Search across external medical sources, the medical-llm retrieval service,
    and the local vector layer.
    """

    _RESEARCH_HINTS = (
        "latest", "recent", "current", "guideline", "guidelines",
        "trial", "study", "evidence", "pubmed", "nih", "who", "2024", "2025", "2026",
    )
    _PHI_PATTERNS = {
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE),
        "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"),
        "mrn": re.compile(r"\b(?:mrn|medical record number)[:#\s-]*[a-z0-9-]{4,}\b", re.IGNORECASE),
    }
    _AUTHORITY = {
        "who.int": 0.99,
        "nih.gov": 0.98,
        "cdc.gov": 0.97,
        "pubmed": 0.96,
        "vector_db": 0.9,
        "guidelines": 0.95,
        "arxiv": 0.76,
    }

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._cache = _SearchCache(
            max_size=500,
            ttl_seconds=float(getattr(settings, "search_cache_ttl_seconds", 600)),
        )
        self._pubmed_last_call = 0.0
        self._pubmed_min_interval = 0.35
        self._pubmed_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        query: str,
        sources: list[str] | None = None,
        max_results: int = 10,
    ) -> SearchResponse:
        """Search across all enabled sources using the advanced retrieval plan."""
        plan = self._analyze_query(query, sources)

        cache_key = (
            f"{plan.redacted_query[:200]}:{','.join(sorted(plan.sources))}:{max_results}"
        )
        cached = self._cache.get(cache_key)
        if cached:
            cached.cached = True
            return cached

        delegated = await self._search_via_medical_backend(plan, max_results=max_results)
        if delegated is not None:
            self._cache.set(cache_key, delegated)
            return delegated

        tasks = []
        if "pubmed" in plan.sources:
            tasks.append(("pubmed", self._search_pubmed(plan, max_results)))
        if "web" in plan.sources:
            tasks.append(("web", self._search_web(plan, max_results)))
        if "vector_db" in plan.sources:
            tasks.append(("vector_db", self._search_vector_db(plan.redacted_query, max_results)))
        if "guidelines" in plan.sources:
            tasks.append(("guidelines", self._search_guidelines(plan, max_results)))

        all_results: list[SearchResult] = []
        sources_queried: list[str] = []

        for (source_name, _), result in zip(
            tasks,
            await asyncio.gather(*(task for _, task in tasks), return_exceptions=True),
        ):
            sources_queried.append(source_name)
            if isinstance(result, Exception):
                logger.warning("Search source %s failed: %s", source_name, result)
                continue
            all_results.extend(result)

        ranked = self._rank_results(plan.redacted_query, _deduplicate(all_results))
        response = SearchResponse(
            query=query,
            results=ranked[:max_results],
            total=len(ranked),
            sources_queried=sources_queried,
            query_analysis={
                "intent": plan.intent,
                "sanitized_query": plan.sanitized_query,
                "redacted_query": plan.redacted_query,
                "rewritten_queries": plan.rewritten_queries,
                "needs_freshness": plan.needs_freshness,
                "removed_phi": plan.removed_phi,
            },
            context=self._build_context(ranked[:max_results]),
            warnings=[
                "Protected health information was redacted before external search."
                for _ in [1] if plan.removed_phi
            ],
        )
        self._cache.set(cache_key, response)
        return response

    async def _search_via_medical_backend(
        self,
        plan: _SearchPlan,
        *,
        max_results: int,
    ) -> SearchResponse | None:
        try:
            resp = await self._client.post(
                f"{settings.medical_llm_url}/rag/search",
                json={
                    "query": plan.query,
                    "max_results": max_results,
                    "sources": plan.sources,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.info("Delegated retrieval unavailable, using local search fallback: %s", exc)
            return None

        sources = data.get("sources", [])
        results = [
            SearchResult(
                title=item.get("title", ""),
                content=item.get("excerpt") or item.get("content", ""),
                url=item.get("url", ""),
                source=item.get("source", ""),
                relevance_score=float(item.get("confidence", 0.0) or 0.0),
                citation_confidence=float(item.get("confidence", 0.0) or 0.0),
                metadata={
                    "domain": item.get("domain", ""),
                    "type": item.get("type", ""),
                    "published_at": item.get("published_at", ""),
                },
            )
            for item in sources
        ]

        return SearchResponse(
            query=plan.query,
            results=results,
            total=int(data.get("total", len(results))),
            sources_queried=data.get("metadata", {}).get("sources_queried", plan.sources),
            cached=bool(data.get("metadata", {}).get("cached", False)),
            query_analysis=data.get("query_analysis", {}),
            context=data.get("context", ""),
            warnings=data.get("warnings", []),
        )

    async def _search_pubmed(self, plan: _SearchPlan, max_results: int) -> list[SearchResult]:
        async with self._pubmed_lock:
            now = time.monotonic()
            elapsed = now - self._pubmed_last_call
            if elapsed < self._pubmed_min_interval:
                await asyncio.sleep(self._pubmed_min_interval - elapsed)
            self._pubmed_last_call = time.monotonic()

        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": plan.rewritten_queries[0],
            "retmax": min(max_results, 20),
            "retmode": "json",
            "sort": "date" if plan.needs_freshness else "relevance",
        }
        if settings.pubmed_api_key:
            params["api_key"] = settings.pubmed_api_key
            self._pubmed_min_interval = 0.1

        try:
            resp = await self._client.get(f"{base}/esearch.fcgi", params=params)
            resp.raise_for_status()
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            summary_resp = await self._client.get(
                f"{base}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            )
            summary_resp.raise_for_status()
            summaries = summary_resp.json().get("result", {})
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("PubMed search failed: %s", exc)
            return []

        results: list[SearchResult] = []
        for pmid in ids:
            article = summaries.get(pmid, {})
            if not isinstance(article, dict):
                continue
            results.append(
                SearchResult(
                    title=article.get("title", ""),
                    content=article.get("sorttitle", "") or article.get("title", ""),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    source="pubmed",
                    relevance_score=0.8,
                    citation_confidence=0.8,
                    metadata={"domain": "pubmed.ncbi.nlm.nih.gov"},
                ),
            )
        return results

    async def _search_web(self, plan: _SearchPlan, max_results: int) -> list[SearchResult]:
        if not getattr(settings, "searxng_url", ""):
            return []

        domains = getattr(settings, "trusted_medical_domains", [])
        query = f"{plan.rewritten_queries[0]} ({' OR '.join(f'site:{d}' for d in domains[:8])})"

        try:
            resp = await self._client.get(
                f"{settings.searxng_url.rstrip('/')}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": "en-US",
                    "safesearch": 1,
                },
                headers={"User-Agent": "MedAI-Platform-Search/1.0"},
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("SearxNG web search failed: %s", exc)
            return []

        results: list[SearchResult] = []
        for item in payload.get("results", [])[:max_results]:
            url = item.get("url") or ""
            domain = _extract_domain(url)
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    url=url,
                    source="web",
                    relevance_score=0.6,
                    citation_confidence=0.6,
                    metadata={"domain": domain},
                ),
            )
        return results

    async def _search_vector_db(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            resp = await self._client.post(
                f"{settings.medical_llm_url}/rag/query",
                json={"query": query, "top_k": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Vector DB search failed: %s", exc)
            return []

        return [
            SearchResult(
                title=item.get("title", "Knowledge Base"),
                content=item.get("content", ""),
                source="vector_db",
                relevance_score=float(item.get("score", 0.7) or 0.0),
                citation_confidence=float(item.get("score", 0.7) or 0.0),
                metadata={"domain": _extract_domain(item.get("source", ""))},
            )
            for item in data.get("results", [])
        ]

    async def _search_guidelines(self, plan: _SearchPlan, max_results: int) -> list[SearchResult]:
        clean = quote_plus(plan.redacted_query[:120])
        entries = [
            SearchResult(
                title="WHO Clinical Guidelines",
                content=f"Search WHO guidance for: {plan.redacted_query[:120]}",
                url=f"https://www.who.int/publications/search?query={clean}",
                source="guidelines",
                relevance_score=0.74,
                citation_confidence=0.74,
                metadata={"domain": "who.int"},
            ),
            SearchResult(
                title="NIH Clinical Guidance",
                content=f"Search NIH resources for: {plan.redacted_query[:120]}",
                url=f"https://search.nih.gov/search?utf8=%E2%9C%93&affiliate=nih&query={clean}",
                source="guidelines",
                relevance_score=0.72,
                citation_confidence=0.72,
                metadata={"domain": "nih.gov"},
            ),
            SearchResult(
                title="NICE Guidelines",
                content=f"Search NICE guidance for: {plan.redacted_query[:120]}",
                url=f"https://www.nice.org.uk/search?q={clean}",
                source="guidelines",
                relevance_score=0.7,
                citation_confidence=0.7,
                metadata={"domain": "nice.org.uk"},
            ),
        ]
        return entries[:max_results]

    def _analyze_query(self, query: str, requested_sources: list[str] | None) -> _SearchPlan:
        sanitized = _sanitize_query(query)
        redacted = sanitized
        removed_phi: list[str] = []
        for label, pattern in self._PHI_PATTERNS.items():
            for match in pattern.findall(redacted):
                removed_phi.append(f"{label}:{match}")
            redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)

        lowered = sanitized.lower()
        needs_freshness = any(term in lowered for term in ("latest", "recent", "current", "2025", "2026"))
        if "guideline" in lowered or "guidelines" in lowered:
            intent = "guideline_lookup"
        elif any(term in lowered for term in ("drug", "dose", "dosage", "interaction", "side effect")):
            intent = "drug_safety"
        elif any(term in lowered for term in self._RESEARCH_HINTS):
            intent = "evidence_lookup"
        else:
            intent = "clinical_question"

        if requested_sources:
            sources = list(dict.fromkeys(requested_sources))
        else:
            sources = ["vector_db", "pubmed", "guidelines"]
            if any(term in lowered for term in self._RESEARCH_HINTS):
                sources.append("web")

        rewrites = [redacted]
        if intent == "guideline_lookup":
            rewrites.extend([
                f"{redacted} latest clinical guidelines WHO NIH",
                f"{redacted} guideline PubMed review",
            ])
        elif intent == "drug_safety":
            rewrites.extend([
                f"{redacted} medication safety NIH FDA",
                f"{redacted} contraindications PubMed",
            ])
        elif intent == "evidence_lookup":
            rewrites.extend([
                f"{redacted} latest medical evidence PubMed WHO NIH",
                f"{redacted} systematic review guideline 2025 2026",
            ])
        else:
            rewrites.append(f"{redacted} clinical evidence")

        return _SearchPlan(
            query=query,
            sanitized_query=sanitized,
            redacted_query=redacted,
            intent=intent,
            needs_freshness=needs_freshness,
            sources=sources,
            rewritten_queries=list(dict.fromkeys(rewrites))[:5],
            removed_phi=removed_phi,
        )

    def _rank_results(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked: list[SearchResult] = []
        for result in results:
            text_tokens = set(re.findall(r"[a-z0-9]+", f"{result.title} {result.content}".lower()))
            lexical = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            domain = result.metadata.get("domain") or _extract_domain(result.url)
            authority = self._AUTHORITY.get(domain, self._AUTHORITY.get(result.source, 0.65))
            freshness = 0.6
            year_match = re.search(r"\b(20\d{2})\b", f"{result.title} {result.content}")
            if year_match:
                current_year = time.gmtime().tm_year
                age = max(0, current_year - int(year_match.group(1)))
                freshness = max(0.35, 1.0 - (age * 0.15))

            result.authority_score = authority
            result.freshness_score = freshness
            result.relevance_score = max(result.relevance_score, lexical)
            result.citation_confidence = round(
                (result.relevance_score * 0.45) + (authority * 0.35) + (freshness * 0.20),
                4,
            )
            ranked.append(result)

        ranked.sort(key=lambda item: item.citation_confidence, reverse=True)
        return ranked

    def _build_context(self, results: list[SearchResult]) -> str:
        parts: list[str] = []
        for idx, result in enumerate(results[:6], start=1):
            excerpt = result.content[:320]
            url = result.url or "N/A"
            parts.append(
                f"[{idx}] {result.title}\n"
                f"Source: {result.source}\n"
                f"URL: {url}\n"
                f"Excerpt: {excerpt}"
            )
        return "\n\n".join(parts)


def _sanitize_query(query: str) -> str:
    clean = query.replace("\\", "").replace('"', "").replace("'", "")
    clean = re.sub(r"<[^>]+>", "", clean)
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean[:500].strip()


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []

    for result in results:
        key = result.url or hashlib.sha256(result.content[:240].encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)

    return unique


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""
