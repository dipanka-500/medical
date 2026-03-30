"""
Advanced retrieval pipeline for medical search and grounding.

Architecture:
    Query understanding -> multi-source retrieval -> content extraction ->
    reranking/filtering -> smart context builder -> output filtering

The implementation is intentionally dependency-tolerant. Optional components
such as LangGraph, Trafilatura, CrossEncoder rerankers, and PDF parsing are
used when available and gracefully bypassed otherwise.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class RetrievalDocument:
    title: str
    content: str
    url: str = ""
    source: str = ""
    source_type: str = "web"
    relevance_score: float = 0.0
    authority_score: float = 0.0
    freshness_score: float = 0.0
    citation_confidence: float = 0.0
    domain: str = ""
    published_at: str = ""
    query_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def dedupe_key(self) -> str:
        if self.url:
            return self.url.strip().lower()
        text = f"{self.title}::{self.content[:240]}".lower()
        return hashlib.sha256(text.encode()).hexdigest()

    def to_source_payload(self) -> dict[str, Any]:
        excerpt = _normalize_whitespace(self.content)[:280]
        return {
            "title": self.title or self.source or "Untitled source",
            "url": self.url,
            "source": self.source or self.domain or self.source_type,
            "type": self.source_type,
            "domain": self.domain,
            "confidence": round(self.citation_confidence, 3),
            "published_at": self.published_at,
            "excerpt": excerpt,
        }

    def to_rag_evidence(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "metadata": {
                "source": self.url or self.source or self.domain or self.source_type,
                "type": self.source_type,
                "domain": self.domain,
                **self.metadata,
            },
            "relevance": round(self.citation_confidence, 4),
        }


@dataclass
class QueryAnalysis:
    original_query: str
    sanitized_query: str
    redacted_query: str
    intent: str
    needs_search: bool
    needs_freshness: bool
    medical_only: bool
    removed_phi: list[str] = field(default_factory=list)
    rewritten_queries: list[str] = field(default_factory=list)
    selected_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query_analysis: QueryAnalysis
    context: str
    documents: list[RetrievalDocument] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    rag_evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_analysis": self.query_analysis.to_dict(),
            "context": self.context,
            "sources": list(self.sources),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "documents": [doc.to_source_payload() for doc in self.documents],
        }


class _RetrievalCache:
    """Simple in-memory LRU cache with TTL."""

    def __init__(self, max_size: int = 128, ttl_seconds: float = 600.0) -> None:
        self._cache: OrderedDict[str, tuple[RetrievalResult, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get(self, key: str) -> RetrievalResult | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, timestamp = entry
        if (time.monotonic() - timestamp) > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: RetrievalResult) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.monotonic())
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


class MedicalQueryAnalyzer:
    """Decides whether external search is needed and rewrites the query."""

    _RESEARCH_PATTERNS = (
        r"\b(latest|recent|current|new|updated)\b",
        r"\b(guideline|guidelines|consensus|recommendation)\b",
        r"\b(study|studies|research|evidence|literature|trial|review|meta-analysis)\b",
        r"\b(pubmed|nih|who|cdc|nejm|lancet|arxiv)\b",
        r"\b(202[4-9])\b",
    )
    _DRUG_PATTERNS = (
        r"\b(dose|dosage|side effect|side effects|contraindication|interaction)\b",
    )
    _GUIDELINE_PATTERNS = (
        r"\b(guideline|guidelines|standard of care|consensus|position statement)\b",
    )
    _PHI_PATTERNS = {
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE),
        "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"),
        "dob": re.compile(
            r"\b(?:dob|date of birth)[:\s-]*(?:\d{1,2}[/-]){2}\d{2,4}\b",
            re.IGNORECASE,
        ),
        "mrn": re.compile(r"\b(?:mrn|medical record number)[:#\s-]*[a-z0-9-]{4,}\b", re.IGNORECASE),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.max_rewrites = max(2, int(config.get("max_rewrites", 5)))
        self.preferred_sources = list(config.get("preferred_sources", []))

    def analyze(
        self,
        query: str,
        requested_sources: list[str] | None = None,
    ) -> QueryAnalysis:
        sanitized = _normalize_whitespace(query)
        redacted, removed_phi = self._redact_phi(sanitized)
        lowered = sanitized.lower()

        needs_freshness = any(re.search(p, lowered) for p in self._RESEARCH_PATTERNS[:2]) or bool(
            re.search(self._RESEARCH_PATTERNS[-1], lowered),
        )
        is_guideline = any(re.search(p, lowered) for p in self._GUIDELINE_PATTERNS)
        is_drug = any(re.search(p, lowered) for p in self._DRUG_PATTERNS)
        is_research = any(re.search(p, lowered) for p in self._RESEARCH_PATTERNS)

        if is_guideline:
            intent = "guideline_lookup"
        elif is_drug:
            intent = "drug_safety"
        elif is_research:
            intent = "evidence_lookup"
        else:
            intent = "clinical_question"

        needs_search = is_research or len(_tokenize(sanitized)) > 8
        if requested_sources:
            selected_sources = list(dict.fromkeys(requested_sources))
        else:
            selected_sources = ["vector_db"]
            if needs_search:
                selected_sources.extend(["pubmed", "guidelines", "web"])
                if intent == "evidence_lookup":
                    selected_sources.append("arxiv")

        rewrites = self._rewrite_query(
            redacted_query=redacted or sanitized,
            intent=intent,
            needs_freshness=needs_freshness,
        )

        return QueryAnalysis(
            original_query=query,
            sanitized_query=sanitized,
            redacted_query=redacted or sanitized,
            intent=intent,
            needs_search=needs_search,
            needs_freshness=needs_freshness,
            medical_only=True,
            removed_phi=removed_phi,
            rewritten_queries=rewrites,
            selected_sources=selected_sources,
        )

    def _redact_phi(self, query: str) -> tuple[str, list[str]]:
        redacted = query
        removed: list[str] = []
        for label, pattern in self._PHI_PATTERNS.items():
            for match in pattern.findall(redacted):
                removed.append(f"{label}:{match}")
            redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
        return _normalize_whitespace(redacted), removed

    def _rewrite_query(
        self,
        *,
        redacted_query: str,
        intent: str,
        needs_freshness: bool,
    ) -> list[str]:
        candidates = [redacted_query]
        suffix = " clinical evidence"
        if intent == "guideline_lookup":
            candidates.extend([
                f"{redacted_query} latest clinical guidelines WHO NIH",
                f"{redacted_query} guideline site:who.int",
                f"{redacted_query} guideline site:nih.gov",
                f"{redacted_query} standards of care PubMed review",
            ])
        elif intent == "drug_safety":
            candidates.extend([
                f"{redacted_query} drug safety NIH FDA",
                f"{redacted_query} contraindications PubMed",
                f"{redacted_query} medication safety clinical guideline",
            ])
        elif intent == "evidence_lookup":
            candidates.extend([
                f"{redacted_query} latest medical evidence PubMed",
                f"{redacted_query} systematic review NIH WHO",
                f"{redacted_query} clinical trial 2025 2026",
                f"{redacted_query} arxiv biomedical preprint",
            ])
        else:
            candidates.extend([
                f"{redacted_query}{suffix}",
                f"{redacted_query} trusted medical sources",
            ])

        if needs_freshness:
            candidates.append(f"{redacted_query} 2025 2026 current guideline")

        seen: set[str] = set()
        rewrites: list[str] = []
        for candidate in candidates:
            normalized = _normalize_whitespace(candidate)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            rewrites.append(normalized)
            if len(rewrites) >= self.max_rewrites:
                break
        return rewrites


class ContentExtractionPipeline:
    """Fetches trusted pages and extracts clean text when possible."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.timeout_seconds = float(config.get("timeout_seconds", 8.0))
        self.max_fetches = int(config.get("max_fetches", 3))
        self.max_chars = int(config.get("max_chars", 6000))
        self.enabled = bool(config.get("enabled", True))
        self._user_agent = config.get(
            "user_agent",
            "MedAI-Retrieval/1.0 (+https://example.local/search)",
        )

    def enrich(self, documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
        if not self.enabled:
            return documents

        enriched: list[RetrievalDocument] = []
        remaining_fetches = self.max_fetches

        for doc in documents:
            if remaining_fetches <= 0 or not doc.url or doc.source_type in {"vector_db", "pubmed"}:
                enriched.append(doc)
                continue

            fetched = self._extract_url(doc.url)
            if fetched:
                doc.content = fetched
                doc.metadata["extraction_method"] = "full_text"
                remaining_fetches -= 1
            enriched.append(doc)

        return enriched

    def _extract_url(self, url: str) -> str:
        headers = {"User-Agent": self._user_agent}

        try:
            import requests

            response = requests.get(url, timeout=self.timeout_seconds, headers=headers)
            response.raise_for_status()
        except Exception as exc:
            logger.debug("Content extraction failed for %s: %s", url, exc)
            return ""

        content_type = response.headers.get("content-type", "").lower()
        body = response.text if "text" in content_type or "html" in content_type else ""

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return self._extract_pdf(response.content)

        extracted = self._extract_html(body)
        return extracted[: self.max_chars]

    def _extract_html(self, html: str) -> str:
        try:
            import trafilatura

            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
            )
            if extracted:
                return _normalize_whitespace(extracted)
        except Exception:
            pass

        text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        return _normalize_whitespace(text)

    def _extract_pdf(self, payload: bytes) -> str:
        try:
            from io import BytesIO
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(payload))
            pages = []
            for page in reader.pages[:3]:
                pages.append(page.extract_text() or "")
            return _normalize_whitespace(" ".join(pages))[: self.max_chars]
        except Exception:
            return ""


class EvidenceReranker:
    """Authority-aware reranker with optional semantic reranking."""

    _AUTHORITY_HINTS = {
        "who.int": 0.99,
        "nih.gov": 0.98,
        "cdc.gov": 0.97,
        "pubmed": 0.96,
        "ncbi.nlm.nih.gov": 0.96,
        "nice.org.uk": 0.95,
        "cochranelibrary.com": 0.94,
        "nejm.org": 0.93,
        "thelancet.com": 0.93,
        "bmj.com": 0.92,
        "vector_db": 0.9,
        "arxiv.org": 0.76,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.max_results = int(config.get("max_results", 8))
        self.enable_semantic = bool(config.get("enable_semantic_reranking", True))
        self.semantic_model = config.get("semantic_model", "BAAI/bge-reranker-large")
        self.trusted_domains = set(config.get("trusted_domains", []))
        self._semantic_model = None
        self._semantic_disabled = False

    def rerank(
        self,
        query: str,
        documents: list[RetrievalDocument],
        max_results: int | None = None,
    ) -> list[RetrievalDocument]:
        if not documents:
            return []

        filtered = self._filter_untrusted(documents)
        if not filtered:
            return []

        semantic_scores = self._semantic_scores(query, filtered)
        query_tokens = set(_tokenize(query))

        for idx, doc in enumerate(filtered):
            lexical = self._lexical_relevance(query_tokens, doc)
            authority = self._authority_score(doc)
            freshness = self._freshness_score(doc)
            semantic = semantic_scores[idx] if semantic_scores else lexical
            doc.relevance_score = lexical
            doc.authority_score = authority
            doc.freshness_score = freshness
            doc.citation_confidence = round(
                (semantic * 0.45) + (authority * 0.35) + (freshness * 0.20),
                4,
            )

        unique: list[RetrievalDocument] = []
        seen: set[str] = set()
        for doc in sorted(filtered, key=lambda item: item.citation_confidence, reverse=True):
            key = doc.dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)

        return unique[: (max_results or self.max_results)]

    def _filter_untrusted(self, documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
        filtered: list[RetrievalDocument] = []
        for doc in documents:
            if doc.source_type in {"vector_db", "pubmed", "guidelines", "nih", "who", "arxiv"}:
                filtered.append(doc)
                continue

            domain = doc.domain or _extract_domain(doc.url)
            if not domain:
                continue
            if domain.endswith(".gov") or domain.endswith(".edu"):
                filtered.append(doc)
                continue
            if domain in self.trusted_domains:
                filtered.append(doc)
        return filtered

    def _authority_score(self, doc: RetrievalDocument) -> float:
        domain = doc.domain or _extract_domain(doc.url)
        if domain in self._AUTHORITY_HINTS:
            return self._AUTHORITY_HINTS[domain]
        if doc.source_type in self._AUTHORITY_HINTS:
            return self._AUTHORITY_HINTS[doc.source_type]
        if domain.endswith(".gov"):
            return 0.94
        if domain.endswith(".edu"):
            return 0.9
        return 0.65

    def _freshness_score(self, doc: RetrievalDocument) -> float:
        candidates = " ".join(
            [
                doc.published_at,
                str(doc.metadata.get("year", "")),
                doc.title,
            ],
        )
        match = re.search(r"\b(20\d{2})\b", candidates)
        if not match:
            return 0.55

        year = int(match.group(1))
        current_year = time.gmtime().tm_year
        distance = max(0, current_year - year)
        return max(0.35, 1.0 - (distance * 0.15))

    def _lexical_relevance(self, query_tokens: set[str], doc: RetrievalDocument) -> float:
        haystack_tokens = set(_tokenize(f"{doc.title} {doc.content[:1000]}"))
        if not query_tokens or not haystack_tokens:
            return 0.0
        overlap = len(query_tokens & haystack_tokens)
        return min(1.0, overlap / max(len(query_tokens), 1))

    def _semantic_scores(
        self,
        query: str,
        documents: list[RetrievalDocument],
    ) -> list[float]:
        if not self.enable_semantic or self._semantic_disabled or not documents:
            return []

        try:
            if self._semantic_model is None:
                from sentence_transformers import CrossEncoder

                self._semantic_model = CrossEncoder(self.semantic_model)
        except Exception as exc:
            logger.warning("Semantic reranker unavailable, falling back to heuristic scoring: %s", exc)
            self._semantic_disabled = True
            return []

        pairs = [(query, f"{doc.title}\n{doc.content[:1200]}") for doc in documents]

        try:
            predictions = self._semantic_model.predict(pairs)
        except Exception as exc:
            logger.warning("Semantic reranker failed, disabling it for this process: %s", exc)
            self._semantic_disabled = True
            return []

        scores: list[float] = []
        for value in predictions:
            val = _safe_float(value, default=0.0)
            normalized = 1.0 / (1.0 + math.exp(-val))
            scores.append(normalized)
        return scores


class SmartContextBuilder:
    """Builds compact RAG context and citation payloads."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.max_context_chars = int(config.get("max_context_chars", 6000))
        self.max_sources = int(config.get("max_sources", 8))
        self.chunk_words = int(config.get("chunk_words", 180))
        self.chunk_overlap = int(config.get("chunk_overlap", 30))

    def build(
        self,
        query: str,
        documents: list[RetrievalDocument],
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        if not documents:
            return "", [], []

        context_parts: list[str] = []
        source_payloads: list[dict[str, Any]] = []
        evidence_payloads: list[dict[str, Any]] = []
        remaining = self.max_context_chars

        for idx, doc in enumerate(documents[: self.max_sources], start=1):
            excerpt = self._best_excerpt(query, doc.content)
            if not excerpt:
                continue

            source_payloads.append(doc.to_source_payload())
            evidence_payloads.append(doc.to_rag_evidence())

            header = (
                f"[{idx}] {doc.title or doc.source or doc.domain}\n"
                f"Source: {doc.source or doc.domain or doc.source_type}\n"
                f"URL: {doc.url or 'N/A'}\n"
                f"Score: {doc.citation_confidence:.3f}\n"
            )
            block = f"{header}Excerpt: {excerpt}\n"
            if len(block) > remaining:
                truncated = block[:remaining].rstrip()
                if truncated:
                    context_parts.append(truncated)
                break
            context_parts.append(block)
            remaining -= len(block)
            if remaining <= 0:
                break

        return "\n".join(context_parts), source_payloads, evidence_payloads

    def _best_excerpt(self, query: str, text: str) -> str:
        normalized = _normalize_whitespace(text)
        if not normalized:
            return ""

        chunks = self._chunk_text(normalized)
        query_tokens = set(_tokenize(query))
        best_chunk = ""
        best_score = -1.0
        for chunk in chunks:
            overlap = len(query_tokens & set(_tokenize(chunk)))
            if overlap > best_score:
                best_score = float(overlap)
                best_chunk = chunk
        return best_chunk[:900]

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.chunk_words:
            return [text]

        chunks: list[str] = []
        step = max(1, self.chunk_words - self.chunk_overlap)
        start = 0
        while start < len(words):
            end = min(len(words), start + self.chunk_words)
            chunks.append(" ".join(words[start:end]))
            start += step
        return chunks


class RetrievalOutputFilter:
    """Final safety and citation checks over retrieval output."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.require_citations = bool(config.get("require_citations", True))
        self.min_trusted_sources = int(config.get("min_trusted_sources", 1))

    def apply(self, result: RetrievalResult) -> RetrievalResult:
        trusted_sources = [source for source in result.sources if source.get("url") or source.get("source")]
        if self.require_citations and not trusted_sources:
            result.warnings.append("No trusted citations were retained after filtering.")
        elif len(trusted_sources) < self.min_trusted_sources:
            result.warnings.append(
                f"Only {len(trusted_sources)} trusted source(s) available for grounding.",
            )
        return result


class MedicalRetrievalPipeline:
    """End-to-end retrieval pipeline used by the Medical LLM service."""

    def __init__(
        self,
        *,
        rag_engine: Any | None = None,
        pubmed_fetcher: Any | None = None,
        web_search: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        config = config or {}
        self._rag_engine = rag_engine
        self._pubmed_fetcher = pubmed_fetcher
        self._web_search = web_search
        self.config = config
        self.query_analyzer = MedicalQueryAnalyzer(config.get("query_understanding", {}))
        self.extractor = ContentExtractionPipeline(config.get("content_extraction", {}))
        self.reranker = EvidenceReranker(config.get("reranker", {}))
        self.context_builder = SmartContextBuilder(config.get("context_builder", {}))
        self.output_filter = RetrievalOutputFilter(config.get("output_filter", {}))

        cache_cfg = config.get("cache", {})
        self._cache = _RetrievalCache(
            max_size=int(cache_cfg.get("max_size", 128)),
            ttl_seconds=float(cache_cfg.get("ttl_seconds", 600.0)),
        )
        self._parallel_sources = max(1, int(config.get("parallel_sources", 4)))
        self._graph = self._build_graph_if_available()

    def retrieve(
        self,
        query: str,
        *,
        max_results: int = 8,
        requested_sources: list[str] | None = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        cache_key = hashlib.sha256(
            f"{query}::{max_results}::{','.join(sorted(requested_sources or []))}".encode(),
        ).hexdigest()

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached.metadata = {**cached.metadata, "cached": True}
                return cached

        if self._graph is not None:
            try:
                state = self._graph.invoke(
                    {
                        "query": query,
                        "max_results": max_results,
                        "requested_sources": requested_sources,
                    },
                )
                result = state["result"]
            except Exception as exc:
                logger.warning("LangGraph orchestration failed, falling back to sequential pipeline: %s", exc)
                result = self._run_sequential(
                    query=query,
                    max_results=max_results,
                    requested_sources=requested_sources,
                )
        else:
            result = self._run_sequential(
                query=query,
                max_results=max_results,
                requested_sources=requested_sources,
            )

        if use_cache:
            self._cache.set(cache_key, result)
        return result

    def _run_sequential(
        self,
        *,
        query: str,
        max_results: int,
        requested_sources: list[str] | None,
    ) -> RetrievalResult:
        analysis = self.query_analyzer.analyze(query, requested_sources=requested_sources)
        raw_documents, sources_queried = self._retrieve_documents(analysis, max_results=max_results)
        enriched = self.extractor.enrich(raw_documents)
        reranked = self.reranker.rerank(
            analysis.redacted_query,
            enriched,
            max_results=max_results,
        )
        context, sources, rag_evidence = self.context_builder.build(
            analysis.redacted_query,
            reranked,
        )

        warnings: list[str] = []
        if analysis.removed_phi:
            warnings.append("Protected health information was redacted before external search.")
        if not reranked:
            warnings.append("No trusted medical evidence matched the query.")

        result = RetrievalResult(
            query_analysis=analysis,
            context=context,
            documents=reranked,
            sources=sources,
            rag_evidence=rag_evidence,
            warnings=warnings,
            metadata={
                "sources_queried": sources_queried,
                "candidate_documents": len(raw_documents),
                "retained_documents": len(reranked),
                "cached": False,
                "orchestrator": "langgraph" if self._graph is not None else "sequential",
            },
        )
        return self.output_filter.apply(result)

    def _retrieve_documents(
        self,
        analysis: QueryAnalysis,
        *,
        max_results: int,
    ) -> tuple[list[RetrievalDocument], list[str]]:
        tasks: list[tuple[str, Any, tuple[Any, ...], dict[str, Any]]] = []
        selected = analysis.selected_sources or ["vector_db"]

        if "vector_db" in selected and self._rag_engine is not None:
            tasks.append(("vector_db", self._search_vector_db, (analysis.redacted_query, max_results), {}))
        if "pubmed" in selected and self._pubmed_fetcher is not None:
            tasks.append(("pubmed", self._search_pubmed, (analysis, max_results), {}))
        if "guidelines" in selected and self._web_search is not None:
            tasks.append(("guidelines", self._search_guidelines, (analysis, max_results), {}))
        if "web" in selected and self._web_search is not None:
            tasks.append(("web", self._search_web, (analysis, max_results), {}))
        if "arxiv" in selected:
            tasks.append(("arxiv", self._search_arxiv, (analysis, max_results), {}))

        documents: list[RetrievalDocument] = []
        sources_queried: list[str] = []

        if not tasks:
            return documents, sources_queried

        with ThreadPoolExecutor(max_workers=min(self._parallel_sources, len(tasks))) as executor:
            futures = {
                executor.submit(fn, *args, **kwargs): source
                for source, fn, args, kwargs in tasks
            }
            for future in as_completed(futures):
                source = futures[future]
                sources_queried.append(source)
                try:
                    documents.extend(future.result() or [])
                except Exception as exc:
                    logger.warning("Retrieval source %s failed: %s", source, exc)

        return documents, sources_queried

    def _search_vector_db(self, query: str, max_results: int) -> list[RetrievalDocument]:
        try:
            self._rag_engine.initialize()
            results = self._rag_engine.query(query, top_k=max_results)
        except Exception as exc:
            logger.warning("Vector retrieval failed: %s", exc)
            return []

        documents: list[RetrievalDocument] = []
        for item in results:
            metadata = dict(item.get("metadata", {}))
            source = metadata.get("source", "vector_db")
            documents.append(
                RetrievalDocument(
                    title=metadata.get("title") or metadata.get("filename") or "Knowledge Base",
                    content=item.get("content", ""),
                    url=source if str(source).startswith(("http://", "https://")) else "",
                    source="vector_db",
                    source_type="vector_db",
                    domain=_extract_domain(str(source)),
                    relevance_score=_safe_float(item.get("relevance"), default=0.0),
                    query_used=query,
                    metadata=metadata,
                ),
            )
        return documents

    def _search_pubmed(
        self,
        analysis: QueryAnalysis,
        max_results: int,
    ) -> list[RetrievalDocument]:
        sort = "date" if analysis.needs_freshness else "relevance"
        try:
            articles = self._pubmed_fetcher.search(
                analysis.rewritten_queries[0],
                max_results=max_results,
                sort=sort,
            )
        except Exception as exc:
            logger.warning("PubMed retrieval failed: %s", exc)
            return []

        documents: list[RetrievalDocument] = []
        for article in articles:
            content = _normalize_whitespace(
                f"{article.get('title', '')}. {article.get('abstract', '')}",
            )
            documents.append(
                RetrievalDocument(
                    title=article.get("title", ""),
                    content=content,
                    url=article.get("url", ""),
                    source="pubmed",
                    source_type="pubmed",
                    domain="pubmed.ncbi.nlm.nih.gov",
                    published_at=article.get("year", ""),
                    query_used=analysis.rewritten_queries[0],
                    metadata={
                        "pmid": article.get("pmid", ""),
                        "journal": article.get("journal", ""),
                        "year": article.get("year", ""),
                    },
                ),
            )
        return documents

    def _search_guidelines(
        self,
        analysis: QueryAnalysis,
        max_results: int,
    ) -> list[RetrievalDocument]:
        domains = [
            "who.int",
            "nih.gov",
            "cdc.gov",
            "nice.org.uk",
            "cochranelibrary.com",
        ]
        results = self._web_search.search(
            analysis.rewritten_queries[min(1, len(analysis.rewritten_queries) - 1)],
            max_results=max_results,
            domains=domains,
        )
        return [self._doc_from_web_result(result, default_type="guidelines") for result in results]

    def _search_web(
        self,
        analysis: QueryAnalysis,
        max_results: int,
    ) -> list[RetrievalDocument]:
        results = self._web_search.search(
            analysis.rewritten_queries[0],
            max_results=max_results,
        )
        return [self._doc_from_web_result(result, default_type="web") for result in results]

    def _search_arxiv(
        self,
        analysis: QueryAnalysis,
        max_results: int,
    ) -> list[RetrievalDocument]:
        try:
            import requests
            import xml.etree.ElementTree as ET

            query = quote_plus(analysis.rewritten_queries[-1])
            url = (
                "http://export.arxiv.org/api/query?"
                f"search_query=all:{query}&start=0&max_results={min(max_results, 5)}"
                "&sortBy=submittedDate&sortOrder=descending"
            )
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except Exception as exc:
            logger.debug("ArXiv retrieval unavailable: %s", exc)
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        documents: list[RetrievalDocument] = []
        for entry in root.findall("atom:entry", ns):
            title = _normalize_whitespace(entry.findtext("atom:title", default="", namespaces=ns))
            summary = _normalize_whitespace(entry.findtext("atom:summary", default="", namespaces=ns))
            url = entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            documents.append(
                RetrievalDocument(
                    title=title,
                    content=summary,
                    url=url,
                    source="arxiv",
                    source_type="arxiv",
                    domain="arxiv.org",
                    published_at=published[:10],
                    query_used=analysis.rewritten_queries[-1],
                ),
            )
        return documents

    def _doc_from_web_result(
        self,
        result: dict[str, Any],
        *,
        default_type: str,
    ) -> RetrievalDocument:
        url = result.get("href") or result.get("url") or ""
        domain = result.get("source") or _extract_domain(url)
        content = result.get("content") or result.get("body") or ""
        source_type = default_type
        if domain == "who.int":
            source_type = "who"
        elif domain.endswith("nih.gov"):
            source_type = "nih"

        return RetrievalDocument(
            title=result.get("title", ""),
            content=_normalize_whitespace(content),
            url=url,
            source=domain or default_type,
            source_type=source_type,
            domain=domain,
            published_at=result.get("published_date", "") or result.get("publishedDate", ""),
            query_used=result.get("query_used", ""),
            metadata={"is_trusted": bool(result.get("is_trusted", False))},
        )

    def _build_graph_if_available(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        try:
            workflow = StateGraph(dict)
            workflow.add_node("run_pipeline", self._graph_run_pipeline)
            workflow.set_entry_point("run_pipeline")
            workflow.add_edge("run_pipeline", END)
            return workflow.compile()
        except Exception as exc:
            logger.warning("LangGraph setup failed, falling back to sequential pipeline: %s", exc)
            return None

    def _graph_run_pipeline(self, state: dict[str, Any]) -> dict[str, Any]:
        result = self._run_sequential(
            query=state["query"],
            max_results=state["max_results"],
            requested_sources=state.get("requested_sources"),
        )
        return {"result": result}


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""
