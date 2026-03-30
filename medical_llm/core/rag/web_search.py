"""
Web Search — DuckDuckGo medical literature search + fact-checking.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class WebSearch:
    """Medical web search via DuckDuckGo.

    Restricts search to trusted medical domains for reliable evidence.
    Provides fact-checking by cross-referencing claims with search results.
    """

    MEDICAL_DOMAINS = [
        "pubmed.ncbi.nlm.nih.gov",
        "who.int",
        "nih.gov",
        "mayoclinic.org",
        "radiopaedia.org",
        "uptodate.com",
        "nice.org.uk",
        "cochranelibrary.com",
        "medscape.com",
        "bmj.com",
        "thelancet.com",
        "nejm.org",
        "cdc.gov",
        "fda.gov",
    ]

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.max_results = config.get("max_results", 5)
        self.domains = config.get("medical_domains", self.MEDICAL_DOMAINS)
        self.provider = config.get("provider", "searxng")
        self.timeout_seconds = float(config.get("timeout_seconds", 12))
        self.searxng_url = (
            os.getenv("MEDICAL_LLM_SEARXNG_URL")
            or config.get("searxng_url", "")
        ).rstrip("/")

    def search(
        self,
        query: str,
        max_results: int | None = None,
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search medical literature using DuckDuckGo.

        Args:
            query: Medical search query
            max_results: Override max results

        Returns:
            List of search results with title, body, href, source
        """
        max_r = max_results or self.max_results
        selected_domains = domains or self.domains

        if self.searxng_url:
            results = self._search_searxng(
                query=query,
                max_results=max_r,
                domains=selected_domains,
            )
            if results:
                return results

        try:
            from duckduckgo_search import DDGS

            # Construct medical-focused search query
            domain_filter = " OR ".join(f"site:{d}" for d in selected_domains[:6])
            medical_query = f"medical {query} ({domain_filter})"

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(medical_query, max_results=max_r):
                    results.append({
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "content": r.get("body", ""),
                        "href": r.get("href", ""),
                        "source": self._identify_source(r.get("href", ""), selected_domains),
                        "is_trusted": self._is_trusted_source(r.get("href", ""), selected_domains),
                    })

            logger.info("Web search: %d results retrieved", len(results))
            return results

        except ImportError:
            logger.warning("duckduckgo_search not installed — web search disabled")
            return []
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def _search_searxng(
        self,
        *,
        query: str,
        max_results: int,
        domains: list[str],
    ) -> list[dict[str, Any]]:
        try:
            import requests

            site_terms = " OR ".join(f"site:{domain}" for domain in domains[:8])
            search_query = f"{query} ({site_terms}) medical"
            response = requests.get(
                f"{self.searxng_url}/search",
                params={
                    "q": search_query,
                    "format": "json",
                    "language": "en-US",
                    "safesearch": 1,
                },
                timeout=self.timeout_seconds,
                headers={"User-Agent": "MedAI-WebSearch/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("SearxNG search failed, falling back to DuckDuckGo: %s", exc)
            return []

        results = []
        for item in payload.get("results", [])[:max_results]:
            url = item.get("url") or item.get("href") or ""
            results.append({
                "title": item.get("title", ""),
                "body": item.get("content", ""),
                "content": item.get("content", ""),
                "href": url,
                "source": self._identify_source(url, domains),
                "is_trusted": self._is_trusted_source(url, domains),
                "published_date": item.get("publishedDate", ""),
            })

        logger.info("Web search (SearxNG): %d results retrieved", len(results))
        return results

    def fact_check(self, claim: str) -> dict[str, Any]:
        """Fact-check a medical claim via web search.

        Args:
            claim: Medical claim to verify

        Returns:
            Verification result with supporting sources
        """
        results = self.search(claim, max_results=5)

        if not results:
            return {
                "claim": claim,
                "verified": None,
                "confidence": 0.0,
                "sources": [],
                "reason": "No web results found",
            }

        # Score claim support based on keyword overlap
        claim_words = set(claim.lower().split())
        supporting = 0
        trusted_supporting = 0

        for r in results:
            body_words = set(r["body"].lower().split())
            overlap = len(claim_words & body_words) / max(len(claim_words), 1)
            if overlap > 0.25:
                supporting += 1
                if r["is_trusted"]:
                    trusted_supporting += 1

        # Calculate confidence
        confidence = 0.0
        if len(results) > 0:
            confidence = (
                (supporting / len(results)) * 0.5 +
                (trusted_supporting / max(supporting, 1)) * 0.5
            )

        return {
            "claim": claim,
            "verified": supporting > 0,
            "confidence": round(confidence, 3),
            "supporting_sources": supporting,
            "trusted_sources": trusted_supporting,
            "total_sources": len(results),
            "sources": results,
        }

    def format_as_context(self, results: list[dict[str, Any]]) -> str:
        """Format search results as context string."""
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            trusted = " ✓" if r.get("is_trusted") else ""
            summary = r.get("content") or r.get("body") or ""
            parts.append(
                f"[Web {i}]{trusted} {r['title']}\n"
                f"    Source: {r['source']}\n"
                f"    Summary: {summary[:300]}"
            )

        return "\n\n".join(parts)

    def _identify_source(self, url: str, domains: list[str] | None = None) -> str:
        """Identify the source domain from a URL."""
        url_lower = url.lower()
        for domain in domains or self.domains:
            if domain in url_lower:
                return domain
        return "other"

    def _is_trusted_source(self, url: str, domains: list[str] | None = None) -> bool:
        """Check if a URL is from a trusted medical domain."""
        url_lower = url.lower()
        domains_to_check = domains or self.domains
        return (
            any(domain in url_lower for domain in domains_to_check)
            or ".gov/" in url_lower
            or url_lower.endswith(".gov")
            or ".edu/" in url_lower
            or url_lower.endswith(".edu")
        )
