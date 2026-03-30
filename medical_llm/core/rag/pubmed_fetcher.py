"""
PubMed Fetcher — Real-time medical literature retrieval via NCBI E-utilities API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PubMedFetcher:
    """Fetches medical literature from PubMed/PMC via NCBI E-utilities API.

    Features:
    - ESearch + EFetch for PubMed abstract retrieval
    - PMC full-text access for open-access papers
    - Local caching to avoid repeated API calls
    - Auto-ingestion into RAG vector store
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        email: str = "medllm@research.org",
        tool_name: str = "MedicalLLMEngine",
        cache_dir: str = "./data/rag/pubmed_cache",
        max_results: int = 10,
        api_key: str = "",
    ):
        self.email = email
        self.tool_name = tool_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results
        self.api_key = api_key

        # SEC-2: Rate limiting — NCBI allows 3 req/s without key, 10/s with key
        self._min_request_interval = 0.35 if not api_key else 0.1
        self._last_request_time: float = 0.0

    def search(
        self,
        query: str,
        max_results: int | None = None,
        sort: str = "relevance",
    ) -> list[dict[str, Any]]:
        """Search PubMed and fetch abstracts.

        Args:
            query: Medical search query
            max_results: Override max results
            sort: "relevance" or "date"

        Returns:
            List of articles with title, abstract, authors, etc.
        """
        max_r = max_results or self.max_results

        # Check cache first
        cache_key = hashlib.md5(f"{query}_{max_r}_{sort}".encode()).hexdigest()
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info("PubMed cache hit (hash=%s)", cache_key[:8])
            return cached

        try:
            import requests
            import xmltodict

            # SEC-2: Enforce rate limiting
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)

            # Step 1: ESearch — get PMIDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_r,
                "sort": sort,
                "retmode": "json",
                "email": self.email,
                "tool": self.tool_name,
            }
            if self.api_key:
                search_params["api_key"] = self.api_key

            search_url = f"{self.BASE_URL}/esearch.fcgi"
            response = requests.get(search_url, params=search_params, timeout=30)
            self._last_request_time = time.time()
            response.raise_for_status()
            search_data = response.json()

            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                logger.info("No PubMed results found (hash=%s)", cache_key[:8])
                return []

            # SEC-2: Rate limit before second request
            time.sleep(self._min_request_interval)

            # Step 2: EFetch — get article details
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
                "email": self.email,
                "tool": self.tool_name,
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key

            fetch_url = f"{self.BASE_URL}/efetch.fcgi"
            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            self._last_request_time = time.time()
            response.raise_for_status()

            # Parse XML response
            data = xmltodict.parse(response.text)
            articles = self._parse_articles(data)

            # Cache results
            self._cache_results(cache_key, articles)

            logger.info("PubMed: %d results retrieved", len(articles))
            return articles

        except ImportError:
            logger.error("requests and/or xmltodict not installed")
            return []
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def _parse_articles(self, data: dict) -> list[dict[str, Any]]:
        """Parse PubMed XML fetch response into structured articles."""
        articles = []

        pubmed_set = data.get("PubmedArticleSet", {})
        article_list = pubmed_set.get("PubmedArticle", [])
        if isinstance(article_list, dict):
            article_list = [article_list]

        for article_data in article_list:
            try:
                medline = article_data.get("MedlineCitation", {})
                article = medline.get("Article", {})

                # Title
                title = article.get("ArticleTitle", "")
                if isinstance(title, dict):
                    title = title.get("#text", str(title))

                # Abstract
                abstract_data = article.get("Abstract", {}).get("AbstractText", "")
                if isinstance(abstract_data, list):
                    abstract_parts = []
                    for part in abstract_data:
                        if isinstance(part, dict):
                            label = part.get("@Label", "")
                            text = part.get("#text", str(part))
                            abstract_parts.append(f"{label}: {text}" if label else text)
                        else:
                            abstract_parts.append(str(part))
                    abstract = " ".join(abstract_parts)
                elif isinstance(abstract_data, dict):
                    abstract = abstract_data.get("#text", str(abstract_data))
                else:
                    abstract = str(abstract_data)

                # Authors
                author_list = article.get("AuthorList", {}).get("Author", [])
                if isinstance(author_list, dict):
                    author_list = [author_list]
                authors = []
                for auth in author_list[:5]:  # First 5 authors
                    if isinstance(auth, dict):
                        last = auth.get("LastName", "")
                        fore = auth.get("ForeName", "")
                        authors.append(f"{last} {fore}".strip())

                # PMID
                pmid = medline.get("PMID", {})
                if isinstance(pmid, dict):
                    pmid = pmid.get("#text", "")

                # Journal
                journal = article.get("Journal", {}).get("Title", "")

                # Date
                pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                year = pub_date.get("Year", "")
                month = pub_date.get("Month", "")

                articles.append({
                    "pmid": str(pmid),
                    "title": str(title),
                    "abstract": abstract,
                    "authors": authors,
                    "journal": str(journal),
                    "year": str(year),
                    "month": str(month),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "source": "pubmed",
                })

            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue

        return articles

    def fetch_and_ingest(
        self,
        query: str,
        rag_engine: Any,
        max_results: int | None = None,
    ) -> int:
        """Search PubMed and ingest results into RAG engine.

        Args:
            query: Search query
            rag_engine: MedicalRAG instance to ingest into
            max_results: Max articles to fetch

        Returns:
            Number of chunks ingested
        """
        articles = self.search(query, max_results=max_results)

        if not articles:
            return 0

        documents = []
        metadatas = []
        ids = []

        for article in articles:
            text = (
                f"Title: {article['title']}\n"
                f"Authors: {', '.join(article['authors'])}\n"
                f"Journal: {article['journal']} ({article['year']})\n"
                f"Abstract: {article['abstract']}"
            )
            documents.append(text)
            metadatas.append({
                "source": article["url"],
                "type": "pubmed",
                "pmid": article["pmid"],
                "year": article["year"],
            })
            ids.append(f"pubmed_{article['pmid']}")

        return rag_engine.ingest_documents(documents, metadatas, ids)

    def format_as_context(self, articles: list[dict[str, Any]]) -> str:
        """Format articles as context string for prompt injection."""
        if not articles:
            return ""

        parts = []
        for i, article in enumerate(articles, 1):
            parts.append(
                f"[{i}] {article['title']}\n"
                f"    Authors: {', '.join(article['authors'][:3])}\n"
                f"    Journal: {article['journal']} ({article['year']})\n"
                f"    Abstract: {article['abstract'][:500]}\n"
                f"    URL: {article['url']}"
            )

        return "\n\n".join(parts)

    def _get_cached(self, cache_key: str) -> list[dict] | None:
        """Check local cache for results."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                if age_hours < 24:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass
        return None

    def _cache_results(self, cache_key: str, results: list[dict]) -> None:
        """Cache results locally."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
