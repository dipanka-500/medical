"""
OpenRAG Service — Configuration.

Manages all settings for the agentic RAG ingestion pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class OpenRAGConfig:
    """Configuration for the OpenRAG ingestion and retrieval service."""

    # ── Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8006

    # ── Vector DB (Qdrant — shared with existing medical_llm) ─────────
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection_documents: str = "openrag_documents"
    qdrant_collection_chunks: str = "openrag_chunks"

    # ── OpenSearch (full-text search complement) ──────────────────────
    opensearch_url: str = "http://opensearch:9200"
    opensearch_index: str = "medai_documents"
    opensearch_enabled: bool = True

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_batch_size: int = 64

    # ── Chunking ────���─────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_document_tokens: int = 100000

    # ── Re-ranking ────────────────────────────────────────────────────
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    reranker_enabled: bool = True
    reranker_top_k: int = 10

    # ── Docling (document conversion) ─────────────────────────────────
    docling_ocr_engine: str = "easyocr"
    docling_table_engine: str = "default"

    # ── Agentic RAG ───────────────────────────────────────────────────
    max_agent_turns: int = 5
    max_tool_calls_per_turn: int = 3
    agent_timeout_seconds: float = 60.0

    # ── LLM backbone (for agentic query decomposition) ────────────────
    llm_url: str = "http://general-llm:8004"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_max_tokens: int = 2048

    # ── Feature flags ─────────────────────────────────────────────────
    enable_contradiction_check: bool = True
    enable_multi_agent: bool = True
    enable_docling: bool = True

    # ── Granite integration ───────────────────────────────────────────
    granite_url: str = "http://granite-vision:8005/v1"
    granite_enabled: bool = True


def load_config() -> OpenRAGConfig:
    """Load configuration from environment variables."""
    return OpenRAGConfig(
        host=os.getenv("OPENRAG_HOST", "0.0.0.0"),
        port=int(os.getenv("OPENRAG_PORT", "8006")),
        qdrant_url=os.getenv("OPENRAG_QDRANT_URL", "http://qdrant:6333"),
        qdrant_api_key=os.getenv("OPENRAG_QDRANT_API_KEY", ""),
        qdrant_collection_documents=os.getenv("OPENRAG_QDRANT_COLLECTION_DOCS", "openrag_documents"),
        qdrant_collection_chunks=os.getenv("OPENRAG_QDRANT_COLLECTION_CHUNKS", "openrag_chunks"),
        opensearch_url=os.getenv("OPENRAG_OPENSEARCH_URL", "http://opensearch:9200"),
        opensearch_index=os.getenv("OPENRAG_OPENSEARCH_INDEX", "medai_documents"),
        opensearch_enabled=os.getenv("OPENRAG_OPENSEARCH_ENABLED", "true").lower() in {"1", "true"},
        embedding_model=os.getenv("OPENRAG_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        chunk_size=int(os.getenv("OPENRAG_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("OPENRAG_CHUNK_OVERLAP", "64")),
        reranker_model=os.getenv("OPENRAG_RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2"),
        reranker_enabled=os.getenv("OPENRAG_RERANKER_ENABLED", "true").lower() in {"1", "true"},
        reranker_top_k=int(os.getenv("OPENRAG_RERANKER_TOP_K", "10")),
        llm_url=os.getenv("OPENRAG_LLM_URL", "http://general-llm:8004"),
        llm_model=os.getenv("OPENRAG_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        granite_url=os.getenv("GRANITE_VLLM_URL", "http://granite-vision:8005/v1"),
        granite_enabled=os.getenv("GRANITE_ENABLED", "true").lower() in {"1", "true"},
    )
