"""
OpenRAG Ingestion Pipeline — Docling-powered document processing.

Handles:
  1. Document conversion (PDF, DOCX, images) via Docling
  2. Intelligent chunking with overlap and metadata preservation
  3. Dual indexing: Qdrant (vector) + OpenSearch (full-text)
  4. Granite Vision integration for table/KVP extraction
  5. Deduplication and incremental ingestion
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from openrag_service.config import OpenRAGConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk from a processed document."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class IngestedDocument:
    """Result of ingesting a single document."""
    doc_id: str
    filename: str
    num_chunks: int
    num_pages: int
    doc_type: str
    tables_extracted: int
    kvp_extracted: int
    processing_time_ms: float
    errors: List[str] = field(default_factory=list)


class DoclingConverter:
    """Converts documents to structured markdown using Docling."""

    def __init__(self, config: OpenRAGConfig) -> None:
        self._config = config
        self._converter = None

    def _load(self):
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            self._converter = DocumentConverter()
            logger.info("Docling converter initialized")
            return self._converter
        except ImportError:
            logger.warning("Docling not installed — falling back to raw text extraction")
            return None

    def convert(self, file_path: str) -> Dict[str, Any]:
        """Convert a document to structured output."""
        converter = self._load()
        if converter is None:
            return self._fallback_convert(file_path)

        try:
            result = converter.convert(file_path)
            doc = result.document

            # Extract structured content
            text_content = doc.export_to_markdown()
            tables = []
            for table in doc.tables:
                tables.append({
                    "html": table.export_to_html() if hasattr(table, "export_to_html") else "",
                    "csv": table.export_to_dataframe().to_csv() if hasattr(table, "export_to_dataframe") else "",
                    "page": getattr(table, "prov", [{}])[0].get("page_no", 0) if hasattr(table, "prov") else 0,
                })

            return {
                "text": text_content,
                "tables": tables,
                "num_pages": getattr(doc, "num_pages", 1),
                "metadata": {
                    "converter": "docling",
                    "filename": Path(file_path).name,
                },
            }
        except Exception as e:
            logger.warning("Docling conversion failed for %s: %s", file_path, e)
            return self._fallback_convert(file_path)

    def _fallback_convert(self, file_path: str) -> Dict[str, Any]:
        """Fallback: read raw text content."""
        path = Path(file_path)
        text = ""
        if path.suffix.lower() in {".txt", ".md", ".csv", ".json"}:
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                text = ""
        return {
            "text": text,
            "tables": [],
            "num_pages": 1,
            "metadata": {"converter": "fallback", "filename": path.name},
        }


class TextChunker:
    """Intelligent text chunking with overlap and metadata preservation."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split text into overlapping chunks with metadata."""
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        total_estimated = max(1, (len(words) - self.chunk_overlap) // step + 1)

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            if not chunk_text.strip():
                start += step
                continue

            chunk_id = hashlib.sha256(
                f"{doc_id}::{len(chunks)}::{chunk_text[:120]}".encode()
            ).hexdigest()

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=len(chunks),
                total_chunks=0,  # updated below
                metadata={
                    **(metadata or {}),
                    "char_start": sum(len(w) + 1 for w in words[:start]),
                    "char_end": sum(len(w) + 1 for w in words[:end]),
                },
            ))
            start += step

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks


class VectorIndexer:
    """Indexes chunks into Qdrant vector store."""

    def __init__(self, config: OpenRAGConfig) -> None:
        self._config = config
        self._embedder = None
        self._client = None

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(self._config.embedding_model)
        logger.info("Embedder loaded: %s", self._config.embedding_model)
        return self._embedder

    def _load_client(self):
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        self._client = QdrantClient(
            url=self._config.qdrant_url,
            api_key=self._config.qdrant_api_key or None,
            timeout=30.0,
        )

        embedder = self._load_embedder()
        dim = embedder.get_sentence_embedding_dimension()

        for collection in [self._config.qdrant_collection_chunks, self._config.qdrant_collection_documents]:
            existing = {c.name for c in self._client.get_collections().collections}
            if collection not in existing:
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection: %s", collection)

        return self._client

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embed and index chunks into Qdrant."""
        if not chunks:
            return 0

        embedder = self._load_embedder()
        client = self._load_client()

        texts = [c.text for c in chunks]
        embeddings = embedder.encode(
            texts, normalize_embeddings=True,
            batch_size=self._config.embedding_batch_size,
            show_progress_bar=len(texts) > 100,
        )

        from qdrant_client.http.models import PointStruct

        points = []
        for chunk, emb in zip(chunks, embeddings):
            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=emb.tolist(),
                payload={
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "metadata": chunk.metadata,
                },
            ))

        # Batch upsert (Qdrant handles batching internally)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=self._config.qdrant_collection_chunks,
                points=batch,
                wait=True,
            )

        logger.info("Indexed %d chunks into Qdrant", len(points))
        return len(points)

    def search(self, query: str, top_k: int = 10, doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search indexed chunks."""
        embedder = self._load_embedder()
        client = self._load_client()

        query_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()

        filter_condition = None
        if doc_filter:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            filter_condition = Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_filter)),
            ])

        results = client.search(
            collection_name=self._config.qdrant_collection_chunks,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
            query_filter=filter_condition,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "doc_id": hit.payload.get("doc_id", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "metadata": hit.payload.get("metadata", {}),
                "score": float(hit.score),
            }
            for hit in results
        ]


class OpenSearchIndexer:
    """Indexes documents into OpenSearch for full-text search."""

    def __init__(self, config: OpenRAGConfig) -> None:
        self._config = config
        self._client = None

    def _load_client(self):
        if self._client is not None:
            return self._client
        if not self._config.opensearch_enabled:
            return None
        try:
            from opensearchpy import OpenSearch
            self._client = OpenSearch(
                hosts=[self._config.opensearch_url],
                use_ssl=False,
                verify_certs=False,
                timeout=30,
            )
            # Create index if not exists
            if not self._client.indices.exists(index=self._config.opensearch_index):
                self._client.indices.create(
                    index=self._config.opensearch_index,
                    body={
                        "settings": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0,
                            "analysis": {
                                "analyzer": {
                                    "medical_analyzer": {
                                        "type": "custom",
                                        "tokenizer": "standard",
                                        "filter": ["lowercase", "stop", "snowball"],
                                    }
                                }
                            },
                        },
                        "mappings": {
                            "properties": {
                                "doc_id": {"type": "keyword"},
                                "text": {"type": "text", "analyzer": "medical_analyzer"},
                                "filename": {"type": "keyword"},
                                "doc_type": {"type": "keyword"},
                                "ingested_at": {"type": "date"},
                                "metadata": {"type": "object", "enabled": False},
                            }
                        },
                    },
                )
            logger.info("OpenSearch indexer initialized: %s", self._config.opensearch_index)
            return self._client
        except Exception as e:
            logger.warning("OpenSearch unavailable: %s", e)
            return None

    def index_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Index a document for full-text search."""
        client = self._load_client()
        if client is None:
            return False

        try:
            client.index(
                index=self._config.opensearch_index,
                id=doc_id,
                body={
                    "doc_id": doc_id,
                    "text": text[:50000],  # cap at 50k chars
                    "filename": metadata.get("filename", ""),
                    "doc_type": metadata.get("doc_type", "unknown"),
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "metadata": metadata,
                },
                refresh=True,
            )
            return True
        except Exception as e:
            logger.warning("OpenSearch indexing failed: %s", e)
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Full-text search in OpenSearch."""
        client = self._load_client()
        if client is None:
            return []

        try:
            response = client.search(
                index=self._config.opensearch_index,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text^2", "filename"],
                            "type": "best_fields",
                        }
                    },
                    "size": top_k,
                },
            )
            return [
                {
                    "doc_id": hit["_source"]["doc_id"],
                    "text": hit["_source"]["text"][:1000],
                    "filename": hit["_source"].get("filename", ""),
                    "score": float(hit["_score"]),
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            logger.warning("OpenSearch search failed: %s", e)
            return []


class ReRanker:
    """Cross-encoder re-ranker for improving retrieval precision."""

    def __init__(self, config: OpenRAGConfig) -> None:
        self._config = config
        self._ranker = None

    def _load(self):
        if self._ranker is not None:
            return self._ranker
        if not self._config.reranker_enabled:
            return None
        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self._config.reranker_model)
            logger.info("Re-ranker loaded: %s", self._config.reranker_model)
            return self._ranker
        except ImportError:
            logger.warning("flashrank not installed — re-ranking disabled")
            return None

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-rank results using cross-encoder."""
        ranker = self._load()
        if ranker is None or not results:
            return results[:top_k]

        try:
            from flashrank import RerankRequest

            passages = [{"id": i, "text": r["text"], "meta": r} for i, r in enumerate(results)]
            request = RerankRequest(query=query, passages=passages)
            reranked = ranker.rerank(request)

            output = []
            for item in reranked[:top_k]:
                original = item.get("meta", item)
                original["rerank_score"] = float(item.get("score", 0))
                output.append(original)
            return output
        except Exception as e:
            logger.warning("Re-ranking failed: %s", e)
            return results[:top_k]


class IngestionPipeline:
    """Complete document ingestion pipeline.

    Flow:
      1. Docling converts document → structured markdown + tables
      2. TextChunker splits into overlapping chunks
      3. VectorIndexer embeds and stores in Qdrant
      4. OpenSearchIndexer stores full-text for BM25 search
      5. Optional: Granite Vision extracts KVP from structure-heavy docs
    """

    def __init__(self, config: OpenRAGConfig) -> None:
        self.config = config
        self.converter = DoclingConverter(config)
        self.chunker = TextChunker(config.chunk_size, config.chunk_overlap)
        self.vector_indexer = VectorIndexer(config)
        self.opensearch_indexer = OpenSearchIndexer(config)
        self.reranker = ReRanker(config)

    def ingest(
        self,
        file_path: str,
        doc_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestedDocument:
        """Ingest a single document through the full pipeline."""
        start = time.monotonic()
        errors = []

        doc_id = hashlib.sha256(
            f"{Path(file_path).name}::{Path(file_path).stat().st_size}".encode()
        ).hexdigest()[:32]

        # 1. Convert with Docling
        converted = self.converter.convert(file_path)
        text = converted["text"]
        tables = converted.get("tables", [])
        num_pages = converted.get("num_pages", 1)

        if not text.strip():
            errors.append("Document conversion produced empty text")

        # 2. Chunk
        doc_metadata = {
            **(metadata or {}),
            "filename": Path(file_path).name,
            "doc_type": doc_type,
            "num_pages": num_pages,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        chunks = self.chunker.chunk(text, doc_id, doc_metadata)

        # 3. Vector index
        try:
            self.vector_indexer.index_chunks(chunks)
        except Exception as e:
            errors.append(f"Vector indexing failed: {e}")
            logger.error("Vector indexing failed: %s", e)

        # 4. Full-text index
        self.opensearch_indexer.index_document(doc_id, text, doc_metadata)

        elapsed = (time.monotonic() - start) * 1000

        return IngestedDocument(
            doc_id=doc_id,
            filename=Path(file_path).name,
            num_chunks=len(chunks),
            num_pages=num_pages,
            doc_type=doc_type,
            tables_extracted=len(tables),
            kvp_extracted=0,
            processing_time_ms=round(elapsed, 2),
            errors=errors,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = True,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: vector + full-text, merged and re-ranked."""
        # Vector search
        vector_results = self.vector_indexer.search(query, top_k=top_k * 2)

        # Full-text search
        text_results = self.opensearch_indexer.search(query, top_k=top_k)

        # Merge (deduplicate by text hash)
        seen = set()
        merged = []
        for r in vector_results + text_results:
            key = hashlib.md5(r["text"][:200].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                merged.append(r)

        # Re-rank
        if use_reranker and self.config.reranker_enabled:
            return self.reranker.rerank(query, merged, top_k=top_k)

        return merged[:top_k]
