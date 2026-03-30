"""
Medical RAG Engine — FAISS + BGE-large embeddings.
Retrieval-Augmented Generation for evidence-grounded medical reasoning.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MedicalRAG:
    """Medical RAG engine with FAISS vector store + BGE-large embeddings.

    Features:
    - BGE-large-en-v1.5 embeddings (SOTA on MTEB)
    - FAISS for fast approximate nearest neighbor search
    - Document chunking with configurable overlap
    - Claim verification against knowledge base
    - Prompt enrichment with retrieved evidence
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        persist_dir: str = "./data/rag/faiss_index",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        vector_backend: str = "faiss",
        qdrant_url: str = "",
        qdrant_collection: str = "medical_rag",
        qdrant_api_key: str = "",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.embedding_model_name = embedding_model
        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_backend = (vector_backend or "faiss").lower()
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self.qdrant_api_key = qdrant_api_key

        self._embedder = None
        self._index = None
        self._documents: list[dict[str, Any]] = []  # id, text, metadata
        self._qdrant_client = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize the embedding model and FAISS index."""
        if self._is_initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"RAG embedder loaded: {self.embedding_model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed")
            return
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            return

        self._load_or_create_index()
        self._is_initialized = True

    def _load_or_create_index(self) -> None:
        """Load or create the configured vector backend."""
        if self.vector_backend == "qdrant" and self._load_or_create_qdrant():
            return
        self._load_or_create_faiss_index()

    def _load_or_create_faiss_index(self) -> None:
        """Load existing FAISS index or create a new one."""
        import faiss

        index_path = self.persist_dir / "index.faiss"
        docs_path = self.persist_dir / "documents.json"

        if index_path.exists() and docs_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                with open(docs_path, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
                logger.info(
                    "RAG loaded FAISS backend: %d vectors, %d documents",
                    self._index.ntotal,
                    len(self._documents),
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")

        # Create new index
        dim = self._embedder.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)
        self._documents = []
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("RAG created new FAISS index (dim=%d)", dim)

    def _load_or_create_qdrant(self) -> bool:
        """Initialize Qdrant and fall back to FAISS if unavailable."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
        except Exception as exc:
            logger.warning("Qdrant backend unavailable (%s), falling back to FAISS", exc)
            self.vector_backend = "faiss"
            return False

        docs_path = self.persist_dir / "documents.json"
        try:
            if self.qdrant_url:
                self._qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key or None,
                    timeout=10.0,
                )
            else:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self._qdrant_client = QdrantClient(path=str(self.persist_dir / "qdrant"))

            dim = self._embedder.get_sentence_embedding_dimension()
            existing = {c.name for c in self._qdrant_client.get_collections().collections}
            if self.qdrant_collection not in existing:
                self._qdrant_client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )

            if docs_path.exists():
                try:
                    with open(docs_path, "r", encoding="utf-8") as f:
                        self._documents = json.load(f)
                except Exception as exc:
                    logger.warning("Failed to load Qdrant document metadata: %s", exc)
                    self._documents = []
            logger.info(
                "RAG initialized Qdrant backend: collection=%s documents=%d",
                self.qdrant_collection,
                len(self._documents),
            )
            return True
        except Exception as exc:
            logger.warning("Qdrant init failed (%s), falling back to FAISS", exc)
            self.vector_backend = "faiss"
            self._qdrant_client = None
            return False

    def _save_index(self) -> None:
        """Persist FAISS index and document metadata to disk."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self.vector_backend == "faiss" and self._index is not None:
            import faiss

            faiss.write_index(self._index, str(self.persist_dir / "index.faiss"))
        with open(self.persist_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self._documents, f, indent=2, ensure_ascii=False)

    def ingest_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """Ingest documents into the vector store with chunking.

        Deduplicates by doc_id — documents with IDs already in the index
        are skipped to prevent index bloat on repeated ingestion.

        Args:
            documents: Raw text documents
            metadatas: Optional metadata for each document
            ids: Optional document IDs

        Returns:
            Number of NEW chunks ingested (0 if all duplicates)
        """
        if not self._is_initialized:
            self.initialize()
        if self._embedder is None:
            return 0

        # Build set of existing doc_ids for deduplication
        existing_doc_ids: set[str] = set()
        for doc in self._documents:
            meta = doc.get("metadata", {})
            if "doc_id" in meta:
                existing_doc_ids.add(meta["doc_id"])

        all_chunks = []
        all_metas = []

        for i, doc_text in enumerate(documents):
            doc_id = ids[i] if ids and i < len(ids) else hashlib.md5(doc_text[:100].encode()).hexdigest()

            # Skip if this document is already in the index
            if doc_id in existing_doc_ids:
                logger.debug(f"Skipping duplicate doc_id: {doc_id}")
                continue

            chunks = self._chunk_text(doc_text)
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}

            for j, chunk in enumerate(chunks):
                chunk_meta = {
                    **meta,
                    "doc_id": doc_id,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                }
                all_chunks.append(chunk)
                all_metas.append(chunk_meta)

        if not all_chunks:
            logger.info("No new documents to ingest (all duplicates or empty)")
            return 0

        # Embed and add to vector store
        embeddings = self._embedder.encode(
            all_chunks,
            normalize_embeddings=True,
            show_progress_bar=len(all_chunks) > 100,
            batch_size=64,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        start_index = len(self._documents)

        if self.vector_backend == "qdrant" and self._qdrant_client is not None:
            try:
                from qdrant_client.http.models import PointStruct

                points = []
                for i, (chunk, meta) in enumerate(zip(all_chunks, all_metas)):
                    point_id = hashlib.sha256(
                        f"{meta['doc_id']}::{meta['chunk_index']}::{chunk[:120]}".encode(),
                    ).hexdigest()
                    payload = {
                        "text": chunk,
                        "metadata": meta,
                        "index": start_index + i,
                    }
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embeddings[i].tolist(),
                            payload=payload,
                        ),
                    )
                self._qdrant_client.upsert(
                    collection_name=self.qdrant_collection,
                    points=points,
                    wait=True,
                )
            except Exception as exc:
                logger.warning("Qdrant upsert failed (%s), falling back to FAISS backend", exc)
                self.vector_backend = "faiss"
                self._qdrant_client = None
                self._load_or_create_faiss_index()
                self._index.add(embeddings)
        if self.vector_backend == "faiss":
            if self._index is None:
                self._load_or_create_faiss_index()
            self._index.add(embeddings)

        # Store document references
        for i, (chunk, meta) in enumerate(zip(all_chunks, all_metas)):
            self._documents.append({
                "index": start_index + i,
                "text": chunk,
                "metadata": meta,
            })

        self._save_index()
        logger.info(f"Ingested {len(all_chunks)} chunks from {len(documents)} documents")
        return len(all_chunks)

    def ingest_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
    ) -> int:
        """Ingest all matching files from a directory."""
        extensions = extensions or [".txt", ".md", ".json", ".csv"]
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"RAG directory not found: {directory}")
            return 0

        documents = []
        metadatas = []
        ids = []

        for ext in extensions:
            for fpath in dir_path.rglob(f"*{ext}"):
                try:
                    content = fpath.read_text(encoding="utf-8")
                    documents.append(content)
                    metadatas.append({
                        "source": str(fpath),
                        "filename": fpath.name,
                        "type": ext.lstrip("."),
                    })
                    ids.append(f"file_{fpath.stem}")
                except Exception as e:
                    logger.warning(f"Cannot read {fpath}: {e}")

        return self.ingest_documents(documents, metadatas, ids) if documents else 0

    def query(
        self,
        question: str,
        top_k: int = 7,
    ) -> list[dict[str, Any]]:
        """Query the knowledge base.

        Args:
            question: Query text
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with relevance scores
        """
        if not self._is_initialized:
            self.initialize()
        try:
            query_vec = self._embedder.encode(
                [question],
                normalize_embeddings=True,
            )
            query_vec = np.array(query_vec, dtype=np.float32)
            if self.vector_backend == "qdrant" and self._qdrant_client is not None:
                results = self._qdrant_client.search(
                    collection_name=self.qdrant_collection,
                    query_vector=query_vec[0].tolist(),
                    limit=max(1, top_k),
                    with_payload=True,
                )
                formatted = []
                for hit in results:
                    payload = hit.payload or {}
                    formatted.append({
                        "content": payload.get("text", ""),
                        "metadata": payload.get("metadata", {}),
                        "relevance": float(hit.score),
                        "index": int(payload.get("index", 0)),
                    })
                return formatted

            if self._index is None or self._index.ntotal == 0:
                return []

            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query_vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._documents):
                    continue
                doc = self._documents[idx]
                results.append({
                    "content": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "relevance": float(score),
                    "index": int(idx),
                })

            return results
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return []

    def enrich_prompt(
        self,
        question: str,
        top_k: int = 5,
        relevance_threshold: float = 0.3,
    ) -> str:
        """Enrich a prompt with retrieved evidence.

        Args:
            question: Original query
            top_k: Number of documents to retrieve
            relevance_threshold: Minimum relevance score

        Returns:
            Enriched prompt with evidence context
        """
        results = self.query(question, top_k=top_k)

        if not results:
            return question

        references = []
        for i, r in enumerate(results):
            if r["relevance"] >= relevance_threshold:
                source = r["metadata"].get("source", "Unknown")
                references.append(
                    f"[Reference {i + 1}] (relevance: {r['relevance']:.3f}, "
                    f"source: {source}):\n{r['content']}"
                )

        if not references:
            return question

        context = "\n\n".join(references)
        return (
            f"## Retrieved Medical Evidence\n"
            f"Use these references to support your analysis:\n\n"
            f"{context}\n\n"
            f"## Query\n{question}"
        )

    def verify_claim(self, claim: str, threshold: float = 0.5) -> dict[str, Any]:
        """Verify a medical claim against the knowledge base.

        Args:
            claim: Medical claim to verify
            threshold: Minimum relevance to consider supported

        Returns:
            Verification result with evidence
        """
        results = self.query(claim, top_k=3)

        if not results:
            return {
                "claim": claim,
                "verified": None,
                "reason": "No relevant evidence in knowledge base",
                "evidence": [],
            }

        best = results[0]
        is_supported = best["relevance"] >= threshold

        return {
            "claim": claim,
            "verified": is_supported,
            "relevance_score": best["relevance"],
            "supporting_text": best["content"][:300],
            "source": best["metadata"].get("source", "unknown"),
            "evidence": results[:3],
        }

    def get_stats(self) -> dict[str, Any]:
        """Return RAG engine statistics."""
        return {
            "total_vectors": self._vector_count(),
            "total_documents": len(self._documents),
            "embedding_model": self.embedding_model_name,
            "persist_dir": str(self.persist_dir),
            "initialized": self._is_initialized,
            "vector_backend": self.vector_backend,
        }

    def _vector_count(self) -> int:
        if self.vector_backend == "qdrant" and self._qdrant_client is not None:
            try:
                info = self._qdrant_client.get_collection(self.qdrant_collection)
                points_count = getattr(info, "points_count", None)
                return int(points_count or 0)
            except Exception:
                return len(self._documents)
        return self._index.ntotal if self._index else 0

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += step

        return chunks if chunks else [text]
