"""Tests for lightweight RAG utilities that do not require model downloads."""

from __future__ import annotations

import pytest

from core.rag.medical_rag import MedicalRAG
from core.rag.retrieval_pipeline import (
    MedicalQueryAnalyzer,
    RetrievalDocument,
    RetrievalOutputFilter,
    RetrievalResult,
    SmartContextBuilder,
)


class TestMedicalRAG:
    """Unit tests for chunking configuration and helpers."""

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            MedicalRAG(chunk_size=8, chunk_overlap=8)

    def test_chunk_text_uses_positive_step(self):
        rag = MedicalRAG(chunk_size=4, chunk_overlap=1)
        chunks = rag._chunk_text("one two three four five six seven eight")
        assert chunks == [
            "one two three four",
            "four five six seven",
            "seven eight",
        ]


class TestRetrievalPipelineHelpers:
    """Unit tests for retrieval-layer orchestration helpers."""

    def test_query_analyzer_redacts_phi_and_rewrites(self):
        analyzer = MedicalQueryAnalyzer()
        result = analyzer.analyze("Latest diabetes guideline for John, email me at jane@example.com")
        assert result.needs_search is True
        assert result.intent == "guideline_lookup"
        assert "[REDACTED_EMAIL]" in result.redacted_query
        assert len(result.rewritten_queries) >= 2
        assert "guidelines" in " ".join(result.rewritten_queries).lower()

    def test_context_builder_returns_sources_and_evidence(self):
        builder = SmartContextBuilder(
            {"max_context_chars": 1500, "max_sources": 2, "chunk_words": 40, "chunk_overlap": 5},
        )
        documents = [
            RetrievalDocument(
                title="WHO Diabetes Guideline",
                content="Diabetes guideline update recommends individualized HbA1c targets. " * 12,
                url="https://www.who.int/example",
                source="who.int",
                source_type="who",
                domain="who.int",
                citation_confidence=0.95,
            ),
            RetrievalDocument(
                title="PubMed Review",
                content="A systematic review on diabetes management and insulin therapy. " * 10,
                url="https://pubmed.ncbi.nlm.nih.gov/123456/",
                source="pubmed",
                source_type="pubmed",
                domain="pubmed.ncbi.nlm.nih.gov",
                citation_confidence=0.9,
            ),
        ]

        context, sources, evidence = builder.build("latest diabetes guideline", documents)
        assert "WHO Diabetes Guideline" in context
        assert len(sources) == 2
        assert len(evidence) == 2
        assert sources[0]["confidence"] >= 0.9

    def test_output_filter_warns_when_citations_missing(self):
        analysis = MedicalQueryAnalyzer().analyze("metformin dosage")
        result = RetrievalResult(query_analysis=analysis, context="", sources=[], rag_evidence=[])
        filtered = RetrievalOutputFilter({"require_citations": True}).apply(result)
        assert filtered.warnings
