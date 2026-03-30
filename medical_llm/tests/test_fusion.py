"""Tests for Layer 4: Meta-Fusion components."""

from __future__ import annotations

import pytest

from core.fusion.meta_fusion import (
    MetaFusion,
    UncertaintyEstimator,
    ContradictionDetector,
)


class TestMetaFusion:
    """Tests for MetaFusion."""

    def setup_method(self):
        self.fusion = MetaFusion()

    def test_empty_results(self):
        result = self.fusion.fuse([])
        assert "error" in result

    def test_single_result(self):
        results = [{"text": "Patient has pneumonia", "model": "test", "role": "primary", "weight": 0.8}]
        fused = self.fusion.fuse(results)
        assert fused["text"] == "Patient has pneumonia"
        assert fused["model_count"] == 1

    def test_multiple_results(self):
        results = [
            {"text": "Diagnosis: pneumonia", "model": "model_a", "role": "primary", "weight": 0.9},
            {"text": "Assessment: community pneumonia", "model": "model_b", "role": "medical_reasoning", "weight": 0.8},
        ]
        fused = self.fusion.fuse(results)
        assert fused["text"]
        assert fused["model_count"] == 2
        assert "confidence" in fused
        assert "agreement_score" in fused

    def test_strategy_majority_vote(self):
        results = [
            {"text": "Pneumonia confirmed", "model": "a", "role": "primary", "weight": 0.7},
            {"text": "Pneumonia likely based on imaging", "model": "b", "role": "primary", "weight": 0.8},
            {"text": "Totally different diagnosis unrelated", "model": "c", "role": "primary", "weight": 0.6},
        ]
        fused = self.fusion.fuse(results, strategy="majority_vote")
        assert fused["strategy"] == "majority_vote"
        assert fused["text"]

    def test_strategy_merge(self):
        results = [
            {"text": "Primary analysis text", "model": "a", "role": "primary", "weight": 0.9},
            {"text": "Secondary analysis text", "model": "b", "role": "primary", "weight": 0.7},
        ]
        fused = self.fusion.fuse(results, strategy="merge")
        assert fused["strategy"] == "merge"
        assert "Primary Analysis" in fused["text"]

    def test_evidence_scoring_with_rag(self):
        results = [
            {"text": "Pneumonia treatment with amoxicillin", "model": "a", "role": "primary", "weight": 0.8},
        ]
        evidence = [
            {"content": "Amoxicillin is first-line for community-acquired pneumonia treatment"},
        ]
        fused = self.fusion.fuse(results, rag_evidence=evidence)
        assert fused["confidence"] > 0

    def test_all_empty_outputs(self):
        results = [
            {"text": "", "model": "a", "role": "primary", "weight": 0.8},
            {"text": "   ", "model": "b", "role": "primary", "weight": 0.7},
        ]
        fused = self.fusion.fuse(results)
        assert "error" in fused


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator."""

    def setup_method(self):
        self.estimator = UncertaintyEstimator()

    def test_single_model(self):
        results = [{"text": "Some analysis"}]
        unc = self.estimator.estimate(results)
        assert unc["total"] == 0.0

    def test_agreeing_models(self):
        results = [
            {"text": "Patient has pneumonia. Start amoxicillin."},
            {"text": "Diagnosis is pneumonia. Prescribe amoxicillin."},
        ]
        unc = self.estimator.estimate(results)
        assert unc["epistemic"] < 0.5  # High agreement = low epistemic

    def test_disagreeing_models(self):
        results = [
            {"text": "Definitive pneumonia diagnosis confirmed"},
            {"text": "Completely different unrelated condition found"},
        ]
        unc = self.estimator.estimate(results)
        assert unc["epistemic"] > 0.3


class TestContradictionDetector:
    """Tests for ContradictionDetector."""

    def setup_method(self):
        self.detector = ContradictionDetector()

    def test_contradiction_found(self):
        results = [
            {"text": "Findings are normal", "model": "a"},
            {"text": "Findings are abnormal", "model": "b"},
        ]
        result = self.detector.detect(results)
        assert result["has_contradictions"] is True
        assert result["count"] >= 1

    def test_no_contradiction(self):
        results = [
            {"text": "Pneumonia confirmed in right lung", "model": "a"},
            {"text": "Right lung pneumonia diagnosis", "model": "b"},
        ]
        result = self.detector.detect(results)
        # Same findings, no contradicting pairs
        assert result["count"] == 0 or result["severity"] in ("none", "low")
