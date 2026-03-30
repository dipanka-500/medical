"""Tests for Layer 1: Input Understanding components."""

from __future__ import annotations

import pytest

from core.input.query_classifier import QueryClassifier
from core.input.symptom_extractor import SymptomExtractor
from core.input.negation_detector import NegationDetector


class TestQueryClassifier:
    """Tests for QueryClassifier."""

    def setup_method(self):
        self.classifier = QueryClassifier()

    def test_empty_query(self):
        result = self.classifier.classify("")
        assert result["category"] == "conversational"

    def test_emergency_detection(self):
        result = self.classifier.classify("Patient in cardiac arrest, unresponsive")
        assert result["category"] == "emergency"
        assert result["is_emergency"] is True

    def test_crisis_detection(self):
        result = self.classifier.classify("I want to kill myself")
        assert result["category"] == "emergency"
        assert result.get("is_crisis") is True

    def test_crisis_self_harm(self):
        result = self.classifier.classify("I've been cutting myself")
        assert result.get("is_crisis") is True

    def test_suicide_resource_query_not_crisis(self):
        result = self.classifier.classify("What is the suicide prevention hotline?")
        assert result.get("is_crisis") is not True
        assert result["category"] in ("simple_qa", "research", "conversational")

    def test_diagnosis_classification(self):
        result = self.classifier.classify(
            "Patient presents with severe chest pain radiating to the left arm"
        )
        assert result["category"] in ("diagnosis", "emergency")

    def test_drug_info_classification(self):
        result = self.classifier.classify(
            "What are the side effects of metformin 500mg?"
        )
        assert result["category"] == "drug_info"

    def test_research_classification(self):
        result = self.classifier.classify(
            "What does the latest meta-analysis say about SGLT2 inhibitors?"
        )
        assert result["category"] == "research"

    def test_simple_qa_fallback(self):
        result = self.classifier.classify("What is diabetes?")
        assert result["category"] == "simple_qa"

    def test_conversational(self):
        result = self.classifier.classify("Hello, how are you?")
        assert result["category"] == "conversational"

    def test_lab_interpretation(self):
        result = self.classifier.classify(
            "My hemoglobin level is 8.5 g/dL, what does this mean?"
        )
        assert result["category"] == "lab_interpretation"

    def test_all_categories_returned(self):
        cats = self.classifier.get_all_categories()
        assert "emergency" in cats
        assert "diagnosis" in cats
        assert "conversational" in cats


class TestSymptomExtractor:
    """Tests for SymptomExtractor."""

    def setup_method(self):
        self.extractor = SymptomExtractor()

    def test_basic_symptom_extraction(self):
        result = self.extractor.extract_all(
            "Patient has chest pain and shortness of breath for 3 days"
        )
        assert result["symptom_count"] > 0
        symptom_names = [s["symptom"] for s in result["symptoms"]]
        assert any("dyspnea" in s for s in symptom_names)

    def test_synonym_normalization(self):
        result = self.extractor.extract_symptoms("I have a heart attack and high blood pressure")
        symptom_names = [s["symptom"] for s in result]
        assert "myocardial infarction" in symptom_names
        assert "hypertension" in symptom_names

    def test_medication_extraction(self):
        meds = self.extractor.extract_medications(
            "Patient is on Metformin 500 mg PO BID and Lisinopril 10 mg daily"
        )
        assert len(meds) >= 2
        drug_names = [m["drug"].lower() for m in meds]
        assert "metformin" in drug_names

    def test_duration_extraction(self):
        result = self.extractor.extract_all("Pain for 3 days with fever")
        assert result["duration"] is not None
        assert result["duration"]["value"] == 3
        assert result["duration"]["unit"] == "day"

    def test_severity_extraction(self):
        result = self.extractor.extract_all("Severe headache with mild nausea")
        assert "severe" in result["severities"]
        assert "mild" in result["severities"]

    def test_normalize_term(self):
        assert self.extractor.normalize_term("heart attack") == "myocardial infarction"
        assert self.extractor.normalize_term("unknown_term") == "unknown_term"

    def test_empty_text(self):
        result = self.extractor.extract_all("")
        assert result["symptom_count"] == 0


class TestNegationDetector:
    """Tests for NegationDetector."""

    def setup_method(self):
        self.detector = NegationDetector()

    def test_negated_finding(self):
        result = self.detector.detect(
            "No evidence of pneumonia on chest X-ray", "pneumonia"
        )
        assert result.is_negated is True

    def test_affirmed_finding(self):
        result = self.detector.detect(
            "Patient diagnosed with pneumonia", "pneumonia"
        )
        assert result.is_negated is False

    def test_filter_entities(self):
        text = "No evidence of fracture. Pneumonia confirmed on imaging."
        positive, negated = self.detector.filter_entities(
            text, ["fracture", "pneumonia"]
        )
        assert "fracture" in negated
        assert "pneumonia" in positive

    def test_scope_breaker(self):
        result = self.detector.detect(
            "No headache, but patient has pneumonia", "pneumonia"
        )
        assert result.is_negated is False

    def test_post_negation(self):
        result = self.detector.detect(
            "Pulmonary embolism was ruled out", "pulmonary embolism"
        )
        assert result.is_negated is True

    def test_entity_not_found(self):
        result = self.detector.detect("Normal chest X-ray", "fracture")
        assert result.is_negated is False

    def test_batch_detection(self):
        results = self.detector.detect_batch(
            "No pneumonia. Fracture confirmed.",
            ["pneumonia", "fracture"],
        )
        assert results["pneumonia"].is_negated is True
        assert results["fracture"].is_negated is False
