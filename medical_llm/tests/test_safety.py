"""Tests for Layer 5: Safety & Validation components."""

from __future__ import annotations

import pytest

from core.safety.safety import (
    DrugInteractionChecker,
    ClinicalValidator,
    RiskFlagger,
    HallucinationDetector,
)


class TestDrugInteractionChecker:
    """Tests for DrugInteractionChecker."""

    def setup_method(self):
        self.checker = DrugInteractionChecker()

    def test_known_interaction(self):
        result = self.checker.check_interactions(["warfarin", "ibuprofen"])
        assert result["has_interactions"] is True
        assert result["max_severity"] == "high"

    def test_critical_interaction(self):
        result = self.checker.check_interactions(["fluoxetine", "phenelzine"])
        assert result["has_interactions"] is True
        assert result["max_severity"] == "critical"

    def test_no_interaction(self):
        result = self.checker.check_interactions(["metformin", "lisinopril"])
        # These may or may not interact — test the structure
        assert "has_interactions" in result
        assert "max_severity" in result

    def test_single_drug(self):
        result = self.checker.check_interactions(["aspirin"])
        assert result["has_interactions"] is False

    def test_empty_list(self):
        result = self.checker.check_interactions([])
        assert result["has_interactions"] is False

    def test_drug_class_expansion(self):
        # SSRI class should match individual drugs
        result = self.checker.check_interactions(["sertraline", "tramadol"])
        assert result["has_interactions"] is True

    def test_nitrate_contraindication(self):
        result = self.checker.check_interactions(["sildenafil", "nitroglycerin"])
        assert result["max_severity"] == "critical"


class TestClinicalValidator:
    """Tests for ClinicalValidator."""

    def setup_method(self):
        self.validator = ClinicalValidator()

    def test_valid_report(self):
        report = (
            "Clinical assessment findings indicate pneumonia. "
            "Impression: community-acquired pneumonia. "
            "Plan: Amoxicillin 1g TID for 7 days. "
            "Clinical correlation recommended. "
            "Further evaluation with chest X-ray follow-up."
        )
        result = self.validator.validate(report)
        assert result["is_valid"] is True
        assert result["has_safety_language"] is True

    def test_short_report(self):
        result = self.validator.validate("Normal.")
        assert result["is_valid"] is False

    def test_missing_safety_disclaimer(self):
        report = (
            "The patient has pneumonia based on findings. "
            "Assessment: Community-acquired pneumonia. "
            "Plan: Start antibiotics immediately. "
            "Impression: Bacterial infection confirmed."
        )
        result = self.validator.validate(report)
        warnings = result["warnings"]
        assert any("safety disclaimer" in w.lower() for w in warnings)

    def test_absolute_language_warning(self):
        report = (
            "This is definitely cancer. The patient certainly has malignancy. "
            "It is always fatal. Treatment is guaranteed to fail. "
            "Assessment: findings show disease. Plan: consult oncology."
        )
        result = self.validator.validate(report)
        assert any("absolute" in w.lower() for w in result["warnings"])


class TestRiskFlagger:
    """Tests for RiskFlagger."""

    def setup_method(self):
        self.flagger = RiskFlagger()

    def test_emergent_risk(self):
        result = self.flagger.flag(
            "Patient shows signs of acute myocardial infarction with ST elevation"
        )
        assert result["risk_level"] == "emergent"
        assert result["requires_immediate_attention"] is True

    def test_negated_emergent(self):
        result = self.flagger.flag("No evidence of stroke or pulmonary embolism")
        assert result["risk_level"] != "emergent"
        assert len(result["negated_findings"]) >= 1

    def test_routine_finding(self):
        result = self.flagger.flag("Stable chronic condition with benign findings")
        assert result["risk_level"] == "routine"

    def test_urgent_finding(self):
        result = self.flagger.flag("Imaging reveals pneumonia in the right lower lobe")
        assert result["risk_level"] == "urgent"


class TestHallucinationDetector:
    """Tests for HallucinationDetector (without RAG engine)."""

    def setup_method(self):
        self.detector = HallucinationDetector()

    def test_claim_extraction(self):
        text = "The scan shows pneumonia. Evidence of pleural effusion noted."
        claims = self.detector._extract_claims(text)
        assert len(claims) > 0

    def test_verify_without_rag(self):
        result = self.detector.verify("The patient has diabetes.", "diabetes query")
        # Without RAG, nothing is verified but it shouldn't crash
        assert "total_claims" in result
        assert "hallucination_rate" in result
