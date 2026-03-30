from medicscan_ocr.config import load_settings
from medicscan_ocr.routing import RoutePlanner
from medicscan_ocr.schemas import AnalysisResult, DocumentType, LayoutComplexity


def make_analysis(document_type, language_family, language_code="en-IN"):
    return AnalysisResult(
        document_type=document_type,
        handwritten_confidence=0.9 if document_type == DocumentType.HANDWRITTEN else 0.1,
        language_code=language_code,
        language_family=language_family,
        language_confidence=0.9,
        layout_complexity=LayoutComplexity.HIGH,
        needs_table_model=True,
        needs_formula_model=False,
        needs_layout_model=True,
        source_hints={},
        warnings=[],
    )


def test_handwritten_indic_prefers_sarvam_when_key_present(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    settings = load_settings()
    planner = RoutePlanner(settings)
    decision = planner.decide(make_analysis(DocumentType.HANDWRITTEN, "indic"))
    assert decision.primary_backend == "sarvam_vision"
    assert "table_transformer_placeholder" in decision.enrichers


def test_printed_english_uses_firered_route_without_remote_preference(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.setenv("MEDISCAN_PREFER_REMOTE_API", "false")
    settings = load_settings()
    planner = RoutePlanner(settings)
    decision = planner.decide(make_analysis(DocumentType.PRINTED, "english"))
    assert decision.primary_backend == "firered_backend"
    assert "layoutlmv3_placeholder" in decision.enrichers


def test_printed_indic_prefers_surya(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    settings = load_settings()
    planner = RoutePlanner(settings)
    decision = planner.decide(make_analysis(DocumentType.PRINTED, "indic", language_code="ta-IN"))
    assert decision.primary_backend == "surya_command"
    assert "doctr_placeholder" in decision.secondary_backends
