from pathlib import Path

from PIL import Image, ImageDraw

from medicscan_ocr.config import load_settings
from medicscan_ocr.service import MediScanOCRService


def test_service_dry_run_returns_route(tmp_path, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.setenv("MEDISCAN_PREFER_REMOTE_API", "false")
    image_path = tmp_path / "sample_form.png"
    image = Image.new("RGB", (400, 160), color="white")
    drawer = ImageDraw.Draw(image)
    drawer.text((20, 40), "Sample Printed Text", fill="black")
    image.save(image_path)

    settings = load_settings(tmp_path)
    service = MediScanOCRService(settings)
    result = service.process(
        input_path=str(image_path),
        language_hint="en-IN",
        dry_run=True,
    )

    assert result.input_path.endswith("sample_form.png")
    assert result.route.primary_backend == "firered_backend"
    assert result.analysis.language_code == "en-IN"
    assert result.route.enrichers


def test_service_uses_filename_hint_for_handwritten_language(tmp_path, monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    image_path = tmp_path / "tamil_notes_page.png"
    image = Image.new("RGB", (400, 160), color="white")
    drawer = ImageDraw.Draw(image)
    drawer.line((20, 30, 350, 30), fill="black", width=3)
    drawer.line((20, 70, 350, 70), fill="black", width=3)
    image.save(image_path)

    settings = load_settings(tmp_path)
    service = MediScanOCRService(settings)
    result = service.process(
        input_path=str(image_path),
        document_type_hint="handwritten",
        dry_run=True,
    )

    assert result.analysis.language_family == "indic"
    assert result.route.primary_backend == "sarvam_vision"
