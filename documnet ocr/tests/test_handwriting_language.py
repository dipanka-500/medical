from PIL import Image, ImageDraw

from medicscan_ocr.models.language import HandwritingLanguageDetector


def test_filename_hint_detects_indic_language():
    detector = HandwritingLanguageDetector()
    result = detector.detect_from_name("tamil_notes_scan.png")
    assert result["language_family"] == "indic"
    assert result["language_code"] == "ta-IN"


def test_handwriting_script_heuristic_returns_family(tmp_path):
    image_path = tmp_path / "script_sample.png"
    image = Image.new("L", (420, 160), "white")
    draw = ImageDraw.Draw(image)
    for row in (30, 70, 110):
        draw.line((20, row, 380, row), fill="black", width=3)
        for col in range(30, 360, 40):
            draw.rectangle((col, row, col + 18, row + 28), outline="black", width=2)
    image.save(image_path)

    detector = HandwritingLanguageDetector()
    result = detector.detect_from_pages([str(image_path)])
    assert result["language_family"] in {"indic", "english"}
    assert result["confidence"] >= 0.55
