from __future__ import annotations

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pydantic_settings")

from services.voice_service import VoiceService, _normalize_language_tag, _resolve_device


def test_voice_service_defaults_to_disabled_backends() -> None:
    capabilities = VoiceService().capabilities()

    assert capabilities.asr_provider == "disabled"
    assert capabilities.asr_available is False
    assert capabilities.tts_provider == "disabled"
    assert capabilities.tts_available is False


def test_resolve_device_keeps_explicit_choice() -> None:
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"


def test_normalize_language_tag_reduces_locale_variants() -> None:
    assert _normalize_language_tag("en-US") == "en"
    assert _normalize_language_tag("hi_IN") == "hi"
    assert _normalize_language_tag("  ta  ") == "ta"
    assert _normalize_language_tag("") is None
    assert _normalize_language_tag(None) is None
