"""
MediScan AI v5.0 — Sarvam AI Translation Engine
Multilingual report generation for Indian languages (replaces IndicTrans2).
"""
from __future__ import annotations


import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class SarvamTranslator:
    """Sarvam AI-based multilingual translation for medical reports.

    Supports: Hindi, Tamil, Kannada, Telugu, Bengali, Marathi,
              Gujarati, Malayalam, Punjabi, Odia
    """

    LANGUAGE_MAP = {
        "hi": "Hindi", "ta": "Tamil", "kn": "Kannada",
        "te": "Telugu", "bn": "Bengali", "mr": "Marathi",
        "gu": "Gujarati", "ml": "Malayalam", "pa": "Punjabi",
        "or": "Odia", "en": "English",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.sarvam.ai",
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["Content-Type"] = "application/json"
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers=headers,
                timeout=60.0,
            )
        return self._client

    async def translate(
        self,
        text: str,
        target_language: str = "hi",
        source_language: str = "en",
    ) -> dict[str, Any]:
        """Translate text using Sarvam AI API.

        Args:
            text: Source text to translate
            target_language: Target language code (hi, ta, kn, etc.)
            source_language: Source language code (default: en)
        """
        if target_language == source_language:
            return {"translated_text": text, "language": target_language}

        if target_language not in self.LANGUAGE_MAP:
            logger.warning(f"Unsupported language: {target_language}")
            return {"translated_text": text, "error": "Unsupported language"}

        try:
            client = self._get_client()
            response = await client.post(
                "/translate",
                json={
                    "input": text,
                    "source_language_code": source_language,
                    "target_language_code": target_language,
                    "mode": "formal",  # Medical reports should be formal
                    "model": "mayura:v1",
                },
            )

            if response.status_code == 200:
                data = response.json()
                translated = data.get("translated_text", text)
                logger.info(
                    f"Translated to {self.LANGUAGE_MAP.get(target_language, target_language)}: "
                    f"{len(text)} → {len(translated)} chars"
                )
                return {
                    "translated_text": translated,
                    "language": target_language,
                    "language_name": self.LANGUAGE_MAP.get(target_language, ""),
                }
            else:
                logger.error(f"Sarvam API error: {response.status_code}")
                return {"translated_text": text, "error": f"API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {"translated_text": text, "error": str(e)}

    def translate_sync(
        self,
        text: str,
        target_language: str = "hi",
        source_language: str = "en",
    ) -> dict[str, Any]:
        """Synchronous translation — safe to call from any context.

        Creates a fresh httpx client per call to avoid event loop
        ownership issues that occur when reusing an AsyncClient
        across threads or event loops.
        """
        if target_language == source_language:
            return {"translated_text": text, "language": target_language}

        if target_language not in self.LANGUAGE_MAP:
            logger.warning(f"Unsupported language: {target_language}")
            return {"translated_text": text, "error": "Unsupported language"}

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            with httpx.Client(
                base_url=self.api_base, headers=headers, timeout=60.0
            ) as client:
                response = client.post(
                    "/translate",
                    json={
                        "input": text,
                        "source_language_code": source_language,
                        "target_language_code": target_language,
                        "mode": "formal",
                        "model": "mayura:v1",
                    },
                )

            if response.status_code == 200:
                data = response.json()
                translated = data.get("translated_text", text)
                return {
                    "translated_text": translated,
                    "language": target_language,
                    "language_name": self.LANGUAGE_MAP.get(target_language, ""),
                }
            else:
                logger.error(f"Sarvam API error: {response.status_code}")
                return {"translated_text": text, "error": f"API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"Translation (sync) failed: {e}")
            return {"translated_text": text, "error": str(e)}

    async def translate_report(
        self,
        report: dict[str, Any],
        target_language: str = "hi",
    ) -> dict[str, Any]:
        """Translate an entire medical report to the target language."""
        clinical = report.get("clinical_report", {})
        translated_sections = {}

        for section_name, section_text in clinical.items():
            if section_text and isinstance(section_text, str):
                result = await self.translate(section_text, target_language)
                translated_sections[section_name] = result.get("translated_text", section_text)
            else:
                translated_sections[section_name] = section_text

        # Build translated report
        translated_report = report.copy()
        translated_report["clinical_report"] = translated_sections
        translated_report["translation"] = {
            "target_language": target_language,
            "language_name": self.LANGUAGE_MAP.get(target_language, ""),
            "translated": True,
        }

        logger.info(f"Report translated to {self.LANGUAGE_MAP.get(target_language, target_language)}")
        return translated_report

    def get_supported_languages(self) -> dict[str, str]:
        """Return supported languages."""
        return self.LANGUAGE_MAP.copy()


class LanguageDetector:
    """Auto-detect input language."""

    def detect(self, text: str) -> str:
        """Simple language detection based on character scripts."""
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')  # Hindi, Marathi
        tamil = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        telugu = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
        kannada = sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')
        bengali = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        malayalam = sum(1 for c in text if '\u0D00' <= c <= '\u0D7F')
        gujarati = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')

        scores = {
            "hi": devanagari, "ta": tamil, "te": telugu,
            "kn": kannada, "bn": bengali, "ml": malayalam,
            "gu": gujarati,
        }

        max_script = max(scores, key=scores.get)
        if scores[max_script] > len(text) * 0.1:
            return max_script

        return "en"
