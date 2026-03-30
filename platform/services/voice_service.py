"""
Optional speech services for chat voice input/output.

The implementation is intentionally lazy-loaded so the main API can boot
without heavyweight speech dependencies until voice is actually used.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import tempfile
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


def _resolve_device(preferred: str) -> str:
    """Map 'auto' to a sensible runtime default."""
    if preferred != "auto":
        return preferred

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices and cuda_visible_devices != "-1":
        return "cuda"
    return "cpu"


def _normalize_language_tag(language: str | None) -> str | None:
    """Reduce locale-style tags (en-US, hi_IN) to model-friendly primary codes."""
    if not language:
        return None

    normalized = language.strip().replace("_", "-")
    if not normalized:
        return None

    primary = normalized.split("-", 1)[0].strip().lower()
    return primary or None


@dataclass(frozen=True)
class VoiceCapabilities:
    asr_available: bool
    asr_provider: str
    asr_model: str | None
    tts_available: bool
    tts_provider: str
    tts_model: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "asr_available": self.asr_available,
            "asr_provider": self.asr_provider,
            "asr_model": self.asr_model,
            "tts_available": self.tts_available,
            "tts_provider": self.tts_provider,
            "tts_model": self.tts_model,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float | None
    provider: str
    model: str
    segment_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration_seconds": self.duration_seconds,
            "provider": self.provider,
            "model": self.model,
            "segment_count": self.segment_count,
        }


class VoiceServiceUnavailable(RuntimeError):
    """Raised when voice is disabled or dependencies are missing."""


class VoiceService:
    """Lazy voice model wrapper for optional ASR and TTS backends."""

    def __init__(self) -> None:
        self._asr_model: object | None = None
        self._tts_model: object | None = None
        self._asr_lock = asyncio.Lock()
        self._tts_lock = asyncio.Lock()

    def capabilities(self) -> VoiceCapabilities:
        asr_provider = settings.voice_asr_provider
        tts_provider = settings.voice_tts_provider

        asr_available = (
            asr_provider == "faster_whisper"
            and importlib.util.find_spec("faster_whisper") is not None
        )
        tts_available = (
            tts_provider == "coqui"
            and importlib.util.find_spec("TTS") is not None
        )

        return VoiceCapabilities(
            asr_available=asr_available,
            asr_provider=asr_provider,
            asr_model=settings.voice_asr_model if asr_provider != "disabled" else None,
            tts_available=tts_available,
            tts_provider=tts_provider,
            tts_model=settings.voice_tts_model if tts_provider != "disabled" else None,
        )

    async def _get_asr_model(self):
        if self._asr_model is not None:
            return self._asr_model

        async with self._asr_lock:
            if self._asr_model is not None:
                return self._asr_model

            if settings.voice_asr_provider == "disabled":
                raise VoiceServiceUnavailable("Server speech recognition is disabled.")

            if settings.voice_asr_provider != "faster_whisper":
                raise VoiceServiceUnavailable("Unsupported ASR provider configuration.")

            if importlib.util.find_spec("faster_whisper") is None:
                raise VoiceServiceUnavailable(
                    "Install the platform voice extras to enable faster-whisper ASR."
                )

            from faster_whisper import WhisperModel

            device = _resolve_device(settings.voice_asr_device)
            load_kwargs: dict[str, object] = {
                "device": device,
                "compute_type": settings.voice_asr_compute_type,
            }
            if settings.voice_model_cache_dir:
                load_kwargs["download_root"] = settings.voice_model_cache_dir

            self._asr_model = WhisperModel(settings.voice_asr_model, **load_kwargs)
            logger.info(
                "Voice ASR ready [provider=%s model=%s device=%s compute_type=%s]",
                settings.voice_asr_provider,
                settings.voice_asr_model,
                device,
                settings.voice_asr_compute_type,
            )

        return self._asr_model

    async def _get_tts_model(self):
        if self._tts_model is not None:
            return self._tts_model

        async with self._tts_lock:
            if self._tts_model is not None:
                return self._tts_model

            if settings.voice_tts_provider == "disabled":
                raise VoiceServiceUnavailable("Server text-to-speech is disabled.")

            if settings.voice_tts_provider != "coqui":
                raise VoiceServiceUnavailable("Unsupported TTS provider configuration.")

            if importlib.util.find_spec("TTS") is None:
                raise VoiceServiceUnavailable(
                    "Install the platform voice extras to enable Coqui XTTS."
                )

            from TTS.api import TTS

            device = _resolve_device(settings.voice_tts_device)
            model = TTS(settings.voice_tts_model)
            if hasattr(model, "to"):
                model = model.to(device)
            self._tts_model = model
            logger.info(
                "Voice TTS ready [provider=%s model=%s device=%s]",
                settings.voice_tts_provider,
                settings.voice_tts_model,
                device,
            )

        return self._tts_model

    async def transcribe_file(
        self,
        audio_path: str,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        model = await self._get_asr_model()
        language_code = _normalize_language_tag(language)

        def _run() -> TranscriptionResult:
            segments, info = model.transcribe(
                audio_path,
                beam_size=settings.voice_asr_beam_size,
                language=language_code or None,
                condition_on_previous_text=False,
                vad_filter=True,
            )
            segment_list = list(segments)
            text = " ".join(
                segment.text.strip()
                for segment in segment_list
                if getattr(segment, "text", "").strip()
            ).strip()
            return TranscriptionResult(
                text=text,
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", None),
                duration_seconds=getattr(info, "duration", None),
                provider="faster_whisper",
                model=settings.voice_asr_model,
                segment_count=len(segment_list),
            )

        return await asyncio.to_thread(_run)

    async def synthesize_text(
        self,
        text: str,
        *,
        language: str | None = None,
        speaker_wav: str | None = None,
    ) -> bytes:
        model = await self._get_tts_model()

        def _run() -> bytes:
            temp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_path = tmp.name

                synth_kwargs: dict[str, object] = {
                    "text": text,
                    "file_path": temp_path,
                }

                speaker_wav_path = speaker_wav or settings.voice_tts_speaker_wav
                if speaker_wav_path:
                    synth_kwargs["speaker_wav"] = speaker_wav_path
                elif settings.voice_tts_speaker:
                    synth_kwargs["speaker"] = settings.voice_tts_speaker

                synth_language = (
                    _normalize_language_tag(language)
                    or _normalize_language_tag(settings.voice_tts_language)
                )
                if synth_language:
                    synth_kwargs["language"] = synth_language

                model.tts_to_file(**synth_kwargs)

                with open(temp_path, "rb") as generated_audio:
                    return generated_audio.read()
            finally:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._asr_model = None
        self._tts_model = None
