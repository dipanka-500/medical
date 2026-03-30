from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        logger.warning(
            "Invalid float value for %s: '%s', using default %s", name, value, default
        )
        return default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        logger.warning(
            "Invalid int value for %s: '%s', using default %s", name, value, default
        )
        return default


@dataclass
class Settings:
    workspace_dir: Path
    handwriting_checkpoint: Path
    output_dir: Path
    artifacts_dir: Path
    default_language_code: str = "en-IN"
    default_indic_language_code: str = "hi-IN"
    sarvam_output_format: str = "md"
    sarvam_poll_interval_seconds: float = 2.0
    sarvam_wait_timeout_seconds: float = 300.0
    local_command_timeout_seconds: float = 300.0
    sarvam_api_key: str = field(default="", repr=False)
    prefer_remote_api: bool = True
    enable_preprocessing: bool = True
    handwriting_index: int = 1
    handwritten_high_threshold: float = 0.70
    handwritten_low_threshold: float = 0.30
    analysis_page_limit: int = 3
    preserve_artifacts: bool = True

    @property
    def has_sarvam_key(self) -> bool:
        return bool(self.sarvam_api_key.strip())

    def __post_init__(self) -> None:
        if self.sarvam_poll_interval_seconds <= 0:
            logger.warning("sarvam_poll_interval_seconds must be positive, resetting to 2.0")
            self.sarvam_poll_interval_seconds = 2.0
        if self.sarvam_wait_timeout_seconds <= 0:
            logger.warning("sarvam_wait_timeout_seconds must be positive, resetting to 300.0")
            self.sarvam_wait_timeout_seconds = 300.0
        if self.local_command_timeout_seconds <= 0:
            logger.warning("local_command_timeout_seconds must be positive, resetting to 300.0")
            self.local_command_timeout_seconds = 300.0
        if self.analysis_page_limit < 1:
            self.analysis_page_limit = 1
        if not (0.0 <= self.handwritten_low_threshold <= self.handwritten_high_threshold <= 1.0):
            logger.warning(
                "Invalid handwriting thresholds (low=%.2f, high=%.2f), resetting to defaults",
                self.handwritten_low_threshold, self.handwritten_high_threshold,
            )
            self.handwritten_low_threshold = 0.30
            self.handwritten_high_threshold = 0.70
        if self.sarvam_output_format not in {"md", "html"}:
            logger.warning(
                "Invalid sarvam_output_format '%s', resetting to 'md'",
                self.sarvam_output_format,
            )
            self.sarvam_output_format = "md"


def load_settings(base_dir: str | Path | None = None) -> Settings:
    workspace_dir = Path(base_dir or os.getcwd()).resolve()
    output_dir = workspace_dir / "outputs"
    artifacts_dir = workspace_dir / ".mediscan"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        workspace_dir=workspace_dir,
        handwriting_checkpoint=workspace_dir / "binary_handwriting_classifier.pth",
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        default_language_code=os.getenv("MEDISCAN_DEFAULT_LANGUAGE_CODE", "en-IN"),
        default_indic_language_code=os.getenv("MEDISCAN_DEFAULT_INDIC_LANGUAGE_CODE", "hi-IN"),
        sarvam_output_format=os.getenv("MEDISCAN_SARVAM_OUTPUT_FORMAT", "md"),
        sarvam_poll_interval_seconds=_get_float("MEDISCAN_SARVAM_POLL_SECONDS", 2.0),
        sarvam_wait_timeout_seconds=_get_float("MEDISCAN_SARVAM_TIMEOUT_SECONDS", 300.0),
        local_command_timeout_seconds=_get_float("MEDISCAN_LOCAL_COMMAND_TIMEOUT_SECONDS", 300.0),
        sarvam_api_key=os.getenv("SARVAM_API_KEY", ""),
        prefer_remote_api=_get_bool("MEDISCAN_PREFER_REMOTE_API", True),
        enable_preprocessing=_get_bool("MEDISCAN_ENABLE_PREPROCESSING", True),
        handwriting_index=_get_int("MEDISCAN_HANDWRITING_INDEX", 1),
        analysis_page_limit=_get_int("MEDISCAN_ANALYSIS_PAGE_LIMIT", 3),
        preserve_artifacts=_get_bool("MEDISCAN_PRESERVE_ARTIFACTS", True),
    )
