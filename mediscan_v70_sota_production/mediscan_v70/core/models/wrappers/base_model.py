"""
MediScan AI v5.0 — Base Model Abstract Class
All model wrappers inherit from this to ensure consistent interface.
"""
from __future__ import annotations


import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all medical VLM model wrappers."""

    def __init__(self, model_id: str, config: dict[str, Any]):
        self.model_id = model_id
        self.config = config
        self.model = None
        self.processor = None
        self.device = None
        self.is_loaded = False
        self._load_time = 0.0

    @abstractmethod
    def load(self) -> None:
        """Load model and processor into memory."""
        pass

    @abstractmethod
    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        **kwargs,
    ) -> dict[str, Any]:
        """Run inference on the given input.

        Args:
            images: List of PIL Images, numpy arrays, or video frames
            text: Text prompt / question
            modality: "image", "video", "3d", "text"

        Returns:
            Dict with: response, confidence, thinking (if available), metadata
        """
        pass

    def unload(self) -> None:
        """Free model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info(f"Model unloaded: {self.model_id}")

    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        if self.device:
            return self.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_dtype(self) -> torch.dtype:
        """Get the configured dtype."""
        dtype_str = self.config.get("torch_dtype", "bfloat16")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def health_check(self) -> dict[str, Any]:
        """Check model health status."""
        return {
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "load_time_seconds": self._load_time,
            "device": str(self.device) if self.device else "not_loaded",
            "gpu_memory_mb": self._get_gpu_memory() if self.is_loaded else 0,
        }

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def _timed_inference(self, func, *args, **kwargs) -> tuple[Any, float]:
        """Run inference with timing."""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed

    @staticmethod
    def estimate_confidence(response_text: str, base_confidence: float = 0.75) -> float:
        """Estimate confidence from model response text heuristics.

        Analyzes hedging language, assertion strength, and response
        quality to produce a dynamic confidence score rather than
        returning a static hardcoded value.

        Args:
            response_text: The model's generated response
            base_confidence: Starting confidence (model-specific baseline)

        Returns:
            Adjusted confidence in [0.1, 0.99]
        """
        if not response_text or len(response_text.strip()) < 10:
            return 0.3  # Very short / empty response → low confidence

        text_lower = response_text.lower()
        score = base_confidence

        # Hedging language reduces confidence
        hedging = [
            "may suggest", "possibly", "cannot exclude", "uncertain",
            "unclear", "difficult to determine", "limited by", "equivocal",
            "questionable", "indeterminate", "non-specific", "could represent",
        ]
        hedge_count = sum(1 for h in hedging if h in text_lower)
        score -= hedge_count * 0.04

        # Strong assertion language increases confidence
        assertive = [
            "consistent with", "diagnostic of", "characteristic of",
            "highly suggestive", "pathognomonic", "definite", "clearly shows",
            "confirmed", "classic appearance", "typical findings",
        ]
        assert_count = sum(1 for a in assertive if a in text_lower)
        score += assert_count * 0.03

        # Structured/detailed response → higher confidence
        if len(response_text) > 500:
            score += 0.03
        if len(response_text) > 1000:
            score += 0.02

        # Multiple findings mentioned → comprehensive analysis
        finding_indicators = ["finding", "impression", "diagnosis", "recommendation"]
        section_count = sum(1 for f in finding_indicators if f in text_lower)
        score += min(section_count * 0.02, 0.06)

        return max(0.1, min(0.99, score))
