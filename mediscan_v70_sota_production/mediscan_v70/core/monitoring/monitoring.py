"""
MediScan AI v7.0 — Monitoring: Drift Detection, OOD Detection, Performance Metrics
"""
from __future__ import annotations


import logging
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects distribution drift in model confidence scores over time."""

    def __init__(self, window_size: int = 100, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self._baseline: list[float] = []
        self._recent: deque = deque(maxlen=window_size)

    def set_baseline(self, confidence_scores: list[float]) -> None:
        """Set baseline confidence distribution."""
        self._baseline = confidence_scores
        logger.info(f"Drift baseline set: {len(confidence_scores)} samples, mean={np.mean(confidence_scores):.3f}")

    def add_observation(self, confidence: float) -> None:
        """Add a new confidence observation."""
        self._recent.append(confidence)

    def check_drift(self) -> dict[str, Any]:
        """Check for confidence distribution drift."""
        if not self._baseline or len(self._recent) < 10:
            return {"drift_detected": False, "reason": "Insufficient data"}

        baseline_mean = np.mean(self._baseline)
        recent_mean = np.mean(list(self._recent))
        diff = abs(recent_mean - baseline_mean)

        return {
            "drift_detected": diff > self.threshold,
            "baseline_mean": round(float(baseline_mean), 4),
            "recent_mean": round(float(recent_mean), 4),
            "drift_magnitude": round(float(diff), 4),
            "threshold": self.threshold,
            "observations": len(self._recent),
        }


class OODDetector:
    """Detects out-of-distribution inputs."""

    def __init__(self):
        # v7.0: Extended to match all modalities supported by the modality detector
        self._known_modalities = {
            # Radiology
            "xray", "ct", "mri", "mammography", "fluoroscopy", "angiography",
            # Ultrasound
            "ultrasound", "echocardiography", "intravascular_ultrasound",
            "ultrasound_clip",
            # Nuclear Medicine
            "pet", "spect", "nuclear_medicine",
            # Pathology
            "pathology", "cytology", "microbiology", "histopathology",
            "fluorescence_microscopy",
            # Ophthalmology
            "fundoscopy", "oct",
            # Dermatology
            "dermoscopy", "clinical_photo",
            # Dental
            "dental", "dental_intraoral", "dental_panoramic",
            # Cardiology
            "ecg",
            # Endoscopy / Surgery
            "endoscopy", "surgical_video",
            # Advanced
            "dti", "fmri",
            # Generic
            "video", "3d_volume", "general_medical",
        }

    def check(self, modality_info: dict[str, Any], confidence: float) -> dict[str, Any]:
        """Check if input appears out-of-distribution."""
        modality = modality_info.get("modality", "unknown")
        mod_confidence = modality_info.get("confidence", 0)

        is_ood = False
        reasons = []

        if modality not in self._known_modalities:
            is_ood = True
            reasons.append(f"Unknown modality: {modality}")

        if mod_confidence < 0.4:
            is_ood = True
            reasons.append(f"Low modality detection confidence: {mod_confidence:.2f}")

        if confidence < 0.3:
            is_ood = True
            reasons.append(f"Very low model confidence: {confidence:.2f}")

        return {
            "is_ood": is_ood,
            "reasons": reasons,
            "modality": modality,
            "recommendation": "Flag for human review" if is_ood else "Normal",
        }


class PerformanceMetrics:
    """Tracks system performance metrics."""

    def __init__(self):
        self._inference_times: deque = deque(maxlen=1000)
        self._request_count = 0
        self._error_count = 0
        self._start_time = time.time()

    def record_inference(self, duration: float, model_key: str, success: bool = True) -> None:
        """Record an inference event."""
        self._inference_times.append({
            "duration": duration,
            "model": model_key,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._request_count += 1
        if not success:
            self._error_count += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        durations = [t["duration"] for t in self._inference_times if t["success"]]
        uptime = time.time() - self._start_time

        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "uptime_seconds": round(uptime, 1),
            "inference_stats": {
                "count": len(durations),
                "mean_seconds": round(float(np.mean(durations)), 3) if durations else 0,
                "p50_seconds": round(float(np.percentile(durations, 50)), 3) if durations else 0,
                "p95_seconds": round(float(np.percentile(durations, 95)), 3) if durations else 0,
                "p99_seconds": round(float(np.percentile(durations, 99)), 3) if durations else 0,
            },
        }
