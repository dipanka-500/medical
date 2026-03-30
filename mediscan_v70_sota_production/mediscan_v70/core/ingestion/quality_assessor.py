"""
MediScan AI v7.0 — Metadata Extractor & Quality Assessor
"""
from __future__ import annotations


import logging
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts and normalizes metadata from any loaded medical data."""

    def extract(self, data: dict[str, Any]) -> dict[str, Any]:
        """Build a unified metadata record from loaded data."""
        return {
            "file_type": data.get("type", "unknown"),
            "source_path": data.get("source_path", ""),
            "modality_info": data.get("modality_info", {}),
            "patient_info": data.get("patient_info", {}),
            "study_info": data.get("study_info", {}),
            "technical_metadata": data.get("metadata", {}),
            "ingestion_timestamp": datetime.utcnow().isoformat(),
        }


class QualityAssessor:
    """Assesses image quality for medical AI suitability.

    v7.0: Modality-aware quality thresholds for pathology/WSI,
    fundoscopy, dermoscopy, ECG, and mammography.
    """

    # Modality-specific minimum resolution thresholds
    MODALITY_MIN_RESOLUTION = {
        "pathology": (256, 256),
        "pathology_wsi": (256, 256),
        "cytology": (256, 256),
        "microbiology": (256, 256),
        "fluorescence_microscopy": (256, 256),
        "fundoscopy": (1024, 1024),
        "oct": (256, 256),
        "dermoscopy": (600, 600),
        "mammography": (1024, 1024),
        "ecg": (800, 200),   # wide format
        "xray": (512, 512),
        "dental": (256, 256),
    }

    def assess(self, data: dict[str, Any], modality: str = "unknown") -> dict[str, Any]:
        """Run quality checks and return a quality report.

        Args:
            data: Loaded image data dict
            modality: Detected modality for modality-specific thresholds
        """
        checks = {}
        overall_score = 1.0

        file_type = data.get("type", "unknown")

        if file_type == "2d":
            pixel_array = data.get("pixel_array")
            if pixel_array is not None:
                meta = data.get("metadata", {})
                # Standard checks
                checks["resolution"] = self._check_resolution_2d(pixel_array, meta, modality)
                checks["contrast"] = self._check_contrast(pixel_array)
                checks["noise"] = self._check_noise(pixel_array)
                checks["blurriness"] = self._check_blur(pixel_array)
                # Modality-specific checks
                if modality in ("pathology", "pathology_wsi", "cytology"):
                    checks["staining"] = self._check_staining(pixel_array)
                elif modality == "fundoscopy":
                    checks["color_channel"] = self._check_fundus_color(pixel_array)
                elif modality == "dermoscopy":
                    checks["color_calibration"] = self._check_dermoscopy_color(pixel_array)

        elif file_type == "3d":
            volume = data.get("volume")
            if volume is not None:
                checks["volume_shape"] = self._check_volume_shape(volume)
                checks["intensity_range"] = self._check_intensity_range(volume)
                checks["slice_count"] = self._check_slice_count(volume)

        elif file_type == "video":
            frames = data.get("frames", [])
            metadata = data.get("metadata", {})
            checks["frame_count"] = self._check_frame_count(frames)
            checks["duration"] = self._check_duration(metadata)

        # Calculate overall score
        if checks:
            scores = [c.get("score", 1.0) for c in checks.values()]
            overall_score = float(np.mean(scores))

        result = {
            "overall_score": round(overall_score, 3),
            "is_acceptable": overall_score >= 0.5,
            "checks": checks,
            "warnings": [c.get("warning") for c in checks.values() if c.get("warning")],
            "modality": modality,
        }

        level = "INFO" if result["is_acceptable"] else "WARNING"
        logger.log(
            getattr(logging, level),
            f"Quality score: {overall_score:.2f} | Modality: {modality} | Acceptable: {result['is_acceptable']}",
        )
        return result

    def _check_resolution_2d(self, arr: np.ndarray, meta: dict, modality: str = "unknown") -> dict:
        h = meta.get("height", arr.shape[0])
        w = meta.get("width", arr.shape[1] if arr.ndim > 1 else 0)

        # Check modality-specific minimum resolution
        min_res = self.MODALITY_MIN_RESOLUTION.get(modality)
        if min_res:
            min_w, min_h = min_res
            if w < min_w or h < min_h:
                return {"score": 0.4, "warning": f"Below minimum resolution for {modality}: {w}x{h} (need {min_w}x{min_h})"}

        # General checks
        if h < 100 or w < 100:
            return {"score": 0.2, "warning": f"Very low resolution: {w}x{h}"}
        if h < 256 or w < 256:
            return {"score": 0.5, "warning": f"Low resolution: {w}x{h}"}
        return {"score": 1.0}

    def _check_contrast(self, arr: np.ndarray) -> dict:
        if arr.dtype == np.uint8:
            std = float(np.std(arr))
            if std < 10:
                return {"score": 0.3, "warning": f"Very low contrast (std={std:.1f})"}
            if std < 30:
                return {"score": 0.6, "warning": f"Low contrast (std={std:.1f})"}
        return {"score": 1.0}

    def _check_noise(self, arr: np.ndarray) -> dict:
        # Noise estimation via local variance (scikit-image)
        if arr.ndim >= 2:
            try:
                from skimage.restoration import estimate_sigma
                sigma = estimate_sigma(arr.astype(np.float64), average_sigmas=True)
                if sigma > 50:
                    return {"score": 0.4, "warning": f"High noise (σ={sigma:.1f})"}
            except ImportError:
                logger.debug("scikit-image not available — skipping noise check")
            except Exception as e:  # noqa: broad-except logged
                pass
        return {"score": 1.0}

    def _check_blur(self, arr: np.ndarray) -> dict:
        if arr.ndim >= 2:
            gray = arr if arr.ndim == 2 else np.mean(arr, axis=-1)
            laplacian_var = float(np.var(np.gradient(gray.astype(np.float64))))
            if laplacian_var < 10:
                return {"score": 0.4, "warning": "Image appears blurry"}
        return {"score": 1.0}

    def _check_volume_shape(self, vol: np.ndarray) -> dict:
        if any(d < 10 for d in vol.shape):
            return {"score": 0.3, "warning": f"Volume dimension too small: {vol.shape}"}
        return {"score": 1.0}

    def _check_intensity_range(self, vol: np.ndarray) -> dict:
        rng = float(vol.max()) - float(vol.min())
        if rng < 1:
            return {"score": 0.1, "warning": "Volume has no intensity variation"}
        return {"score": 1.0}

    def _check_slice_count(self, vol: np.ndarray) -> dict:
        slices = vol.shape[0]
        if slices < 5:
            return {"score": 0.4, "warning": f"Very few slices: {slices}"}
        return {"score": 1.0}

    def _check_frame_count(self, frames: list) -> dict:
        if len(frames) < 2:
            return {"score": 0.3, "warning": f"Very few frames: {len(frames)}"}
        return {"score": 1.0}

    def _check_duration(self, meta: dict) -> dict:
        dur = meta.get("duration_seconds", 0)
        if dur < 1:
            return {"score": 0.4, "warning": "Video too short"}
        return {"score": 1.0}

    # ── Modality-specific quality checks ──────────────────────

    def _check_staining(self, arr: np.ndarray) -> dict:
        """Check adequacy of H&E staining in pathology images."""
        if arr.ndim < 3 or arr.shape[2] < 3:
            return {"score": 0.5, "warning": "Grayscale image — cannot assess staining"}
        mean_color = np.mean(arr, axis=(0, 1))
        r, g, b = mean_color[0], mean_color[1], mean_color[2]
        # H&E stained tissue: purple-pink hue (R > B > G roughly)
        # Very pale or very dark = poor staining
        intensity = (r + g + b) / 3
        if intensity < 40:
            return {"score": 0.3, "warning": "Image too dark — possible under-staining or poor illumination"}
        if intensity > 230:
            return {"score": 0.4, "warning": "Image too bright — possible over-exposure or wash-out"}
        # Check for some color variation (not all gray)
        color_range = max(r, g, b) - min(r, g, b)
        if color_range < 10:
            return {"score": 0.5, "warning": "Low color variation — possible unstained or monochrome tissue"}
        return {"score": 1.0}

    def _check_fundus_color(self, arr: np.ndarray) -> dict:
        """Check that fundoscopy image has appropriate red-channel dominance."""
        if arr.ndim < 3 or arr.shape[2] < 3:
            return {"score": 0.5, "warning": "Expected color fundus image"}
        mean_color = np.mean(arr, axis=(0, 1))
        r, g, b = mean_color[0], mean_color[1], mean_color[2]
        # Fundus images typically have red-dominant channel
        if r < 50 and g < 50 and b < 50:
            return {"score": 0.3, "warning": "Image too dark for reliable fundus analysis"}
        if r < g or r < b:
            return {"score": 0.6, "warning": "Red channel not dominant — unusual for fundus image"}
        return {"score": 1.0}

    def _check_dermoscopy_color(self, arr: np.ndarray) -> dict:
        """Check dermoscopy image color calibration."""
        if arr.ndim < 3 or arr.shape[2] < 3:
            return {"score": 0.5, "warning": "Expected color dermoscopy image"}
        # Dermoscopy images should have reasonable color variation
        std_per_channel = np.std(arr, axis=(0, 1))
        mean_std = float(np.mean(std_per_channel))
        if mean_std < 10:
            return {"score": 0.4, "warning": "Very low color variation — possible imaging artifact"}
        if mean_std > 100:
            return {"score": 0.6, "warning": "High color noise — check illumination and calibration"}
        return {"score": 1.0}

