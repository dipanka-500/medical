from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


NAME_HINTS = {
    "english": ("en-IN", "english"),
    "latin": ("en-IN", "english"),
    "hindi": ("hi-IN", "indic"),
    "devanagari": ("hi-IN", "indic"),
    "marathi": ("mr-IN", "indic"),
    "tamil": ("ta-IN", "indic"),
    "telugu": ("te-IN", "indic"),
    "kannada": ("kn-IN", "indic"),
    "malayalam": ("ml-IN", "indic"),
    "gujarati": ("gu-IN", "indic"),
    "bengali": ("bn-IN", "indic"),
    "urdu": ("ur-IN", "indic"),
    "punjabi": ("pa-IN", "indic"),
    "odia": ("od-IN", "indic"),
    "oriya": ("od-IN", "indic"),
    "assamese": ("as-IN", "indic"),
}


class HandwritingLanguageDetector:
    def __init__(self, default_indic_language_code: str = "hi-IN") -> None:
        self.default_indic_language_code = default_indic_language_code

    def detect_from_name(self, path: str | Path) -> Dict[str, object]:
        name = Path(path).stem.lower()
        for token, (language_code, family) in NAME_HINTS.items():
            if token in name:
                return {
                    "language_code": language_code,
                    "language_family": family,
                    "confidence": 0.85,
                    "source": "filename_hint",
                }
        return {
            "language_code": "unknown",
            "language_family": "unknown",
            "confidence": 0.0,
            "source": "filename_hint_missing",
        }

    def _features(self, image_path: str) -> Dict[str, float]:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {}

        _, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        total_ink = float(thresh.sum() / 255.0) + 1e-6
        height, width = thresh.shape

        horizontal_projection = thresh.sum(axis=1) / 255.0
        top_band_ratio = float(horizontal_projection[: max(1, height // 5)].sum() / total_ink)
        top_peak_strength = float(horizontal_projection[: max(1, height // 3)].max() / (horizontal_projection.mean() + 1e-6))

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        aspects: List[float] = []
        fills: List[float] = []
        areas: List[float] = []
        for contour in contours[:1000]:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3:
                continue
            region = thresh[y : y + h, x : x + w]
            fill_ratio = float(region.mean() / 255.0)
            aspects.append(float(w / max(h, 1)))
            fills.append(fill_ratio)
            areas.append(float((w * h) / max(width * height, 1)))

        if not aspects:
            return {}

        aspects_array = np.array(aspects, dtype=np.float32)
        fills_array = np.array(fills, dtype=np.float32)
        areas_array = np.array(areas, dtype=np.float32)
        return {
            "top_band_ratio": top_band_ratio,
            "top_peak_strength": min(top_peak_strength / 6.0, 1.0),
            "wide_ratio": float(np.mean(aspects_array > 0.95)),
            "slender_ratio": float(np.mean(aspects_array < 0.55)),
            "fill_mean": float(np.mean(fills_array)),
            "area_mean": min(float(np.mean(areas_array) * 1500.0), 1.0),
            "component_density": min(float(len(aspects) / max(width / 10.0, 1.0)), 1.0),
        }

    def detect_from_pages(self, page_paths: List[str]) -> Dict[str, object]:
        page_paths = [path for path in page_paths if path]
        if not page_paths:
            return {
                "language_code": "unknown",
                "language_family": "unknown",
                "confidence": 0.0,
                "source": "no_pages",
            }

        feature_sets = [self._features(path) for path in page_paths[:3]]
        feature_sets = [features for features in feature_sets if features]
        if not feature_sets:
            return {
                "language_code": "unknown",
                "language_family": "unknown",
                "confidence": 0.0,
                "source": "feature_extraction_failed",
            }

        averaged = {}
        for key in feature_sets[0].keys():
            averaged[key] = float(np.mean([features[key] for features in feature_sets]))

        indic_score = (
            0.30 * averaged["top_peak_strength"]
            + 0.20 * averaged["top_band_ratio"]
            + 0.20 * averaged["wide_ratio"]
            + 0.15 * averaged["fill_mean"]
            + 0.15 * averaged["area_mean"]
        )
        english_score = (
            0.35 * averaged["slender_ratio"]
            + 0.20 * (1.0 - averaged["top_band_ratio"])
            + 0.20 * averaged["component_density"]
            + 0.15 * (1.0 - averaged["wide_ratio"])
            + 0.10 * (1.0 - averaged["area_mean"])
        )

        if indic_score >= english_score:
            family = "indic"
            language_code = self.default_indic_language_code
        else:
            family = "english"
            language_code = "en-IN"

        margin = abs(indic_score - english_score)
        confidence = min(0.55 + margin, 0.90)
        return {
            "language_code": language_code,
            "language_family": family,
            "confidence": confidence,
            "source": "handwriting_script_heuristic",
            "scores": {
                "indic": round(indic_score, 4),
                "english": round(english_score, 4),
            },
        }
