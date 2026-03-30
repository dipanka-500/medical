from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchvision import models, transforms
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]

from medicscan_ocr.schemas import DocumentType
from medicscan_ocr.utils.files import is_image_path

logger = logging.getLogger(__name__)


class HandwritingClassifier:
    def __init__(self, checkpoint_path: str, handwritten_index: int = 1) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.handwritten_index = handwritten_index
        self._model: Optional[nn.Module] = None
        self._transform = None
        if transforms is not None:
            self._transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def available(self) -> bool:
        return (
            self.checkpoint_path.exists()
            and torch is not None
            and transforms is not None
            and models is not None
            and nn is not None
            and Image is not None
        )

    def _build_model(self) -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model.eval()

    def _load_model(self) -> nn.Module:
        if self._model is not None:
            return self._model
        model = self._build_model()
        load_kwargs = {"map_location": "cpu"}
        try:
            state_dict = torch.load(
                str(self.checkpoint_path),
                weights_only=True,
                **load_kwargs
            )
        except TypeError:
            state_dict = torch.load(str(self.checkpoint_path), **load_kwargs)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                cleaned[key[6:]] = value
            else:
                cleaned[key] = value
        model.load_state_dict(cleaned, strict=True)
        self._model = model.eval()
        return self._model

    def _heuristic_score(self, image_path: str) -> float:
        if cv2 is None or np is None:
            return 0.5
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0.5
        _, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.5
        heights = []
        widths = []
        fills = []
        for contour in contours[:800]:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3:
                continue
            region = thresh[y : y + h, x : x + w]
            fill_ratio = float(region.mean() / 255.0)
            widths.append(w)
            heights.append(h)
            fills.append(fill_ratio)
        if not heights:
            return 0.5
        height_var = float(np.std(heights) / (np.mean(heights) + 1e-6))
        width_var = float(np.std(widths) / (np.mean(widths) + 1e-6))
        fill_mean = float(np.mean(fills))
        raw = 0.4 * height_var + 0.3 * width_var + 0.3 * fill_mean
        return max(0.0, min(raw, 1.0))

    def predict_handwritten_probability(self, image_path: str) -> Dict[str, float]:
        if not is_image_path(image_path):
            return {"probability": 0.5, "source": "unsupported"}
        if not self.available():
            if self.checkpoint_path.exists() and torch is None:
                logger.info(
                    "Handwriting checkpoint found, but torch/torchvision is unavailable; using heuristic fallback"
                )
            return {
                "probability": self._heuristic_score(image_path),
                "source": "heuristic",
            }
        try:
            model = self._load_model()
            with Image.open(image_path) as _img:
                image = _img.convert("RGB")
            tensor = self._transform(image).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
            raw_probability = float(probs[self.handwritten_index].item())
            heuristic_probability = self._heuristic_score(image_path)
            flipped_probability = 1.0 - raw_probability

            if abs(raw_probability - heuristic_probability) > 0.60 and abs(flipped_probability - heuristic_probability) < abs(raw_probability - heuristic_probability):
                blended = 0.7 * flipped_probability + 0.3 * heuristic_probability
                return {
                    "probability": blended,
                    "source": "checkpoint_flipped_by_heuristic",
                }

            blended = 0.85 * raw_probability + 0.15 * heuristic_probability
            return {"probability": blended, "source": "checkpoint_blended"}
        except Exception:
            return {
                "probability": self._heuristic_score(image_path),
                "source": "heuristic_fallback",
            }

    def classify(
        self,
        image_path: str,
        high_threshold: float = 0.70,
        low_threshold: float = 0.30,
    ) -> Dict[str, object]:
        result = self.predict_handwritten_probability(image_path)
        probability = float(result["probability"])
        if probability >= high_threshold:
            doc_type = DocumentType.HANDWRITTEN
        elif probability <= low_threshold:
            doc_type = DocumentType.PRINTED
        else:
            doc_type = DocumentType.MIXED
        return {
            "document_type": doc_type,
            "confidence": probability,
            "source": result["source"],
        }
