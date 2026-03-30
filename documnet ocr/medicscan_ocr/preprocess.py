from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from medicscan_ocr.schemas import PreprocessResult
from medicscan_ocr.utils.files import ensure_directory, is_image_path

logger = logging.getLogger(__name__)


def _estimate_skew(binary_image: np.ndarray) -> float:
    coords = np.column_stack(np.where(binary_image > 0))
    if coords.size == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_file(
    input_path: str,
    artifacts_dir: str,
    enabled: bool = True,
) -> PreprocessResult:
    source = Path(input_path).resolve()
    if not enabled or not is_image_path(source):
        return PreprocessResult(
            input_path=str(source),
            processed_path=str(source),
            was_modified=False,
            applied_steps=[],
            metadata={"reason": "preprocessing_skipped"},
        )

    image = cv2.imread(str(source))
    if image is None:
        logger.warning("Failed to load image for preprocessing: %s", source)
        return PreprocessResult(
            input_path=str(source),
            processed_path=str(source),
            was_modified=False,
            applied_steps=[],
            metadata={"reason": "image_load_failed"},
        )

    applied_steps = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    applied_steps.append("clahe")

    denoised = cv2.fastNlMeansDenoising(contrast, None, 10, 7, 21)
    applied_steps.append("denoise")

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    applied_steps.append("adaptive_threshold")

    inverted = cv2.bitwise_not(binary)
    angle = _estimate_skew(inverted)

    if abs(angle) >= 0.5:
        image = _rotate_image(image, angle)
        applied_steps.append("deskew")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = clahe.apply(l_channel)
    l_sharpened = cv2.addWeighted(
        l_enhanced, 1.5,
        cv2.GaussianBlur(l_enhanced, (0, 0), 3), -0.5,
        0,
    )
    enhanced_lab = cv2.merge([l_sharpened, a_channel, b_channel])
    result_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    applied_steps.append("color_enhance")
    applied_steps.append("sharpen")

    target_dir = ensure_directory(Path(artifacts_dir) / "preprocessed")
    # Use hash of full path to avoid same-stem collisions from different directories
    import hashlib
    path_hash = hashlib.md5(str(source).encode()).hexdigest()[:8]
    output_path = target_dir / "{0}_{1}_preprocessed.png".format(source.stem, path_hash)
    cv2.imwrite(str(output_path), result_image)

    return PreprocessResult(
        input_path=str(source),
        processed_path=str(output_path),
        was_modified=True,
        applied_steps=applied_steps,
        metadata={
            "angle": round(angle, 3),
            "shape": list(result_image.shape),
        },
    )
