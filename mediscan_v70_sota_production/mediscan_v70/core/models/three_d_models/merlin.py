"""
MediScan AI v7.0 — Merlin 3D CT Vision-Language Model Wrapper
Based on official implementation from:
  https://github.com/StanfordMIMI/Merlin
  HuggingFace: StanfordMIMI/Merlin

Architecture: 3D Vision Transformer encoder + LLaMA decoder
Speciality:  3D CT volume interpretation — organ segmentation, abnormality
             detection, radiology report generation, visual question answering.
Pretraining: 15,000+ CT volumes (AbdomenAtlas + RadGenome-ChestCT).

Key differences from Med3DVLM:
  - Merlin uses a 3D ViT encoder (not DCFormer/SigLIP)
  - Trained specifically on CT (not general MRI+CT)
  - Excels at organ-level findings and structured report generation
  - Provides grounded localisation of abnormalities

Official inference API:
  model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
  processor = AutoProcessor.from_pretrained(...)
  Volumes resized to (32, 256, 256) by default.
"""
from __future__ import annotations


import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from ...wrappers.base_model import BaseModel

logger = logging.getLogger(__name__)


class MerlinModel(BaseModel):
    """
    Merlin: A 3D Vision Language Model for CT Volume Understanding.

    From Stanford MIMI.  Uses a 3D ViT encoder with LLaMA decoder.
    Excels at CT-specific tasks: organ segmentation, finding detection,
    radiology report generation, and structured QA.

    Input: NIfTI (.nii.gz) → SimpleITK → numpy → Resize(32,256,256) → tensor
    """

    IMAGE_SIZE = (32, 256, 256)
    MAX_NEW_TOKENS = 2048

    def load(self) -> None:
        """Load Merlin using official HuggingFace API."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        start = time.time()
        logger.info(f"Loading Merlin 3D CT VLM: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.get_dtype(),
            device_map=self.config.get("device_map", "auto"),
            trust_remote_code=True,
        )

        self.device = next(self.model.parameters()).device
        self.is_loaded = True
        self._load_time = time.time() - start
        logger.info(f"Merlin loaded in {self._load_time:.1f}s on {self.device}")

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "3d",
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.9,
        **kwargs,
    ) -> dict[str, Any]:
        """Run Merlin inference on a 3D CT volume.

        Args:
            images: Not used directly — use volume_array or volume_path in kwargs
            text: The medical question/prompt
            modality: Should be "3d" for this model
            **kwargs:
                volume_array: Pre-loaded 3D numpy array
                volume_path / nii_path / source_path: Path to NIfTI file
        """
        if not self.is_loaded:
            self.load()

        volume = self._get_volume(images, kwargs)

        # Preprocess: CT-specific windowing + resize
        image_size = kwargs.get("image_size", self.IMAGE_SIZE)
        volume_tensor = self._preprocess_volume(volume, image_size)

        # Build prompt
        prompt = self._build_prompt(text)

        # Process inputs via processor
        inputs = self.processor(
            text=prompt, images=volume_tensor, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        response = self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        )

        return {
            "model": self.model_id,
            "response": response.strip(),
            "answer": response.strip(),
            "thinking": "",
            "modality": "3d",
            "confidence": self.estimate_confidence(response.strip(), base_confidence=0.80),
            "metadata": {
                "volume_shape": list(volume.shape),
                "image_size": list(image_size),
                "max_new_tokens": max_new_tokens,
                "model_type": "merlin_3d_ct",
            },
        }

    def _preprocess_volume(
        self, volume: np.ndarray, target_size: tuple[int, ...]
    ) -> torch.Tensor:
        """Preprocess 3D volume for Merlin.

        Steps:
          1. Clip to CT HU range [-1024, 3071]
          2. Normalize to [0, 1]
          3. Resize to target dimensions via trilinear interpolation
          4. Add batch and channel dims: (1, 1, D, H, W)
        """
        volume = np.clip(volume, -1024, 3071).astype(np.float32)
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)

        try:
            from monai.transforms import Resize
            resize_fn = Resize(spatial_size=target_size, mode="trilinear")
            tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)
            tensor = resize_fn(tensor)
        except ImportError:
            tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(
                tensor, size=target_size, mode="trilinear", align_corners=False
            )
            tensor = tensor.squeeze(0)

        return tensor.unsqueeze(0).to(dtype=self.get_dtype())

    def _build_prompt(self, question: str) -> str:
        """Build a structured prompt for Merlin CT analysis."""
        if not question or question.strip() == "":
            question = (
                "Analyze this CT volume comprehensively. Describe all visible "
                "organs, identify any abnormalities, and provide a structured "
                "radiology report with findings, impressions, and recommendations."
            )
        return question

    def _get_volume(self, images: Optional[list], kwargs: dict) -> np.ndarray:
        """Extract 3D volume from various input formats."""
        volume_array = kwargs.get("volume_array")
        if volume_array is not None:
            return np.asarray(volume_array, dtype=np.float32)

        volume_path = (
            kwargs.get("volume_path")
            or kwargs.get("nii_path")
            or kwargs.get("source_path")
        )
        if volume_path:
            import SimpleITK as sitk
            logger.info(f"Loading 3D volume from: ...{Path(str(volume_path)).name}")
            img = sitk.ReadImage(str(volume_path))
            return sitk.GetArrayFromImage(img).astype(np.float32)

        if images and isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], np.ndarray) and images[0].ndim == 3:
                return images[0].astype(np.float32)

        raise ValueError(
            "Merlin requires 3D CT volume input. Provide volume_array, "
            "volume_path, nii_path, or source_path."
        )

    def extract_organ_findings(self, volume: np.ndarray, text: str = "") -> dict[str, Any]:
        """Run organ-specific finding extraction leveraging AbdomenAtlas training."""
        organ_prompt = (
            "For each visible organ in this CT volume, provide: "
            "1) organ name, 2) appearance (normal/abnormal), "
            "3) specific findings if abnormal, 4) measurements if relevant. "
            "Format as a structured list."
        )
        combined = f"{text}\n{organ_prompt}" if text else organ_prompt
        result = self.analyze(text=combined, volume_array=volume)
        result["metadata"]["analysis_type"] = "organ_findings"
        return result

    def extract_slices_for_display(
        self, volume: np.ndarray
    ) -> dict[str, list[Image.Image]]:
        """Extract axial/coronal/sagittal slices for visualization."""
        def normalize_slice(s: np.ndarray) -> Image.Image:
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                norm = ((s - s_min) / (s_max - s_min) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(s, dtype=np.uint8)
            return Image.fromarray(norm).convert("L")

        return {
            "axial": [normalize_slice(volume[i]) for i in range(volume.shape[0])],
            "coronal": [normalize_slice(volume[:, i]) for i in range(volume.shape[1])],
            "sagittal": [normalize_slice(volume[:, :, i]) for i in range(volume.shape[2])],
        }
