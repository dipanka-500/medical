"""
MediScan AI v5.0 — Med3DVLM Model Wrapper
Based STRICTLY on official API from:
  https://github.com/mirthAI/Med3DVLM
  https://github.com/mirthAI/Med3DVLM/blob/main/app.py
  HuggingFace: MagicXin/Med3DVLM-Qwen-2.5-7B

Architecture: DCFormer encoder + SigLIP + Dual-stream MLP-Mixer + Qwen-2.5-7B
Speciality: 3D volumetric medical image analysis (CT/MRI NIfTI)
"""
from __future__ import annotations


import logging
import time
from typing import Any, Optional
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Resize
from PIL import Image

from ...wrappers.base_model import BaseModel

logger = logging.getLogger(__name__)


class Med3DVLMModel(BaseModel):
    """
    Med3DVLM: Efficient Vision-Language Model for 3D Medical Image Analysis.

    Official API: AutoModelForCausalLM + AutoProcessor (trust_remote_code=True)
    Uses <im_patch> token injection and MONAI Resize for 3D volume preprocessing.
    Input: NIfTI (.nii.gz) → SimpleITK → numpy → MONAI Resize(128,256,256) → tensor
    """

    IMAGE_SIZE = (128, 256, 256)  # Official default from app.py
    MAX_LENGTH = 1024

    def load(self) -> None:
        """Load Med3DVLM using official API from app.py."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        start = time.time()
        logger.info(f"Loading Med3DVLM: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            max_length=self.MAX_LENGTH,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.get_dtype(),
            device_map=self.config.get("device_map", "auto"),
            trust_remote_code=True,
        )

        # Get projection output number from model config (official code)
        try:
            self.proj_out_num = (
                self.model.get_model().config.proj_out_num
                if hasattr(self.model.get_model().config, "proj_out_num")
                else 256
            )
        except Exception as e:
            logger.warning(f"Could not read proj_out_num from model config: {e}")
            self.proj_out_num = 256

        self.device = next(self.model.parameters()).device
        self.is_loaded = True
        self._load_time = time.time() - start
        logger.info(
            f"Med3DVLM loaded in {self._load_time:.1f}s | "
            f"proj_out_num={self.proj_out_num}"
        )

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "3d",
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> dict[str, Any]:
        """Run Med3DVLM inference following official app.py implementation.

        Args:
            images: Not used directly — use volume_array or volume_path in kwargs
            text: The medical question/prompt
            modality: Should be "3d" for this model
            **kwargs:
                volume_array: Pre-loaded 3D numpy array
                volume_path: Path to NIfTI file
        """
        if not self.is_loaded:
            self.load()

        # Get the 3D volume
        volume = self._get_volume(images, kwargs)

        # Preprocess volume exactly as in official app.py:
        # 1. Expand dims to add channel: (1, D, H, W)
        # 2. MONAI Resize to IMAGE_SIZE
        # 3. Unsqueeze for batch: (1, 1, D, H, W)
        image_size = kwargs.get("image_size", self.IMAGE_SIZE)
        resize_transform = Resize(spatial_size=image_size, mode="bilinear")
        image_input = np.expand_dims(volume.copy(), axis=0)  # (1, D, H, W)
        image_input = resize_transform(image_input)  # MONAI Resize
        image_input = image_input.data.unsqueeze(0).to(
            device=self.device, dtype=self.get_dtype()
        )  # (1, 1, D, H, W)

        # Build input text with <im_patch> tokens (official code)
        image_tokens = "<im_patch>" * self.proj_out_num
        input_txt = image_tokens + text
        input_ids = self.processor(input_txt, return_tensors="pt")["input_ids"].to(
            device=self.device
        )

        # Generate — official API passes images= separately
        with torch.no_grad():
            generation = self.model.generate(
                images=image_input,
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
            )

        response = self.processor.decode(generation[0], skip_special_tokens=True)

        return {
            "model": self.model_id,
            "response": response.strip(),
            "answer": response.strip(),
            "thinking": "",
            "modality": "3d",
            "confidence": self.estimate_confidence(response.strip(), base_confidence=0.75),
            "metadata": {
                "volume_shape": list(volume.shape),
                "image_size": list(image_size),
                "proj_out_num": self.proj_out_num,
                "max_new_tokens": max_new_tokens,
            },
        }

    def _get_volume(self, images: Optional[list], kwargs: dict) -> np.ndarray:
        """Extract 3D volume from various input formats.

        Path resolution order:
            1. volume_array (pre-loaded numpy array)
            2. volume_path / nii_path / source_path (file paths)
            3. images list (pre-loaded 3D numpy arrays)
        """
        # Direct numpy array
        volume_array = kwargs.get("volume_array")
        if volume_array is not None:
            return np.asarray(volume_array, dtype=np.float32)

        # From path using SimpleITK (official method)
        # Check multiple keys: volume_path (legacy), nii_path, source_path
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

        # From preprocessed data dict
        if images and isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], np.ndarray) and images[0].ndim == 3:
                return images[0].astype(np.float32)

        raise ValueError(
            "Med3DVLM requires 3D volume input. Provide volume_array, "
            "volume_path, nii_path, or source_path."
        )

    def extract_slices_for_display(
        self, volume: np.ndarray
    ) -> dict[str, list[Image.Image]]:
        """Extract axial/coronal/sagittal slices for visualization (from official app.py)."""
        def normalize_slice(s: np.ndarray) -> Image.Image:
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                norm = ((s - s_min) / (s_max - s_min) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(s, dtype=np.uint8)
            return Image.fromarray(norm).convert("L")

        return {
            "axial": [normalize_slice(volume[i, :, :]) for i in range(volume.shape[0])],
            "coronal": [normalize_slice(volume[:, i, :]) for i in range(volume.shape[1])],
            "sagittal": [normalize_slice(volume[:, :, i]) for i in range(volume.shape[2])],
        }
