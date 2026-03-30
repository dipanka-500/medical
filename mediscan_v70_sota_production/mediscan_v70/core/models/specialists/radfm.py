"""
MediScan AI v7.0 — RadFM Radiology Foundation Model Wrapper
Based on official implementation from:
  https://github.com/chaoyi-wu/RadFM
  https://huggingface.co/chaoyi-wu/RadFM

Architecture: MedKLIP vision encoder + LLaMA-2-7B language decoder
             with a cross-modal perceiver bridge.

Speciality:  Multi-modal radiology understanding across 2D and 3D:
             - X-ray, CT, MRI interpretation
             - Radiology report generation
             - Visual question answering
             - Rationale-grounded diagnosis

Pretraining: 16M+ multi-modal radiology data samples covering
             X-rays, CTs, MRIs across multiple anatomical regions.

Key advantage: RadFM is one of the few models trained on BOTH
  2D (X-ray) and 3D (CT/MRI) data with a unified architecture,
  making it a strong generalist radiology VLM.

Official API:
  model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
  processor = AutoProcessor.from_pretrained(...)
  Supports both 2D PIL images and 3D numpy volume inputs.
"""
from __future__ import annotations


import logging
import time
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from ...wrappers.base_model import BaseModel

logger = logging.getLogger(__name__)


class RadFMModel(BaseModel):
    """
    RadFM: A Radiology Foundation Model for multi-modal understanding.

    Handles both 2D radiographs and 3D volumetric scans through a
    unified perceiver bridge architecture.

    Official API: AutoModelForCausalLM + AutoProcessor
    """

    MAX_NEW_TOKENS = 2048

    def load(self) -> None:
        """Load RadFM model."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        start = time.time()
        logger.info(f"Loading RadFM: {self.model_id}")

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
        logger.info(f"RadFM loaded in {self._load_time:.1f}s on {self.device}")

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.9,
        **kwargs,
    ) -> dict[str, Any]:
        """Run RadFM inference on 2D or 3D radiology images.

        Supports:
          - 2D: PIL images (X-rays, single CT/MRI slices)
          - 3D: NIfTI volumes via volume_array / volume_path kwargs
        """
        if not self.is_loaded:
            self.load()

        if not text or text.strip() == "":
            text = (
                "Analyze this radiology image comprehensively. "
                "Provide a detailed report with findings, impression, "
                "and recommendations."
            )

        # Determine if 3D or 2D input
        volume_array = kwargs.get("volume_array")
        volume_path = kwargs.get("volume_path") or kwargs.get("nii_path") or kwargs.get("source_path")

        if volume_array is not None or (volume_path and modality == "3d"):
            return self._analyze_3d(text, volume_array, volume_path, max_new_tokens, temperature, top_p, **kwargs)
        else:
            return self._analyze_2d(images, text, max_new_tokens, temperature, top_p, **kwargs)

    def _analyze_2d(
        self, images, text, max_new_tokens, temperature, top_p, **kwargs
    ) -> dict[str, Any]:
        """Analyze 2D radiology images (X-rays, single slices)."""
        if not images or len(images) == 0:
            logger.warning("RadFM _analyze_2d called without images — text-only fallback")

        content = []
        if images:
            for img in images:
                if isinstance(img, Image.Image):
                    content.append({"type": "image", "image": img.convert("RGB")})
                elif isinstance(img, str):
                    content.append({"type": "image", "image": Image.open(img).convert("RGB")})
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]

        try:
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.debug(f"apply_chat_template unavailable ({e}), using raw prompt")
            text_prompt = f"USER: {text}\nASSISTANT:"

        image_list = [
            img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            for img in (images or [])
        ]

        processor_kwargs = {"text": text_prompt, "return_tensors": "pt"}
        if image_list:
            processor_kwargs["images"] = image_list[0] if len(image_list) == 1 else image_list

        inputs = self.processor(**processor_kwargs).to(self.device, dtype=self.get_dtype())
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        response = self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        )

        return {
            "model": self.model_id,
            "response": response.strip(),
            "answer": response.strip(),
            "thinking": "",
            "modality": "image",
            "confidence": self.estimate_confidence(response.strip(), base_confidence=0.75),
            "metadata": {
                "input_type": "2d",
                "model_type": "radfm_radiology",
            },
        }

    def _analyze_3d(
        self, text, volume_array, volume_path, max_new_tokens, temperature, top_p, **kwargs
    ) -> dict[str, Any]:
        """Analyze 3D volumetric data (CT/MRI).

        RadFM handles 3D by processing representative slices through
        its perceiver bridge — extract axial/coronal/sagittal views.
        """
        # Load volume if needed
        if volume_array is None and volume_path:
            import SimpleITK as sitk
            img = sitk.ReadImage(str(volume_path))
            volume_array = sitk.GetArrayFromImage(img).astype(np.float32)

        if volume_array is None:
            raise ValueError("RadFM 3D requires volume_array or volume_path")

        # Extract representative slices for multi-view analysis
        slices = self._extract_representative_slices(volume_array)

        # Process as multi-image input
        processor_kwargs = {
            "text": f"USER: <image>\n{text}\nASSISTANT:",
            "images": slices,
            "return_tensors": "pt",
        }
        inputs = self.processor(**processor_kwargs).to(self.device, dtype=self.get_dtype())
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        response = self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        )
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        return {
            "model": self.model_id,
            "response": response.strip(),
            "answer": response.strip(),
            "thinking": "",
            "modality": "3d",
            "confidence": self.estimate_confidence(response.strip(), base_confidence=0.72),
            "metadata": {
                "input_type": "3d",
                "volume_shape": list(volume_array.shape),
                "num_slices_used": len(slices),
                "model_type": "radfm_radiology",
            },
        }

    def _extract_representative_slices(
        self, volume: np.ndarray, num_slices: int = 5
    ) -> list[Image.Image]:
        """Extract representative axial slices from a 3D volume."""
        d = volume.shape[0]
        indices = np.linspace(d * 0.2, d * 0.8, num_slices, dtype=int)

        slices = []
        for idx in indices:
            s = volume[idx]
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                norm = ((s - s_min) / (s_max - s_min) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(s, dtype=np.uint8)
            pil_img = Image.fromarray(norm).convert("RGB")
            slices.append(pil_img)

        return slices
