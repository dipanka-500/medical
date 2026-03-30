"""
MediScan AI v7.0 — CheXagent Model Wrapper
Based on official implementation from:
  https://github.com/Stanford-AIMI/CheXagent
  https://huggingface.co/StanfordAIMI/CheXagent-2-3b
  https://huggingface.co/StanfordAIMI/CheXagent-2-8b

Architecture: Qwen2.5-VL backbone fine-tuned for chest X-ray tasks
Capabilities:
  - Abnormality detection & grounding (bounding boxes)
  - View classification (AP/PA/lateral)
  - Radiology report generation (findings + impressions)
  - Visual question answering
  - Report quality assessment and error detection

CheXagent-2 achieves SOTA on CheXpert, MIMIC-CXR, and PadChest benchmarks.
Sizes: 3B and 8B variants.

Official API (from model card):
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
  processor = AutoProcessor.from_pretrained(...)
  Uses Qwen2.5-VL chat template with image tokens.
"""
from __future__ import annotations


import logging
import time
from typing import Any, Optional

import torch
from PIL import Image

from ...wrappers.base_model import BaseModel

logger = logging.getLogger(__name__)


class CheXagentModel(BaseModel):
    """
    CheXagent-2: Chest X-ray Foundation Agent from Stanford AIMI.

    Fine-tuned Qwen2.5-VL with expert-level CXR understanding.
    Supports abnormality detection, grounding, report generation,
    view classification, and visual QA.

    Official API: Qwen2_5_VLForConditionalGeneration + AutoProcessor
    """

    # Default task prompts aligned with CheXagent training objectives
    TASK_PROMPTS = {
        "report": (
            "Generate a detailed radiology report for this chest X-ray. "
            "Include FINDINGS and IMPRESSION sections."
        ),
        "abnormality": (
            "Identify all abnormalities visible in this chest X-ray. "
            "For each abnormality, describe its location, size, and severity."
        ),
        "grounding": (
            "Detect and localize all abnormalities in this chest X-ray. "
            "Provide bounding box coordinates [x1, y1, x2, y2] for each finding."
        ),
        "view_classification": (
            "What is the view/projection of this chest X-ray? "
            "(AP, PA, lateral, or other)"
        ),
        "quality": (
            "Assess the technical quality of this chest X-ray. "
            "Comment on positioning, exposure, rotation, and any artifacts."
        ),
    }

    def load(self) -> None:
        """Load CheXagent using Qwen2.5-VL API."""
        start = time.time()
        logger.info(f"Loading CheXagent: {self.model_id}")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.get_dtype(),
                device_map=self.config.get("device_map", "auto"),
                trust_remote_code=True,
            )
        except ImportError:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=self.get_dtype(),
                device_map=self.config.get("device_map", "auto"),
                trust_remote_code=True,
            )

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
        )

        self.device = next(self.model.parameters()).device
        self.is_loaded = True
        self._load_time = time.time() - start
        logger.info(f"CheXagent loaded in {self._load_time:.1f}s on {self.device}")

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
        task: str = "report",
        **kwargs,
    ) -> dict[str, Any]:
        """Run CheXagent inference on chest X-ray images.

        Args:
            images: List of PIL Images (chest X-rays)
            text: Custom prompt or question
            task: One of 'report', 'abnormality', 'grounding', 'view_classification', 'quality'
        """
        if not self.is_loaded:
            self.load()

        if not images or len(images) == 0:
            raise ValueError(
                "CheXagent is a chest X-ray specialist and requires at least one image. "
                "Provide a PIL Image or file path in the images list."
            )

        if not text or text.strip() == "":
            text = self.TASK_PROMPTS.get(task, self.TASK_PROMPTS["report"])

        messages = self._build_messages(images, text)

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = self._process_images(images)
        processor_kwargs = {
            "text": [text_prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs

        inputs = self.processor(**processor_kwargs).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
            )

        response = self.processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        )

        return {
            "model": self.model_id,
            "response": response.strip(),
            "answer": response.strip(),
            "thinking": "",
            "modality": "xray",
            "confidence": self.estimate_confidence(response.strip(), base_confidence=0.82),
            "metadata": {
                "task": task,
                "max_new_tokens": max_new_tokens,
                "model_type": "chexagent_cxr",
            },
        }

    def _build_messages(self, images: Optional[list], text: str) -> list[dict]:
        """Build Qwen2.5-VL chat messages."""
        content = []
        if images:
            for img in images:
                if isinstance(img, Image.Image):
                    content.append({"type": "image", "image": img})
                elif isinstance(img, str):
                    content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _process_images(self, images: Optional[list]) -> Optional[list]:
        """Process PIL images for model input."""
        if not images:
            return None
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                processed.append(img.convert("RGB"))
            elif isinstance(img, str):
                processed.append(Image.open(img).convert("RGB"))
        return processed if processed else None
