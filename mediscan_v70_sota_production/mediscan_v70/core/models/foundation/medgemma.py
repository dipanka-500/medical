"""
MediScan AI v5.0 — MedGemma Model Wrapper
Based STRICTLY on official API from:
  https://huggingface.co/google/medgemma-1.5-4b-it
  https://huggingface.co/google/medgemma-27b-it
  https://github.com/Google-Health/medgemma

Architecture: Gemma 3 decoder-only transformer with SigLIP vision encoder
Models: medgemma-1.5-4b-it, medgemma-27b-it
API: AutoModelForImageTextToText + AutoProcessor

MedGemma 1.5 supports:
  - 2D medical images (X-ray, dermatology, fundus, pathology)
  - 3D volumetric imaging (CT, MRI) via multi-slice input
  - Whole-slide histopathology (WSI) via patch-based input
  - Longitudinal comparison (current vs prior images)
  - Anatomical localization with bounding boxes
  - Document understanding (lab reports → structured JSON)
  - EHR/FHIR data interpretation
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


class MedGemmaModel(BaseModel):
    """
    MedGemma: Google's medical foundation model built on Gemma 3.

    Official API uses AutoModelForImageTextToText (NOT AutoModelForCausalLM).
    Uses processor.apply_chat_template() for input preparation.

    Supports system prompts from official docs:
      {"role": "system", "content": [{"type": "text", "text": "..."}]}
    """

    # Default system prompts from official documentation examples
    SYSTEM_PROMPTS = {
        "radiologist": "You are an expert radiologist.",
        "medical_assistant": "You are a helpful medical assistant.",
        "pathologist": "You are an expert pathologist analyzing histopathology images.",
        "dermatologist": "You are an expert dermatologist.",
        "ophthalmologist": "You are an expert ophthalmologist analyzing fundus images.",
    }

    def load(self) -> None:
        """Load MedGemma using official HuggingFace API.

        From official docs:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto")
            processor = AutoProcessor.from_pretrained(model_id)
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor

        start = time.time()
        logger.info(f"Loading MedGemma: {self.model_id}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.get_dtype(),
            device_map=self.config.get("device_map", "auto"),
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.device = next(self.model.parameters()).device
        self.is_loaded = True
        self._load_time = time.time() - start
        logger.info(f"MedGemma loaded in {self._load_time:.1f}s on {self.device}")

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        max_new_tokens: int = 2000,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run MedGemma inference following official API exactly.

        From official docs (Run the model directly):
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)

        Args:
            images: List of PIL Image objects
            text: Medical question/prompt
            modality: "image" or "text"
            max_new_tokens: Max output tokens (default 2000)
            system_prompt: Optional system prompt. If None, uses role-based default.
                          Pass "" to disable system prompt.
        """
        if not self.is_loaded:
            self.load()

        # Build messages in official format (with system prompt from docs)
        messages = self._build_messages(images, text, modality, system_prompt)

        # Official API: processor.apply_chat_template()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.get_dtype())

        input_len = inputs["input_ids"].shape[-1]

        # Generate — official API uses do_sample=False for deterministic output
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get("do_sample", False),
            )

        # Decode only the generated tokens (skip input) — official method
        generated_tokens = generation[0][input_len:]
        response = self.processor.decode(generated_tokens, skip_special_tokens=True)

        # MedGemma 1.5 may include thinking traces between special tokens
        # Strip them if present (from official localization notebook)
        clean_response = self._strip_thinking_trace(response)

        return {
            "model": self.model_id,
            "response": clean_response.strip(),
            "answer": clean_response.strip(),
            "thinking": "",
            "modality": modality,
            "confidence": self.estimate_confidence(clean_response.strip(), base_confidence=0.78),
            "metadata": {
                "max_new_tokens": max_new_tokens,
                "input_length": input_len,
                "output_length": len(generated_tokens),
            },
        }

    def analyze_with_pipeline(
        self,
        images: Optional[list] = None,
        text: str = "",
        max_new_tokens: int = 2000,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Alternative: Use HuggingFace pipeline API (simpler but less control).

        From official MedGemma 1.5 docs:
            pipe = pipeline("image-text-to-text", model=model_id,
                            model_kwargs=dict(dtype=torch.bfloat16, device_map="auto"))
            output = pipe(text=messages, max_new_tokens=2000)
        """
        from transformers import pipeline

        # Official MedGemma 1.5 pipeline kwargs format
        pipe = pipeline(
            "image-text-to-text",
            model=self.model_id,
            model_kwargs=dict(
                dtype=self.get_dtype(),
                device_map="auto",
            ),
        )

        content = []
        if images:
            for img in images:
                if isinstance(img, Image.Image):
                    content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})

        messages = []

        # Add system prompt if provided (from official docs)
        if system_prompt is None:
            system_prompt = self.SYSTEM_PROMPTS["medical_assistant"]
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        messages.append({"role": "user", "content": content})

        output = pipe(text=messages, max_new_tokens=max_new_tokens)
        response = output[0]["generated_text"][-1]["content"]

        return {
            "model": self.model_id,
            "response": response.strip(),
            "answer": response.strip(),
            "thinking": "",
            "modality": "image" if images else "text",
            "confidence": self.estimate_confidence(response.strip(), base_confidence=0.78),
            "metadata": {"method": "pipeline"},
        }

    def _build_messages(
        self,
        images: Optional[list],
        text: str,
        modality: str,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Build messages in MedGemma's official format.

        From official docs, messages include:
          1. System message (optional but recommended):
             {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]}
          2. User message:
             {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "..."}]}
        """
        messages = []

        # ── System prompt (from official examples) ──
        if system_prompt is None:
            # Auto-select based on modality context
            if modality == "text":
                system_prompt = self.SYSTEM_PROMPTS["medical_assistant"]
            else:
                system_prompt = self.SYSTEM_PROMPTS["radiologist"]

        if system_prompt:  # Allow empty string to skip system prompt
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        # ── User content ──
        content = []

        if images and modality != "text":
            for img in images:
                if isinstance(img, Image.Image):
                    content.append({"type": "image", "image": img})
                elif isinstance(img, str):
                    content.append({
                        "type": "image",
                        "image": Image.open(img).convert("RGB"),
                    })

        content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content})

        return messages

    def pad_image_to_square(self, image: Image.Image) -> Image.Image:
        """Pad image to square format for bounding box / localization tasks.

        From official MedGemma 1.5 localization notebook:
            Maintains consistency with original preprocessing in model
            training and evaluation.
        """
        img_array = np.array(image)

        # Handle grayscale
        if len(img_array.shape) < 3:
            img_array = np.stack([img_array] * 3, axis=-1)

        # Handle RGBA
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        h, w = img_array.shape[:2]

        if h < w:
            dh = w - h
            img_array = np.pad(
                img_array, ((dh // 2, dh - dh // 2), (0, 0), (0, 0))
            )
        elif w < h:
            dw = h - w
            img_array = np.pad(
                img_array, ((0, 0), (dw // 2, dw - dw // 2), (0, 0))
            )

        return Image.fromarray(img_array.astype(np.uint8))

    def _strip_thinking_trace(self, response: str) -> str:
        """Strip thinking trace from MedGemma 1.5 output.

        From official localization notebook:
            if '<end_of_turn>' in response:
                response = response.split('<end_of_turn>', 1)[1].lstrip()
        """
        import re
        # Remove thinking blocks delimited by special tokens
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return cleaned if cleaned else response
