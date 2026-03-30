"""
MediScan AI v5.0 — MediX-R1 Model Wrapper
Based STRICTLY on official API from:
  https://github.com/mbzuai-oryx/MediX-R1
  https://huggingface.co/MBZUAI/MediX-R1-8B
  https://huggingface.co/MBZUAI/MediX-R1-30B

Architecture: ALL variants are Qwen3-VL based (NOT Gemma or Qwen2.5):
  - MediX-R1-2B  → Qwen3-VL-2B           (qwen3_vl)
  - MediX-R1-8B  → Qwen3-VL-8B-Instruct  (qwen3_vl)
  - MediX-R1-30B → Qwen3-VL-30B-A3B-Instruct (qwen3_vl_moe, MoE architecture)

Key Feature: RL-trained chain-of-thought reasoning with <think>/<thinking> tags
Uses composite reward design (LLM accuracy + embedding similarity + format + modality).

Official inference code uses:
  - AutoProcessor.from_pretrained()
  - processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  - process_vision_info(messages) from qwen_vl_utils
  - _strip_think_tokens: splits on </think> or </thinking> and takes the LAST part
"""
from __future__ import annotations


import logging
import re
import time
from typing import Any, Optional

import torch
from PIL import Image

from ...wrappers.base_model import BaseModel

logger = logging.getLogger(__name__)


class MediXR1Model(BaseModel):
    """
    MediX-R1: Open-Ended Medical Reinforcement Learning.

    ALL model sizes (2B, 8B, 30B) are based on Qwen3-VL architecture.
    The 30B variant uses Qwen3-VL-30B-A3B (MoE architecture).

    Official API: AutoModelForImageTextToText + AutoProcessor
    Uses qwen_vl_utils.process_vision_info() for image processing.
    Produces chain-of-thought reasoning in <think>...</think> blocks.
    """

    def load(self) -> None:
        """Load MediX-R1 using official Qwen3-VL API.

        From official eval code (models/Qwen3_VL.py):
            processor = AutoProcessor.from_pretrained(model_path)

        All variants (2B, 8B, 30B) use the same Qwen3-VL architecture.
        HuggingFace tags: Image-Text-to-Text for all variants.
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor

        start = time.time()
        logger.info(f"Loading MediX-R1: {self.model_id}")

        # All MediX-R1 variants are Qwen3-VL based (Image-Text-to-Text)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.get_dtype(),
            device_map=self.config.get("device_map", "auto"),
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self.device = next(self.model.parameters()).device
        self.is_loaded = True
        self._load_time = time.time() - start
        logger.info(f"MediX-R1 loaded in {self._load_time:.1f}s on {self.device}")

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        enable_reasoning: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Run MediX-R1 inference with chain-of-thought reasoning.

        From official eval code (models/Qwen3_VL.py):
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

        The model naturally produces <think>reasoning</think> before answers
        due to RL training with format rewards.
        """
        if not self.is_loaded:
            self.load()

        # Build prompt — optionally encourage reasoning
        prompt = text
        if enable_reasoning and "step by step" not in text.lower():
            prompt = (
                "Please reason step by step about this medical case, "
                "then provide your final assessment.\n\n" + text
            )

        # Build messages in Qwen3-VL format
        messages = self._build_messages(images, prompt, modality)

        # Process inputs following official Qwen3-VL API
        # Step 1: Apply chat template to get text prompt
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Step 2: Process vision info using qwen_vl_utils (official method)
        image_inputs = self._process_vision_info(images, modality)

        # Step 3: Tokenize with processor
        processor_kwargs = {
            "text": [text_prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs

        inputs = self.processor(**processor_kwargs).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Generate
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )

        # Decode only generated tokens (skip input)
        generated_tokens = generation[0][input_len:]
        full_response = self.processor.decode(
            generated_tokens, skip_special_tokens=True
        )

        # Parse thinking and answer from response
        thinking, answer = self._parse_thinking(full_response)

        return {
            "model": self.model_id,
            "response": full_response.strip(),
            "answer": answer,
            "thinking": thinking,
            "modality": modality,
            "confidence": self.estimate_confidence(answer, base_confidence=0.80),
            "metadata": {
                "enable_reasoning": enable_reasoning,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "has_thinking": bool(thinking),
            },
        }

    def _build_messages(
        self, images: Optional[list], text: str, modality: str
    ) -> list[dict]:
        """Build messages in Qwen3-VL format.

        From official eval code:
            messages.append({"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]})

        Multi-image support (from official code):
            for i, image in enumerate(images):
                content.append({"type": "text", "text": f"<image_{i+1}>: "})
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
        """
        content = []

        if images and modality != "text":
            if len(images) == 1:
                # Single image
                img = images[0]
                if isinstance(img, Image.Image):
                    content.append({"type": "image", "image": img})
                elif isinstance(img, str):
                    content.append({"type": "image", "image": img})
            else:
                # Multi-image: official format with <image_N> labels
                for i, img in enumerate(images):
                    content.append({"type": "text", "text": f"<image_{i+1}>: "})
                    if isinstance(img, Image.Image):
                        content.append({"type": "image", "image": img})
                    elif isinstance(img, str):
                        content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _process_vision_info(
        self, images: Optional[list], modality: str
    ) -> Optional[list]:
        """Process images for Qwen3-VL input.

        From official eval code:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)

        Falls back to direct PIL image list if qwen_vl_utils is not available.
        """
        if not images or modality == "text":
            return None

        # Try to use official qwen_vl_utils
        try:
            from qwen_vl_utils import process_vision_info

            messages = self._build_messages(images, "", modality)
            image_inputs, _ = process_vision_info(messages)
            return image_inputs if image_inputs else None
        except ImportError:
            logger.debug("qwen_vl_utils not available, using direct PIL images")

        # Fallback: ensure all images are PIL Image objects
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                processed.append(img.convert("RGB"))
            elif isinstance(img, str):
                processed.append(Image.open(img).convert("RGB"))
        return processed if processed else None

    def _parse_thinking(self, response: str) -> tuple[str, str]:
        """Extract thinking content from response.

        From official eval code (_strip_think_tokens):
            if '</think>' in text:
                text = text.split('</think>')[-1].strip()
            elif '</thinking>' in text:
                text = text.split('</thinking>')[-1].strip()

        Supports both <think>...</think> and <thinking>...</thinking> tags.
        """
        # Method 1: <think>...</think> (primary format)
        think_match = re.search(
            r"<think>(.*?)</think>", response, re.DOTALL
        )
        if think_match:
            thinking = think_match.group(1).strip()
            answer = response.split("</think>")[-1].strip()
            return thinking, answer

        # Method 2: <thinking>...</thinking> (alternate format)
        thinking_match = re.search(
            r"<thinking>(.*?)</thinking>", response, re.DOTALL
        )
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            answer = response.split("</thinking>")[-1].strip()
            return thinking, answer

        # No thinking tags — entire response is the answer
        return "", response.strip()
