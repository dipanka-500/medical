"""
MediScan AI v7.0 — PathGen Pathology Wrapper

Supports two runtime modes:
  1. Legacy generative PathGen/LLaVA-style checkpoints
  2. Public PathGen-CLIP HF checkpoints (classification/similarity mode)

The public Hugging Face assets currently available for PathGen are CLIP-style
checkpoints rather than the older LLaVA-style repo that this project
originally referenced. To keep the pathology route operational on real public
weights, this wrapper can transparently fall back to CLIP-style inference.
"""
from __future__ import annotations


import logging
import time
from typing import Any, Optional

import torch
from PIL import Image

from ...wrappers.base_model import BaseModel

logger = logging.getLogger(__name__)


class PathGenModel(BaseModel):
    """
    PathGen-1.6B: Pathology Vision-Language Model.

    Uses CONCH vision encoder + Phi-2 LLM for computational pathology.
    Excels at tissue classification, grading, IHC scoring, and
    morphological description of histopathology images.

    Official API: LlavaForConditionalGeneration + AutoProcessor
    """

    PATHOLOGY_PROMPTS = {
        "general": (
            "Describe the histopathological features visible in this image. "
            "Include tissue type, cellular composition, architectural patterns, "
            "and any abnormalities."
        ),
        "grading": (
            "Assess the grade of this tissue sample. Describe the "
            "differentiation level, mitotic activity, and nuclear features."
        ),
        "ihc": (
            "Analyze this immunohistochemistry (IHC) stained image. "
            "Describe the staining pattern, intensity, and distribution. "
            "Provide an estimated percentage of positive cells."
        ),
        "classification": (
            "Classify this tissue sample. Identify the tissue type, "
            "determine if it is benign, pre-malignant, or malignant, "
            "and provide the specific histological diagnosis."
        ),
    }

    CLIP_TASK_LABELS = {
        "general": [
            "benign histopathology tissue",
            "adenocarcinoma histopathology",
            "squamous cell carcinoma histopathology",
            "high cellularity malignant histopathology",
            "necrotic tumor histopathology",
            "inflammatory histopathology",
            "fibrotic stroma histopathology",
            "lymphoid tissue histopathology",
            "normal glandular epithelium histopathology",
            "atypical dysplastic histopathology",
            "hematoxylin and eosin stained tumor patch",
            "immunohistochemistry stained pathology patch",
        ],
        "classification": [
            "benign tissue histopathology",
            "adenocarcinoma histopathology",
            "squamous cell carcinoma histopathology",
            "poorly differentiated carcinoma histopathology",
            "lymphoma histopathology",
            "metastatic tumor histopathology",
            "normal tissue histopathology",
            "pre-malignant dysplasia histopathology",
        ],
        "grading": [
            "low grade dysplasia histopathology",
            "intermediate grade dysplasia histopathology",
            "high grade dysplasia histopathology",
            "well differentiated carcinoma histopathology",
            "moderately differentiated carcinoma histopathology",
            "poorly differentiated carcinoma histopathology",
        ],
        "ihc": [
            "negative immunohistochemistry staining",
            "focal weak immunohistochemistry staining",
            "heterogeneous moderate immunohistochemistry staining",
            "diffuse strong immunohistochemistry staining",
            "patchy positive immunohistochemistry staining",
        ],
    }

    def _use_clip_runtime(self) -> bool:
        runtime_mode = str(self.config.get("runtime_mode", "")).strip().lower()
        model_id_lower = self.model_id.lower()
        return runtime_mode == "clip" or "pathgenclip" in model_id_lower

    @staticmethod
    def _coerce_image(image_like: Any) -> Image.Image:
        if isinstance(image_like, Image.Image):
            return image_like.convert("RGB")
        if isinstance(image_like, str):
            return Image.open(image_like).convert("RGB")
        raise ValueError("PathGen requires a PIL image or image path.")

    def load(self) -> None:
        """Load PathGen using CLIP or legacy LLaVA-compatible API."""
        start = time.time()
        logger.info(f"Loading PathGen: {self.model_id}")

        if self._use_clip_runtime():
            from open_clip import create_model_from_pretrained, get_tokenizer

            clip_id = self.model_id
            if not clip_id.startswith("hf-hub:"):
                clip_id = f"hf-hub:{clip_id}"

            self.model, self.preprocess = create_model_from_pretrained(clip_id)
            self.tokenizer = get_tokenizer(clip_id)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.processor = None
            self._runtime_mode = "clip"
            self.is_loaded = True
            self._load_time = time.time() - start
            logger.info(f"PathGen CLIP runtime loaded in {self._load_time:.1f}s on {self.device}")
            return

        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.get_dtype(),
                device_map=self.config.get("device_map", "auto"),
                trust_remote_code=True,
            )
        except (ImportError, ValueError):
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.model = AutoModelForCausalLM.from_pretrained(
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
        self._runtime_mode = "generative"
        self.is_loaded = True
        self._load_time = time.time() - start
        logger.info(f"PathGen loaded in {self._load_time:.1f}s on {self.device}")

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        task: str = "general",
        **kwargs,
    ) -> dict[str, Any]:
        """Run PathGen inference on pathology images.

        Args:
            images: List of PIL Images (H&E / IHC stained patches)
            text: Custom prompt or question
            task: 'general', 'grading', 'ihc', or 'classification'
        """
        if not self.is_loaded:
            self.load()

        if getattr(self, "_runtime_mode", "") == "clip":
            return self._analyze_with_clip(
                images=images,
                text=text,
                modality=modality,
                task=task,
            )

        if not text or text.strip() == "":
            text = self.PATHOLOGY_PROMPTS.get(task, self.PATHOLOGY_PROMPTS["general"])

        # PathGen uses LLaVA-style conversation format
        prompt = f"USER: <image>\n{text}\nASSISTANT:"

        if images:
            image = self._coerce_image(images[0])
        else:
            raise ValueError("PathGen requires at least one pathology image.")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, dtype=self.get_dtype())

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
            )

        response = self.processor.decode(
            generation[0], skip_special_tokens=True
        )
        # Strip the prompt portion from response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        return {
            "model": self.model_id,
            "response": response.strip(),
            "answer": response.strip(),
            "thinking": "",
            "modality": "pathology",
            "confidence": self.estimate_confidence(response.strip(), base_confidence=0.78),
            "metadata": {
                "task": task,
                "max_new_tokens": max_new_tokens,
                "model_type": "pathgen_pathology",
                "runtime_mode": "generative",
            },
        }

    def _analyze_with_clip(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        task: str = "general",
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("PathGen requires at least one pathology image.")

        image = self._coerce_image(images[0])
        labels = list(self.CLIP_TASK_LABELS.get(task, self.CLIP_TASK_LABELS["general"]))
        if text and text.strip():
            labels.insert(0, text.strip())

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = self.tokenizer(labels).to(self.device)

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(image_tensor, text_inputs)
            logits = (logit_scale * image_features @ text_features.t()).squeeze(0)
            probs = logits.softmax(dim=-1)

        ranked = sorted(
            (
                {"label": labels[idx], "confidence": float(probs[idx].item())}
                for idx in range(len(labels))
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )
        top = ranked[0]
        differentials = ", ".join(item["label"] for item in ranked[1:4]) or "none"
        answer = (
            f"PathGen analysis: Most similar pathology pattern is '{top['label']}' "
            f"(confidence: {top['confidence']:.2%}). Top differentials: {differentials}."
        )

        return {
            "model": self.model_id,
            "response": answer,
            "answer": answer,
            "thinking": "",
            "modality": modality,
            "confidence": top["confidence"],
            "classifications": ranked[:10],
            "metadata": {
                "task": task,
                "num_labels": len(labels),
                "model_type": "pathgen_pathology",
                "runtime_mode": "clip",
            },
        }
