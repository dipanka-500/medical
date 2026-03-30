"""
MediScan AI v5.0 — BiomedCLIP Model Wrapper
Based STRICTLY on official API from:
  https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

Architecture: PubMedBERT text encoder + ViT-B/16 vision encoder (open_clip)
Use Case: Zero-shot medical image classification, image-text similarity,
          verification/anti-hallucination scoring
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


class BiomedCLIPModel(BaseModel):
    """
    BiomedCLIP: Contrastive learning model for biomedical image-text matching.

    Official API: open_clip.create_model_from_pretrained() + get_tokenizer()
    NOT a generative model — used for classification and similarity scoring.
    """

    CONTEXT_LENGTH = 256

    # Comprehensive medical image labels for zero-shot classification
    MEDICAL_LABELS = [
        # Chest
        "normal chest X-ray",
        "pneumonia on chest X-ray",
        "pneumothorax on chest X-ray",
        "pleural effusion on chest X-ray",
        "cardiomegaly on chest X-ray",
        "lung nodule on chest X-ray",
        "pulmonary edema on chest X-ray",
        "atelectasis on chest X-ray",
        "consolidation on chest X-ray",
        # Brain
        "normal brain MRI",
        "brain tumor on MRI",
        "intracranial hemorrhage on CT",
        "ischemic stroke on MRI",
        "brain metastasis on MRI",
        # Pathology
        "adenocarcinoma histopathology",
        "squamous cell carcinoma histopathology",
        "hematoxylin and eosin histopathology",
        "immunohistochemistry histopathology",
        "benign tissue histopathology",
        # Other modalities
        "bone fracture on X-ray",
        "bone X-ray normal",
        "abdominal CT scan",
        "liver lesion on CT",
        "kidney stone on CT",
        "retinal fundoscopy normal",
        "diabetic retinopathy fundoscopy",
        "dermoscopy melanoma",
        "dermoscopy benign nevus",
        "ultrasound normal",
    ]

    def load(self) -> None:
        """Load BiomedCLIP using official open_clip API."""
        from open_clip import create_model_from_pretrained, get_tokenizer

        start = time.time()
        logger.info("Loading BiomedCLIP")

        model_id = self.config.get(
            "model_id",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )

        self.model, self.preprocess = create_model_from_pretrained(model_id)
        self.tokenizer = get_tokenizer(model_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        self._load_time = time.time() - start
        logger.info(f"BiomedCLIP loaded in {self._load_time:.1f}s on {self.device}")

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        labels: Optional[list[str]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run BiomedCLIP zero-shot classification.

        Args:
            images: List of PIL Images
            text: Not used for classification (use labels instead)
            labels: Custom labels for classification. If None, uses MEDICAL_LABELS.

        Returns:
            Dict with classification results and confidence scores.
        """
        if not self.is_loaded:
            self.load()

        if not images:
            raise ValueError("BiomedCLIP requires image input")

        labels = labels or self.MEDICAL_LABELS
        template = "this is a photo of "

        # Process images — official API
        image_tensors = torch.stack(
            [self.preprocess(img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")) for img in images]
        ).to(self.device)

        # Process text labels — official API
        text_inputs = self.tokenizer(
            [template + label for label in labels],
            context_length=self.CONTEXT_LENGTH,
        ).to(self.device)

        # Compute similarities — official API
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(
                image_tensors, text_inputs
            )
            logits = (logit_scale * image_features @ text_features.t()).detach()
            probs = logits.softmax(dim=-1)

        # Build results for each image
        results = []
        probs_np = probs.cpu().numpy()
        for i in range(len(images)):
            sorted_indices = np.argsort(probs_np[i])[::-1]
            classifications = [
                {"label": labels[idx], "confidence": float(probs_np[i][idx])}
                for idx in sorted_indices[:10]  # Top 10
            ]
            results.append({
                "top_prediction": labels[sorted_indices[0]],
                "top_confidence": float(probs_np[i][sorted_indices[0]]),
                "classifications": classifications,
            })

        return {
            "model": "BiomedCLIP",
            "response": results[0]["top_prediction"] if len(results) == 1 else str(results),
            "answer": results[0]["top_prediction"] if len(results) == 1 else "",
            "thinking": "",
            "modality": modality,
            "classifications": results,
            "metadata": {
                "num_images": len(images),
                "num_labels": len(labels),
            },
        }

    def compute_similarity(
        self, image: Image.Image, text_queries: list[str]
    ) -> list[float]:
        """Compute image-text similarity scores for verification/anti-hallucination.

        Used by the fusion engine to verify model outputs against image content.
        """
        if not self.is_loaded:
            self.load()

        image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        text_inputs = self.tokenizer(
            text_queries, context_length=self.CONTEXT_LENGTH
        ).to(self.device)

        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(
                image_tensor, text_inputs
            )
            similarities = (logit_scale * image_features @ text_features.t()).squeeze(0)
            scores = similarities.softmax(dim=-1).cpu().numpy().tolist()

        return scores

    def verify_finding(
        self, image: Image.Image, finding: str, threshold: float = 0.3
    ) -> dict[str, Any]:
        """Verify if a specific finding is consistent with the image.

        Used by anti-hallucination engine to cross-check VLM outputs.
        """
        queries = [
            f"this is a photo of {finding}",
            f"this is a photo of normal anatomy without {finding}",
        ]
        scores = self.compute_similarity(image, queries)

        is_consistent = scores[0] > scores[1] and scores[0] > threshold

        return {
            "finding": finding,
            "finding_score": scores[0],
            "normal_score": scores[1],
            "is_consistent": is_consistent,
            "confidence": abs(scores[0] - scores[1]),
        }
