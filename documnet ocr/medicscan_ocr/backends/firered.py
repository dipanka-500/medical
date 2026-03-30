from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from medicscan_ocr.backends.base import OCRBackend
from medicscan_ocr.schemas import AnalysisResult, BackendResult, Section
from medicscan_ocr.utils.files import is_image_path
from medicscan_ocr.utils.sorting import natural_sorted_paths
from medicscan_ocr.utils.text import normalize_text

logger = logging.getLogger(__name__)

MODEL_NAME = "FireRedTeam/FireRed-OCR"


class FireRedBackend(OCRBackend):
    name = "firered_backend"

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._model = None
        self._processor = None

    def _dependencies_available(self) -> bool:
        return bool(importlib.util.find_spec("transformers"))

    def availability(self) -> tuple[bool, str | None]:
        if torch is None:
            return False, "torch is not installed"
        if not self._dependencies_available():
            return False, "transformers is not installed"
        return True, None

    def _load_model(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            dtype = torch.bfloat16 if bf16_supported else torch.float16
        else:
            dtype = torch.float32

        logger.info("Loading FireRed model %s on %s (%s)", MODEL_NAME, device, dtype)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if device == "cpu":
            model = model.to(device)
        model.eval()

        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        self._model = model
        self._processor = processor
        logger.info("FireRed model loaded successfully")
        return self._model, self._processor

    def _collect_images(self, input_path: str):
        path = Path(input_path).resolve()
        if path.is_dir():
            return natural_sorted_paths(
                [candidate for candidate in path.iterdir() if candidate.is_file() and is_image_path(candidate)]
            )
        if is_image_path(path):
            return [path]
        return []

    def run(
        self,
        input_path: str,
        analysis: AnalysisResult,
        output_dir: Optional[str] = None,
    ) -> BackendResult:
        images = self._collect_images(input_path)
        if not images:
            return BackendResult(
                backend=self.name,
                status="failed",
                error="FireRed backend expects page images. Install PDF splitting support or provide images.",
            )

        if not self._dependencies_available():
            return BackendResult(
                backend=self.name,
                status="failed",
                error="FireRed requires transformers and model weights. Install the FireRed runtime to enable this backend.",
                metadata={"docs_url": "https://github.com/FireRedTeam/FireRed-OCR"},
            )

        try:
            model, processor = self._load_model()
        except Exception as exc:
            logger.exception("FireRed model loading failed")
            return BackendResult(
                backend=self.name,
                status="failed",
                error="FireRed model loading failed: {0}".format(exc),
            )

        sections = []
        outputs = []
        try:
            for index, image_path in enumerate(images, start=1):
                logger.info("FireRed processing page %d/%d: %s", index, len(images), image_path.name)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {
                                "type": "text",
                                "text": (
                                    "Extract this document page into faithful structured markdown. "
                                    "Preserve reading order, tables, formulas, and section hierarchy. "
                                    "Do not explain the page."
                                ),
                            },
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if hasattr(inputs, "to"):
                    inputs = inputs.to(model.device if hasattr(model, "device") else "cpu")

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=8192)

                trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                output_text = normalize_text(output_text)
                outputs.append(output_text)
                sections.append(
                    Section(
                        type="page",
                        text=output_text,
                        confidence=0.86,
                        data={"page_number": index, "source_name": image_path.name},
                    )
                )
        except Exception as exc:
            logger.exception("FireRed inference failed on page %d", index if 'index' in dir() else 0)
            return BackendResult(
                backend=self.name,
                status="failed",
                error="FireRed inference failed: {0}".format(exc),
            )

        raw_text = normalize_text("\n\n".join(outputs))
        return BackendResult(
            backend=self.name,
            status="completed",
            raw_text=raw_text,
            sections=sections,
            confidence=0.86,
            metadata={"page_count": len(images), "model": MODEL_NAME},
        )
