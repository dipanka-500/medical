"""
IBM Granite 4.0 3B Vision — document extraction backend.

Deployed as a sidecar (vLLM OpenAI-compatible endpoint) for:
  - Table extraction (HTML / JSON / OTSL)
  - Key-Value Pair extraction with strict JSON schemas
  - Chart-to-CSV / chart-to-summary
  - Contradiction review against primary OCR output

Only invoked for structure-heavy documents: lab reports, prescriptions,
discharge summaries, referral letters, claims, intake forms.

Every extracted field carries provenance: page number, source snippet,
confidence, and model name.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from medicscan_ocr.backends.base import OCRBackend
from medicscan_ocr.schemas import AnalysisResult, BackendResult, Section

logger = logging.getLogger(__name__)

MODEL_ID = "ibm-granite/granite-4.0-3b-vision"

# ── Document-type JSON schemas for strict KVP extraction ──────────────────

DOCUMENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "lab_report": {
        "type": "object",
        "properties": {
            "patient_name": {"type": "string", "description": "Full name of the patient"},
            "patient_id": {"type": "string", "description": "Patient ID or MRN"},
            "report_date": {"type": "string", "description": "Date when the report was issued (YYYY-MM-DD)"},
            "ordering_physician": {"type": "string", "description": "Name of the ordering physician"},
            "lab_name": {"type": "string", "description": "Name of the laboratory"},
            "specimen_type": {"type": "string", "description": "Type of specimen collected"},
            "collection_date": {"type": "string", "description": "Date specimen was collected (YYYY-MM-DD)"},
            "tests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "test_name": {"type": "string"},
                        "result_value": {"type": "string"},
                        "unit": {"type": "string"},
                        "reference_range": {"type": "string"},
                        "flag": {"type": "string", "description": "H=High, L=Low, N=Normal, C=Critical"},
                    },
                },
            },
            "summary": {"type": "string", "description": "Overall interpretation or summary"},
            "critical_values": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of critical or abnormal values requiring attention",
            },
        },
    },
    "prescription": {
        "type": "object",
        "properties": {
            "patient_name": {"type": "string"},
            "patient_id": {"type": "string"},
            "prescriber_name": {"type": "string"},
            "prescriber_license": {"type": "string"},
            "prescription_date": {"type": "string", "description": "YYYY-MM-DD"},
            "diagnosis": {"type": "string"},
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string"},
                        "dosage": {"type": "string"},
                        "frequency": {"type": "string"},
                        "route": {"type": "string"},
                        "duration": {"type": "string"},
                        "quantity": {"type": "string"},
                        "refills": {"type": "integer"},
                        "special_instructions": {"type": "string"},
                    },
                },
            },
            "allergies_noted": {"type": "array", "items": {"type": "string"}},
            "follow_up": {"type": "string"},
        },
    },
    "discharge_summary": {
        "type": "object",
        "properties": {
            "patient_name": {"type": "string"},
            "patient_id": {"type": "string"},
            "admission_date": {"type": "string", "description": "YYYY-MM-DD"},
            "discharge_date": {"type": "string", "description": "YYYY-MM-DD"},
            "attending_physician": {"type": "string"},
            "admitting_diagnosis": {"type": "string"},
            "discharge_diagnosis": {"type": "string"},
            "procedures_performed": {"type": "array", "items": {"type": "string"}},
            "hospital_course": {"type": "string"},
            "discharge_medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string"},
                        "dosage": {"type": "string"},
                        "frequency": {"type": "string"},
                        "duration": {"type": "string"},
                    },
                },
            },
            "discharge_instructions": {"type": "string"},
            "follow_up_appointments": {"type": "array", "items": {"type": "string"}},
            "condition_at_discharge": {"type": "string"},
        },
    },
    "referral_letter": {
        "type": "object",
        "properties": {
            "patient_name": {"type": "string"},
            "patient_id": {"type": "string"},
            "referring_physician": {"type": "string"},
            "referred_to": {"type": "string"},
            "referral_date": {"type": "string"},
            "reason_for_referral": {"type": "string"},
            "clinical_history": {"type": "string"},
            "current_medications": {"type": "array", "items": {"type": "string"}},
            "relevant_investigations": {"type": "array", "items": {"type": "string"}},
            "urgency": {"type": "string", "description": "routine, urgent, emergency"},
        },
    },
    "insurance_claim": {
        "type": "object",
        "properties": {
            "claim_number": {"type": "string"},
            "patient_name": {"type": "string"},
            "patient_id": {"type": "string"},
            "policy_number": {"type": "string"},
            "provider_name": {"type": "string"},
            "provider_npi": {"type": "string"},
            "date_of_service": {"type": "string"},
            "diagnosis_codes": {"type": "array", "items": {"type": "string"}},
            "procedure_codes": {"type": "array", "items": {"type": "string"}},
            "total_charge": {"type": "string"},
            "amount_paid": {"type": "string"},
            "patient_responsibility": {"type": "string"},
        },
    },
    "intake_form": {
        "type": "object",
        "properties": {
            "patient_name": {"type": "string"},
            "date_of_birth": {"type": "string"},
            "gender": {"type": "string"},
            "address": {"type": "string"},
            "phone": {"type": "string"},
            "email": {"type": "string"},
            "emergency_contact": {"type": "string"},
            "insurance_provider": {"type": "string"},
            "policy_number": {"type": "string"},
            "chief_complaint": {"type": "string"},
            "current_medications": {"type": "array", "items": {"type": "string"}},
            "allergies": {"type": "array", "items": {"type": "string"}},
            "medical_history": {"type": "array", "items": {"type": "string"}},
            "surgical_history": {"type": "array", "items": {"type": "string"}},
            "family_history": {"type": "array", "items": {"type": "string"}},
        },
    },
}

# Document types that should be routed to Granite for structured extraction
STRUCTURE_HEAVY_TYPES = frozenset({
    "lab_report", "prescription", "discharge_summary",
    "referral_letter", "insurance_claim", "intake_form",
})

# ── Granite extraction task tags ──────────────────────────────────────────

TASK_TAGS = {
    "table_html": "<tables_html>",
    "table_json": "<tables_json>",
    "table_otsl": "<tables_otsl>",
    "chart_csv": "<chart2csv>",
    "chart_summary": "<chart2summary>",
    "chart_code": "<chart2code>",
}


def _build_kvp_prompt(schema: Dict[str, Any]) -> str:
    """Build a strict KVP extraction prompt from a JSON schema."""
    return (
        "Extract structured data from this document.\n"
        "Return a JSON object matching this schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Return null for fields you cannot find.\n"
        "Return ONLY valid JSON.\n"
        "Return an instance of the JSON with extracted values, not the schema itself."
    )


def _attach_provenance(
    extracted: Dict[str, Any],
    page_number: int,
    source_snippet: str,
    confidence: float,
    model_name: str,
) -> Dict[str, Any]:
    """Attach provenance metadata to every extracted field."""
    return {
        "extracted_data": extracted,
        "provenance": {
            "page_number": page_number,
            "source_snippet": source_snippet[:500],
            "confidence": round(confidence, 4),
            "model_name": model_name,
            "extraction_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def _validate_against_schema(data: Any, schema: Dict[str, Any]) -> bool:
    """Lightweight schema conformance check (no jsonschema dependency)."""
    if not isinstance(data, dict):
        return False
    props = schema.get("properties", {})
    if not props:
        return True
    # At minimum, we should have at least 1 non-null field
    non_null = sum(1 for k in props if data.get(k) is not None)
    return non_null >= 1


class GraniteVisionBackend(OCRBackend):
    """IBM Granite 4.0 3B Vision backend for structured document extraction.

    Operates in two modes:
      1. LOCAL: Load model via transformers + PEFT (GPU required)
      2. VLLM:  Call an OpenAI-compatible vLLM sidecar (preferred in production)

    Production deployment uses vLLM sidecar on port 8005.
    """

    name = "granite_vision"

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._model = None
        self._processor = None
        # vLLM sidecar endpoint (preferred)
        self._vllm_url = os.getenv("GRANITE_VLLM_URL", "http://granite-vision:8005/v1")
        self._vllm_api_key = os.getenv("GRANITE_VLLM_API_KEY", "EMPTY")
        self._use_vllm = os.getenv("GRANITE_USE_VLLM", "true").lower() in {"1", "true", "yes"}
        self._timeout = float(os.getenv("GRANITE_TIMEOUT_SECONDS", "120"))
        self._max_tokens = int(os.getenv("GRANITE_MAX_TOKENS", "4096"))

    def availability(self) -> tuple[bool, str | None]:
        if self._use_vllm:
            # vLLM mode: always available if configured (health checked at runtime)
            return True, None
        # Local mode: need torch + transformers + peft
        try:
            import torch
            if not importlib.util.find_spec("transformers"):
                return False, "transformers not installed"
            if not importlib.util.find_spec("peft"):
                return False, "peft not installed"
            return True, None
        except ImportError:
            return False, "torch not installed"

    # ── vLLM inference (production path) ──────────────────────────────

    def _vllm_inference(
        self, image_path: str, prompt: str
    ) -> str:
        """Call the Granite vLLM sidecar with OpenAI-compatible API."""
        import base64
        import httpx

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Determine mime type
        ext = Path(image_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "tiff": "image/tiff", "tif": "image/tiff", "bmp": "image/bmp"
                }.get(ext.lstrip("."), "image/png")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = httpx.post(
            f"{self._vllm_url}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": messages,
                "max_tokens": self._max_tokens,
                "temperature": 0,
            },
            headers={"Authorization": f"Bearer {self._vllm_api_key}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ── Local inference (dev/testing path) ────────────────────────────

    def _load_local_model(self):
        if self._model is not None:
            return self._model, self._processor

        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        logger.info("Loading Granite Vision model %s on %s", MODEL_ID, device)
        self._processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=dtype, device_map=device,
        ).eval()
        # Merge LoRA for faster inference
        if hasattr(self._model, "merge_lora_adapters"):
            self._model.merge_lora_adapters()
        logger.info("Granite Vision model loaded successfully")
        return self._model, self._processor

    def _local_inference(self, image_path: str, prompt: str) -> str:
        import torch
        from PIL import Image

        model, processor = self._load_local_model()
        img = Image.open(image_path).convert("RGB")

        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[img], return_tensors="pt", padding=True, do_pad=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=self._max_tokens, use_cache=True)
        gen = outputs[0, inputs["input_ids"].shape[1]:]
        return processor.decode(gen, skip_special_tokens=True)

    # ── Unified inference dispatcher ──────────────────────────────────

    def _infer(self, image_path: str, prompt: str) -> str:
        if self._use_vllm:
            return self._vllm_inference(image_path, prompt)
        return self._local_inference(image_path, prompt)

    # ── Image collection ──────────────────────────────────────────────

    def _collect_images(self, input_path: str) -> List[Path]:
        from medicscan_ocr.utils.files import is_image_path
        from medicscan_ocr.utils.sorting import natural_sorted_paths

        path = Path(input_path).resolve()
        if path.is_dir():
            return natural_sorted_paths(
                [c for c in path.iterdir() if c.is_file() and is_image_path(c)]
            )
        if path.is_file() and is_image_path(path):
            return [path]
        return []

    # ── Main run method ───────────────────────────────────────────────

    def run(
        self,
        input_path: str,
        analysis: AnalysisResult,
        output_dir: Optional[str] = None,
        document_type_hint: Optional[str] = None,
    ) -> BackendResult:
        images = self._collect_images(input_path)
        if not images:
            return BackendResult(
                backend=self.name, status="failed",
                error="Granite Vision backend requires image inputs.",
            )

        # Determine document type for schema selection
        doc_type = document_type_hint or self._detect_document_type(analysis)
        schema = DOCUMENT_SCHEMAS.get(doc_type)

        all_sections: List[Section] = []
        all_kvp: List[Dict[str, Any]] = []
        all_tables: List[str] = []
        raw_parts: List[str] = []
        extraction_errors: List[str] = []

        for page_idx, image_path in enumerate(images, start=1):
            logger.info(
                "Granite Vision processing page %d/%d: %s (doc_type=%s)",
                page_idx, len(images), image_path.name, doc_type,
            )

            try:
                # 1. Table extraction (always)
                table_html = self._infer(str(image_path), TASK_TAGS["table_html"])
                if table_html.strip():
                    all_tables.append(table_html)
                    all_sections.append(Section(
                        type="table",
                        text=table_html,
                        confidence=0.88,
                        data={
                            "page_number": page_idx,
                            "format": "html",
                            "model": MODEL_ID,
                        },
                    ))

                # 2. KVP extraction (if schema available)
                if schema:
                    kvp_prompt = _build_kvp_prompt(schema)
                    kvp_raw = self._infer(str(image_path), kvp_prompt)
                    try:
                        kvp_data = json.loads(kvp_raw)
                        if _validate_against_schema(kvp_data, schema):
                            provenance = _attach_provenance(
                                extracted=kvp_data,
                                page_number=page_idx,
                                source_snippet=kvp_raw[:500],
                                confidence=0.855,
                                model_name=MODEL_ID,
                            )
                            all_kvp.append(provenance)
                            all_sections.append(Section(
                                type="key_value_pairs",
                                text=json.dumps(kvp_data, indent=2),
                                confidence=0.855,
                                data={
                                    "page_number": page_idx,
                                    "document_type": doc_type,
                                    "schema_used": doc_type,
                                    "provenance": provenance["provenance"],
                                },
                            ))
                        else:
                            extraction_errors.append(
                                f"Page {page_idx}: KVP output failed schema validation"
                            )
                            logger.warning(
                                "Granite KVP output failed schema validation on page %d", page_idx
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        extraction_errors.append(
                            f"Page {page_idx}: KVP JSON parse error: {e}"
                        )
                        logger.warning("Granite KVP JSON parse error on page %d: %s", page_idx, e)
                else:
                    # No schema — do general text extraction
                    general_text = self._infer(
                        str(image_path),
                        "Extract all text from this document page faithfully. "
                        "Preserve structure, headings, and reading order.",
                    )
                    raw_parts.append(general_text)
                    all_sections.append(Section(
                        type="page",
                        text=general_text,
                        confidence=0.85,
                        data={"page_number": page_idx, "model": MODEL_ID},
                    ))

            except Exception as exc:
                logger.exception("Granite Vision failed on page %d", page_idx)
                extraction_errors.append(f"Page {page_idx}: {exc}")

        # Compose final output
        raw_text = "\n\n".join(raw_parts) if raw_parts else ""
        if all_kvp and not raw_text:
            raw_text = json.dumps(all_kvp, indent=2, ensure_ascii=False)

        confidence = 0.855 if all_kvp else (0.88 if all_tables else 0.82)

        return BackendResult(
            backend=self.name,
            status="completed" if (all_kvp or all_tables or raw_parts) else "failed",
            raw_text=raw_text,
            sections=all_sections,
            confidence=confidence,
            error="; ".join(extraction_errors) if extraction_errors and not (all_kvp or all_tables) else None,
            metadata={
                "model": MODEL_ID,
                "page_count": len(images),
                "document_type": doc_type,
                "schema_used": doc_type if schema else None,
                "kvp_extracted": len(all_kvp),
                "tables_extracted": len(all_tables),
                "extraction_errors": extraction_errors,
                "vllm_mode": self._use_vllm,
                "provenance_attached": True,
            },
        )

    def _detect_document_type(self, analysis: AnalysisResult) -> str:
        """Detect document type from analysis hints."""
        hints = analysis.source_hints
        filename = hints.get("filename", "").lower()

        type_keywords = {
            "lab_report": ["lab", "report", "blood", "cbc", "panel", "test_result", "pathology"],
            "prescription": ["prescription", "rx", "script", "medication"],
            "discharge_summary": ["discharge", "summary", "disch", "hospital"],
            "referral_letter": ["referral", "refer", "consult"],
            "insurance_claim": ["claim", "insurance", "billing", "invoice"],
            "intake_form": ["intake", "registration", "admission", "form", "questionnaire"],
        }

        for doc_type, keywords in type_keywords.items():
            if any(kw in filename for kw in keywords):
                return doc_type

        # Default to generic if no match
        return "unknown"


class GraniteContradictionReviewer:
    """Uses Granite Vision to review primary OCR output for contradictions.

    Compares primary OCR text extraction against Granite's own extraction
    to flag discrepancies in tables, numbers, dates, and key values.
    """

    def __init__(self, granite_backend: GraniteVisionBackend) -> None:
        self._backend = granite_backend

    def review(
        self,
        image_path: str,
        primary_ocr_text: str,
        page_number: int = 1,
    ) -> Dict[str, Any]:
        """Review primary OCR output for contradictions."""
        prompt = (
            "You are a medical document verification system.\n"
            "Compare the following OCR-extracted text against what you see in this image.\n"
            "Report ONLY factual contradictions — wrong numbers, dates, names, or values.\n"
            "Do NOT report formatting or style differences.\n\n"
            f"PRIMARY OCR TEXT:\n{primary_ocr_text[:2000]}\n\n"
            "Respond in JSON format:\n"
            '{"contradictions": [{"field": "...", "ocr_value": "...", "image_value": "...", "severity": "high|medium|low"}], '
            '"overall_agreement": "high|medium|low"}'
        )

        try:
            result = self._backend._infer(image_path, prompt)
            return json.loads(result)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Contradiction review failed: %s", e)
            return {
                "contradictions": [],
                "overall_agreement": "unknown",
                "error": str(e),
            }
