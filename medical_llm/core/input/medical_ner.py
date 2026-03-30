"""
Medical NER — Entity Recognition & Linking
Uses SciSpacy for biomedical NER and BioGPT for entity linking to UMLS concepts.

Pipeline:
    Raw text → SciSpacy NER → Entity spans → BioGPT linking → Structured entities
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MedicalEntity:
    """A single extracted medical entity."""
    text: str
    label: str                    # DISEASE, CHEMICAL, GENE, etc.
    start: int = 0
    end: int = 0
    cui: str = ""                 # UMLS Concept Unique Identifier
    canonical_name: str = ""      # Normalized name from UMLS
    confidence: float = 1.0
    source: str = "scispacy"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "cui": self.cui,
            "canonical_name": self.canonical_name or self.text,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "metadata": self.metadata,
        }


class MedicalNER:
    """Biomedical Named Entity Recognition using SciSpacy.

    Extracts diseases, chemicals/drugs, genes, anatomical structures,
    procedures, and lab values from medical text.
    """

    # Entity label mapping for different SciSpacy models
    ENTITY_LABELS = {
        "DISEASE": "disease",
        "CHEMICAL": "drug",
        "GENE_OR_GENE_PRODUCT": "gene",
        "ORGANISM": "organism",
        "CELL_TYPE": "cell",
        "CELL_LINE": "cell_line",
        "DNA": "dna",
        "RNA": "rna",
        "PROTEIN": "protein",
        "SIMPLE_CHEMICAL": "drug",
        "AMINO_ACID": "amino_acid",
    }

    # Regex patterns for entities that NER models often miss
    LAB_VALUE_PATTERN = re.compile(
        r'(?P<name>[A-Za-z\s]+?)\s*[:=]\s*'
        r'(?P<value>[\d.,]+)\s*'
        r'(?P<unit>mg/[dL]+|mmol/L|g/dL|%|U/L|IU/L|mEq/L|ng/mL|pg/mL|µg/L|cells/µL|mm/hr)?',
        re.IGNORECASE,
    )
    VITAL_SIGNS_PATTERN = re.compile(
        r'(?:BP|blood pressure|HR|heart rate|RR|respiratory rate|SpO2|temperature|temp)\s*'
        r'[:=]?\s*[\d/.,]+\s*(?:mmHg|bpm|°[CF]|%)?',
        re.IGNORECASE,
    )

    def __init__(
        self,
        ner_model_name: str = "en_ner_bc5cdr_md",
        sci_model_name: str = "en_core_sci_lg",
        enable_linker: bool = True,
    ):
        self.ner_model_name = ner_model_name
        self.sci_model_name = sci_model_name
        self.enable_linker = enable_linker
        self._ner_nlp = None
        self._sci_nlp = None
        self._linker = None
        self._is_loaded = False

    def load(self) -> None:
        """Load SciSpacy models and optional UMLS linker."""
        if self._is_loaded:
            return

        try:
            import spacy

            # Primary NER model (disease + chemical)
            try:
                self._ner_nlp = spacy.load(self.ner_model_name)
                logger.info(f"Loaded NER model: {self.ner_model_name}")
            except OSError:
                logger.warning(
                    f"NER model '{self.ner_model_name}' not found. "
                    f"Install with: pip install {self.ner_model_name}"
                )

            # Broader scientific model
            try:
                self._sci_nlp = spacy.load(self.sci_model_name)
                logger.info(f"Loaded science NLP model: {self.sci_model_name}")
            except OSError:
                logger.warning(
                    f"Science model '{self.sci_model_name}' not found. "
                    f"Install with: pip install {self.sci_model_name}"
                )

            # UMLS entity linker
            if self.enable_linker and self._sci_nlp is not None:
                try:
                    from scispacy.linking import EntityLinker
                    self._sci_nlp.add_pipe(
                        "scispacy_linker",
                        config={
                            "resolve_abbreviations": True,
                            "linker_name": "umls",
                        },
                    )
                    self._linker = self._sci_nlp.get_pipe("scispacy_linker")
                    logger.info("UMLS entity linker initialized")
                except Exception as e:
                    logger.warning(f"UMLS linker not available: {e}")

            self._is_loaded = True

        except ImportError:
            logger.error(
                "spacy/scispacy not installed. "
                "Install with: pip install spacy scispacy"
            )

    def extract(self, text: str) -> list[MedicalEntity]:
        """Extract all medical entities from text.

        Returns combined entities from NER model, science model,
        and regex-based extractors (lab values, vital signs).
        """
        if not self._is_loaded:
            self.load()

        entities: list[MedicalEntity] = []
        seen_spans: set[tuple[int, int]] = set()

        # Pass 1: Dedicated NER model (diseases + chemicals)
        if self._ner_nlp is not None:
            doc = self._ner_nlp(text)
            for ent in doc.ents:
                span_key = (ent.start_char, ent.end_char)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    entities.append(MedicalEntity(
                        text=ent.text,
                        label=self.ENTITY_LABELS.get(ent.label_, ent.label_),
                        start=ent.start_char,
                        end=ent.end_char,
                        source="scispacy_ner",
                    ))

        # Pass 2: Science model + UMLS linking
        if self._sci_nlp is not None:
            doc = self._sci_nlp(text)
            for ent in doc.ents:
                span_key = (ent.start_char, ent.end_char)
                if span_key not in seen_spans:
                    seen_spans.add(span_key)

                    entity = MedicalEntity(
                        text=ent.text,
                        label=self.ENTITY_LABELS.get(ent.label_, "medical_term"),
                        start=ent.start_char,
                        end=ent.end_char,
                        source="scispacy_sci",
                    )

                    # Add UMLS linking info
                    if self._linker and hasattr(ent, "_") and hasattr(ent._, "kb_ents"):
                        kb_ents = ent._.kb_ents
                        if kb_ents:
                            top_cui, top_score = kb_ents[0]
                            entity.cui = top_cui
                            entity.confidence = float(top_score)
                            # Resolve canonical name
                            if hasattr(self._linker, "kb"):
                                try:
                                    concept = self._linker.kb.cui_to_entity.get(top_cui)
                                    if concept:
                                        entity.canonical_name = concept.canonical_name
                                except Exception:
                                    pass

                    entities.append(entity)

        # Pass 3: Regex extraction for lab values
        for match in self.LAB_VALUE_PATTERN.finditer(text):
            span_key = (match.start(), match.end())
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                entities.append(MedicalEntity(
                    text=match.group(0).strip(),
                    label="lab_value",
                    start=match.start(),
                    end=match.end(),
                    source="regex",
                    metadata={
                        "name": match.group("name").strip(),
                        "value": match.group("value"),
                        "unit": match.group("unit") or "",
                    },
                ))

        # Pass 4: Vital signs extraction
        for match in self.VITAL_SIGNS_PATTERN.finditer(text):
            span_key = (match.start(), match.end())
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                entities.append(MedicalEntity(
                    text=match.group(0).strip(),
                    label="vital_sign",
                    start=match.start(),
                    end=match.end(),
                    source="regex",
                ))

        # Sort by position
        entities.sort(key=lambda e: e.start)
        return entities

    def extract_structured(self, text: str) -> dict[str, Any]:
        """Extract entities and return structured output grouped by type."""
        entities = self.extract(text)

        grouped: dict[str, list[dict]] = {}
        for ent in entities:
            label = ent.label
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(ent.to_dict())

        return {
            "raw_text": text,
            "entity_count": len(entities),
            "entities_by_type": grouped,
            "all_entities": [e.to_dict() for e in entities],
            "diseases": [e.to_dict() for e in entities if e.label == "disease"],
            "drugs": [e.to_dict() for e in entities if e.label == "drug"],
            "genes": [e.to_dict() for e in entities if e.label == "gene"],
            "lab_values": [e.to_dict() for e in entities if e.label == "lab_value"],
            "vital_signs": [e.to_dict() for e in entities if e.label == "vital_sign"],
        }


class BioGPTLinker:
    """Entity linking and concept disambiguation using BioGPT.

    For entities that SciSpacy cannot link to UMLS, uses BioGPT
    to disambiguate and normalize medical terms.
    """

    def __init__(self, model_id: str = "microsoft/biogpt"):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    def load(self) -> None:
        """Load BioGPT model and tokenizer."""
        if self._is_loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._is_loaded = True
            logger.info(f"BioGPT linker loaded: {self.model_id}")

        except Exception as e:
            logger.error(f"Failed to load BioGPT: {e}")

    def link_entity(self, entity_text: str, context: str = "") -> dict[str, Any]:
        """Disambiguate a medical entity using BioGPT.

        Args:
            entity_text: The entity text to disambiguate
            context: Surrounding text for context

        Returns:
            Dict with canonical name, definition, and related concepts
        """
        if not self._is_loaded:
            self.load()

        if not self._is_loaded:
            return {
                "entity": entity_text,
                "canonical_name": entity_text,
                "linked": False,
            }

        prompt = (
            f"Define the medical term '{entity_text}'"
            f"{f' in the context: {context[:200]}' if context else ''}. "
            f"Provide: 1) Standard medical name 2) Brief definition."
        )

        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with __import__("torch").no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                )

            generated = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            return {
                "entity": entity_text,
                "canonical_name": entity_text,
                "definition": generated,
                "linked": True,
                "source": "biogpt",
            }

        except Exception as e:
            logger.warning(f"BioGPT linking failed for '{entity_text}': {e}")
            return {
                "entity": entity_text,
                "canonical_name": entity_text,
                "linked": False,
                "error": str(e),
            }

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
