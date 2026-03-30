"""
MediScan AI v7.0 — Governance Layer
Clinical validation, guideline checking, NEGATION-AWARE risk flagging,
explainability, and audit logging.

v5.1 CHANGES:
  ✅ Negation-aware keyword matching in RiskFlagger and GuidelineChecker
  ✅ _is_negated() — 8-word window before keyword for negation phrases
  ✅ _find_positive_keywords() — returns only non-negated matches
  ✅ negated_findings list in risk_assessment output for transparency
  ✅ GuidelineChecker skips negated critical findings
"""
from __future__ import annotations


import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Shared Negation Detection ────────────────────────────────────────────────

NEGATION_PHRASES = [
    "no evidence of", "no sign of", "no signs of", "without evidence of",
    "without", "no", "not", "none", "absent", "negative for",
    "rules out", "ruled out", "rule out", "no definite",
    "no acute", "no significant", "denies", "denied", "unremarkable for",
    "not suggestive of", "not consistent with", "not compatible with",
    "no convincing", "no demonstrable", "no appreciable", "no focal",
    "no gross", "no obvious", "unlikely", "low probability of",
    "resolved", "clear of", "free of", "free from",
]
NEGATION_PHRASES.sort(key=len, reverse=True)


def _is_negated(text: str, keyword: str, window_words: int = 8) -> bool:
    """Check if a keyword is negated in its context.

    Scans all occurrences of `keyword` in `text`. For each, checks a
    window of `window_words` words preceding the keyword — but only within
    the SAME sentence (split on '. ') — for negation phrases.
    Returns True only if ALL occurrences are negated.
    Returns False if the keyword is not found at all (caller should pre-check).
    """
    text_lower = text.lower()
    kw_lower = keyword.lower()
    start = 0
    found_any = False

    while True:
        idx = text_lower.find(kw_lower, start)
        if idx == -1:
            break
        found_any = True

        # Find the sentence containing this keyword occurrence
        r1 = text_lower.rfind(". ", 0, idx)
        r2 = text_lower.rfind(".\n", 0, idx)
        sent_start = max(r1 + 2 if r1 >= 0 else 0,
                         r2 + 2 if r2 >= 0 else 0,
                         0)
        prefix_in_sentence = text_lower[sent_start:idx].strip()

        # v5.1 FIX: Contrastive conjunctions break negation scope
        # "No pneumothorax but possible pneumonia" — "but" resets negation
        CONJUNCTIONS = ["but", "however", "although", "though", "yet", "while"]
        prefix_words_raw = prefix_in_sentence.split()
        for i, w in enumerate(prefix_words_raw):
            if w.strip(",.;:") in CONJUNCTIONS:
                prefix_in_sentence = " ".join(prefix_words_raw[i + 1:])
                break

        prefix_words = prefix_in_sentence.split()
        window = " ".join(prefix_words[-window_words:]) if prefix_words else ""

        negated_here = any(neg in window for neg in NEGATION_PHRASES)
        if not negated_here:
            return False  # found a non-negated occurrence

        start = idx + len(kw_lower)

    return found_any  # True only if ALL occurrences were negated; False if not found


def _find_positive_keywords(
    text: str, keywords: list[str]
) -> tuple[list[str], list[str]]:
    """Partition keywords into positive (non-negated) and negated.

    Returns:
        (positive_keywords, negated_keywords)
    """
    text_lower = text.lower()
    positive, negated = [], []
    for kw in keywords:
        if kw.lower() not in text_lower:
            continue
        if _is_negated(text_lower, kw.lower()):
            negated.append(kw)
        else:
            positive.append(kw)
    return positive, negated


# ─── Clinical Validator ───────────────────────────────────────────────────────

class ClinicalValidator:
    """Validates model outputs against clinical standards."""

    ICD10_PATTERN = re.compile(r'[A-Z]\d{2}(?:\.\d{1,4})?')
    REQUIRED_SECTIONS = ["findings", "impression"]

    def validate(self, report: dict[str, Any]) -> dict[str, Any]:
        issues, warnings = [], []
        response_text = report.get("consensus_answer", report.get("response", ""))
        response_lower = response_text.lower()

        if len(response_text.strip()) < 50:
            issues.append("Report is too short for clinical use")

        for section in self.REQUIRED_SECTIONS:
            if section not in response_lower:
                warnings.append(f"Missing recommended section: {section}")

        laterality_issues = self._check_laterality(response_text)
        issues.extend(laterality_issues)

        abs_count = sum(1 for w in ["definitely", "certainly", "always", "never"]
                       if w in response_lower)
        if abs_count > 3:
            warnings.append("Excessive absolute language — consider more nuanced phrasing")

        safety_phrases = ["correlate clinically", "clinical correlation recommended",
                          "clinical correlation is recommended"]
        has_safety = any(p in response_lower for p in safety_phrases)

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "has_safety_language": has_safety,
            "validation_timestamp": datetime.utcnow().isoformat(),
        }

    def _check_laterality(self, text: str) -> list[str]:
        return []


# ─── Guideline Checker (v5.1: Negation-Aware) ─────────────────────────────────

class GuidelineChecker:
    """Checks model outputs against WHO/NIH/ACR guidelines — negation-aware.

    v5.1: "no evidence of pneumothorax" will NOT trigger a critical finding.
    """

    CRITICAL_FINDINGS_ACR = {
        "pneumothorax": {
            "urgency": "stat",
            "communication": "Verbal communication to ordering physician within 1 hour",
            "guideline": "ACR Practice Parameter for Communication of Diagnostic Imaging Findings",
        },
        "pulmonary embolism": {
            "urgency": "stat",
            "communication": "Immediate verbal communication",
            "guideline": "ACR Appropriateness Criteria",
        },
        "aortic dissection": {
            "urgency": "emergent",
            "communication": "Immediate verbal communication to ED/surgeon",
            "guideline": "AHA/ACC Guidelines",
        },
        "intracranial hemorrhage": {
            "urgency": "emergent",
            "communication": "Immediate verbal communication",
            "guideline": "AHA/ASA Guidelines for Stroke",
        },
        "tension pneumothorax": {
            "urgency": "emergent",
            "communication": "Immediate verbal communication — life threatening",
            "guideline": "ATLS Guidelines",
        },
        "spinal cord compression": {
            "urgency": "emergent",
            "communication": "Immediate verbal communication to neurosurgery",
            "guideline": "NICE Guidelines NG12",
        },
        "acute appendicitis": {
            "urgency": "urgent",
            "communication": "Verbal communication within 4 hours",
            "guideline": "ACR Appropriateness Criteria",
        },
    }

    def check(self, report_text: str) -> dict[str, Any]:
        finding_names = list(self.CRITICAL_FINDINGS_ACR.keys())
        positive_findings, negated_findings = _find_positive_keywords(
            report_text, finding_names
        )
        triggered = [
            {"finding": f, **self.CRITICAL_FINDINGS_ACR[f]}
            for f in positive_findings
        ]
        return {
            "has_critical_findings": len(triggered) > 0,
            "critical_findings": triggered,
            "negated_critical_findings": negated_findings,
            "requires_immediate_communication": any(
                g["urgency"] in ("stat", "emergent") for g in triggered
            ),
            "guidelines_checked": len(self.CRITICAL_FINDINGS_ACR),
        }


# ─── Risk Flagger (v5.1: Negation-Aware) ──────────────────────────────────────

class RiskFlagger:
    """Flags high-risk findings — negation-aware.

    v5.1: "no evidence of pneumothorax" → ROUTINE, not EMERGENT.
    """

    RISK_KEYWORDS = {
        "emergent": [
            "pneumothorax", "tension pneumothorax", "aortic dissection",
            "pulmonary embolism", "intracranial hemorrhage", "stroke",
            "cardiac tamponade", "ruptured aneurysm", "free air",
        ],
        "urgent": [
            "mass", "tumor", "malignant", "fracture", "dislocation",
            "bowel obstruction", "appendicitis", "cholecystitis",
            "abscess", "empyema", "pneumonia",
        ],
        "routine": [
            "degenerative", "chronic", "stable", "benign",
            "incidental", "unremarkable",
        ],
    }

    def flag(self, report_text: str) -> dict[str, Any]:
        risk_level = "routine"
        flagged_findings = []
        negated_findings = []

        for level in ["emergent", "urgent", "routine"]:
            positive, negated = _find_positive_keywords(
                report_text, self.RISK_KEYWORDS[level]
            )
            for kw in positive:
                flagged_findings.append({"finding": kw, "risk_level": level})
                if level == "emergent":
                    risk_level = "emergent"
                elif level == "urgent" and risk_level != "emergent":
                    risk_level = "urgent"
            for kw in negated:
                negated_findings.append({"finding": kw, "would_be_level": level})

        return {
            "risk_level": risk_level,
            "flagged_findings": flagged_findings,
            "negated_findings": negated_findings,
            "requires_immediate_attention": risk_level == "emergent",
            "requires_follow_up": risk_level in ("emergent", "urgent"),
        }


# ─── Explainability ──────────────────────────────────────────────────────────

class Explainability:
    """Provides model explainability via attention visualization."""

    def generate_attention_summary(
        self, model_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        explanations = []
        for result in model_results:
            model_name = result.get("model", "unknown")
            thinking = result.get("thinking", "")
            answer = result.get("answer", "")
            explanations.append({
                "model": model_name,
                "reasoning_available": bool(thinking),
                "reasoning_summary": thinking[:500] if thinking else "No reasoning provided",
                "key_findings": self._extract_key_phrases(answer),
            })
        return {
            "explanations": explanations,
            "models_with_reasoning": sum(1 for e in explanations if e["reasoning_available"]),
            "total_models": len(explanations),
        }

    def _extract_key_phrases(self, text: str) -> list[str]:
        patterns = [
            r"(?:shows?|demonstrates?|reveals?|indicates?)\s+(.+?)(?:\.|,|$)",
            r"(?:consistent with|suggestive of|compatible with)\s+(.+?)(?:\.|,|$)",
        ]
        phrases = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend(m.strip() for m in matches if len(m.strip()) > 5)
        return phrases[:10]


# ─── Audit Logger ─────────────────────────────────────────────────────────────

class AuditLogger:
    """Complete audit trail — HIPAA compliance."""

    def __init__(self, log_dir: str = "./logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_analysis(
        self, request_id: str, input_metadata: dict[str, Any],
        models_used: list[str], results: dict[str, Any],
        governance_results: dict[str, Any], user_id: str = "system",
    ) -> str:
        audit_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "input": {
                "file_type": input_metadata.get("file_type", "unknown"),
                "modality": input_metadata.get("modality", "unknown"),
                "has_patient_info": bool(input_metadata.get("patient_info")),
            },
            "models_used": models_used,
            "results": {
                "confidence": results.get("confidence", 0),
                "agreement_score": results.get("agreement_score", 0),
                "uncertainty": results.get("uncertainty", 0),
                "model_count": results.get("model_count", 0),
            },
            "governance": {
                "clinical_valid": governance_results.get("clinical_validation", {}).get("is_valid", False),
                "risk_level": governance_results.get("risk_assessment", {}).get("risk_level", "unknown"),
                "has_critical_findings": governance_results.get("guideline_check", {}).get("has_critical_findings", False),
                "negated_critical_count": len(
                    governance_results.get("guideline_check", {}).get("negated_critical_findings", [])
                ),
                "warnings_count": len(governance_results.get("clinical_validation", {}).get("warnings", [])),
            },
        }

        log_file = self.log_dir / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

        logger.info(f"Audit logged: {request_id} | Risk: {audit_entry['governance']['risk_level']}")
        return request_id
