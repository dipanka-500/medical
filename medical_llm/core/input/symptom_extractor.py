"""
Symptom Extractor — Extracts symptoms, conditions, medications, and lab values.
Normalizes medical synonyms to standard terminology.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SymptomExtractor:
    """Extracts and normalizes medical symptoms, conditions, and medications.

    Uses a combination of keyword matching, regex patterns, and synonym
    normalization to extract structured medical information from free text.
    """

    # Comprehensive synonym mapping: colloquial → standard medical term
    SYMPTOM_SYNONYMS: dict[str, str] = {
        # Cardiovascular
        "heart attack": "myocardial infarction",
        "chest pain": "angina pectoris",
        "high blood pressure": "hypertension",
        "low blood pressure": "hypotension",
        "fast heartbeat": "tachycardia",
        "slow heartbeat": "bradycardia",
        "irregular heartbeat": "arrhythmia",
        "heart failure": "congestive heart failure",
        "swollen legs": "peripheral edema",
        "swollen ankles": "pedal edema",
        # Respiratory
        "shortness of breath": "dyspnea",
        "difficulty breathing": "dyspnea",
        "trouble breathing": "dyspnea",
        "can't breathe": "dyspnea",
        "out of breath": "dyspnea",
        "coughing up blood": "hemoptysis",
        "blood in sputum": "hemoptysis",
        "wheezing": "bronchospasm",
        "stuffy nose": "nasal congestion",
        "runny nose": "rhinorrhea",
        # Neurological
        "headache": "cephalgia",
        "migraine": "migraine",
        "dizziness": "vertigo",
        "lightheaded": "presyncope",
        "fainting": "syncope",
        "passed out": "syncope",
        "seizure": "seizure",
        "convulsion": "seizure",
        "numbness": "paresthesia",
        "tingling": "paresthesia",
        "pins and needles": "paresthesia",
        "memory loss": "amnesia",
        "confusion": "altered mental status",
        "slurred speech": "dysarthria",
        # Gastrointestinal
        "stomach ache": "abdominal pain",
        "tummy ache": "abdominal pain",
        "belly pain": "abdominal pain",
        "nausea": "nausea",
        "throwing up": "emesis",
        "vomiting": "emesis",
        "diarrhea": "diarrhea",
        "constipation": "constipation",
        "blood in stool": "hematochezia",
        "black stool": "melena",
        "heartburn": "gastroesophageal reflux",
        "acid reflux": "gastroesophageal reflux",
        "difficulty swallowing": "dysphagia",
        "bloating": "abdominal distension",
        "loss of appetite": "anorexia",
        "jaundice": "icterus",
        "yellow skin": "icterus",
        # Musculoskeletal
        "back pain": "dorsalgia",
        "joint pain": "arthralgia",
        "muscle pain": "myalgia",
        "body aches": "myalgia",
        "stiff neck": "neck rigidity",
        "swollen joint": "joint effusion",
        # General
        "fever": "pyrexia",
        "chills": "rigors",
        "weight loss": "unintentional weight loss",
        "fatigue": "fatigue",
        "tiredness": "fatigue",
        "weakness": "asthenia",
        "night sweats": "nocturnal diaphoresis",
        "swollen lymph nodes": "lymphadenopathy",
        "rash": "dermatitis",
        "itching": "pruritus",
        "swelling": "edema",
        # Urological
        "blood in urine": "hematuria",
        "painful urination": "dysuria",
        "frequent urination": "polyuria",
        "can't hold urine": "urinary incontinence",
        # Ophthalmological
        "blurry vision": "blurred vision",
        "double vision": "diplopia",
        "eye pain": "ophthalmalgia",
        "red eye": "conjunctival injection",
        # Psychiatric
        "depression": "major depressive disorder",
        "anxiety": "generalized anxiety disorder",
        "insomnia": "insomnia",
        "can't sleep": "insomnia",
    }

    # Standard symptom keywords to look for
    SYMPTOM_KEYWORDS: list[str] = [
        "pain", "ache", "discomfort", "tenderness", "swelling", "edema",
        "fever", "chills", "fatigue", "weakness", "malaise",
        "nausea", "vomiting", "diarrhea", "constipation",
        "cough", "dyspnea", "wheezing", "hemoptysis",
        "headache", "dizziness", "syncope", "seizure",
        "rash", "pruritus", "lesion", "erythema",
        "bleeding", "bruising", "petechiae",
        "numbness", "tingling", "paresthesia",
        "anxiety", "depression", "insomnia",
        "polyuria", "dysuria", "hematuria",
        "tachycardia", "bradycardia", "palpitations",
        "dysphagia", "odynophagia", "regurgitation",
        "diplopia", "photophobia", "scotoma",
        "tinnitus", "vertigo", "hearing loss",
        "arthralgia", "myalgia", "stiffness",
        "anorexia", "weight loss", "weight gain",
        "diaphoresis", "rigors", "flushing",
    ]

    # Medication form patterns
    MEDICATION_PATTERN = re.compile(
        r'(?P<drug>[A-Za-z][\w-]+)\s+'
        r'(?P<dose>\d+(?:\.\d+)?)\s*'
        r'(?P<unit>mg|mcg|µg|g|mL|IU|units?)\s*'
        r'(?P<route>PO|IV|IM|SC|SQ|SL|PR|topical|inhaled|oral|intravenous)?'
        r'\s*(?P<frequency>(?:once|twice|three times|QD|BID|TID|QID|PRN|q\d+h|daily|weekly))?',
        re.IGNORECASE,
    )

    # Duration patterns
    DURATION_PATTERN = re.compile(
        r'(?:for|since|over the (?:past|last)|x)\s*'
        r'(\d+)\s*'
        r'(days?|weeks?|months?|years?|hours?|minutes?)',
        re.IGNORECASE,
    )

    # Severity patterns
    SEVERITY_PATTERN = re.compile(
        r'(mild|moderate|severe|extreme|slight|significant|marked|profound|'
        r'minimal|maximal|intense|acute|chronic|intermittent|constant|progressive|'
        r'worsening|improving|stable|persistent)',
        re.IGNORECASE,
    )

    # Pain scale pattern
    PAIN_SCALE_PATTERN = re.compile(
        r'(?:pain|discomfort)\s*(?:score|level|rating|scale)?\s*'
        r'(?:of|:|\s)\s*(\d{1,2})\s*(?:/\s*10|out of 10)?',
        re.IGNORECASE,
    )

    def extract_symptoms(self, text: str) -> list[dict[str, Any]]:
        """Extract symptoms with severity and duration."""
        text_lower = text.lower()
        symptoms: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Check synonym mappings first
        for colloquial, standard in self.SYMPTOM_SYNONYMS.items():
            if colloquial in text_lower:
                if standard not in seen:
                    seen.add(standard)
                    symptoms.append({
                        "symptom": standard,
                        "original_text": colloquial,
                        "normalized": True,
                    })

        # Check standard keywords
        for keyword in self.SYMPTOM_KEYWORDS:
            if keyword in text_lower and keyword not in seen:
                seen.add(keyword)
                symptoms.append({
                    "symptom": keyword,
                    "original_text": keyword,
                    "normalized": False,
                })

        # Extract severity for each symptom
        severity_matches = self.SEVERITY_PATTERN.findall(text)
        global_severity = severity_matches[0].lower() if severity_matches else None

        # Extract duration
        duration_match = self.DURATION_PATTERN.search(text)
        duration_info = None
        if duration_match:
            duration_info = {
                "value": int(duration_match.group(1)),
                "unit": duration_match.group(2).lower().rstrip("s"),
            }

        # Extract pain scale
        pain_match = self.PAIN_SCALE_PATTERN.search(text)
        pain_score = int(pain_match.group(1)) if pain_match else None

        # Enrich symptoms
        for symptom in symptoms:
            symptom["severity"] = global_severity
            symptom["duration"] = duration_info
            if "pain" in symptom["symptom"].lower() and pain_score is not None:
                symptom["pain_score"] = pain_score

        return symptoms

    def extract_medications(self, text: str) -> list[dict[str, Any]]:
        """Extract medication mentions with dose, route, and frequency."""
        medications = []
        for match in self.MEDICATION_PATTERN.finditer(text):
            medications.append({
                "drug": match.group("drug"),
                "dose": match.group("dose"),
                "unit": match.group("unit"),
                "route": (match.group("route") or "").upper() or None,
                "frequency": match.group("frequency") or None,
            })
        return medications

    def extract_all(self, text: str) -> dict[str, Any]:
        """Extract all medical information from text.

        Returns:
            Dict with symptoms, medications, durations, severities
        """
        symptoms = self.extract_symptoms(text)
        medications = self.extract_medications(text)

        # Extract duration separately
        duration_match = self.DURATION_PATTERN.search(text)
        duration = None
        if duration_match:
            duration = {
                "value": int(duration_match.group(1)),
                "unit": duration_match.group(2).lower().rstrip("s"),
                "raw": duration_match.group(0),
            }

        # Extract severity
        severity_matches = self.SEVERITY_PATTERN.findall(text)

        return {
            "symptoms": symptoms,
            "symptom_count": len(symptoms),
            "medications": medications,
            "medication_count": len(medications),
            "duration": duration,
            "severities": [s.lower() for s in severity_matches],
            "normalized_terms": [
                s["symptom"] for s in symptoms if s.get("normalized")
            ],
        }

    def normalize_term(self, term: str) -> str:
        """Normalize a single medical term to standard terminology."""
        term_lower = term.lower().strip()
        return self.SYMPTOM_SYNONYMS.get(term_lower, term)
