"""
Query Classifier — Classifies medical queries into routing categories.
Uses keyword heuristics for fast classification + optional BioMistral for ambiguous cases.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Classifies medical queries to determine routing strategy.

    Categories:
        - diagnosis: Symptom description seeking diagnosis
        - research: Literature/evidence-based questions
        - simple_qa: Factual medical knowledge questions
        - conversational: General health chat
        - drug_info: Drug/medication questions
        - emergency: Time-critical emergency queries
        - differential: Differential diagnosis requests
        - treatment: Treatment/management questions
        - lab_interpretation: Lab result interpretation
    """

    # Emergency signal keywords (high priority)
    EMERGENCY_SIGNALS = [
        "emergency", "urgent", "critical", "life threatening", "life-threatening",
        "dying", "cardiac arrest", "can't breathe", "not breathing",
        "severe chest pain", "stroke", "seizure right now",
        "massive bleeding", "unresponsive", "unconscious",
        "overdose", "poisoning", "anaphylaxis", "choking",
        "severe allergic reaction", "heart attack right now",
    ]

    # Explicit self-harm / suicidal ideation patterns that require immediate safety response
    CRISIS_PATTERNS = [
        r"\bi (?:want to|wanna|need to) (?:die|kill myself|end my life)\b",
        r"\bi(?:'m| am)?\s+(?:suicidal|thinking about suicide)\b",
        r"\bi(?:'m| am)?\s+(?:thinking about|planning on)\s+(?:killing myself|ending my life)\b",
        r"\bkill myself\b",
        r"\bend my life\b",
        r"\bwant to die\b",
        r"\bhurting myself\b",
        r"\bcutting myself\b",
        r"\bself[- ]harm(?:ing)?\b",
        r"\bnot worth living\b",
    ]

    # Diagnosis-seeking patterns
    DIAGNOSIS_PATTERNS = [
        r"what (?:could|might|is|does) .+ (?:mean|indicate|suggest|cause)",
        r"(?:diagnos|differenti|assess|evaluat)",
        r"(?:i have|i'm having|i've been having|patient (?:has|presents|complains))",
        r"(?:symptoms? (?:of|include|are)|what (?:is|are) the symptoms)",
        r"(?:could (?:this|it|i) (?:be|have))",
        r"what (?:condition|disease|disorder|illness)",
        r"(?:presenting with|chief complaint|cc:)",
    ]

    # Research/literature patterns
    RESEARCH_PATTERNS = [
        r"(?:study|studies|research|evidence|literature|trial|meta-analysis)",
        r"(?:pubmed|journal|paper|article|review)",
        r"(?:mechanism of action|pathophysiology|etiology|epidemiology)",
        r"(?:statistics|prevalence|incidence|mortality rate)",
        r"(?:latest|recent|current) (?:research|findings|evidence)",
        r"(?:what does the evidence say|evidence-based)",
        r"(?:clinical trial|randomized|systematic review)",
    ]

    # Drug/medication patterns
    DRUG_PATTERNS = [
        r"(?:drug|medication|medicine|prescription|pharmaceutical)",
        r"(?:dose|dosage|dosing|side ?effects?|adverse (?:effects?|reactions?))",
        r"(?:interaction|contraindication|precaution)",
        r"(?:can i take|should i take|is it safe to take)",
        r"(?:overdose|toxicity|therapeutic (?:range|index|level))",
        r"(?:generic|brand name|active ingredient)",
        r"(?:mg |mcg |µg |tablet|capsule|injection|infusion)",
    ]

    # Treatment/management patterns
    TREATMENT_PATTERNS = [
        r"(?:treatment|therapy|management|intervention|procedure)",
        r"(?:how (?:to|do you) treat|what is the treatment for)",
        r"(?:first.?line|second.?line|standard of care|protocol)",
        r"(?:surgery|surgical|operation|procedure)",
        r"(?:conservative|supportive|palliative)",
        r"(?:prognosis|outcome|recovery|remission|cure)",
    ]

    # Lab interpretation patterns
    LAB_PATTERNS = [
        r"(?:lab|laboratory|blood (?:test|work)|CBC|BMP|CMP|LFT|TFT)",
        r"(?:result|level|count|ratio|titer|value)\s*(?:is|of|:)",
        r"(?:high|low|elevated|decreased|abnormal|normal|borderline)\s+\w+\s*(?:level|count)",
        r"\d+\s*(?:mg/dL|mmol/L|g/dL|U/L|IU/L|mEq/L|ng/mL|cells/µL|mm/hr|%)",
        r"(?:what does .+ (?:level|count|result) mean)",
        r"(?:hemoglobin|hematocrit|WBC|RBC|platelet|creatinine|BUN|glucose|HbA1c)",
    ]

    # Simple Q&A patterns
    SIMPLE_QA_PATTERNS = [
        r"^what is ",
        r"^define ",
        r"^explain ",
        r"^describe ",
        r"^how (?:does|do|is|are)",
        r"^why (?:does|do|is|are)",
        r"^when (?:should|do|does)",
        r"^where (?:is|are|does)",
        r"^is it (?:true|normal|common|safe|possible)",
    ]

    # Conversational/casual patterns
    CONVERSATIONAL_PATTERNS = [
        r"^(?:hi|hello|hey|thanks|thank you|ok|okay|goodbye|bye)",
        r"(?:your opinion|what do you think|in general|generally speaking)",
        r"(?:tell me about|can you explain|i'm curious about)",
        r"(?:healthy|wellness|prevention|diet|exercise|lifestyle)",
        r"(?:should i (?:worry|be concerned)|is this normal|am i okay)",
    ]

    def __init__(self, model_engine: Optional[Any] = None):
        """
        Args:
            model_engine: Optional BioMistral engine for ambiguous query classification.
                          If None, uses heuristics only.
        """
        self._model_engine = model_engine

        # Compile patterns for performance
        self._compiled = {
            "emergency": [re.compile(re.escape(s), re.IGNORECASE) for s in self.EMERGENCY_SIGNALS],
            "diagnosis": [re.compile(p, re.IGNORECASE) for p in self.DIAGNOSIS_PATTERNS],
            "research": [re.compile(p, re.IGNORECASE) for p in self.RESEARCH_PATTERNS],
            "drug_info": [re.compile(p, re.IGNORECASE) for p in self.DRUG_PATTERNS],
            "treatment": [re.compile(p, re.IGNORECASE) for p in self.TREATMENT_PATTERNS],
            "lab_interpretation": [re.compile(p, re.IGNORECASE) for p in self.LAB_PATTERNS],
            "simple_qa": [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_QA_PATTERNS],
            "conversational": [re.compile(p, re.IGNORECASE) for p in self.CONVERSATIONAL_PATTERNS],
        }
        self._crisis_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.CRISIS_PATTERNS
        ]

    def classify(self, query: str) -> dict[str, Any]:
        """Classify a medical query into a category.

        Uses a scoring system: each category gets points based on
        pattern matches. The category with the highest score wins.
        Emergency always takes priority.

        Args:
            query: The user's medical query

        Returns:
            Dict with category, confidence, and reasoning
        """
        query_stripped = query.strip()
        if not query_stripped:
            return {
                "category": "conversational",
                "confidence": 1.0,
                "reasoning": "Empty query",
                "scores": {},
            }

        # Crisis / self-harm check (highest priority — triggers safety response)
        for pattern in self._crisis_patterns:
            match = pattern.search(query_stripped)
            if match:
                return {
                    "category": "emergency",
                    "confidence": 0.99,
                    "reasoning": f"Crisis signal detected: {match.group(0)}",
                    "scores": {"emergency": 10.0},
                    "is_emergency": True,
                    "is_crisis": True,
                }

        # Emergency check first (always highest priority)
        for pattern in self._compiled["emergency"]:
            if pattern.search(query_stripped):
                return {
                    "category": "emergency",
                    "confidence": 0.95,
                    "reasoning": f"Emergency signal detected: {pattern.pattern}",
                    "scores": {"emergency": 10.0},
                    "is_emergency": True,
                }

        # Score each category
        scores: dict[str, float] = {}
        match_reasons: dict[str, list[str]] = {}

        for category, patterns in self._compiled.items():
            if category == "emergency":
                continue

            score = 0.0
            reasons = []
            for pattern in patterns:
                matches = pattern.findall(query_stripped)
                if matches:
                    score += 1.0
                    reasons.append(pattern.pattern)

            if score > 0:
                scores[category] = score
                match_reasons[category] = reasons

        if not scores:
            # No pattern matched — try model-based classification
            if self._model_engine is not None:
                return self._classify_with_model(query_stripped)

            return {
                "category": "simple_qa",
                "confidence": 0.4,
                "reasoning": "No strong pattern match, defaulting to simple_qa",
                "scores": scores,
            }

        # Select highest scoring category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        total_score = sum(scores.values())
        confidence = min(0.95, best_score / max(total_score, 1.0))

        # Boost confidence if score is high
        if best_score >= 3.0:
            confidence = max(confidence, 0.85)

        return {
            "category": best_category,
            "confidence": round(confidence, 4),
            "reasoning": f"Pattern matches: {match_reasons.get(best_category, [])}",
            "scores": {k: round(v, 2) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
            "is_emergency": False,
        }

    def _classify_with_model(self, query: str) -> dict[str, Any]:
        """Use BioMistral for ambiguous queries."""
        categories = [
            "diagnosis", "research", "simple_qa", "conversational",
            "drug_info", "treatment", "lab_interpretation", "emergency",
        ]

        prompt = (
            f"Classify the following medical query into exactly one category.\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"Query: {query}\n\n"
            f"Respond with ONLY the category name, nothing else."
        )

        try:
            result = self._model_engine.generate(prompt, max_new_tokens=32)
            response = result.get("text", "").strip().lower()

            # Find which category the model chose
            for cat in categories:
                if cat in response:
                    return {
                        "category": cat,
                        "confidence": 0.75,
                        "reasoning": "Classified by BioMistral model",
                        "scores": {},
                        "model_response": response,
                    }

        except Exception as e:
            logger.warning(f"Model-based classification failed: {e}")

        return {
            "category": "simple_qa",
            "confidence": 0.3,
            "reasoning": "Model classification failed, fallback to simple_qa",
            "scores": {},
        }

    def get_all_categories(self) -> list[str]:
        """Return all possible classification categories."""
        return [
            "emergency",
            "diagnosis",
            "differential",
            "treatment",
            "drug_info",
            "lab_interpretation",
            "research",
            "simple_qa",
            "conversational",
        ]
