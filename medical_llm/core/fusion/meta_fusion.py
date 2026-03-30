"""
Meta-Fusion — Multi-model consensus engine.
Normalizes outputs, scores confidence, and selects/merges the best answer.

Confidence Formula:
    confidence = reasoning_score * 0.4 + medical_score * 0.4 + evidence_score * 0.2
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MetaFusion:
    """Multi-model fusion layer — the 'secret weapon'.

    Takes outputs from multiple medical LLMs and produces a single
    consensus answer with calibrated confidence scores.

    Strategies:
    - weighted_consensus: Confidence-weighted selection
    - majority_vote: Agreement-based selection
    - best_confidence: Highest individual score
    - merge: Combines key sections from multiple outputs
    """

    # Role-based weights for confidence formula
    ROLE_WEIGHTS = {
        "reasoning": {"reasoning_score": 0.6, "medical_score": 0.3, "evidence_score": 0.1},
        "medical_reasoning": {"reasoning_score": 0.3, "medical_score": 0.5, "evidence_score": 0.2},
        "literature": {"reasoning_score": 0.1, "medical_score": 0.3, "evidence_score": 0.6},
        "fast_engine": {"reasoning_score": 0.3, "medical_score": 0.5, "evidence_score": 0.2},
        "validator": {"reasoning_score": 0.2, "medical_score": 0.6, "evidence_score": 0.2},
        "conversational": {"reasoning_score": 0.2, "medical_score": 0.4, "evidence_score": 0.4},
    }

    # Hedging → lower confidence, Confident language → higher confidence
    HEDGING_PHRASES = [
        "might be", "could be", "possibly", "uncertain", "unclear",
        "may represent", "cannot exclude", "differential includes",
        "nonspecific", "equivocal", "indeterminate", "suggest",
        "further evaluation", "clinical correlation",
    ]
    CONFIDENT_PHRASES = [
        "consistent with", "diagnostic of", "compatible with",
        "characteristic of", "represents", "confirms", "demonstrates",
        "classic appearance", "pathognomonic", "definitive",
        "strongly suggests", "highly likely", "evidence supports",
    ]

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.fusion_weights = config.get("confidence_weights", {
            "reasoning_score": 0.4,
            "medical_score": 0.4,
            "evidence_score": 0.2,
        })
        self.strategy = config.get("strategy", "weighted_consensus")
        self.agreement_threshold = config.get("agreement_threshold", 0.6)

    def fuse(
        self,
        model_results: list[dict[str, Any]],
        rag_evidence: list[dict[str, Any]] | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Fuse multiple model outputs into a consensus result.

        Args:
            model_results: List of dicts with {text, model, role, weight, ...}
            rag_evidence: Retrieved evidence for evidence scoring
            strategy: Override fusion strategy

        Returns:
            Fused result with consensus answer, confidence, attribution
        """
        if not model_results:
            return {"error": "No model results to fuse", "text": ""}

        active_strategy = strategy or self.strategy
        answers = [r.get("text", "") for r in model_results]
        valid_results = [r for r in model_results if r.get("text", "").strip()]

        if not valid_results:
            return {"error": "All model outputs are empty", "text": ""}

        # Step 1: Score each model output
        scores = self._score_outputs(valid_results, rag_evidence)

        # Step 2: Select/merge based on strategy
        if active_strategy == "weighted_consensus":
            best_idx = int(np.argmax(scores))
        elif active_strategy == "majority_vote":
            best_idx = self._majority_vote([r.get("text", "") for r in valid_results])
        elif active_strategy == "best_confidence":
            best_idx = int(np.argmax(scores))
        elif active_strategy == "merge":
            merged = self._merge_outputs(valid_results, scores)
            return merged
        else:
            best_idx = int(np.argmax(scores))

        best_result = valid_results[best_idx]
        agreement = self._compute_agreement([r.get("text", "") for r in valid_results])

        # Build individual results
        individual = []
        for i, r in enumerate(valid_results):
            individual.append({
                "model": r.get("model", "unknown"),
                "role": r.get("role", "primary"),
                "score": float(scores[i]),
                "excerpt": r.get("text", "")[:400],
                "tokens": r.get("tokens_generated", 0),
                "latency": r.get("latency", 0),
            })

        fused = {
            "text": best_result.get("text", ""),
            "consensus_answer": best_result.get("text", ""),
            "best_model": best_result.get("model", "unknown"),
            "confidence": float(scores[best_idx]),
            "agreement_score": agreement,
            "uncertainty": 1.0 - agreement,
            "strategy": active_strategy,
            "individual_results": individual,
            "model_count": len(valid_results),
            "reasoning_chain": best_result.get("reasoning_chain"),
        }

        logger.info(
            f"Fusion: best={best_result.get('model')} "
            f"conf={scores[best_idx]:.3f} agree={agreement:.3f}"
        )
        return fused

    def _score_outputs(
        self,
        results: list[dict[str, Any]],
        rag_evidence: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Score each model output using the confidence formula.

        confidence = reasoning_score * w1 + medical_score * w2 + evidence_score * w3
        """
        n = len(results)
        scores = np.zeros(n)

        for i, result in enumerate(results):
            text = result.get("text", "")
            role = result.get("role", "primary")
            model_weight = result.get("weight", 0.5)

            # Reasoning score: based on reasoning chain quality
            reasoning_score = self._compute_reasoning_score(result)

            # Medical score: based on language quality + model reliability
            medical_score = self._compute_medical_score(text, model_weight)

            # Evidence score: based on RAG alignment
            evidence_score = self._compute_evidence_score(text, rag_evidence)

            # Cross-model agreement bonus
            agreement_bonus = 0.0
            for j in range(n):
                if i != j:
                    sim = self._text_similarity(text, results[j].get("text", ""))
                    agreement_bonus += sim
            agreement_bonus = agreement_bonus / max(n - 1, 1)

            # Role-specific weight distribution
            role_weights = self.ROLE_WEIGHTS.get(role, self.fusion_weights)

            confidence = (
                reasoning_score * role_weights.get("reasoning_score", 0.4) +
                medical_score * role_weights.get("medical_score", 0.4) +
                evidence_score * role_weights.get("evidence_score", 0.2)
            )

            # Add agreement bonus
            scores[i] = confidence * 0.7 + agreement_bonus * 0.3

        return scores

    def _compute_reasoning_score(self, result: dict[str, Any]) -> float:
        """Score based on reasoning chain quality."""
        text = result.get("text", "")

        # Has reasoning chain (self-reflection)?
        if result.get("reasoning_chain"):
            chain = result["reasoning_chain"]
            passes_completed = len(chain)
            base = min(0.9, 0.5 + passes_completed * 0.15)
        else:
            base = 0.5

        # Check for structured reasoning indicators
        reasoning_markers = [
            "differential diagnosis", "step 1", "step 2",
            "evidence", "therefore", "because", "given that",
            "considering", "based on", "in conclusion",
            "assessment", "plan", "impression",
        ]
        marker_count = sum(1 for m in reasoning_markers if m in text.lower())
        marker_bonus = min(0.2, marker_count * 0.03)

        return min(1.0, base + marker_bonus)

    def _compute_medical_score(self, text: str, model_weight: float) -> float:
        """Score based on medical language quality and model reliability."""
        text_lower = text.lower()

        hedge_count = sum(1 for p in self.HEDGING_PHRASES if p in text_lower)
        confident_count = sum(1 for p in self.CONFIDENT_PHRASES if p in text_lower)

        language_score = 0.6 + (confident_count * 0.04) - (hedge_count * 0.06)
        language_score = max(0.1, min(1.0, language_score))

        # Combine with model reliability weight
        return language_score * 0.5 + model_weight * 0.5

    def _compute_evidence_score(
        self, text: str, rag_evidence: list[dict[str, Any]] | None
    ) -> float:
        """Score based on alignment with RAG evidence."""
        if not rag_evidence:
            return 0.5  # Neutral if no evidence

        text_words = set(re.findall(r'\w+', text.lower()))
        overlap_scores = []

        for evidence in rag_evidence[:5]:
            ev_text = evidence.get("content", "")
            ev_words = set(re.findall(r'\w+', ev_text.lower()))
            if ev_words:
                overlap = len(text_words & ev_words) / len(ev_words)
                overlap_scores.append(overlap)

        if not overlap_scores:
            return 0.5

        return min(1.0, max(overlap_scores) * 1.5)

    def _majority_vote(self, answers: list[str]) -> int:
        """Find the answer with most agreement with others."""
        n = len(answers)
        scores = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    scores[i] += self._text_similarity(answers[i], answers[j])
        return int(np.argmax(scores))

    def _compute_agreement(self, answers: list[str]) -> float:
        """Compute pairwise agreement between answers."""
        if len(answers) < 2:
            return 1.0
        sims = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                sims.append(self._text_similarity(answers[i], answers[j]))
        return float(np.mean(sims)) if sims else 0.0

    def _merge_outputs(
        self,
        results: list[dict[str, Any]],
        scores: np.ndarray,
    ) -> dict[str, Any]:
        """Merge outputs by combining sections from top-scoring models."""
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        merged_parts = []

        # Take best answer as base
        base = results[sorted_indices[0]]
        merged_parts.append(f"## Primary Analysis (by {base.get('model', 'Model 1')})")
        merged_parts.append(base.get("text", ""))

        # Add unique insights from other models
        for idx in sorted_indices[1:3]:  # Top 3 only
            other = results[idx]
            other_text = other.get("text", "")
            if other_text:
                merged_parts.append(
                    f"\n## Additional Perspective (by {other.get('model', 'Model')})"
                )
                merged_parts.append(other_text[:500])

        merged_text = "\n\n".join(merged_parts)
        best_score = float(scores[sorted_indices[0]])

        return {
            "text": merged_text,
            "consensus_answer": merged_text,
            "best_model": base.get("model", "unknown"),
            "confidence": best_score,
            "strategy": "merge",
            "model_count": len(results),
        }

    def _text_similarity(self, a: str, b: str) -> float:
        """Hybrid lexical similarity for lightly paraphrased clinical text."""
        words_a = set(re.findall(r'\w+', a.lower()))
        words_b = set(re.findall(r'\w+', b.lower()))
        if not words_a or not words_b:
            return 0.0

        jaccard = len(words_a & words_b) / len(words_a | words_b)
        sequence = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        return (0.35 * jaccard) + (0.65 * sequence)


class UncertaintyEstimator:
    """Estimates prediction uncertainty from ensemble disagreement."""

    def estimate(self, model_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute uncertainty from multi-model results."""
        answers = [r.get("text", "") for r in model_results if r.get("text")]

        if len(answers) < 2:
            return {"epistemic": 0.0, "aleatoric": 0.0, "total": 0.0, "interpretation": "Single model — no ensemble uncertainty"}

        fusion = MetaFusion()

        # Epistemic: model disagreement
        agreement = fusion._compute_agreement(answers)
        epistemic = 1.0 - agreement

        # Aleatoric: hedging language
        hedge_ratios = []
        for ans in answers:
            ans_lower = ans.lower()
            hedge_count = sum(1 for p in fusion.HEDGING_PHRASES if p in ans_lower)
            word_count = max(len(ans.split()), 1)
            hedge_ratios.append(min(1.0, hedge_count / (word_count / 100)))
        aleatoric = float(np.mean(hedge_ratios))

        total = 0.6 * epistemic + 0.4 * aleatoric

        return {
            "epistemic": round(epistemic, 4),
            "aleatoric": round(aleatoric, 4),
            "total": round(total, 4),
            "interpretation": self._interpret(total),
        }

    def _interpret(self, u: float) -> str:
        if u < 0.2:
            return "High confidence — strong model agreement"
        elif u < 0.4:
            return "Moderate confidence — minor disagreement"
        elif u < 0.6:
            return "Moderate uncertainty — consider additional review"
        return "High uncertainty — significant disagreement, expert review required"


class ContradictionDetector:
    """Detects contradictions between model outputs."""

    CONTRADICTION_PAIRS = [
        ("normal", "abnormal"), ("present", "absent"),
        ("benign", "malignant"), ("positive", "negative"),
        ("increased", "decreased"), ("enlarged", "small"),
        ("acute", "chronic"), ("stable", "progressive"),
        ("left", "right"), ("superior", "inferior"),
        ("improve", "worsen"), ("proximal", "distal"),
        ("systolic", "diastolic"), ("hyper", "hypo"),
    ]

    def detect(self, model_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect contradictions between model outputs."""
        answers = [r.get("text", "") for r in model_results]
        model_names = [r.get("model", f"model_{i}") for i, r in enumerate(model_results)]

        contradictions = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                found = self._find_contradictions(answers[i], answers[j])
                if found:
                    contradictions.append({
                        "model_a": model_names[i],
                        "model_b": model_names[j],
                        "contradictions": found,
                    })

        severity = "none"
        total = sum(len(c["contradictions"]) for c in contradictions)
        if total >= 4:
            severity = "high"
        elif total >= 2:
            severity = "moderate"
        elif total >= 1:
            severity = "low"

        return {
            "has_contradictions": len(contradictions) > 0,
            "count": total,
            "severity": severity,
            "details": contradictions,
        }

    def _find_contradictions(self, a: str, b: str) -> list[dict]:
        found = []
        a_lower, b_lower = a.lower(), b.lower()
        for w1, w2 in self.CONTRADICTION_PAIRS:
            if (w1 in a_lower and w2 in b_lower) or (w2 in a_lower and w1 in b_lower):
                found.append({
                    "type": f"{w1}_vs_{w2}",
                    "description": f"One model says '{w1}' while another says '{w2}'",
                })
        return found
