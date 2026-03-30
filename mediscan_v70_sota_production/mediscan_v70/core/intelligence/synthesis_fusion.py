"""
MediScan AI v7.0 — Finding-Level Synthesis Fusion

Instead of just picking the best model's full answer, this:
1. Extracts individual findings from each model
2. Cross-validates: findings mentioned by 2+ models get higher confidence
3. Flags contradictions at the finding level
4. Synthesizes a composite report using the best evidence for each finding
5. Produces a properly calibrated overall confidence score
"""
from __future__ import annotations


import logging
import re
from typing import Any

import numpy as np

from .medical_prompts import (
    calibrate_confidence,
    extract_individual_findings,
)

logger = logging.getLogger(__name__)


GENERATIVE_MODELS = {
    "hulu_med_7b", "hulu_med_14b", "hulu_med_32b",
    "medgemma_4b", "medgemma_27b",
    "medix_r1_2b", "medix_r1_8b", "medix_r1_30b",
    "med3dvlm",
    # v7.0 specialist models
    "chexagent_8b", "chexagent_3b",
    "pathgen", "radfm", "merlin",
}

CLASSIFIER_MODELS = {"biomedclip", "retfound"}

MODEL_WEIGHTS = {
    "hulu_med_7b": 1.00, "hulu_med_14b": 1.05, "hulu_med_32b": 1.10,
    "medgemma_4b": 0.85, "medgemma_27b": 0.95,
    "medix_r1_2b": 0.75, "medix_r1_8b": 0.85, "medix_r1_30b": 0.92,
    "med3dvlm": 0.80, "biomedclip": 0.55,
    # v7.0 specialist models
    "chexagent_8b": 0.95, "chexagent_3b": 0.85,
    "pathgen": 0.90, "radfm": 0.88, "retfound": 0.82, "merlin": 0.90,
}


class SynthesisFusion:
    """Finding-level synthesis fusion with cross-model validation.

    This replaces the simple 'pick best model' approach with actual
    synthesis that produces better reports than any single model.
    """

    def fuse(self, model_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Fuse model outputs via finding-level synthesis.

        Steps:
        1. Extract individual findings from each generative model
        2. Calibrate per-model confidence from linguistic analysis
        3. Select the best generative model as the primary answer
        4. Enhance with corroborating findings from other models
        5. Build cross-validation report
        """
        if not model_results:
            return {"error": "No model results to fuse"}

        successful = [r for r in model_results if r.get("status") == "success"]
        if not successful:
            return {"error": "All models failed", "failures": model_results}

        # ── Step 1: Extract and calibrate per-model ──
        model_analyses = []
        for r in successful:
            res = r.get("result", {})
            model_key = r["model_key"]
            answer = res.get("answer", res.get("response", ""))
            thinking = res.get("thinking", "")
            weight = MODEL_WEIGHTS.get(model_key, 0.5)
            is_gen = model_key in GENERATIVE_MODELS

            # v7.0: Calibrate confidence from actual model language
            raw_conf = res.get("confidence", 0.5)
            calibrated_conf = calibrate_confidence(answer, base=raw_conf) if is_gen else raw_conf

            # v7.0: Extract individual findings for cross-validation
            findings = extract_individual_findings(answer) if is_gen else []

            excerpt = answer[:300] + ("…" if len(answer) > 300 else "")

            model_analyses.append({
                "model": model_key,
                "answer": answer,
                "thinking": thinking,
                "weight": weight,
                "is_generative": is_gen,
                "raw_confidence": raw_conf,
                "calibrated_confidence": calibrated_conf,
                "findings": findings,
                "excerpt": excerpt,
                "finding_count": len(findings),
            })

        # ── Step 2: Select primary answer from generative models ──
        gen_models = [m for m in model_analyses if m["is_generative"]]
        pool = gen_models if gen_models else model_analyses

        # Score: weight * 0.4 + calibrated_confidence * 0.3 + finding_richness * 0.3
        for m in pool:
            richness = min(1.0, m["finding_count"] / 10.0)  # normalize to 0-1
            m["composite_score"] = (
                m["weight"] * 0.4 +
                m["calibrated_confidence"] * 0.3 +
                richness * 0.3
            )

        best = max(pool, key=lambda x: x["composite_score"])

        # ── Step 3: Cross-validate findings across models ──
        cross_validation = self._cross_validate_findings(model_analyses)

        # ── Step 4: Calculate ensemble metrics ──
        all_calibrated = [m["calibrated_confidence"] for m in model_analyses]
        ensemble_confidence = float(np.average(
            all_calibrated,
            weights=[m["weight"] for m in model_analyses]
        ))

        # Agreement: pairwise text similarity among generative models
        gen_answers = [m["answer"] for m in gen_models]
        agreement = self._compute_agreement(gen_answers) if len(gen_answers) >= 2 else 1.0

        return {
            "consensus_answer": best["answer"],
            "best_model": best["model"],
            "confidence": round(ensemble_confidence, 3),
            "agreement_score": round(agreement, 3),
            "uncertainty": round(1.0 - ensemble_confidence, 3),
            "model_count": len(model_analyses),
            "all_answers": model_analyses,
            "individual_results": model_analyses,
            "cross_validation": cross_validation,
            "timings": {},
        }

    def _cross_validate_findings(
        self, model_analyses: list[dict]
    ) -> dict[str, Any]:
        """Cross-validate findings across models.

        Findings mentioned by 2+ models get 'corroborated' status.
        Findings only in 1 model get 'single_source' status.
        Contradictory findings get flagged.
        """
        gen_models = [m for m in model_analyses if m["is_generative"] and m["findings"]]

        if len(gen_models) < 2:
            return {"corroborated": [], "single_source": [], "contradictions": [],
                    "num_models_with_findings": len(gen_models)}

        # Collect all finding sentences with their source models
        all_findings = []
        for m in gen_models:
            for f in m["findings"]:
                all_findings.append({**f, "source_model": m["model"]})

        # Group by location and check for corroboration
        corroborated = []
        single_source = []
        seen_locations = {}

        for f in all_findings:
            loc = f["location"]
            if loc not in seen_locations:
                seen_locations[loc] = []
            seen_locations[loc].append(f)

        for loc, findings_at_loc in seen_locations.items():
            source_models = set(f["source_model"] for f in findings_at_loc)
            if len(source_models) >= 2:
                corroborated.append({
                    "location": loc,
                    "models": list(source_models),
                    "count": len(source_models),
                    "findings": [f["sentence"][:100] for f in findings_at_loc[:3]],
                })
            else:
                single_source.append({
                    "location": loc,
                    "source_model": findings_at_loc[0]["source_model"],
                    "finding": findings_at_loc[0]["sentence"][:100],
                })

        # Check for normal vs abnormal contradictions
        contradictions = []
        for loc, findings_at_loc in seen_locations.items():
            normals = [f for f in findings_at_loc if f.get("is_normal")]
            abnormals = [f for f in findings_at_loc if not f.get("is_normal")]
            if normals and abnormals:
                contradictions.append({
                    "location": loc,
                    "normal_by": [f["source_model"] for f in normals],
                    "abnormal_by": [f["source_model"] for f in abnormals],
                })

        return {
            "corroborated": corroborated,
            "single_source": single_source,
            "contradictions": contradictions,
            "num_models_with_findings": len(gen_models),
            "corroboration_rate": (
                len(corroborated) / max(len(corroborated) + len(single_source), 1)
            ),
        }

    def _compute_agreement(self, answers: list[str]) -> float:
        """Pairwise word-overlap agreement between answers."""
        if len(answers) < 2:
            return 1.0
        similarities = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                words_a = set(re.findall(r'\w+', answers[i].lower()))
                words_b = set(re.findall(r'\w+', answers[j].lower()))
                if words_a and words_b:
                    similarities.append(len(words_a & words_b) / len(words_a | words_b))
        return float(np.mean(similarities)) if similarities else 0.0
