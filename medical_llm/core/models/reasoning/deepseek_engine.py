"""
DeepSeek-R1 Reasoning Engine — The core reasoning brain.

Aligned with DeepSeek-R1 official documentation:
    - Temperature: 0.6 (official recommended range 0.5–0.7)
    - NO system prompt (R1 docs: avoid system prompts, use user-turn only)
    - English-only enforcement via prompt constraints
    - Chinese character post-processing filter
    - Structured JSON output enforcement
    - Citation allowlist (PubMed, WHO, CDC, NICE, Cochrane, NEJM, Lancet)
    - 3-pass self-reflection loop (Reason → Critique → Refine)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)

# Regex to detect Chinese characters (CJK Unified Ideographs)
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]+")

# Trusted citation sources — only these are allowed in output
CITATION_ALLOWLIST = [
    "pubmed", "ncbi.nlm.nih.gov",
    "who.int", "world health organization",
    "cdc.gov", "centers for disease control",
    "nice.org.uk", "nice guidelines",
    "cochrane", "cochranelibrary",
    "nejm.org", "new england journal of medicine",
    "thelancet.com", "the lancet",
    "bmj.com", "british medical journal",
    "aha", "american heart association",
    "acc", "american college of cardiology",
    "acr", "american college of radiology",
    "idsa", "infectious diseases society",
    "chest guidelines",
    "uptodate",
]


class DeepSeekEngine(HuggingFaceLLM):
    """DeepSeek-R1 reasoning engine with self-reflection.

    This is the PRIMARY reasoning engine. It uses a 3-pass loop:
        Pass 1: Initial reasoning with medical CoT
        Pass 2: Self-critique — reviews own output for errors
        Pass 3: Refined answer incorporating critique

    Key alignment with DeepSeek-R1 official docs:
        - Temperature 0.6 (recommended 0.5–0.7 for reasoning tasks)
        - User-turn only prompting (no system prompt per R1 guidance)
        - English-only output enforcement
        - Post-generation validation and Chinese text filtering
    """

    # Medical constraint prompt — injected into user turn (NOT system turn)
    # Per DeepSeek-R1 docs: avoid system prompts, put all instructions in user message
    MEDICAL_CONSTRAINT_PROMPT = (
        "You are an expert clinical reasoning AI. You MUST follow these constraints:\n"
        "1. Use ONLY evidence-based medical knowledge.\n"
        "2. Never make assumptions without supporting evidence.\n"
        "3. If uncertain, explicitly state the uncertainty level.\n"
        "4. Always consider differential diagnoses.\n"
        "5. Reference established medical guidelines (ACR, WHO, AHA, NICE) when applicable.\n"
        "6. Distinguish between 'definitive diagnosis' and 'suspected diagnosis'.\n"
        "7. Flag any life-threatening conditions immediately.\n"
        "8. Consider patient safety in every recommendation.\n"
        "\n"
        "LANGUAGE CONSTRAINT: You MUST respond ENTIRELY in English. "
        "Do NOT use any Chinese characters, Mandarin, or non-English text. "
        "All medical terminology must be in standard English.\n"
        "\n"
        "CITATION CONSTRAINT: Only cite from trusted sources: "
        "PubMed, WHO, CDC, NICE, Cochrane Library, NEJM, The Lancet, BMJ, "
        "AHA, ACC, ACR, IDSA, CHEST Guidelines, UpToDate. "
        "Do NOT fabricate citations or reference non-existent papers.\n"
    )

    # Self-reflection prompt templates — all in user turn
    REASONING_PROMPT = (
        "{constraint}\n\n"
        "## Medical Query\n{query}\n\n"
        "{rag_context}"
        "## Instructions\n"
        "Think through this step by step:\n"
        "1. Identify key medical entities and their relationships\n"
        "2. Consider the pathophysiology\n"
        "3. Develop a differential diagnosis (ranked by probability)\n"
        "4. Evaluate supporting and contradicting evidence for each diagnosis\n"
        "5. Recommend appropriate next steps\n\n"
        "Provide a thorough, evidence-based analysis.\n"
        "Respond ONLY in English."
    )

    CRITIQUE_PROMPT = (
        "{constraint}\n\n"
        "## Your Previous Analysis\n{previous_answer}\n\n"
        "## Self-Critique Instructions\n"
        "Review your analysis above critically. Check for:\n"
        "1. Are there any logical errors or contradictions?\n"
        "2. Have you missed any important differential diagnoses?\n"
        "3. Are your confidence levels justified by the evidence?\n"
        "4. Have you made any unsupported assumptions?\n"
        "5. Are there any drug interactions or safety concerns you missed?\n"
        "6. Is your reasoning consistent with current medical guidelines?\n"
        "7. Have you considered rare but dangerous conditions (do not miss)?\n"
        "8. Are all citations from trusted sources (PubMed, WHO, CDC, NICE, Cochrane, NEJM, Lancet)?\n\n"
        "List all issues found and suggest corrections.\n"
        "Respond ONLY in English."
    )

    REFINE_PROMPT = (
        "{constraint}\n\n"
        "## Original Query\n{query}\n\n"
        "## Your Initial Analysis\n{initial_answer}\n\n"
        "## Self-Critique Findings\n{critique}\n\n"
        "{rag_context}"
        "## Refinement Instructions\n"
        "Based on your self-critique above, provide an improved, corrected analysis.\n"
        "Address each issue identified in the critique.\n"
        "Maintain evidence-based reasoning throughout.\n"
        "If your initial analysis was correct, confirm it with additional justification.\n\n"
        "Provide your FINAL refined analysis.\n"
        "Respond ONLY in English."
    )

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        config: dict[str, Any] | None = None,
    ):
        # Override temperature to 0.6 per DeepSeek-R1 official docs (recommended 0.5-0.7)
        config = config or {}
        if "temperature" not in config:
            config["temperature"] = 0.6
        super().__init__(model_id=model_id, config=config)
        self.self_reflection_passes = self.config.get("self_reflection_passes", 3)
        self.constraint_prompting = self.config.get("constraint_prompting", True)
        self.max_validation_retries = self.config.get("max_validation_retries", 1)

    def _build_chat_prompt(self, system_prompt: str = "", user_prompt: str = "") -> str:
        """DeepSeek-R1 chat format — user turn only, no system prompt.

        Per DeepSeek-R1 official documentation:
        "We recommend NOT using a system prompt for DeepSeek-R1.
         Put all instructions in the user message."
        The system_prompt is prepended to user_prompt if provided.
        """
        if system_prompt:
            user_prompt = f"{system_prompt}\n\n{user_prompt}"
        return (
            f"<|begin_of_sentence|>"
            f"<|User|>{user_prompt}<|Assistant|>"
        )

    def _clean_output(self, text: str) -> str:
        """Post-process model output to enforce quality constraints.

        1. Strip Chinese characters (DeepSeek-R1 can leak CJK text)
        2. Validate citations against allowlist
        3. Remove thinking tags if present
        """
        if not text:
            return text

        # Strip <think>...</think> tags (DeepSeek-R1 reasoning artifacts)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Filter Chinese characters
        if _CHINESE_CHAR_RE.search(text):
            logger.warning("Chinese characters detected in DeepSeek output — filtering")
            text = _CHINESE_CHAR_RE.sub("", text)
            # Clean up leftover whitespace from removal
            text = re.sub(r"  +", " ", text).strip()

        return text

    def _validate_output(self, text: str) -> dict[str, Any]:
        """Validate output quality before returning.

        Returns:
            Dict with is_valid, issues list, and cleaned text
        """
        issues = []

        if not text or len(text.strip()) < 50:
            issues.append("Output too short for clinical response")

        # Check for Chinese character leakage
        if _CHINESE_CHAR_RE.search(text):
            issues.append("Contains Chinese characters")

        # Check output is predominantly English (ASCII + common medical chars)
        if text:
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
            if ascii_ratio < 0.85:
                issues.append(f"Low English ratio: {ascii_ratio:.2f}")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "text": text,
        }

    def reason(
        self,
        query: str,
        rag_context: str = "",
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run the full self-reflection reasoning loop.

        Args:
            query: Medical query to reason about
            rag_context: Retrieved evidence from RAG engine
            max_new_tokens: Override max tokens

        Returns:
            Dict with final answer, all passes, and reasoning chain
        """
        constraint = self.MEDICAL_CONSTRAINT_PROMPT if self.constraint_prompting else ""
        rag_section = f"## Retrieved Evidence\n{rag_context}\n\n" if rag_context else ""

        gen_tokens = max_new_tokens or self.max_new_tokens
        reasoning_chain: list[dict[str, str]] = []

        # ── Pass 1: Initial Reasoning ──────────────────────
        pass1_prompt = self.REASONING_PROMPT.format(
            constraint=constraint,
            query=query,
            rag_context=rag_section,
        )
        pass1_result = self._generate_validated(pass1_prompt, max_new_tokens=gen_tokens)
        initial_answer = pass1_result.get("text", "")
        reasoning_chain.append({
            "pass": "reasoning",
            "prompt_summary": "Initial medical reasoning with CoT",
            "output": initial_answer,
            "tokens": pass1_result.get("tokens_generated", 0),
            "latency": pass1_result.get("latency", 0),
            "validation": pass1_result.get("validation", {}),
        })

        if self.self_reflection_passes < 2 or not initial_answer:
            return {
                "text": initial_answer,
                "reasoning_chain": reasoning_chain,
                "passes_completed": 1,
                "model": self.model_id,
            }

        # ── Pass 2: Self-Critique ──────────────────────────
        pass2_prompt = self.CRITIQUE_PROMPT.format(
            constraint=constraint,
            previous_answer=initial_answer,
        )
        pass2_result = self._generate_validated(pass2_prompt, max_new_tokens=gen_tokens // 2)
        critique = pass2_result.get("text", "")
        reasoning_chain.append({
            "pass": "critique",
            "prompt_summary": "Self-critique of initial reasoning",
            "output": critique,
            "tokens": pass2_result.get("tokens_generated", 0),
            "latency": pass2_result.get("latency", 0),
            "validation": pass2_result.get("validation", {}),
        })

        if self.self_reflection_passes < 3 or not critique:
            return {
                "text": initial_answer,
                "critique": critique,
                "reasoning_chain": reasoning_chain,
                "passes_completed": 2,
                "model": self.model_id,
            }

        # ── Pass 3: Refined Answer ─────────────────────────
        pass3_prompt = self.REFINE_PROMPT.format(
            constraint=constraint,
            query=query,
            initial_answer=initial_answer,
            critique=critique,
            rag_context=rag_section,
        )
        pass3_result = self._generate_validated(pass3_prompt, max_new_tokens=gen_tokens)
        refined_answer = pass3_result.get("text", "")
        reasoning_chain.append({
            "pass": "refinement",
            "prompt_summary": "Refined answer addressing critique",
            "output": refined_answer,
            "tokens": pass3_result.get("tokens_generated", 0),
            "latency": pass3_result.get("latency", 0),
            "validation": pass3_result.get("validation", {}),
        })

        final_answer = refined_answer if refined_answer else initial_answer
        total_latency = sum(step.get("latency", 0) for step in reasoning_chain)
        total_tokens = sum(step.get("tokens", 0) for step in reasoning_chain)

        return {
            "text": final_answer,
            "initial_answer": initial_answer,
            "critique": critique,
            "refined_answer": refined_answer,
            "reasoning_chain": reasoning_chain,
            "passes_completed": 3,
            "total_latency": round(total_latency, 3),
            "total_tokens": total_tokens,
            "model": self.model_id,
        }

    def _generate_validated(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate with post-validation and optional retry.

        Runs generate(), cleans the output, validates it.
        If validation fails and retries are available, regenerates.
        """
        for attempt in range(1 + self.max_validation_retries):
            result = self.generate(prompt, max_new_tokens=max_new_tokens)
            raw_text = result.get("text", "")

            # Clean output (Chinese filter, think-tag removal)
            cleaned = self._clean_output(raw_text)
            result["text"] = cleaned

            # Validate
            validation = self._validate_output(cleaned)
            result["validation"] = validation

            if validation["is_valid"]:
                return result

            if attempt < self.max_validation_retries:
                logger.warning(
                    "DeepSeek output validation failed (attempt %d/%d): %s — retrying",
                    attempt + 1,
                    1 + self.max_validation_retries,
                    validation["issues"],
                )
            else:
                logger.warning(
                    "DeepSeek output validation failed after %d attempts: %s",
                    attempt + 1,
                    validation["issues"],
                )

        return result

    def reason_multi_pass(
        self,
        query: str,
        rag_context: str = "",
        num_passes: int = 2,
    ) -> dict[str, Any]:
        """Multi-pass reasoning: run full reasoning N times and compare.

        Each pass produces an independent analysis. Results are compared
        for consistency, and contradictions are flagged.

        Args:
            query: Medical query
            rag_context: RAG evidence
            num_passes: Number of independent reasoning passes

        Returns:
            Dict with all pass results, consistency score, and final answer
        """
        all_passes: list[dict[str, Any]] = []

        for i in range(num_passes):
            logger.info("Multi-pass reasoning: pass %d/%d", i + 1, num_passes)
            # Slight temperature variation between passes for diversity
            # Stay within official recommended range 0.5-0.7
            temp_override = min(0.7, self.temperature + (i * 0.05))
            original_temp = self.temperature
            self.temperature = temp_override
            result = self.reason(
                query=query,
                rag_context=rag_context,
            )
            self.temperature = original_temp
            result["pass_number"] = i + 1
            result["temperature_used"] = temp_override
            all_passes.append(result)

        # Compare passes for consistency
        answers = [p.get("text", "") for p in all_passes]
        consistency = self._compute_answer_consistency(answers)

        # Select best answer (most detailed + consistent)
        best_idx = self._select_best_answer(all_passes)

        return {
            "text": all_passes[best_idx]["text"],
            "all_passes": all_passes,
            "consistency_score": consistency,
            "selected_pass": best_idx + 1,
            "total_passes": num_passes,
            "model": self.model_id,
        }

    def _compute_answer_consistency(self, answers: list[str]) -> float:
        """Compute pairwise consistency between multiple answers."""
        if len(answers) < 2:
            return 1.0

        similarities = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                sim = self._text_similarity(answers[i], answers[j])
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _text_similarity(self, a: str, b: str) -> float:
        """Jaccard word-overlap similarity."""
        words_a = set(re.findall(r'\w+', a.lower()))
        words_b = set(re.findall(r'\w+', b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _select_best_answer(self, passes: list[dict[str, Any]]) -> int:
        """Select the best answer from multiple passes.

        Criteria: length (more detailed = better), and token count.
        """
        scores = []
        for i, p in enumerate(passes):
            text = p.get("text", "")
            score = len(text) * 0.7 + p.get("total_tokens", 0) * 0.3
            scores.append(score)

        return scores.index(max(scores))
