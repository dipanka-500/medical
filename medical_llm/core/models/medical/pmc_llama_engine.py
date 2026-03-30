"""
PMC-LLaMA — Medical literature comprehension engine.
Specialized for understanding and synthesizing PubMed/PMC research papers.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models.base_model import HuggingFaceLLM

logger = logging.getLogger(__name__)


class PMCLLaMAEngine(HuggingFaceLLM):
    """PMC-LLaMA literature understanding engine.

    Trained on PubMed Central full-text articles. Excels at:
    - Summarizing medical research papers
    - Synthesizing evidence across multiple studies
    - Interpreting clinical trial results
    - Literature-grounded medical reasoning
    """

    SYSTEM_PROMPT = (
        "You are PMC-LLaMA, a medical literature AI trained on PubMed Central articles. "
        "Your expertise is in medical evidence synthesis and research interpretation.\n"
        "Your role:\n"
        "1. Summarize complex research findings accurately\n"
        "2. Evaluate strength of evidence (RCT vs. observational vs. case report)\n"
        "3. Identify key statistics (NNT, NNH, absolute risk reduction)\n"
        "4. Note limitations and potential biases in studies\n"
        "5. Synthesize findings across multiple sources into coherent conclusions\n"
        "6. Distinguish between correlation and causation\n"
        "7. Rate evidence quality (GRADE criteria when applicable)"
    )

    SYNTHESIS_TEMPLATE = (
        "## Research Question\n{query}\n\n"
        "## Retrieved Literature\n{evidence}\n\n"
        "## Task: Evidence Synthesis\n"
        "Please provide:\n\n"
        "### Key Findings\n"
        "- Summarize the main findings from the evidence\n\n"
        "### Evidence Quality\n"
        "- Rate the overall evidence quality (high/moderate/low/very low)\n"
        "- Note study designs and sample sizes\n\n"
        "### Clinical Implications\n"
        "- What do these findings mean for clinical practice?\n\n"
        "### Limitations\n"
        "- Key limitations and gaps in the evidence\n\n"
        "### Recommendations\n"
        "- Evidence-based recommendations with confidence levels"
    )

    def __init__(
        self,
        model_id: str = "axiong/PMC_LLaMA_13B",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(model_id=model_id, config=config or {})

    def synthesize_evidence(
        self,
        query: str,
        evidence: str = "",
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Synthesize medical evidence from retrieved literature.

        Args:
            query: Research question
            evidence: Retrieved papers/abstracts from RAG

        Returns:
            Evidence synthesis result
        """
        prompt = self.SYNTHESIS_TEMPLATE.format(
            query=query,
            evidence=evidence or "No specific literature retrieved. Answer based on your training data.",
        )

        result = self.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
        )

        return {
            "text": result.get("text", ""),
            "model": self.model_id,
            "role": "literature",
            "tokens_generated": result.get("tokens_generated", 0),
            "latency": result.get("latency", 0),
        }
