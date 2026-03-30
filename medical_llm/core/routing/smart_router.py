"""
Smart Router — Intelligent query-to-model routing engine.
Determines which models to activate based on query classification,
complexity estimation, and available resources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class SmartRouter:
    """Routes medical queries to the optimal model ensemble.

    Routing strategy:
        1. Receives classified query (from QueryClassifier)
        2. Looks up routing table for the query category
        3. Adjusts model selection based on complexity and available models
        4. Returns execution plan with model keys, roles, and priorities
    """

    # Default routing table — overridable via pipeline_config.yaml
    DEFAULT_ROUTING_TABLE: dict[str, dict[str, list[str]]] = {
        "simple_qa": {
            "primary": ["biomistral_7b"],
            "medical": [],
            "verifier": [],
            "reasoning": [],
        },
        "diagnosis": {
            "primary": ["deepseek_r1", "openbiollm_70b"],
            "medical": ["meditron_70b"],
            "verifier": ["clinical_camel_70b", "med42_70b"],
            "reasoning": ["deepseek_r1"],
        },
        "differential": {
            "primary": ["deepseek_r1", "openbiollm_70b"],
            "medical": ["meditron_70b"],
            "verifier": ["med42_70b"],
            "reasoning": ["deepseek_r1"],
        },
        "treatment": {
            "primary": ["openbiollm_70b", "meditron_70b"],
            "medical": ["biomistral_7b"],
            "verifier": ["clinical_camel_70b", "med42_70b"],
            "reasoning": [],
        },
        "drug_info": {
            "primary": ["biomistral_7b", "openbiollm_70b"],
            "medical": [],
            "verifier": ["med42_70b"],
            "reasoning": [],
        },
        "lab_interpretation": {
            "primary": ["openbiollm_70b", "biomistral_7b"],
            "medical": ["meditron_70b"],
            "verifier": [],
            "reasoning": [],
        },
        "research": {
            "primary": ["pmc_llama_13b", "openbiollm_70b"],
            "medical": [],
            "verifier": [],
            "reasoning": [],
        },
        "conversational": {
            "primary": ["mellama_13b", "chatdoctor"],
            "medical": [],
            "verifier": [],
            "reasoning": [],
        },
        "emergency": {
            "primary": ["deepseek_r1", "openbiollm_70b"],
            "medical": ["meditron_70b"],
            "verifier": ["clinical_camel_70b", "med42_70b"],
            "reasoning": ["deepseek_r1"],
        },
    }

    # Complexity escalation: add more models for complex queries
    COMPLEXITY_ESCALATION = {
        "simple": {
            "max_primary": 1,
            "enable_verifier": False,
            "enable_reasoning": False,
            "enable_rag": False,
        },
        "standard": {
            "max_primary": 2,
            "enable_verifier": True,
            "enable_reasoning": True,
            "enable_rag": True,
        },
        "complex": {
            "max_primary": 4,
            "enable_verifier": True,
            "enable_reasoning": True,
            "enable_rag": True,
        },
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.available_models: set[str] = set()
        self.min_models = self.config.get("min_models_per_task", 1)
        self.max_models = self.config.get("max_models_per_task", 6)

        # Load custom routing table from config if provided
        self.routing_table = self.config.get(
            "routing_table", self.DEFAULT_ROUTING_TABLE
        )

    def register_model(self, model_key: str) -> None:
        """Register a model as available for routing."""
        self.available_models.add(model_key)
        logger.info(f"Router: Registered model '{model_key}'")

    def register_models(self, model_keys: list[str]) -> None:
        """Register multiple models."""
        for key in model_keys:
            self.register_model(key)

    def route(
        self,
        query_category: str,
        complexity: str = "standard",
        enable_rag: bool = True,
        force_models: list[str] | None = None,
    ) -> dict[str, Any]:
        """Determine which models to use for a query.

        Args:
            query_category: Classification from QueryClassifier
            complexity: "simple", "standard", or "complex"
            enable_rag: Whether to include RAG in the pipeline
            force_models: Optional override to force specific models

        Returns:
            Execution plan with model assignments by role
        """
        # Handle forced model selection
        if force_models:
            return {
                "primary": force_models,
                "medical": [],
                "verifier": [],
                "reasoning": [],
                "enable_rag": enable_rag,
                "enable_self_reflection": False,
                "complexity": complexity,
                "source": "forced",
            }

        # Look up routing table
        route_key = query_category if query_category in self.routing_table else "simple_qa"
        base_route = {
            k: list(v) for k, v in self.routing_table[route_key].items()
        }

        # Apply complexity escalation
        escalation = self.COMPLEXITY_ESCALATION.get(
            complexity, self.COMPLEXITY_ESCALATION["standard"]
        )

        # Trim or expand based on complexity
        if not escalation["enable_verifier"]:
            base_route["verifier"] = []
        if not escalation["enable_reasoning"]:
            base_route["reasoning"] = []

        max_primary = escalation["max_primary"]
        base_route["primary"] = base_route["primary"][:max_primary]

        # Filter to available models only
        if self.available_models:
            filtered = {}
            for role, models in base_route.items():
                filtered[role] = [m for m in models if m in self.available_models]
            base_route = filtered

        # Ensure at least one model
        total_models = sum(len(v) for v in base_route.values())
        if total_models == 0 and self.available_models:
            # Fallback: use any available model
            fallback = next(iter(self.available_models))
            base_route["primary"] = [fallback]
            logger.warning(f"No models matched route. Falling back to: {fallback}")

        # Determine special pipeline features
        enable_self_reflection = (
            "deepseek_r1" in base_route.get("reasoning", [])
            and complexity != "simple"
        )
        enable_rag_query = enable_rag and escalation.get("enable_rag", True)

        result = {
            **base_route,
            "enable_rag": enable_rag_query,
            "enable_self_reflection": enable_self_reflection,
            "complexity": complexity,
            "source": "routing_table",
            "query_category": query_category,
        }

        logger.info(
            f"Router: {query_category} ({complexity}) → "
            f"primary={base_route.get('primary', [])} "
            f"medical={base_route.get('medical', [])} "
            f"verifier={base_route.get('verifier', [])} "
            f"rag={enable_rag_query}"
        )

        return result


class FallbackManager:
    """Manages model fallback chains when primary models fail.

    Degradation strategy: larger models fall back to smaller ones.
    Tracks failure counts and auto-disables unstable models.
    """

    # Degradation chains: big → small
    FALLBACK_CHAINS: dict[str, list[str]] = {
        "deepseek_r1": ["openbiollm_70b", "meditron_70b", "biomistral_7b"],
        "openbiollm_70b": ["meditron_70b", "med42_70b", "biomistral_7b"],
        "meditron_70b": ["openbiollm_70b", "biomistral_7b"],
        "med42_70b": ["clinical_camel_70b", "biomistral_7b"],
        "clinical_camel_70b": ["med42_70b", "biomistral_7b"],
        "mellama_13b": ["chatdoctor", "biomistral_7b"],
        "pmc_llama_13b": ["biomistral_7b"],
        "chatdoctor": ["mellama_13b", "biomistral_7b"],
        "biomistral_7b": [],  # Last resort, no fallback
    }

    # Maximum consecutive failures before auto-disable
    MAX_FAILURES_BEFORE_DISABLE = 5

    def __init__(self):
        self.failure_counts: dict[str, int] = {}
        self.disabled_models: set[str] = set()

    def get_fallback(self, failed_model: str) -> str | None:
        """Get next fallback model for a failed model.

        Args:
            failed_model: Key of the model that failed

        Returns:
            Key of fallback model, or None if no fallback available
        """
        chain = self.FALLBACK_CHAINS.get(failed_model, [])
        for fallback in chain:
            if fallback not in self.disabled_models:
                if self.failure_counts.get(fallback, 0) < self.MAX_FAILURES_BEFORE_DISABLE:
                    logger.info(f"Fallback: {failed_model} → {fallback}")
                    return fallback

        logger.warning(f"No fallback available for {failed_model}")
        return None

    def record_failure(self, model_key: str) -> None:
        """Record a model failure."""
        self.failure_counts[model_key] = self.failure_counts.get(model_key, 0) + 1
        count = self.failure_counts[model_key]

        if count >= self.MAX_FAILURES_BEFORE_DISABLE:
            self.disabled_models.add(model_key)
            logger.error(
                f"Model {model_key} auto-disabled after {count} consecutive failures"
            )
        else:
            logger.warning(f"Model {model_key} failed (count: {count})")

    def record_success(self, model_key: str) -> None:
        """Reset failure count on success."""
        self.failure_counts[model_key] = 0
        self.disabled_models.discard(model_key)

    def get_status(self) -> dict[str, Any]:
        """Get status of all models."""
        return {
            "failure_counts": dict(self.failure_counts),
            "disabled_models": list(self.disabled_models),
        }

    def reset(self) -> None:
        """Reset all failure tracking."""
        self.failure_counts.clear()
        self.disabled_models.clear()
