"""
Governance & Audit — Logging, audit trails, and compliance tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditLogger:
    """HIPAA-compliant audit logging for all pipeline operations.

    Logs:
    - Query metadata (anonymized)
    - Models used and their outputs (summarized)
    - Confidence scores and agreement levels
    - Safety validation results
    - Drug interaction alerts
    - Timestamps and execution details
    """

    def __init__(
        self,
        log_dir: str = "./logs/audit",
        enabled: bool = True,
        enable_detailed: bool = True,
        enable_console: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_detailed = enable_detailed
        self.enable_console = enable_console
        self._log_count = 0

    def log_analysis(
        self,
        query: str,
        query_category: str,
        routing_decision: dict[str, Any],
        model_results: list[dict[str, Any]],
        fused_result: dict[str, Any],
        safety_result: dict[str, Any],
        report: dict[str, Any],
        execution_time: float,
        **extra_fields,
    ) -> str:
        """Log a complete analysis pipeline execution.

        Args:
            query: Original query (may be truncated for privacy)
            query_category: Classification result
            routing_decision: Router output
            model_results: All model outputs
            fused_result: Fusion output
            safety_result: Safety validation output
            report: Final report
            execution_time: Total pipeline time

        Returns:
            Audit entry ID
        """
        if not self.enabled:
            return "audit_disabled"

        timestamp = datetime.now(timezone.utc)
        entry_id = f"audit_{timestamp.strftime('%Y%m%d_%H%M%S')}_{self._log_count:04d}"
        self._log_count += 1

        audit_entry = {
            "audit_id": entry_id,
            "timestamp": timestamp.isoformat(),
            "query_hash": self._hash_query(query),
            "query_word_count": len(query.split()),  # SEC-3: No raw query text logged
            "query_category": query_category,
            "routing": {
                "primary_models": routing_decision.get("primary", []),
                "medical_models": routing_decision.get("medical", []),
                "verifier_models": routing_decision.get("verifier", []),
                "rag_enabled": routing_decision.get("enable_rag", False),
                "self_reflection": routing_decision.get("enable_self_reflection", False),
            },
            "execution": {
                "models_used": [r.get("model", "unknown") for r in model_results],
                "total_models": len(model_results),
                "total_tokens": sum(r.get("tokens_generated", 0) for r in model_results),
                "execution_time_seconds": round(execution_time, 3),
            },
            "fusion": {
                "best_model": fused_result.get("best_model"),
                "confidence": fused_result.get("confidence"),
                "agreement": fused_result.get("agreement_score"),
                "strategy": fused_result.get("strategy"),
            },
            "safety": {
                "risk_level": safety_result.get("risk_level"),
                "flagged_findings": len(safety_result.get("flagged_findings", [])),
                "is_valid": safety_result.get("is_valid", True),
                "drug_interactions": self._count_drug_interactions(safety_result),
            },
            **extra_fields,
        }

        # Write to file
        log_file = self.log_dir / f"{timestamp.strftime('%Y-%m-%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Audit log write failed: {e}")

        if self.enable_console:
            logger.info(
                f"AUDIT [{entry_id}] cat={query_category} "
                f"conf={fused_result.get('confidence', 0):.3f} "
                f"risk={safety_result.get('risk_level')} "
                f"time={execution_time:.1f}s"
            )

        return entry_id

    def _count_drug_interactions(self, safety_result: dict[str, Any]) -> int:
        """Normalize interaction counts across legacy and current schemas."""
        if "drug_interactions" in safety_result:
            return int(safety_result.get("drug_interactions", 0) or 0)

        drug_check = safety_result.get("drug_check") or {}
        interactions = drug_check.get("interactions") or []
        return len(interactions)

    def _hash_query(self, query: str) -> str:
        """Hash query for privacy-preserving logging."""
        import hashlib
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def get_logs(self, date: str | None = None, limit: int = 100) -> list[dict]:
        """Retrieve audit logs."""
        if not self.enabled:
            return []

        if date:
            log_file = self.log_dir / f"{date}.jsonl"
            if not log_file.exists():
                return []
            files = [log_file]
        else:
            files = sorted(self.log_dir.glob("*.jsonl"), reverse=True)

        entries = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        entries.append(json.loads(line.strip()))
                        if len(entries) >= limit:
                            return entries
            except Exception:
                continue

        return entries
