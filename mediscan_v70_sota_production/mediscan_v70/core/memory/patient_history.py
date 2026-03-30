"""
MediScan AI v7.0 — Patient History & Case Tracking
Maintains patient analysis history for longitudinal tracking.
"""
from __future__ import annotations


import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PatientHistory:
    """Tracks patient analysis history for comparison and longitudinal care."""

    def __init__(self, storage_dir: str = "./data/patient_history"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_patient_id(patient_id: str) -> str:
        """Sanitize patient_id to prevent path traversal and OS-reserved names."""
        import hashlib
        import re
        # Strip any path separators and null bytes
        safe = patient_id.replace("/", "").replace("\\", "").replace("\x00", "")
        # Reject if empty or matches Windows reserved names
        _RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
            f"{d}{n}" for d in ("COM", "LPT") for n in range(1, 10)
        }
        if not safe or safe.upper().split(".")[0] in _RESERVED:
            # Fall back to a hash of the original ID
            safe = hashlib.sha256(patient_id.encode()).hexdigest()[:24]
        # Only allow alphanumeric, dash, underscore, dot
        safe = re.sub(r"[^\w\-.]", "_", safe)
        return safe

    def add_record(
        self,
        patient_id: str,
        report: dict[str, Any],
        anonymize: bool = True,
    ) -> str:
        """Add an analysis record to patient history."""
        record = {
            "record_id": report.get("report_id", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "modality": report.get("study", {}).get("modality", "unknown"),
            "risk_level": report.get("governance", {}).get("risk_level", "routine"),
            "confidence": report.get("ai_metadata", {}).get("confidence", 0),
            "impression": report.get("clinical_report", {}).get("impression", ""),
            "models_used": report.get("ai_metadata", {}).get("models_used", []),
        }

        if anonymize:
            record.pop("patient_name", None)

        # Sanitize patient_id to prevent path traversal
        safe_id = self._sanitize_patient_id(patient_id)
        patient_file = self.storage_dir / f"{safe_id}.jsonl"
        # Verify the resolved path is still within our storage directory
        if not patient_file.resolve().is_relative_to(self.storage_dir.resolve()):
            raise ValueError(f"Invalid patient_id: path traversal detected")
        with open(patient_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info("Patient history updated: %s", safe_id)
        return record["record_id"]

    def get_history(
        self, patient_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Retrieve patient analysis history."""
        safe_id = self._sanitize_patient_id(patient_id)
        patient_file = self.storage_dir / f"{safe_id}.jsonl"
        if not patient_file.exists():
            return []

        records = []
        with open(patient_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Return most recent records
        return records[-limit:]

    def get_comparison_context(self, patient_id: str) -> str:
        """Generate comparison context from previous studies for the report."""
        history = self.get_history(patient_id, limit=5)
        if not history:
            return "No prior studies available for comparison."

        last = history[-1]
        return (
            f"Comparison: {last.get('modality', 'Prior')} study from "
            f"{last.get('timestamp', 'unknown date')}. "
            f"Previous impression: {last.get('impression', 'N/A')[:200]}"
        )


class CaseTracker:
    """Tracks cases for the doctor dashboard."""

    def __init__(self, storage_dir: str = "./data/cases"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_case(
        self,
        report: dict[str, Any],
        assigned_doctor: str = "",
    ) -> dict[str, Any]:
        """Create a new case from a report."""
        case = {
            "case_id": report.get("report_id", ""),
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending_review",
            "assigned_doctor": assigned_doctor,
            "patient_id": report.get("patient", {}).get("patient_id", ""),
            "modality": report.get("study", {}).get("modality", ""),
            "risk_level": report.get("governance", {}).get("risk_level", "routine"),
            "ai_confidence": report.get("ai_metadata", {}).get("confidence", 0),
            "impression": report.get("clinical_report", {}).get("impression", ""),
            "has_critical_findings": report.get("governance", {}).get(
                "critical_findings", []
            ) != [],
        }

        case_file = self.storage_dir / f"{case['case_id']}.json"
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2)

        logger.info(f"Case created: {case['case_id']} | Risk: {case['risk_level']}")
        return case

    def update_status(self, case_id: str, status: str, notes: str = "") -> bool:
        """Update case status (pending_review → reviewed → confirmed → archived)."""
        case_file = self.storage_dir / f"{case_id}.json"
        if not case_file.exists():
            return False

        with open(case_file, "r") as f:
            case = json.load(f)

        case["status"] = status
        case["updated_at"] = datetime.utcnow().isoformat()
        if notes:
            case.setdefault("notes", []).append({
                "timestamp": datetime.utcnow().isoformat(),
                "text": notes,
            })

        with open(case_file, "w") as f:
            json.dump(case, f, indent=2)

        return True

    def list_cases(
        self, status: Optional[str] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List cases, optionally filtered by status."""
        cases = []
        for case_file in sorted(self.storage_dir.glob("*.json"), reverse=True):
            try:
                with open(case_file, "r") as f:
                    case = json.load(f)
                if status is None or case.get("status") == status:
                    cases.append(case)
                if len(cases) >= limit:
                    break
            except Exception as e:  # noqa: broad-except logged
                continue
        return cases
