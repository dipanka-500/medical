"""
MediScan AI v5.0 — Security: HIPAA Compliance & Access Control
"""
from __future__ import annotations


import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Persistent salt for deterministic de-identification within a deployment.
# MEDISCAN_PHI_SALT MUST be set in production for consistent anonymized IDs.
_PHI_SALT = os.environ.get("MEDISCAN_PHI_SALT", "")
if not _PHI_SALT:
    logger.warning(
        "MEDISCAN_PHI_SALT is not set — anonymized patient IDs will NOT be "
        "consistent across process restarts. Set this env var in production."
    )
    _PHI_SALT = secrets.token_hex(16)


class HIPAACompliance:
    """HIPAA-compliant data handling for medical information."""

    PHI_FIELDS = [
        "patient_name", "patient_id", "birth_date", "address",
        "phone", "email", "ssn", "medical_record_number",
        "health_plan_number", "account_number",
    ]

    def anonymize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove or hash PHI (Protected Health Information) from data."""
        sanitized = data.copy()

        for field in self.PHI_FIELDS:
            if field in sanitized:
                sanitized[field] = self._hash_value(str(sanitized[field]))

        # Recursively sanitize nested dicts
        for key, value in sanitized.items():
            if isinstance(value, dict):
                sanitized[key] = self.anonymize(value)

        return sanitized

    def _hash_value(self, value: str) -> str:
        """Hash a PHI value for de-identification.

        Uses HMAC-SHA256 with a deployment-wide salt to prevent
        rainbow table attacks. Returns a prefixed string so that
        check_compliance() can distinguish anonymized from raw values.
        """
        import hmac
        mac = hmac.new(_PHI_SALT.encode(), value.encode(), hashlib.sha256)
        return f"ANON:{mac.hexdigest()[:32]}"

    def check_compliance(self, data: dict[str, Any]) -> dict[str, Any]:
        """Check if data handling meets HIPAA requirements."""
        issues = []

        # Check for unencrypted PHI
        flat_data = json.dumps(data)
        for field in self.PHI_FIELDS:
            if field in data and len(str(data[field])) > 0:
                raw_value = str(data[field])
                # Anonymized values start with "ANON:" prefix
                if not raw_value.startswith("ANON:") and len(raw_value) > 2:
                    issues.append(f"Unprotected PHI: {field}")

        return {
            "is_compliant": len(issues) == 0,
            "issues": issues,
            "checked_at": datetime.utcnow().isoformat(),
        }


class AccessControl:
    """Role-based access control for the API."""

    ROLES = {
        "admin": ["read", "write", "delete", "manage_users", "view_audit"],
        "doctor": ["read", "write", "view_patients"],
        "radiologist": ["read", "write", "review_cases"],
        "nurse": ["read", "view_patients"],
        "patient": ["read_own"],
        "api": ["read", "write"],
    }

    def __init__(self):
        self._tokens: dict[str, dict] = {}

    def generate_token(self, user_id: str, role: str) -> str:
        """Generate an access token for a user."""
        if role not in self.ROLES:
            raise ValueError(f"Invalid role: {role}")

        token = secrets.token_urlsafe(32)
        self._tokens[token] = {
            "user_id": user_id,
            "role": role,
            "permissions": self.ROLES[role],
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
        }
        return token

    def validate_token(self, token: str) -> Optional[dict[str, Any]]:
        """Validate a token and return user info."""
        token_data = self._tokens.get(token)
        if not token_data:
            return None

        # Check expiry
        expires = datetime.fromisoformat(token_data["expires_at"])
        if datetime.utcnow() > expires:
            del self._tokens[token]
            return None

        return token_data

    def check_permission(self, token: str, required_permission: str) -> bool:
        """Check if a token has a specific permission."""
        token_data = self.validate_token(token)
        if not token_data:
            return False
        return required_permission in token_data.get("permissions", [])
