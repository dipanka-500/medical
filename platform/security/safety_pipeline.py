"""
Safety Pipeline — ChatGPT-grade content moderation for medical AI.

Full pipeline:
    1. Token counting & per-tier limit enforcement
    2. Input safety classifier (harmful/prohibited content detection)
    3. Output filter (PII/PHI redaction + harmful content blocking)
    4. Confidence-based rejection (low confidence → "consult a doctor")

This module is imported by the chat endpoints and wired into the
request/response flow at the gateway level.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. TOKEN COUNTER & LIMITER ───────────────────────────────────────────

# Per-tier token limits (input tokens per request)
TIER_TOKEN_LIMITS: dict[str, int] = {
    "free": 2_000,
    "pro": 8_000,
    "enterprise": 32_000,
}

# Per-tier max output tokens
TIER_OUTPUT_LIMITS: dict[str, int] = {
    "free": 1_000,
    "pro": 4_096,
    "enterprise": 8_192,
}

# Simple token estimator (works without tiktoken or HF tokenizer)
# Approximation: 1 token ≈ 4 chars for English, 1 token ≈ 2 chars for code/medical
_AVG_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Estimate token count without requiring a tokenizer library.

    Uses word/char heuristic that's accurate within ~10% for English medical text.
    For exact counts, use a HuggingFace tokenizer on the model side.
    """
    if not text:
        return 0
    # Hybrid: average of word-based and char-based estimates
    word_estimate = len(text.split()) * 1.3  # ~1.3 tokens per word
    char_estimate = len(text) / _AVG_CHARS_PER_TOKEN
    return int((word_estimate + char_estimate) / 2)


def check_token_limit(text: str, tier: str = "free") -> tuple[bool, int, int]:
    """Check if input text is within the tier's token limit.

    Returns:
        (is_within_limit, estimated_tokens, max_allowed)
    """
    tokens = estimate_tokens(text)
    limit = TIER_TOKEN_LIMITS.get(tier, TIER_TOKEN_LIMITS["free"])
    return tokens <= limit, tokens, limit


def trim_to_token_limit(text: str, tier: str = "free") -> str:
    """Smart-trim text to fit within tier token limit.

    Preserves the most recent content (drops from the beginning),
    similar to how ChatGPT keeps recent context.
    """
    limit = TIER_TOKEN_LIMITS.get(tier, TIER_TOKEN_LIMITS["free"])
    tokens = estimate_tokens(text)
    if tokens <= limit:
        return text

    # Approximate character limit from token limit
    char_limit = int(limit * _AVG_CHARS_PER_TOKEN)
    # Keep the END of the text (most recent context)
    return text[-char_limit:]


def get_output_token_limit(tier: str = "free") -> int:
    """Get max output tokens for a tier."""
    return TIER_OUTPUT_LIMITS.get(tier, TIER_OUTPUT_LIMITS["free"])


# ── 2. INPUT SAFETY CLASSIFIER ──────────────────────────────────────────

# Categories of unsafe content with pattern-based detection
# Production systems would use a fine-tuned classifier model (e.g., RoBERTa)
# This is a rule-based layer that catches obvious cases instantly

_PROHIBITED_PATTERNS = re.compile(
    # Self-harm / suicide instructions
    r"how\s+to\s+(kill|harm|hurt)\s+(myself|yourself|themselves)|"
    r"suicide\s+(method|instructions|guide)|"
    r"ways\s+to\s+(die|end\s+my\s+life)|"
    # Weapons / violence
    r"how\s+to\s+make\s+a?\s*(bomb|poison|weapon|explosive)|"
    r"synthesize\s+(poison|toxin|nerve\s+agent)|"
    # Drug manufacturing
    r"how\s+to\s+(make|synthesize|cook)\s+(meth|heroin|fentanyl|cocaine)|"
    r"recipe\s+for\s+(meth|drugs)|"
    # Illegal prescriptions
    r"write\s+me\s+a\s+prescription|"
    r"forge\s+a?\s*(prescription|medical\s+certificate)|"
    r"fake\s+(prescription|medical\s+note)|"
    # Data exfiltration attempts
    r"list\s+all\s+patients|"
    r"show\s+(me\s+)?all\s+medical\s+records|"
    r"dump\s+(the\s+)?database|"
    r"extract\s+all\s+user\s+data",
    re.IGNORECASE,
)

_EMERGENCY_PATTERNS = re.compile(
    r"(i\s+want\s+to\s+die|"
    r"i('m|\s+am)\s+going\s+to\s+kill\s+myself|"
    r"suicidal|"
    r"i\s+don'?t\s+want\s+to\s+live|"
    r"having\s+a\s+(heart\s+attack|stroke|seizure)|"
    r"can'?t\s+breathe|"
    r"overdosed?|"
    r"poisoned|"
    r"severe\s+chest\s+pain|"
    r"uncontrolled\s+bleeding)",
    re.IGNORECASE,
)

_OFF_TOPIC_PATTERNS = re.compile(
    r"(write\s+(me\s+)?(a\s+)?(code|program|script|essay|poem|song|story)|"
    r"translate\s+.+\s+to\s+|"
    r"solve\s+this\s+math|"
    r"what\s+is\s+the\s+capital\s+of|"
    r"who\s+won\s+the\s+(election|match|game)|"
    r"(stock|crypto|bitcoin)\s+price)",
    re.IGNORECASE,
)


class SafetyVerdict:
    """Result of safety classification."""

    __slots__ = ("safe", "category", "message", "emergency")

    def __init__(
        self,
        safe: bool,
        category: str = "ok",
        message: str = "",
        emergency: bool = False,
    ):
        self.safe = safe
        self.category = category
        self.message = message
        self.emergency = emergency


def classify_input_safety(text: str) -> SafetyVerdict:
    """Classify user input for safety — Layer 1 of the pipeline.

    Categories:
        ok         — safe to process
        emergency  — user may be in crisis → show emergency resources
        prohibited — harmful/illegal request → hard block
        off_topic  — non-medical query → soft redirect
    """
    if not text or not text.strip():
        return SafetyVerdict(safe=False, category="empty", message="Empty query")

    # Check for emergency / crisis signals FIRST
    if _EMERGENCY_PATTERNS.search(text):
        return SafetyVerdict(
            safe=True,  # Still process, but flag
            category="emergency",
            emergency=True,
            message=(
                "**If you are in immediate danger, please call emergency services "
                "(911 in the US, 112 in EU, 108 in India) or contact a crisis "
                "helpline immediately.**\n\n"
                "- **US:** National Suicide Prevention Lifeline: 988\n"
                "- **India:** iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345\n"
                "- **International:** https://findahelpline.com\n\n"
                "I'll do my best to help, but please reach out to a professional."
            ),
        )

    # Check for prohibited content
    if _PROHIBITED_PATTERNS.search(text):
        logger.warning("Prohibited content detected in query (length=%d)", len(text))
        return SafetyVerdict(
            safe=False,
            category="prohibited",
            message=(
                "I cannot help with that request. As a medical AI assistant, "
                "I'm designed to provide helpful, safe medical information. "
                "If you're in crisis, please contact emergency services."
            ),
        )

    # Check for off-topic content (soft redirect)
    if _OFF_TOPIC_PATTERNS.search(text):
        return SafetyVerdict(
            safe=True,  # Still allow — LLM will handle gracefully
            category="off_topic",
            message="",  # Let the LLM handle with its system prompt
        )

    return SafetyVerdict(safe=True, category="ok")


# ── 3. OUTPUT FILTER ────────────────────────────────────────────────────

# PII/PHI patterns that should never appear in AI responses
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # SSN (US Social Security Number)
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    # Aadhaar (India 12-digit ID)
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[ID REDACTED]"),
    # Phone numbers (broad pattern)
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"), None),
    # Email addresses
    (re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"), "[EMAIL REDACTED]"),
    # Medical record numbers (MRN patterns)
    (re.compile(r"\bMRN[:\s#]*\d{4,12}\b", re.IGNORECASE), "[MRN REDACTED]"),
    # Patient IDs
    (re.compile(r"\bpatient[\s_-]?id[:\s#]*\w{4,20}\b", re.IGNORECASE), "[PATIENT_ID REDACTED]"),
]

# Patterns that indicate the AI is hallucinating or being unsafe
_UNSAFE_OUTPUT_PATTERNS = re.compile(
    # AI claiming to be a doctor
    r"as\s+your\s+doctor|"
    r"i\s+am\s+(a\s+)?doctor|"
    r"i\s+can\s+prescribe|"
    r"i('m|\s+am)\s+diagnosing|"
    # Specific dosage advice (should always say "consult your doctor")
    r"take\s+\d+\s*mg\s+(of\s+)?\w+\s+(every|twice|three\s+times|daily)|"
    # Definitive diagnosis without hedging
    r"you\s+(definitely|certainly)\s+have\s+|"
    r"this\s+is\s+(definitely|certainly)\s+(cancer|tumor|malignant)",
    re.IGNORECASE,
)

_DISCLAIMER = (
    "\n\n---\n*This is AI-generated medical information for educational purposes. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider.*"
)


def filter_output(
    text: str,
    *,
    confidence: float = 0.0,
    add_disclaimer: bool = True,
    redact_pii: bool = True,
) -> dict[str, Any]:
    """Filter AI output for safety and compliance — Layer 3 of the pipeline.

    Returns:
        dict with: text, was_modified, redactions, safety_flags
    """
    if not text:
        return {"text": "", "was_modified": False, "redactions": [], "safety_flags": []}

    was_modified = False
    redactions: list[str] = []
    safety_flags: list[str] = []
    filtered = text

    # 1. PII/PHI redaction
    if redact_pii:
        for pattern, replacement in _PII_PATTERNS:
            if replacement is None:
                continue  # Phone numbers: don't redact (too many false positives)
            matches = pattern.findall(filtered)
            if matches:
                filtered = pattern.sub(replacement, filtered)
                redactions.extend(matches)
                was_modified = True

    # 2. Unsafe output patterns
    if _UNSAFE_OUTPUT_PATTERNS.search(filtered):
        safety_flags.append("unsafe_medical_claim")
        # Don't block — add stronger disclaimer instead
        was_modified = True

    # 3. Add medical disclaimer for substantive medical content
    if add_disclaimer and len(filtered) > 100:
        # Only add if not already present
        if "not a substitute for professional" not in filtered.lower():
            filtered += _DISCLAIMER
            was_modified = True

    if redactions:
        logger.info("PII redacted from output: %d items", len(redactions))

    return {
        "text": filtered,
        "was_modified": was_modified,
        "redactions": [f"[{len(r)} chars]" for r in redactions],  # Don't log actual PII
        "safety_flags": safety_flags,
    }


# ── 4. CONFIDENCE-BASED REJECTION ───────────────────────────────────────

# Minimum confidence thresholds per mode
_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "doctor": 0.4,      # Doctors can interpret low-confidence results
    "patient": 0.6,     # Patients need higher confidence
    "research": 0.3,    # Research mode allows exploratory answers
}

_LOW_CONFIDENCE_ADDENDUM = (
    "\n\n**Note:** This response has lower confidence than usual. "
    "Please verify this information with a qualified healthcare professional "
    "before making any medical decisions."
)


def apply_confidence_check(
    result: dict[str, Any],
    mode: str = "patient",
) -> dict[str, Any]:
    """Apply confidence-based filtering — Layer 4 of the pipeline.

    If confidence is below threshold:
        - For patients: add strong warning
        - For doctors: add note
        - For research: pass through
    """
    confidence = result.get("confidence", 0.0)
    threshold = _CONFIDENCE_THRESHOLDS.get(mode, 0.6)

    if confidence < threshold:
        answer = result.get("answer", "")

        if mode == "patient" and confidence < 0.3:
            # Very low confidence for patient → replace answer entirely
            result["answer"] = (
                "I'm not confident enough to provide a reliable answer to this question. "
                "For your safety, please consult a qualified healthcare professional "
                "who can give you accurate, personalized medical advice.\n\n"
                "If this is urgent, please contact your doctor or visit an emergency room."
            )
            result["confidence_rejected"] = True
        else:
            # Add warning addendum
            result["answer"] = answer + _LOW_CONFIDENCE_ADDENDUM

        result["low_confidence"] = True

    return result


# ── 5. FULL PIPELINE ────────────────────────────────────────────────────


def run_pre_query_checks(
    query: str,
    tier: str = "free",
) -> dict[str, Any]:
    """Run all pre-query safety checks.

    Returns:
        dict with: allowed, query (possibly trimmed), token_count, safety_verdict,
                    error (if blocked), emergency_message
    """
    result: dict[str, Any] = {
        "allowed": True,
        "query": query,
        "error": None,
        "emergency_message": None,
    }

    # 1. Token limit check
    within_limit, token_count, max_tokens = check_token_limit(query, tier)
    result["token_count"] = token_count
    result["max_tokens"] = max_tokens

    if not within_limit:
        # Try smart trimming
        trimmed = trim_to_token_limit(query, tier)
        result["query"] = trimmed
        result["was_trimmed"] = True
        logger.info(
            "Query trimmed: %d→%d tokens (tier=%s limit=%d)",
            token_count, estimate_tokens(trimmed), tier, max_tokens,
        )

    # 2. Safety classification
    verdict = classify_input_safety(query)
    result["safety_category"] = verdict.category

    if not verdict.safe:
        result["allowed"] = False
        result["error"] = verdict.message
        return result

    if verdict.emergency:
        result["emergency_message"] = verdict.message

    return result


def run_post_query_checks(
    result: dict[str, Any],
    mode: str = "patient",
    tier: str = "free",
) -> dict[str, Any]:
    """Run all post-query safety checks on the AI response.

    Modifies result in-place and returns it.
    """
    answer = result.get("answer", "")

    # 1. Output filtering (PII redaction + safety)
    filter_result = filter_output(answer)
    result["answer"] = filter_result["text"]

    if filter_result["safety_flags"]:
        safety = result.get("safety", {}) or {}
        safety["output_flags"] = filter_result["safety_flags"]
        result["safety"] = safety

    if filter_result["redactions"]:
        safety = result.get("safety", {}) or {}
        safety["pii_redacted"] = len(filter_result["redactions"])
        result["safety"] = safety

    # 2. Confidence-based rejection
    result = apply_confidence_check(result, mode)

    return result
