"""
Negation Detector — Context-aware medical negation detection.
Implements NegEx-inspired algorithm with configurable window sizes and phrase lists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NegationResult:
    """Result of negation analysis for a single entity."""
    entity: str
    is_negated: bool
    negation_cue: str = ""
    window_text: str = ""


class NegationDetector:
    """Context-aware negation detection for medical text.

    Uses a sliding-window approach inspired by NegEx to detect
    negated medical findings. Handles pre-negation (before entity)
    and post-negation (after entity) cues.

    The detector scans a configurable character window around each
    entity occurrence to find negation phrases.
    """

    # Pre-negation phrases (appear BEFORE the entity)
    PRE_NEGATION_PHRASES = [
        "no evidence of",
        "no radiographic evidence of",
        "no signs of",
        "no significant",
        "no definite",
        "no acute",
        "no obvious",
        "no ",
        "not ",
        "without ",
        "negative for ",
        "ruled out ",
        "rules out ",
        "rule out ",
        "denies ",
        "deny ",
        "denied ",
        "absence of ",
        "absent ",
        "unlikely ",
        "never ",
        "fails to reveal ",
        "failed to reveal ",
        "not demonstrate ",
        "not seen ",
        "not identified ",
        "cannot see ",
        "unremarkable ",
        "free of ",
        "resolved ",
        "no longer ",
        "has not been ",
        "was not ",
        "were not ",
        "did not ",
        "does not ",
        "do not ",
        "cannot ",
        "could not ",
        "rather than ",
        "as opposed to ",
        "with no ",
        "no further ",
        "no new ",
        "no interval ",
        "no recurrence of ",
        "no residual ",
        "no remaining ",
        "no gross ",
        "no suspicious ",
    ]

    # Post-negation phrases (appear AFTER the entity)
    POST_NEGATION_PHRASES = [
        "is absent",
        "was absent",
        "are absent",
        "were absent",
        "is ruled out",
        "was ruled out",
        "is unlikely",
        "was unlikely",
        "is negative",
        "was negative",
        "is not present",
        "was not present",
        "is not seen",
        "was not seen",
        "not identified",
        "not demonstrated",
        "has been excluded",
        "has been ruled out",
        "was excluded",
    ]

    # Pseudo-negation phrases (look like negation but are not)
    PSEUDO_NEGATION_PHRASES = [
        "no increase",
        "no decrease",
        "no change",
        "not only",
        "not necessarily",
        "no longer a concern",
        "gram negative",
    ]

    # Conjunctions that break negation scope
    SCOPE_BREAKERS = [
        "but",
        "however",
        "although",
        "though",
        "except",
        "apart from",
        "aside from",
        "nevertheless",
        "nonetheless",
        "yet",
        "still",
        "with the exception of",
    ]

    # Sentence boundaries that should stop negation scope.
    SENTENCE_BREAKERS = [".", "!", "?", ";", "\n"]

    def __init__(
        self,
        pre_window_chars: int = 80,
        post_window_chars: int = 60,
        pre_negation_phrases: list[str] | None = None,
        post_negation_phrases: list[str] | None = None,
    ):
        self.pre_window_chars = pre_window_chars
        self.post_window_chars = post_window_chars

        if pre_negation_phrases is not None:
            self.PRE_NEGATION_PHRASES = pre_negation_phrases
        if post_negation_phrases is not None:
            self.POST_NEGATION_PHRASES = post_negation_phrases

        # Sort phrases by length (longest first) for greedy matching
        self.PRE_NEGATION_PHRASES = sorted(
            self.PRE_NEGATION_PHRASES, key=len, reverse=True
        )
        self.POST_NEGATION_PHRASES = sorted(
            self.POST_NEGATION_PHRASES, key=len, reverse=True
        )

    def detect(self, text: str, entity: str) -> NegationResult:
        """Detect if an entity is negated in the given text.

        Checks all occurrences of the entity. If ALL occurrences
        are negated, returns negated=True. If any occurrence is
        affirmed, returns negated=False.

        Args:
            text: Full medical text
            entity: Entity string to check

        Returns:
            NegationResult with negation status and evidence
        """
        text_lower = text.lower()
        entity_lower = entity.lower()

        # Find all occurrences
        positions: list[int] = []
        start = 0
        while True:
            pos = text_lower.find(entity_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + len(entity_lower)

        if not positions:
            return NegationResult(entity=entity, is_negated=False)

        # Check each occurrence
        negated_count = 0
        last_cue = ""
        last_window = ""

        for pos in positions:
            is_neg, cue, window = self._check_single_occurrence(
                text_lower, entity_lower, pos
            )
            if is_neg:
                negated_count += 1
                last_cue = cue
                last_window = window

        # All occurrences must be negated for entity to be considered negated
        all_negated = negated_count == len(positions)

        return NegationResult(
            entity=entity,
            is_negated=all_negated,
            negation_cue=last_cue if all_negated else "",
            window_text=last_window if all_negated else "",
        )

    def _check_single_occurrence(
        self, text_lower: str, entity_lower: str, pos: int
    ) -> tuple[bool, str, str]:
        """Check if a single occurrence of an entity is negated."""

        # Check for pseudo-negation first
        pre_window_start = max(0, pos - self.pre_window_chars)
        pre_window = text_lower[pre_window_start:pos]
        pre_window = self._trim_to_latest_boundary(pre_window)

        for pseudo in self.PSEUDO_NEGATION_PHRASES:
            if pseudo in pre_window:
                return False, "", ""

        # Check for scope breakers between negation and entity
        has_scope_breaker = False
        for breaker in self.SCOPE_BREAKERS:
            if breaker in pre_window:
                # Find the last scope breaker position
                breaker_pos = pre_window.rfind(breaker)
                # Only the text AFTER the breaker matters
                pre_window = pre_window[breaker_pos + len(breaker):]
                has_scope_breaker = True
                break

        # Check pre-negation phrases
        for phrase in self.PRE_NEGATION_PHRASES:
            if phrase in pre_window:
                return True, phrase, pre_window.strip()

        # Check post-negation phrases
        entity_end = pos + len(entity_lower)
        post_window_end = min(len(text_lower), entity_end + self.post_window_chars)
        post_window = text_lower[entity_end:post_window_end]
        post_window = self._trim_to_earliest_boundary(post_window)

        for phrase in self.POST_NEGATION_PHRASES:
            if phrase in post_window:
                return True, phrase, post_window.strip()

        return False, "", ""

    def _trim_to_latest_boundary(self, window: str) -> str:
        """Limit a pre-window to the current sentence/clause."""
        last_boundary = max(window.rfind(marker) for marker in self.SENTENCE_BREAKERS)
        if last_boundary == -1:
            return window
        return window[last_boundary + 1:]

    def _trim_to_earliest_boundary(self, window: str) -> str:
        """Limit a post-window to the current sentence/clause."""
        boundary_positions = [
            window.find(marker) for marker in self.SENTENCE_BREAKERS
            if window.find(marker) != -1
        ]
        if not boundary_positions:
            return window
        return window[:min(boundary_positions)]

    def detect_batch(
        self, text: str, entities: list[str]
    ) -> dict[str, NegationResult]:
        """Detect negation for multiple entities at once."""
        return {entity: self.detect(text, entity) for entity in entities}

    def filter_entities(
        self, text: str, entities: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split entities into positive (affirmed) and negated groups.

        Args:
            text: Medical text
            entities: List of entity strings

        Returns:
            Tuple of (positive_entities, negated_entities)
        """
        positive = []
        negated = []

        for entity in entities:
            result = self.detect(text, entity)
            if result.is_negated:
                negated.append(entity)
            else:
                positive.append(entity)

        return positive, negated

    def annotate_text(self, text: str, entities: list[str]) -> str:
        """Return annotated text with [NEG] markers for negated entities."""
        results = self.detect_batch(text, entities)
        annotated = text

        # Process in reverse order to preserve character positions
        for entity in sorted(results.keys(), key=lambda e: -text.lower().find(e.lower())):
            result = results[entity]
            if result.is_negated:
                pattern = re.compile(re.escape(entity), re.IGNORECASE)
                annotated = pattern.sub(f"[NEG]{entity}[/NEG]", annotated)

        return annotated
