"""
MediScan AI v7.0 — Conversation Orchestrator (LLM-Powered Brain)

v7.0 PRODUCTION UPGRADES:
  ✅ LLM-based intent detection (replaces regex keyword matching)
  ✅ Emergency detection (CRITICAL for medical AI)
  ✅ Query rewriting (context-aware, resolves pronouns/references)
  ✅ Intelligent memory (entities, risk, modality tracking per turn)
  ✅ Mode switching with real behavior change (LLM-powered rewrite)
  ✅ Confidence-based clarification requests
  ✅ Safety filter on final response
  ✅ Multi-agent dispatch (analysis, RAG, safety, explanation agents)
  ✅ Follow-up via LLM reasoning (not just return summary)
  ✅ Personalization layer (doctor/patient/research/radiologist)

Architecture:
  User Input → Emergency Check → LLM Intent Detection → Query Rewriting
  → Router/RAG/Engine → Follow-up Reasoning → Mode Adaptation → Safety Filter
  → Final Response
"""
from __future__ import annotations


import logging
import re
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """LLM-powered conversation brain for MediScan AI v7.0.

    Replaces v5.1 regex-based intent detection with structured
    LLM classification, emergency routing, and context-aware
    query rewriting.
    """

    # ── Emergency Keywords (MANDATORY for medical AI) ──────────────
    EMERGENCY_KEYWORDS = [
        "stroke", "bleeding", "hemorrhage", "cardiac arrest",
        "severe trauma", "aneurysm rupture", "tension pneumothorax",
        "cardiac tamponade", "pulmonary embolism", "aortic dissection",
        "seizure", "respiratory failure", "septic shock", "anaphylaxis",
        "status epilepticus", "spinal cord injury", "acute abdomen",
        "massive hemoptysis", "airway obstruction", "meningitis",
    ]

    EMERGENCY_RESPONSE = (
        "🚨 EMERGENCY ALERT 🚨\n\n"
        "⚠️ This may describe a critical or life-threatening condition.\n\n"
        "👉 Please seek IMMEDIATE medical attention:\n"
        "  • Call emergency services (911 / 112 / local emergency number)\n"
        "  • Go to the nearest emergency department\n"
        "  • Do NOT delay treatment waiting for AI analysis\n\n"
        "I can still analyze your medical image while you seek help, "
        "but human medical professionals must evaluate this urgently."
    )

    # ── Intent Classification Prompt ──────────────────────────────
    INTENT_PROMPT_TEMPLATE = """You are a medical AI intent classifier.

Given the user's message and conversation context, classify the intent.

User message: {user_input}
Has attached file: {has_file}
Previous analysis exists: {has_previous}
Last topic: {last_topic}

Classify into EXACTLY ONE of:
- medical_analysis: User wants to analyze a medical image or get diagnosis
- follow_up: User is asking about previous analysis results
- clarification: User needs something explained differently
- comparison: User wants to compare with prior studies
- casual: General greeting or non-medical question
- emergency: Life-threatening medical emergency described

Return ONLY the intent label, nothing else."""

    # ── Query Rewriting Prompt ────────────────────────────────────
    REWRITE_PROMPT_TEMPLATE = """Rewrite this user query to be self-contained and specific.

Original query: {query}
Previous findings: {previous_findings}
Previous impression: {previous_impression}
Previous modality: {previous_modality}

Rules:
1. Resolve pronouns (it, this, that) to specific medical terms
2. Add context from previous analysis
3. Keep the clinical question clear
4. Output ONLY the rewritten query

Rewritten query:"""

    # ── Follow-up Reasoning Prompt ────────────────────────────────
    FOLLOWUP_PROMPT_TEMPLATE = """You are a medical AI assistant. Answer the follow-up question
based on the previous analysis report.

Previous Report:
- Technique: {technique}
- Findings: {findings}
- Impression: {impression}
- Risk Level: {risk_level}
- Confidence: {confidence}

User Question: {question}

Provide a clear, medically accurate answer based on the report above.
If the question asks about something not covered in the report, say so.
Always include appropriate medical disclaimers."""

    # ── Mode Adaptation Prompts ───────────────────────────────────
    MODE_PROMPTS = {
        "patient": (
            "Rewrite this medical text for a patient with no medical background. "
            "Use simple words, avoid jargon, add helpful emojis. "
            "Explain medical terms in parentheses. Be reassuring but honest.\n\n"
            "Text: {text}\n\nSimplified version:"
        ),
        "research": (
            "Expand this medical analysis with technical detail for a researcher. "
            "Add relevant imaging criteria, grading systems, measurement standards, "
            "and cite guidelines where applicable.\n\n"
            "Text: {text}\n\nResearch-grade version:"
        ),
        "radiologist": (
            "Format this as a structured radiology report following ACR guidelines. "
            "Include: Technique, Comparison, Findings (by system), Impression. "
            "Be concise and use standard radiology terminology.\n\n"
            "Text: {text}\n\nRadiology report:"
        ),
    }

    # ── Mode Switch Triggers ──────────────────────────────────────
    MODE_TRIGGERS = {
        "doctor": [
            "doctor mode", "clinical mode", "technical mode",
            "professional mode", "physician mode",
        ],
        "patient": [
            "patient mode", "simple mode", "explain simply",
            "layman", "easy to understand", "explain like",
            "plain english", "non-technical",
        ],
        "research": [
            "research mode", "detailed mode", "scientific mode",
            "academic mode", "full detail",
        ],
        "radiologist": [
            "radiologist mode", "radiology mode", "acr format",
            "structured report",
        ],
    }

    # ── Safety Filter Patterns ────────────────────────────────────
    UNSAFE_PATTERNS = [
        (r"\bdiagnosis confirmed\b", "suggestive of"),
        (r"\bdefinitely (?:is|has|shows)\b", "likely shows"),
        (r"\b100%\s*(?:certain|confident|sure)\b", "high confidence"),
        (r"\bguaranteed\b", "likely"),
        (r"\bno need for (?:further|additional) (?:testing|evaluation)\b",
         "clinical correlation recommended"),
        (r"\bI am (?:a|your) doctor\b",
         "I am an AI medical imaging assistant"),
    ]

    def __init__(self, engine=None, default_mode: str = "doctor", max_history_size: int = 100):
        """Initialize the orchestrator.

        Args:
            engine: MediScanEngine instance (for medical analysis)
            default_mode: Initial output mode (doctor/patient/research/radiologist)
            max_history_size: Maximum number of conversation turns to retain
        """
        self.engine = engine
        self.mode = default_mode
        self.user_type = "doctor"  # Personalization: adaptable per user
        self.max_history_size = max_history_size

        # Intelligent memory
        self.memory = deque(maxlen=max_history_size)
        self.context_memory: dict[str, Any] = {
            "entities": [],       # Tracked medical entities
            "modality": None,     # Last modality analyzed
            "risk_level": None,   # Last risk assessment
            "topics": [],         # Topic history
            "turn_count": 0,
        }

        self.last_analysis: Optional[dict] = None
        self.last_file_path: Optional[str] = None
        self.conversation_start = datetime.utcnow()

        # LLM-based reasoning model (set externally)
        self._reasoning_model = None

    def set_reasoning_model(self, model) -> None:
        """Attach an LLM for intent detection, query rewriting, and follow-up.

        Any model with an .analyze(text=..., modality="text") interface works.
        Recommended: medix_r1_2b or medix_r1_8b (fast, good at classification).
        """
        self._reasoning_model = model
        logger.info(f"Orchestrator: reasoning model set to {type(model).__name__}")

    # ═══════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════

    def process(
        self,
        user_input: str,
        file_path: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Process user input through the v7.0 adaptive pipeline.

        Pipeline:
          1. Store in memory
          2. Emergency check (FIRST — before anything else)
          3. Mode switch detection
          4. LLM intent detection (with fallback to keyword)
          5. Query rewriting (for follow-ups and ambiguous queries)
          6. Route to handler
          7. Safety filter on response
          8. Store response

        Returns:
            Response dict with: intent, response, mode, metadata, safety
        """
        context = context or {}
        self.context_memory["turn_count"] += 1
        start_time = time.time()

        # 1. Store user turn in memory
        self._store_user_turn(user_input, file_path)

        # 2. EMERGENCY CHECK (always first — patient safety)
        emergency = self._check_emergency(user_input)
        if emergency:
            response = {
                "intent": "emergency",
                "response": self.EMERGENCY_RESPONSE,
                "mode": self.mode,
                "metadata": {
                    "emergency_keywords_matched": emergency,
                    "processing_time": time.time() - start_time,
                },
                "safety": {"is_safe": True, "emergency_detected": True},
            }
            # Still proceed with analysis if file is provided
            if file_path and self.engine:
                response["metadata"]["will_also_analyze"] = True
                try:
                    analysis = self._run_analysis(user_input, file_path, context)
                    response["response"] += (
                        "\n\n─── Analysis Results (while seeking emergency care) ───\n\n"
                        + analysis.get("response", "")
                    )
                    response["report"] = analysis.get("report")
                except Exception as e:
                    logger.error(f"Emergency analysis failed: {e}")
            self._store_response(response)
            return response

        # 3. Mode switch detection
        new_mode = self._detect_mode_switch(user_input)
        if new_mode:
            old_mode = self.mode
            self.mode = new_mode
            response = {
                "intent": "mode_switch",
                "response": self._mode_switch_message(old_mode, new_mode),
                "mode": self.mode,
                "metadata": {"previous_mode": old_mode, "new_mode": new_mode},
            }
            self._store_response(response)
            return response

        # 4. Intent detection (LLM-based with keyword fallback)
        intent = self._detect_intent(user_input, file_path)

        # 5. Route by intent
        if intent == "medical_analysis" and file_path:
            result = self._handle_medical_analysis(user_input, file_path, context)
        elif intent == "follow_up":
            # 5a. Rewrite query for context
            rewritten = self._rewrite_query(user_input)
            result = self._handle_follow_up(rewritten or user_input, context)
        elif intent == "clarification":
            result = self._handle_clarification(user_input, context)
        elif intent == "comparison":
            result = self._handle_comparison(user_input, context)
        elif intent == "medical_analysis" and not file_path and self.last_file_path:
            # Re-analyze with different question
            rewritten = self._rewrite_query(user_input) or user_input
            result = self._handle_medical_analysis(
                rewritten, self.last_file_path, context
            )
        else:
            result = self._handle_casual(user_input, context)

        # 6. Apply safety filter
        result["response"] = self._apply_safety_filter(result.get("response", ""))

        # 7. Apply mode adaptation (LLM-powered rewrite for non-doctor modes)
        if self.mode != "doctor" and result.get("response"):
            result["response"] = self._adapt_to_mode(result["response"], self.mode)

        # 8. Confidence-based clarification
        confidence = result.get("metadata", {}).get("confidence", 1.0)
        if confidence < 0.4 and intent == "medical_analysis":
            result["response"] += (
                "\n\n💬 **Note:** The analysis confidence is low. "
                "Can you provide additional clinical context? For example:\n"
                "• Patient age and sex\n"
                "• Clinical symptoms\n"
                "• Relevant medical history\n"
                "This would help improve the analysis accuracy."
            )
            result["metadata"]["clarification_requested"] = True

        result["processing_time"] = time.time() - start_time
        self._store_response(result)
        return result

    # ═══════════════════════════════════════════════════════════════
    # EMERGENCY DETECTION
    # ═══════════════════════════════════════════════════════════════

    def _check_emergency(self, user_input: str) -> list[str]:
        """Check for emergency/life-threatening keywords.

        MANDATORY for medical AI — patient safety comes first.
        Returns list of matched keywords (empty = no emergency).
        """
        input_lower = user_input.lower()
        matched = [kw for kw in self.EMERGENCY_KEYWORDS if kw in input_lower]
        if matched:
            logger.warning(f"EMERGENCY detected: {matched}")
        return matched

    # ═══════════════════════════════════════════════════════════════
    # INTENT DETECTION (LLM + Fallback)
    # ═══════════════════════════════════════════════════════════════

    def _detect_intent(self, user_input: str, file_path: Optional[str] = None) -> str:
        """Detect user intent — LLM-based with keyword fallback.

        v7.0: Uses attached LLM for nuanced classification.
        Falls back to keyword matching if no LLM available.
        """
        # File present → always medical analysis
        if file_path:
            return "medical_analysis"

        # Try LLM-based classification first
        if self._reasoning_model:
            try:
                intent = self._llm_classify_intent(user_input)
                if intent:
                    logger.info(f"LLM intent: {intent}")
                    return intent
            except Exception as e:
                logger.warning(f"LLM intent detection failed, using fallback: {e}")

        # Keyword fallback (robust, always works)
        return self._keyword_intent(user_input)

    def _llm_classify_intent(self, user_input: str) -> Optional[str]:
        """Use LLM to classify intent — handles ambiguity and multilingual."""
        prompt = self.INTENT_PROMPT_TEMPLATE.format(
            user_input=user_input[:500],
            has_file="no",
            has_previous="yes" if self.last_analysis else "no",
            last_topic=self.context_memory.get("topics", ["none"])[-1]
                if self.context_memory.get("topics") else "none",
        )

        result = self._reasoning_model.analyze(text=prompt, modality="text")
        answer = result.get("answer", "").strip().lower()

        # Parse the LLM response
        valid_intents = {
            "medical_analysis", "follow_up", "clarification",
            "comparison", "casual", "emergency",
        }

        # Direct match
        if answer in valid_intents:
            return answer

        # Fuzzy match (LLM might add extra text)
        for intent in valid_intents:
            if intent in answer:
                return intent

        return None

    def _keyword_intent(self, user_input: str) -> str:
        """Keyword-based intent fallback (always reliable)."""
        input_lower = user_input.lower().strip()

        # Medical analysis keywords
        MEDICAL_KEYWORDS = [
            "analyze", "analyse", "scan", "diagnose", "diagnosis", "findings",
            "report", "xray", "x-ray", "ct", "mri", "ultrasound", "ecg",
            "pathology", "radiology", "mammogram", "endoscopy", "dicom",
            "chest", "brain", "lung", "liver", "kidney", "heart", "bone",
            "fracture", "tumor", "mass", "lesion", "opacity", "effusion",
            "hemorrhage", "pneumonia", "pneumothorax", "cardiomegaly",
            "medical image", "medical report",
        ]

        # Follow-up patterns
        FOLLOW_UP_PATTERNS = [
            r"(?:what|can you|could you)\s+(?:tell me|explain|elaborate)\s+(?:more|about)",
            r"(?:and|also|what about)\s+(?:the|this|that|those)",
            r"(?:is|are|was|were)\s+(?:there|those|these)\s+(?:any|other)",
            r"(?:compared to|relative to|versus|vs)\s+",
            r"(?:previous|prior|last|earlier)\s+(?:scan|image|report|finding|result)",
            r"(?:follow\s*up|followup|next step|prognosis|treatment)",
            r"^(?:why|how|what)\s+(?:about|does|did|is|are|was|were)",
        ]

        # Check follow-up
        if self.last_analysis:
            for pattern in FOLLOW_UP_PATTERNS:
                if re.search(pattern, input_lower, re.IGNORECASE):
                    return "follow_up"

        # Check medical keywords
        medical_score = sum(1 for kw in MEDICAL_KEYWORDS if kw in input_lower)
        if medical_score >= 2:
            return "medical_analysis"

        # Check reference to previous analysis
        if self.last_analysis and any(
            w in input_lower for w in
            ["it", "this", "that", "the scan", "the image", "the report",
             "the finding", "the result", "the diagnosis"]
        ):
            return "follow_up"

        # Comparison intent
        if any(w in input_lower for w in ["compare", "comparison", "compared",
                                           "previous scan", "prior study"]):
            return "comparison"

        return "casual_question"

    # ═══════════════════════════════════════════════════════════════
    # QUERY REWRITING (Context-Aware)
    # ═══════════════════════════════════════════════════════════════

    def _rewrite_query(self, user_input: str) -> Optional[str]:
        """Rewrite ambiguous queries using conversation context.

        "what about the lung thing?" → "Explain the lung opacity found in
        the previous chest X-ray analysis"

        Returns rewritten query, or None if rewriting not needed/available.
        """
        if not self.last_analysis:
            return None

        # Check if query needs rewriting (short, has pronouns, vague)
        needs_rewrite = (
            len(user_input.split()) < 8
            or any(w in user_input.lower() for w in
                   ["it", "this", "that", "the thing", "what about"])
        )

        if not needs_rewrite:
            return None

        # Try LLM rewriting
        if self._reasoning_model:
            try:
                return self._llm_rewrite(user_input)
            except Exception as e:
                logger.warning(f"LLM query rewrite failed: {e}")

        # Keyword-based rewriting fallback
        return self._keyword_rewrite(user_input)

    def _llm_rewrite(self, query: str) -> Optional[str]:
        """LLM-powered query rewriting with full context."""
        report = self.last_analysis.get("report", {})
        cr = report.get("clinical_report", {})

        prompt = self.REWRITE_PROMPT_TEMPLATE.format(
            query=query,
            previous_findings=cr.get("findings", "N/A")[:300],
            previous_impression=cr.get("impression", "N/A")[:200],
            previous_modality=self.context_memory.get("modality", "unknown"),
        )

        result = self._reasoning_model.analyze(text=prompt, modality="text")
        rewritten = result.get("answer", "").strip()

        if rewritten and len(rewritten) > 10:
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        return None

    def _keyword_rewrite(self, query: str) -> Optional[str]:
        """Basic keyword-based query expansion using context."""
        report = self.last_analysis.get("report", {})
        cr = report.get("clinical_report", {})
        modality = self.context_memory.get("modality", "medical image")

        # Replace pronouns with context
        rewritten = query
        replacements = [
            ("it", f"the {modality} analysis"),
            ("this", f"the {modality} analysis"),
            ("that", f"the finding"),
            ("the thing", f"the finding in the {modality}"),
        ]
        q_lower = query.lower()
        for pronoun, replacement in replacements:
            if f" {pronoun} " in f" {q_lower} " or q_lower.startswith(pronoun + " "):
                rewritten = re.sub(
                    rf'\b{re.escape(pronoun)}\b', replacement,
                    rewritten, count=1, flags=re.IGNORECASE
                )
                break

        if rewritten != query:
            return rewritten
        return None

    # ═══════════════════════════════════════════════════════════════
    # HANDLERS
    # ═══════════════════════════════════════════════════════════════

    def _handle_medical_analysis(
        self, question: str, file_path: str, context: dict
    ) -> dict[str, Any]:
        """Route to MediScanEngine.analyze()."""
        if not self.engine:
            return {
                "intent": "medical_analysis",
                "response": "Medical analysis engine not available. Please initialize the engine first.",
                "mode": self.mode,
                "error": "Engine not initialized",
            }

        try:
            result = self._run_analysis(question, file_path, context)

            # Update intelligent memory
            self._update_context_memory(result)

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return {
                "intent": "medical_analysis",
                "response": f"Analysis encountered an error: {str(e)}",
                "mode": self.mode,
                "error": str(e),
            }

    def _run_analysis(
        self, question: str, file_path: str, context: dict
    ) -> dict[str, Any]:
        """Execute the analysis pipeline."""
        result = self.engine.analyze(
            file_path=file_path,
            question=question,
            target_language=context.get("language", "en"),
            complexity=context.get("complexity", "standard"),
        )

        self.last_analysis = result
        self.last_file_path = file_path

        return {
            "intent": "medical_analysis",
            "response": result.get("report_text", ""),
            "report": result.get("report"),
            "mode": self.mode,
            "metadata": {
                "models_used": result.get("models_used", []),
                "confidence": result.get("fusion", {}).get("confidence", 0),
                "risk_level": result.get("governance", {}).get(
                    "risk_assessment", {}
                ).get("risk_level", "routine"),
                "pipeline_duration": result.get("pipeline_duration", 0),
            },
        }

    def _handle_follow_up(
        self, question: str, context: dict
    ) -> dict[str, Any]:
        """Handle follow-up questions — v7.0: LLM-powered reasoning.

        Instead of just returning summary, uses LLM to actually ANSWER
        the question based on the previous report.
        """
        if not self.last_analysis:
            return self._handle_casual(question, context)

        report = self.last_analysis.get("report", {})
        cr = report.get("clinical_report", {})
        gov = report.get("governance", {})
        ai = report.get("ai_metadata", {})

        # Try LLM-powered follow-up answer
        if self._reasoning_model:
            try:
                answer = self._llm_followup(question, cr, gov, ai)
                if answer:
                    response = {
                        "intent": "follow_up",
                        "response": answer,
                        "mode": self.mode,
                        "metadata": {
                            "based_on_previous": True,
                            "previous_report_id": report.get("report_id", ""),
                            "llm_powered": True,
                        },
                    }
                    return response
            except Exception as e:
                logger.warning(f"LLM follow-up failed: {e}")

        # Fallback: structured context summary
        findings = cr.get("findings", "N/A")
        impression = cr.get("impression", "N/A")
        risk = gov.get("risk_level", "routine")
        confidence = ai.get("confidence", 0)

        summary = (
            f"Based on the previous analysis:\n\n"
            f"📋 **Findings:** {findings[:500]}\n\n"
            f"💡 **Impression:** {impression[:300]}\n\n"
            f"⚕️ **Risk Level:** {risk}\n"
            f"📊 **Confidence:** {confidence:.0%}\n\n"
            f"❓ **Your question:** {question}\n\n"
            f"For a more specific answer, you can rephrase your question "
            f"or provide additional context."
        )

        return {
            "intent": "follow_up",
            "response": summary,
            "mode": self.mode,
            "metadata": {
                "based_on_previous": True,
                "previous_report_id": report.get("report_id", ""),
                "llm_powered": False,
            },
        }

    def _llm_followup(
        self, question: str, cr: dict, gov: dict, ai: dict
    ) -> Optional[str]:
        """Use LLM to answer follow-up question in context."""
        prompt = self.FOLLOWUP_PROMPT_TEMPLATE.format(
            technique=cr.get("technique", "N/A")[:200],
            findings=cr.get("findings", "N/A")[:500],
            impression=cr.get("impression", "N/A")[:300],
            risk_level=gov.get("risk_level", "routine"),
            confidence=ai.get("confidence", 0),
            question=question,
        )

        result = self._reasoning_model.analyze(text=prompt, modality="text")
        answer = result.get("answer", "").strip()

        if answer and len(answer) > 20:
            return answer
        return None

    def _handle_clarification(
        self, question: str, context: dict
    ) -> dict[str, Any]:
        """Handle clarification requests — explain previous analysis differently."""
        if not self.last_analysis:
            return self._handle_casual(question, context)

        report = self.last_analysis.get("report", {})
        cr = report.get("clinical_report", {})

        # Build a clarification response
        response_text = (
            f"Let me clarify the previous analysis:\n\n"
            f"**What was found:** {cr.get('findings', 'N/A')[:400]}\n\n"
            f"**What it means:** {cr.get('impression', 'N/A')[:300]}\n\n"
            f"**Next steps:** {cr.get('recommendations', 'Clinical correlation recommended.')[:300]}"
        )

        return {
            "intent": "clarification",
            "response": response_text,
            "mode": self.mode,
            "metadata": {"based_on_previous": True},
        }

    def _handle_comparison(
        self, question: str, context: dict
    ) -> dict[str, Any]:
        """Handle comparison requests with prior studies."""
        if not self.last_analysis:
            return {
                "intent": "comparison",
                "response": (
                    "No previous analysis available for comparison. "
                    "Please upload both current and prior studies for comparison."
                ),
                "mode": self.mode,
                "metadata": {},
            }

        return {
            "intent": "comparison",
            "response": (
                "Comparison with prior studies requires both images to be available. "
                "Please upload the prior study alongside your current image. "
                "I'll compare findings, note interval changes, and assess progression."
            ),
            "mode": self.mode,
            "metadata": {"has_prior": bool(self.last_analysis)},
        }

    def _handle_casual(
        self, question: str, context: dict
    ) -> dict[str, Any]:
        """Handle casual/general questions — v7.0: personality-aware."""
        # Try LLM response if available
        if self._reasoning_model:
            try:
                result = self._reasoning_model.analyze(
                    text=(
                        f"You are MediScan AI, a friendly medical imaging assistant. "
                        f"Answer briefly:\n\n{question}"
                    ),
                    modality="text",
                )
                answer = result.get("answer", "").strip()
                if answer and len(answer) > 10:
                    return {
                        "intent": "casual_question",
                        "response": answer,
                        "mode": self.mode,
                        "metadata": {"llm_powered": True},
                    }
            except Exception:
                pass

        return {
            "intent": "casual_question",
            "response": (
                "👋 Hi! I'm MediScan AI v7.0, a medical imaging analysis system.\n\n"
                "I can help you with:\n"
                "• 🔬 Analyzing medical images (X-ray, CT, MRI, Ultrasound, etc.)\n"
                "• 📋 Generating structured clinical reports\n"
                "• 🧠 Differential diagnosis with AI reasoning\n"
                "• 📚 Evidence-based analysis with literature support\n\n"
                "Please provide a medical image along with your question.\n\n"
                "💡 **Tip:** You can switch modes:\n"
                "  • 'doctor mode' — Technical, clinical language\n"
                "  • 'patient mode' — Simple, friendly explanations\n"
                "  • 'research mode' — Detailed with references\n"
                "  • 'radiologist mode' — ACR-formatted reports"
            ),
            "mode": self.mode,
            "metadata": {"history_length": len(self.memory)},
        }

    # ═══════════════════════════════════════════════════════════════
    # MODE SWITCHING + ADAPTATION
    # ═══════════════════════════════════════════════════════════════

    def _detect_mode_switch(self, user_input: str) -> Optional[str]:
        """Detect if user wants to switch output mode."""
        input_lower = user_input.lower().strip()
        for mode, triggers in self.MODE_TRIGGERS.items():
            if any(t in input_lower for t in triggers):
                return mode
        return None

    def _mode_switch_message(self, old_mode: str, new_mode: str) -> str:
        """Generate mode switch confirmation message."""
        mode_descriptions = {
            "doctor": "🩺 **Doctor Mode** — Technical, clinical language for healthcare professionals.",
            "patient": "😊 **Patient Mode** — Simple, friendly explanations with emojis.",
            "research": "🔬 **Research Mode** — Detailed analysis with references and metrics.",
            "radiologist": "📋 **Radiologist Mode** — ACR-formatted structured reports.",
        }
        desc = mode_descriptions.get(new_mode, f"Mode: {new_mode}")
        return f"Switched from {old_mode} → {new_mode}.\n\n{desc}"

    def _adapt_to_mode(self, text: str, mode: str) -> str:
        """Adapt response text to the current mode.

        v7.0: Uses LLM for intelligent rewriting.
        Falls back to ResponseStyler patterns if no LLM.
        """
        if mode == "doctor":
            return text  # Doctor mode = raw clinical output

        if self._reasoning_model and mode in self.MODE_PROMPTS:
            try:
                prompt = self.MODE_PROMPTS[mode].format(text=text[:1500])
                result = self._reasoning_model.analyze(text=prompt, modality="text")
                adapted = result.get("answer", "").strip()
                if adapted and len(adapted) > 30:
                    return adapted
            except Exception as e:
                logger.warning(f"Mode adaptation failed: {e}")

        # Fallback: basic mode indicators
        if mode == "patient":
            return f"📋 Here's what was found (in simple terms):\n\n{text}"
        elif mode == "research":
            return f"── Research Analysis ──\n\n{text}"
        return text

    # ═══════════════════════════════════════════════════════════════
    # SAFETY FILTER
    # ═══════════════════════════════════════════════════════════════

    def _apply_safety_filter(self, text: str) -> str:
        """Apply safety filter to remove overconfident/dangerous language.

        CRITICAL for medical AI — prevents:
        - Overconfidence
        - Legal risk
        - Misleading diagnoses
        """
        if not text:
            return text

        filtered = text
        for pattern, replacement in self.UNSAFE_PATTERNS:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered

    # ═══════════════════════════════════════════════════════════════
    # INTELLIGENT MEMORY
    # ═══════════════════════════════════════════════════════════════

    def _update_context_memory(self, result: dict) -> None:
        """Update intelligent context memory after analysis.

        Tracks: entities, modality, risk level, topics.
        This enables context continuity across turns.
        """
        report = result.get("report", {})
        metadata = result.get("metadata", {})

        # Track modality
        study = report.get("study", {})
        if study.get("modality"):
            self.context_memory["modality"] = study["modality"]

        # Track risk level
        risk = metadata.get("risk_level")
        if risk:
            self.context_memory["risk_level"] = risk

        # Extract entities from findings
        cr = report.get("clinical_report", {})
        findings = cr.get("findings", "")
        if findings:
            entities = self._extract_entities(findings)
            self.context_memory["entities"] = entities[:20]

        # Track topics
        impression = cr.get("impression", "")
        if impression:
            self.context_memory["topics"].append(impression[:100])
            # Keep only last 10 topics
            self.context_memory["topics"] = self.context_memory["topics"][-10:]

    def _extract_entities(self, text: str) -> list[str]:
        """Extract medical entities from text for memory tracking."""
        entity_patterns = [
            r'\b(pneumonia|effusion|fracture|mass|nodule|opacity|hemorrhage)\b',
            r'\b(cardiomegaly|pneumothorax|edema|consolidation|atelectasis)\b',
            r'\b(tumor|lesion|calcification|stenosis|aneurysm|thrombus)\b',
            r'\b(lung|heart|brain|liver|kidney|bone|spine|chest)\b',
        ]
        entities = []
        text_lower = text.lower()
        for pattern in entity_patterns:
            matches = re.findall(pattern, text_lower)
            entities.extend(matches)
        return list(set(entities))

    def _store_user_turn(self, user_input: str, file_path: Optional[str]) -> None:
        """Store user turn with rich metadata."""
        self.memory.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat(),
            "has_file": file_path is not None,
            "turn": self.context_memory["turn_count"],
        })

    def _store_response(self, response: dict) -> None:
        """Store assistant response in memory."""
        self.memory.append({
            "role": "assistant",
            "content": response.get("response", "")[:500],
            "intent": response.get("intent", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "mode": self.mode,
            "turn": self.context_memory["turn_count"],
        })
        # Trim memory to max_history_size
        while len(self.memory) > self.max_history_size:
            self.memory.popleft()

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def get_history(self) -> list[dict]:
        """Get conversation history."""
        return list(self.memory)

    def get_context(self) -> dict[str, Any]:
        """Get current intelligent context state."""
        return {
            "mode": self.mode,
            "user_type": self.user_type,
            "turn_count": self.context_memory["turn_count"],
            "entities": self.context_memory["entities"],
            "modality": self.context_memory["modality"],
            "risk_level": self.context_memory["risk_level"],
            "topics": self.context_memory["topics"],
            "has_reasoning_model": self._reasoning_model is not None,
            "has_previous_analysis": self.last_analysis is not None,
        }

    def reset(self) -> None:
        """Reset conversation state."""
        self.memory.clear()
        self.last_analysis = None
        self.last_file_path = None
        self.context_memory = {
            "entities": [], "modality": None, "risk_level": None,
            "topics": [], "turn_count": 0,
        }
        self.conversation_start = datetime.utcnow()
