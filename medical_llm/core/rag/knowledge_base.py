"""
Knowledge Base — WHO/NICE guideline ingestion and medical knowledge management.

Production hardening:
    - All guidelines include version, date, and source metadata
    - Staleness detection warns when guidelines exceed their review period
    - Stroke guidelines updated from 2019 → 2023 (AHA/ASA 2023 update)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum age (in days) before a guideline is flagged as potentially stale
_DEFAULT_STALENESS_DAYS = 730  # 2 years


class KnowledgeBase:
    """Medical knowledge base manager.

    Handles ingestion of:
    - WHO clinical guidelines
    - NICE protocols
    - Drug databases (interactions, dosages)
    - Medical textbook excerpts
    - Custom institution-specific protocols

    All built-in knowledge includes version metadata for staleness tracking.
    """

    # Pre-built medical knowledge chunks for common conditions
    # Each entry includes version, date, and review_period_days for staleness detection
    BUILT_IN_KNOWLEDGE: list[dict[str, Any]] = [
        {
            "topic": "Chest Pain Evaluation",
            "source": "AHA/ACC Guidelines",
            "version": "2021 AHA/ACC Chest Pain Guideline",
            "guideline_date": "2021-10-28",
            "review_period_days": 1095,  # 3 years
            "content": (
                "Acute chest pain evaluation should include: "
                "1) 12-lead ECG within 10 minutes of presentation. "
                "2) Serial cardiac troponins (high-sensitivity preferred). "
                "3) Risk stratification using HEART score or TIMI score. "
                "4) Consider PE if pleuritic pain, tachycardia, unilateral leg swelling. "
                "5) Consider aortic dissection if sudden onset, tearing quality, BP differential. "
                "Red flags: ST-elevation, hemodynamic instability, new murmur."
            ),
        },
        {
            "topic": "Pneumonia Management",
            "source": "IDSA/ATS Guidelines",
            "version": "2019 ATS/IDSA CAP Guideline",
            "guideline_date": "2019-10-01",
            "review_period_days": 1095,
            "content": (
                "Community-Acquired Pneumonia (CAP) management: "
                "Outpatient (no comorbidities): Amoxicillin 1g TID or Doxycycline 100mg BID. "
                "Outpatient (with comorbidities): Amoxicillin-clavulanate + Macrolide, or Respiratory FQ. "
                "Inpatient (non-ICU): Beta-lactam + Macrolide, or Respiratory FQ alone. "
                "Inpatient (ICU): Beta-lactam + Macrolide or Beta-lactam + Respiratory FQ. "
                "CURB-65 for severity: Confusion, Urea>7, RR≥30, BP<90/60, Age≥65. "
                "Score ≥3: Consider ICU admission."
            ),
        },
        {
            "topic": "Diabetes Management",
            "source": "ADA Standards of Care 2024",
            "version": "ADA Standards of Care 2024",
            "guideline_date": "2024-01-01",
            "review_period_days": 365,  # Annual updates
            "content": (
                "Type 2 Diabetes Management: "
                "First-line: Metformin + lifestyle modifications. "
                "HbA1c target: <7% for most adults, <8% for elderly/comorbid. "
                "If HbA1c >9% at diagnosis, consider dual therapy. "
                "SGLT2 inhibitors preferred if heart failure or CKD. "
                "GLP-1 RAs preferred if ASCVD or high CV risk. "
                "Insulin: Consider if HbA1c >10% or symptomatic hyperglycemia. "
                "Monitor: HbA1c every 3 months, annual eye/foot/kidney screening."
            ),
        },
        {
            "topic": "Stroke Assessment",
            "source": "AHA/ASA Guidelines 2023",
            "version": "2023 AHA/ASA Acute Ischemic Stroke Guideline Update",
            "guideline_date": "2023-07-01",
            "review_period_days": 1095,
            "content": (
                "Acute Ischemic Stroke Protocol (2023 Update): "
                "NIHSS assessment immediately. "
                "CT head without contrast to rule out hemorrhage. "
                "IV alteplase if within 4.5 hours of onset and eligible. "
                "Tenecteplase (0.25 mg/kg, max 25 mg) is now an alternative to alteplase "
                "for patients eligible for mechanical thrombectomy (Class 2a recommendation). "
                "Mechanical thrombectomy if LVO confirmed on CTA/MRA and within 24 hours "
                "(with perfusion imaging selection beyond 6 hours). "
                "BP management: Allow permissive hypertension (up to 220/120) unless thrombolysis given. "
                "If thrombolysis: Maintain BP <185/110 before and <180/105 for 24h after. "
                "Endovascular: Eligible patients should receive thrombectomy even if IV thrombolysis given. "
                "Mobile stroke units with CT capability can reduce door-to-needle time. "
                "Contraindications to IV thrombolysis: Active bleeding, recent major surgery (<14 days), "
                "platelet <100K, INR >1.7, glucose <50 mg/dL."
            ),
        },
        {
            "topic": "Sepsis Management",
            "source": "Surviving Sepsis Campaign 2021",
            "version": "SSC 2021 Guidelines",
            "guideline_date": "2021-10-02",
            "review_period_days": 1460,  # 4 years
            "content": (
                "Sepsis Hour-1 Bundle: "
                "1) Measure lactate (remeasure if >2 mmol/L). "
                "2) Obtain blood cultures before antibiotics. "
                "3) Administer broad-spectrum antibiotics. "
                "4) Begin rapid crystalloid infusion (30 mL/kg) for hypotension/lactate ≥4. "
                "5) Apply vasopressors (norepinephrine first-line) if hypotensive during/after fluids. "
                "qSOFA: RR≥22, altered mentation, SBP≤100. "
                "SOFA ≥2 points change = organ dysfunction."
            ),
        },
        {
            "topic": "Heart Failure Management",
            "source": "AHA/ACC/HFSA 2022 Guidelines",
            "version": "2022 AHA/ACC/HFSA Heart Failure Guideline",
            "guideline_date": "2022-04-01",
            "review_period_days": 1095,
            "content": (
                "HFrEF (LVEF ≤40%) Guideline-Directed Medical Therapy: "
                "Four pillars: 1) ACEi/ARB/ARNI 2) Beta-blocker 3) MRA 4) SGLT2i. "
                "ARNI (sacubitril/valsartan) preferred over ACEi/ARB. "
                "Beta-blockers: Carvedilol, Bisoprolol, or Metoprolol succinate. "
                "MRA: Spironolactone or Eplerenone (check K+ and renal function). "
                "SGLT2i: Dapagliflozin or Empagliflozin regardless of diabetes status. "
                "Diuretics for volume management (not mortality benefit). "
                "CRT/ICD: Consider if LVEF ≤35%, LBBB, QRS ≥150ms."
            ),
        },
        {
            "topic": "Anticoagulation Therapy",
            "source": "CHEST Guidelines",
            "version": "CHEST 2021 VTE Guideline Update",
            "guideline_date": "2021-01-01",
            "review_period_days": 1095,
            "content": (
                "Venous Thromboembolism (VTE) Treatment: "
                "DOACs preferred over warfarin for DVT/PE (unless mechanical valve, APL). "
                "Rivaroxaban: 15mg BID x21 days, then 20mg daily. "
                "Apixaban: 10mg BID x7 days, then 5mg BID. "
                "Duration: Provoked VTE: 3 months. Unprovoked VTE: ≥3 months, consider extended. "
                "Cancer-associated: LMWH or edoxaban/rivaroxaban. "
                "Reversal agents: Idarucizumab (dabigatran), Andexanet alfa (Xa inhibitors)."
            ),
        },
        {
            "topic": "Drug Interaction Alert",
            "source": "FDA Drug Safety Communications",
            "version": "FDA DSC Compilation 2024",
            "guideline_date": "2024-01-15",
            "review_period_days": 365,
            "content": (
                "Critical Drug Interactions: "
                "1) Warfarin + NSAIDs: Increased bleeding risk. "
                "2) ACEi + Potassium-sparing diuretics: Hyperkalemia risk. "
                "3) SSRIs + MAOIs: Serotonin syndrome (contraindicated). "
                "4) Methotrexate + NSAIDs: Increased methotrexate toxicity. "
                "5) Digoxin + Amiodarone: Increased digoxin levels (reduce dose 50%). "
                "6) Statins + Macrolides: Rhabdomyolysis risk. "
                "7) Fluoroquinolones + QT-prolonging drugs: Torsades de pointes. "
                "8) Clopidogrel + PPIs (esp. omeprazole): Reduced antiplatelet effect. "
                "9) Opioids + Benzodiazepines: FDA black box — respiratory depression. "
                "10) DOACs + strong CYP3A4/P-gp inhibitors: Increased bleeding risk."
            ),
        },
    ]

    def __init__(self, data_dir: str = "./data/knowledge"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def check_staleness(self) -> list[dict[str, Any]]:
        """Check all built-in guidelines for staleness.

        Returns:
            List of stale guideline warnings with topic, age, and recommended action
        """
        stale: list[dict[str, Any]] = []
        now = time.time()

        for item in self.BUILT_IN_KNOWLEDGE:
            guideline_date = item.get("guideline_date", "")
            review_period = item.get("review_period_days", _DEFAULT_STALENESS_DAYS)

            if not guideline_date:
                stale.append({
                    "topic": item["topic"],
                    "warning": "No guideline date specified — unable to assess staleness",
                    "action": "Add guideline_date metadata",
                })
                continue

            try:
                # Parse ISO date
                parts = guideline_date.split("-")
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                import datetime
                pub_date = datetime.datetime(year, month, day)
                age_days = (datetime.datetime.now() - pub_date).days

                if age_days > review_period:
                    stale.append({
                        "topic": item["topic"],
                        "source": item["source"],
                        "version": item.get("version", "unknown"),
                        "guideline_date": guideline_date,
                        "age_days": age_days,
                        "review_period_days": review_period,
                        "warning": f"Guideline is {age_days} days old (review period: {review_period} days)",
                        "action": "Check for updated guideline version",
                    })
            except (ValueError, IndexError):
                stale.append({
                    "topic": item["topic"],
                    "warning": f"Invalid guideline_date format: {guideline_date}",
                    "action": "Fix date format to YYYY-MM-DD",
                })

        if stale:
            logger.warning(
                "Staleness check: %d guideline(s) may need updating", len(stale)
            )
        else:
            logger.info("Staleness check: all guidelines within review period")

        return stale

    def ingest_built_in(self, rag_engine: Any) -> int:
        """Ingest built-in medical knowledge into RAG engine.

        Args:
            rag_engine: MedicalRAG instance

        Returns:
            Number of chunks ingested
        """
        # Run staleness check on startup
        stale = self.check_staleness()
        if stale:
            for s in stale:
                logger.warning(
                    "STALE GUIDELINE: %s (%s) — %s",
                    s.get("topic", "?"),
                    s.get("version", "?"),
                    s.get("warning", ""),
                )

        documents = [item["content"] for item in self.BUILT_IN_KNOWLEDGE]
        metadatas = [
            {
                "source": item["source"],
                "topic": item["topic"],
                "type": "guideline",
                "version": item.get("version", ""),
                "guideline_date": item.get("guideline_date", ""),
            }
            for item in self.BUILT_IN_KNOWLEDGE
        ]
        ids = [f"builtin_{i}" for i in range(len(self.BUILT_IN_KNOWLEDGE))]

        count = rag_engine.ingest_documents(documents, metadatas, ids)
        logger.info("Ingested %d built-in knowledge chunks", count)
        return count

    def ingest_guidelines_file(
        self,
        file_path: str,
        rag_engine: Any,
        source: str = "custom",
    ) -> int:
        """Ingest a guidelines file into RAG.

        Supports .txt, .md, .json formats.

        Args:
            file_path: Path to guidelines file
            rag_engine: MedicalRAG instance
            source: Source label

        Returns:
            Number of chunks ingested
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("Guidelines file not found: %s", file_path)
            return 0

        try:
            if path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    documents = [
                        item.get("content", str(item)) for item in data
                    ]
                    metadatas = [
                        {
                            "source": item.get("source", source),
                            "topic": item.get("topic", path.stem),
                            "type": "guideline",
                            "version": item.get("version", ""),
                            "guideline_date": item.get("guideline_date", ""),
                        }
                        for item in data
                    ]
                else:
                    documents = [json.dumps(data, indent=2)]
                    metadatas = [{"source": source, "topic": path.stem, "type": "guideline"}]
            else:
                content = path.read_text(encoding="utf-8")
                documents = [content]
                metadatas = [{"source": source, "topic": path.stem, "type": "guideline"}]

            ids = [f"guideline_{path.stem}_{i}" for i in range(len(documents))]
            return rag_engine.ingest_documents(documents, metadatas, ids)

        except Exception as e:
            logger.error("Failed to ingest %s: %s", file_path, e)
            return 0

    def ingest_directory(self, directory: str, rag_engine: Any) -> int:
        """Ingest all supported files from a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        total = 0
        for ext in [".txt", ".md", ".json"]:
            for fpath in dir_path.rglob(f"*{ext}"):
                total += self.ingest_guidelines_file(
                    str(fpath), rag_engine, source=fpath.parent.name
                )

        logger.info("Ingested %d chunks from directory: %s", total, directory)
        return total
