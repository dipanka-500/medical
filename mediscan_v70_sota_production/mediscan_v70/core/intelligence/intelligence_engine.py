"""
MediScan AI v7.0 — Intelligence Engine
Production-grade reasoning, safety, and explainability layer.

v7.0: Multi-specialty knowledge graph covering radiology, neurology,
cardiology, ophthalmology, dermatology, dental, pathology, and GI.

Modules:
  1. MedicalKnowledgeEngine  — Causal reasoning via embedded medical knowledge graph
  2. EnhancedMedicalRAG      — Pre-seeded knowledge + PubMed search
  3. MedicalReasoningEngine   — CoT reasoning: Extract → Verify → Rank → Explain
  4. DynamicFusionEngine      — Uncertainty-aware fusion with model debate
  5. ClinicalSafetyLayer      — Hallucination detection, confidence gating
  6. SelfReflectionLoop       — Post-analysis critique and refinement
  7. MultiAgentOrchestrator   — Specialist role assignment
  8. ExplainabilityEngine     — "Why this diagnosis?" attribution
"""
from __future__ import annotations


import logging
import re
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("mediscan.intelligence")

# ═══════════════════════════════════════════════════════════
#  MEDICAL KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════

MEDICAL_KNOWLEDGE_GRAPH = {
    "findings_to_dx": {
        # ── Chest / Pulmonary ──────────────────────────────────
        "opacity": ["pneumonia", "atelectasis", "pulmonary edema", "lung mass", "pleural effusion"],
        "consolidation": ["pneumonia", "hemorrhage", "organizing pneumonia", "lung cancer"],
        "ground glass": ["covid-19", "interstitial pneumonia", "pulmonary hemorrhage", "drug toxicity"],
        "nodule": ["lung cancer", "granuloma", "metastasis", "hamartoma", "infection"],
        "mass": ["lung cancer", "lymphoma", "metastasis", "abscess"],
        "effusion": ["heart failure", "pneumonia", "malignancy", "liver disease", "kidney disease"],
        "cardiomegaly": ["heart failure", "cardiomyopathy", "pericardial effusion", "valvular disease"],
        "pneumothorax": ["trauma", "spontaneous", "iatrogenic", "bullous disease"],
        "fracture": ["trauma", "osteoporosis", "pathologic fracture", "stress fracture"],
        "edema": ["heart failure", "fluid overload", "ARDS", "renal failure"],
        "cavitation": ["tuberculosis", "lung abscess", "squamous cell carcinoma", "fungal infection"],
        "calcification": ["granuloma", "old infection", "atherosclerosis", "mesothelioma"],
        "air bronchograms": ["pneumonia", "alveolar hemorrhage", "lymphoma"],
        "widened mediastinum": ["aortic dissection", "lymphoma", "mediastinal mass", "trauma"],
        "hilar enlargement": ["sarcoidosis", "lymphoma", "lung cancer", "pulmonary hypertension"],
        "interstitial markings": ["pulmonary fibrosis", "interstitial pneumonia", "sarcoidosis"],
        # ── Neuro ─────────────────────────────────────────────
        "midline shift": ["intracranial hemorrhage", "brain tumor", "cerebral edema", "subdural hematoma"],
        "mass effect": ["brain tumor", "abscess", "hemorrhage", "edema"],
        "ring enhancement": ["brain abscess", "glioblastoma", "metastasis", "toxoplasmosis"],
        "restricted diffusion": ["acute stroke", "abscess", "epidermoid cyst", "CJD"],
        "white matter lesions": ["multiple sclerosis", "small vessel disease", "ADEM", "PML"],
        "hydrocephalus": ["obstructive hydrocephalus", "NPH", "meningitis", "subarachnoid hemorrhage"],
        "cerebral atrophy": ["alzheimer disease", "frontotemporal dementia", "aging", "chronic alcohol"],
        # ── Abdominal ─────────────────────────────────────────
        "hepatomegaly": ["hepatitis", "cirrhosis", "heart failure", "fatty liver", "hepatic malignancy"],
        "ascites": ["cirrhosis", "malignancy", "heart failure", "tuberculosis", "nephrotic syndrome"],
        "bowel wall thickening": ["inflammatory bowel disease", "infection", "ischemia", "lymphoma"],
        "free air": ["bowel perforation", "post-surgical", "peptic ulcer perforation"],
        "renal mass": ["renal cell carcinoma", "angiomyolipoma", "oncocytoma", "cyst"],
        "gallstones": ["cholelithiasis", "cholecystitis", "choledocholithiasis", "gallstone pancreatitis"],
        "pancreatic mass": ["pancreatic adenocarcinoma", "neuroendocrine tumor", "pancreatitis", "IPMN"],
        # ── MSK ───────────────────────────────────────────────
        "joint effusion": ["septic arthritis", "osteoarthritis", "rheumatoid arthritis", "gout", "trauma"],
        "bone erosion": ["rheumatoid arthritis", "gout", "infection", "malignancy"],
        "disc herniation": ["lumbar disc disease", "cervical disc disease", "radiculopathy"],
        "ligament tear": ["ACL tear", "MCL tear", "rotator cuff tear", "ankle sprain"],
        "bone marrow edema": ["stress fracture", "osteomyelitis", "contusion", "avascular necrosis"],
        # ── Cardiac ───────────────────────────────────────────
        "wall motion abnormality": ["myocardial infarction", "cardiomyopathy", "myocarditis"],
        "pericardial effusion": ["pericarditis", "malignancy", "uremia", "post-surgical", "autoimmune"],
        "valve calcification": ["aortic stenosis", "mitral stenosis", "degenerative valve disease"],
        "st elevation": ["STEMI", "pericarditis", "early repolarization", "left ventricular aneurysm"],
        "st depression": ["ischemia", "NSTEMI", "digoxin effect", "LVH strain pattern"],
        "prolonged qt": ["drug-induced", "electrolyte imbalance", "congenital long QT syndrome"],
        # ── Ophthalmology ─────────────────────────────────────
        "disc cupping": ["glaucoma", "optic neuropathy", "normal variant"],
        "macular edema": ["diabetic macular edema", "retinal vein occlusion", "uveitis", "post-surgical"],
        "cotton wool spots": ["hypertensive retinopathy", "diabetic retinopathy", "HIV retinopathy", "lupus"],
        "drusen": ["age-related macular degeneration", "optic disc drusen"],
        "retinal detachment": ["rhegmatogenous detachment", "tractional detachment", "exudative detachment"],
        "neovascularization": ["proliferative diabetic retinopathy", "retinal vein occlusion", "sickle cell"],
        "subretinal fluid": ["central serous retinopathy", "wet AMD", "Vogt-Koyanagi-Harada"],
        # ── Dermatology ───────────────────────────────────────
        "asymmetry": ["melanoma", "dysplastic nevus", "basal cell carcinoma"],
        "border irregularity": ["melanoma", "squamous cell carcinoma", "dysplastic nevus"],
        "color variegation": ["melanoma", "dysplastic nevus", "seborrheic keratosis"],
        "blue-white veil": ["melanoma", "blue nevus"],
        "pigment network irregular": ["melanoma", "dysplastic nevus", "lentigo maligna"],
        "ulceration": ["squamous cell carcinoma", "basal cell carcinoma", "melanoma", "pyoderma gangrenosum"],
        # ── Dental ────────────────────────────────────────────
        "periapical radiolucency": ["periapical abscess", "periapical granuloma", "radicular cyst"],
        "bone loss": ["periodontitis", "aggressive periodontitis", "medication-related osteonecrosis"],
        "radiopaque lesion": ["odontoma", "cementoblastoma", "osteoma", "torus"],
        # ── Pathology ─────────────────────────────────────────
        "high mitotic rate": ["high-grade malignancy", "lymphoma", "sarcoma"],
        "nuclear pleomorphism": ["carcinoma", "sarcoma", "dysplasia"],
        "glandular architecture": ["adenocarcinoma", "adenoma", "endometriosis"],
        "koilocytes": ["HPV infection", "cervical dysplasia"],
        # ── GI / Endoscopy ────────────────────────────────────
        "mucosal erythema": ["gastritis", "esophagitis", "inflammatory bowel disease"],
        "polyp": ["adenomatous polyp", "hyperplastic polyp", "sessile serrated lesion", "carcinoma"],
        "ulcer": ["peptic ulcer", "malignancy", "Crohn disease", "NSAID ulcer"],
        "stricture": ["malignancy", "Crohn disease", "radiation stricture", "eosinophilic esophagitis"],
        # ── Nuclear Medicine ──────────────────────────────────
        "focal uptake": ["malignancy", "infection", "inflammation", "fracture"],
        "diffuse uptake": ["thyroiditis", "diffuse large B-cell lymphoma", "sarcoidosis"],
        "photopenic area": ["infarction", "necrosis", "cyst", "artifact"],
    },
    "dx_to_findings": {
        # ── Chest ─────────────────────────────────────────────
        "pneumonia": {"required": ["opacity", "consolidation"], "supporting": ["air bronchograms", "fever", "cough"],
                      "location": "lobar or segmental", "severity_markers": ["bilateral", "multilobar", "cavitation"]},
        "heart failure": {"required": ["cardiomegaly"], "supporting": ["effusion", "edema", "cephalization"],
                         "location": "bilateral", "severity_markers": ["bilateral effusion", "pulmonary edema"]},
        "lung cancer": {"required": ["mass", "nodule"], "supporting": ["lymphadenopathy", "effusion", "bone destruction"],
                       "location": "unilateral", "severity_markers": ["metastasis", "effusion"]},
        "tuberculosis": {"required": ["opacity"], "supporting": ["cavitation", "upper lobe", "lymphadenopathy"],
                        "location": "upper lobes", "severity_markers": ["miliary", "bilateral", "cavitation"]},
        "pulmonary embolism": {"required": [], "supporting": ["hampton hump", "westermark sign", "effusion"],
                              "location": "peripheral", "severity_markers": ["bilateral", "saddle embolus"]},
        "aortic dissection": {"required": ["widened mediastinum"], "supporting": ["intimal flap", "double lumen"],
                             "location": "mediastinal", "severity_markers": ["type a", "pericardial effusion"]},
        "covid-19": {"required": ["ground glass"], "supporting": ["bilateral", "peripheral", "crazy paving"],
                    "location": "bilateral peripheral", "severity_markers": ["bilateral", ">50% involvement"]},
        # ── Neuro ─────────────────────────────────────────────
        "acute stroke": {"required": ["restricted diffusion"], "supporting": ["vessel occlusion", "perfusion mismatch"],
                        "location": "vascular territory", "severity_markers": ["large vessel", "hemorrhagic transformation"]},
        "brain tumor": {"required": ["mass", "mass effect"], "supporting": ["enhancement", "edema", "midline shift"],
                       "location": "variable", "severity_markers": ["ring enhancement", "hemorrhage", "hydrocephalus"]},
        "multiple sclerosis": {"required": ["white matter lesions"], "supporting": ["periventricular", "dawson fingers", "enhancement"],
                              "location": "periventricular, juxtacortical, infratentorial", "severity_markers": ["enhancing lesion", "cord lesion"]},
        # ── Cardiac ───────────────────────────────────────────
        "STEMI": {"required": ["st elevation"], "supporting": ["reciprocal changes", "q waves", "wall motion abnormality"],
                 "location": "coronary territory", "severity_markers": ["anterior", "extensive", "cardiogenic shock"]},
        "aortic stenosis": {"required": ["valve calcification"], "supporting": ["LVH", "reduced EF", "gradient elevation"],
                           "location": "aortic valve", "severity_markers": ["gradient >40mmHg", "AVA <1.0cm2"]},
        # ── Ophthalmology ─────────────────────────────────────
        "diabetic retinopathy": {"required": ["cotton wool spots", "neovascularization"],
                                "supporting": ["microaneurysms", "hemorrhages", "hard exudates", "macular edema"],
                                "location": "retinal", "severity_markers": ["proliferative", "macular edema", "vitreous hemorrhage"]},
        "glaucoma": {"required": ["disc cupping"], "supporting": ["elevated IOP", "RNFL thinning", "visual field defect"],
                    "location": "optic nerve", "severity_markers": ["C/D ratio >0.7", "advanced field loss"]},
        "wet AMD": {"required": ["subretinal fluid", "drusen"], "supporting": ["PED", "hemorrhage", "CNV"],
                   "location": "macular", "severity_markers": ["subfoveal CNV", "disciform scar"]},
        # ── Dermatology ───────────────────────────────────────
        "melanoma": {"required": ["asymmetry", "border irregularity"], "supporting": ["color variegation", "blue-white veil", "regression"],
                    "location": "skin", "severity_markers": ["ulceration", "depth >1mm", "satellite lesions"]},
        # ── Dental ────────────────────────────────────────────
        "periapical abscess": {"required": ["periapical radiolucency"], "supporting": ["tooth decay", "pain", "swelling"],
                              "location": "tooth apex", "severity_markers": ["sinus tract", "cellulitis"]},
        "periodontitis": {"required": ["bone loss"], "supporting": ["pocket depth", "bleeding", "mobility"],
                         "location": "periodontal", "severity_markers": ["generalized", "severe bone loss"]},
    },
    "urgency": {
        "emergent": [
            # Radiology
            "aortic dissection", "pulmonary embolism", "pneumothorax", "cardiac tamponade",
            "tension pneumothorax",
            # Neuro
            "intracranial hemorrhage", "stroke", "acute stroke", "subarachnoid hemorrhage",
            "status epilepticus", "spinal cord compression",
            # Cardiac
            "STEMI", "ventricular fibrillation", "cardiac arrest",
            # Ophthalmology
            "retinal detachment", "acute angle-closure glaucoma", "central retinal artery occlusion",
            # GI
            "bowel perforation", "massive GI bleed",
            # Dermatology
            "necrotizing fasciitis",
        ],
        "urgent": [
            # Radiology
            "pneumonia", "lung cancer", "heart failure", "fracture", "tuberculosis",
            # Neuro
            "brain tumor", "brain abscess", "hydrocephalus",
            # Cardiac
            "NSTEMI", "unstable angina", "aortic stenosis",
            # Ophthalmology
            "proliferative diabetic retinopathy", "wet AMD", "endophthalmitis",
            # Dermatology
            "melanoma", "squamous cell carcinoma",
            # GI
            "bowel obstruction", "appendicitis",
            # Pathology
            "high-grade malignancy",
            # Dental
            "periapical abscess", "Ludwig angina",
        ],
        "routine": [
            "granuloma", "degenerative changes", "atherosclerosis", "old fracture",
            "benign cyst", "lipoma", "seborrheic keratosis", "dental caries",
            "mild periodontitis", "senile cataract", "drusen", "hyperplastic polyp",
        ],
    },
}

MEDICAL_KNOWLEDGE_BASE = [
    # ── Radiology / Chest ─────────────────────────────────────
    "Normal chest X-ray shows clear lung fields bilaterally, normal cardiac silhouette with cardiothoracic ratio less than 0.5, clear costophrenic angles, intact bony structures, and midline trachea.",
    "Pneumonia typically presents as lobar or segmental consolidation with air bronchograms. Common organisms include Streptococcus pneumoniae.",
    "Congestive heart failure on CXR shows cardiomegaly (CTR > 0.5), cephalization of vessels, Kerley B lines, bilateral pleural effusions.",
    "Pneumothorax shows visceral pleural line with absent lung markings peripherally. Tension pneumothorax requires immediate decompression.",
    "Pulmonary nodule workup: < 6mm low risk, 6-8mm CT 6-12 months, > 8mm PET/CT or biopsy per Fleischner Society guidelines.",
    "CT pulmonary angiography is gold standard for PE. Direct sign: intraluminal filling defect. RV/LV > 1.0 suggests right heart strain.",
    "Stanford Type A aortic dissection involves ascending aorta (surgical emergency). Type B involves descending aorta only.",
    "Costophrenic angle blunting suggests at least 200-300mL of pleural fluid. Meniscus sign confirms free-flowing effusion.",
    "Normal mediastinal width on PA CXR is less than 8cm. Widened mediastinum raises concern for aortic pathology.",
    "Lung zones: Upper (above 2nd anterior rib), Mid (2nd-4th rib), Lower (below 4th rib). TB favors upper lobes.",
    # ── CT / MRI ──────────────────────────────────────────────
    "CT Hounsfield units: air -1000, fat -100, water 0, soft tissue 40-80, bone 400-1000. Window settings must match tissue of interest.",
    "MRI sequences: T1 (anatomy, fat bright), T2 (pathology, fluid bright), FLAIR (fluid suppressed, lesions bright), DWI (acute infarct/abscess bright).",
    "Brain MRI for stroke: DWI shows acute infarct as restricted diffusion (bright). ADC map shows corresponding dark signal.",
    # ── Mammography ───────────────────────────────────────────
    "BIRADS: 0-incomplete, 1-negative, 2-benign, 3-probably benign, 4-suspicious, 5-highly suggestive of malignancy, 6-known malignancy.",
    "Breast density categories: a (fatty), b (scattered), c (heterogeneously dense, may obscure lesions), d (extremely dense, lowers mammographic sensitivity).",
    # ── Ultrasound ────────────────────────────────────────────
    "Ultrasound: hypoechoic (darker than surroundings), hyperechoic (brighter), anechoic (black, fluid-filled). Posterior acoustic shadowing suggests calcification or dense tissue.",
    "Thyroid ultrasound: ACR TI-RADS scoring for nodules. TR1 benign, TR2 not suspicious, TR3 mildly suspicious, TR4 moderate, TR5 highly suspicious.",
    "Echocardiography: normal LVEF 55-70%. Wall motion scoring: 1=normal, 2=hypokinetic, 3=akinetic, 4=dyskinetic. TAPSE >17mm suggests normal RV function.",
    # ── Nuclear Medicine ──────────────────────────────────────
    "PET-CT: SUVmax >2.5 generally suspicious for malignancy. Deauville score for lymphoma: 1-3 favorable, 4-5 unfavorable. PERCIST for treatment response.",
    "Bone scan: increased uptake in metastases, fractures, infection. 'Superscan' with diffuse uptake suggests widespread metastatic disease.",
    "Thyroid scan: hot nodule (increased uptake) usually benign. Cold nodule (decreased uptake) requires biopsy to rule out malignancy.",
    # ── Ophthalmology ─────────────────────────────────────────
    "Diabetic retinopathy grading (ETDRS): mild NPDR (microaneurysms only), moderate NPDR (more than microaneurysms), severe NPDR (4-2-1 rule), PDR (neovascularization).",
    "OCT interpretation: normal foveal contour with central depression. Intraretinal fluid (cystoid spaces), subretinal fluid (dome-shaped elevation), sub-RPE fluid (PED).",
    "Glaucoma: C/D ratio >0.5 suspicious, >0.7 highly suspicious. RNFL thinning on OCT precedes visual field loss. Normal IOP 10-21 mmHg.",
    "AMD grading (AREDS): Category 1 (no drusen), 2 (small drusen), 3 (intermediate drusen or GA), 4 (advanced AMD: CNV or central GA).",
    "Central retinal artery occlusion: cherry-red spot on fundoscopy. Retinal whitening from ischemia. 90-minute window for intervention.",
    # ── Dermatology ───────────────────────────────────────────
    "ABCDE criteria for melanoma: Asymmetry, Border irregularity, Color variegation (>3 colors), Diameter >6mm, Evolution (change over time).",
    "Dermoscopy: typical pigment network (benign), atypical network (concerning). Blue-white veil strongly associated with melanoma.",
    "Basal cell carcinoma: pearly papule with telangiectasia, rolled borders. Dermoscopy shows arborizing vessels, leaf-like structures.",
    # ── Dental ────────────────────────────────────────────────
    "Dental caries classification: Class I (occlusal), II (interproximal posterior), III (interproximal anterior), IV (incisal edge), V (cervical).",
    "Periodontal bone loss: horizontal (generalized, even) vs vertical (localized angular defects). Furcation involvement graded I-III.",
    "Panoramic radiograph: evaluates mandible, maxilla, TMJ, sinuses, teeth. Limited for caries detection — periapicals needed.",
    # ── Pathology ──────────────────────────────────────────────
    "H&E staining: hematoxylin stains nuclei blue/purple, eosin stains cytoplasm/ECM pink. Most common histological stain.",
    "Tumor grading: well-differentiated (low grade, resembles normal tissue), moderately differentiated, poorly differentiated (high grade, aggressive).",
    "Pap smear Bethesda system: NILM (negative), ASC-US, ASC-H, LSIL, HSIL, AGC, squamous cell carcinoma.",
    "Immunohistochemistry (IHC): Ki-67 for proliferation index, ER/PR for breast cancer, HER2 for targeted therapy, PD-L1 for immunotherapy.",
    # ── Endoscopy / GI ────────────────────────────────────────
    "Paris classification for GI lesions: 0-Ip (pedunculated), 0-Is (sessile), 0-IIa (slightly elevated), 0-IIb (flat), 0-IIc (depressed), 0-III (excavated).",
    "Barrett esophagus: Prague classification C_M_ (circumferential and maximal extent). Intestinal metaplasia on biopsy confirms diagnosis.",
    "Colonoscopy polyp management: diminutive (<5mm) usually hyperplastic in rectosigmoid. Adenomas need complete removal. Surveillance per USMSTF guidelines.",
    # ── Cardiology / ECG ──────────────────────────────────────
    "ECG: normal sinus rhythm 60-100 bpm. PR 120-200ms. QRS <120ms. QTc <440ms (male), <460ms (female).",
    "STEMI criteria: ≥1mm ST elevation in ≥2 contiguous limb leads, or ≥2mm in ≥2 contiguous precordial leads. New LBBB equivalent.",
    "Atrial fibrillation: irregularly irregular, absent P waves. CHA2DS2-VASc score determines anticoagulation need.",
    # ── General ───────────────────────────────────────────────
    "Critical findings requiring immediate communication: tension pneumothorax, aortic dissection, PE, intracranial hemorrhage, retinal detachment, STEMI.",
    "AI-assisted diagnosis should always include confidence levels and recommendation for clinical correlation.",
    "Differential diagnosis should always be provided with supporting evidence. Report uncertainty explicitly.",
]

# ═══════════════════════════════════════════════════════════
#  FINDING EXTRACTION (shared utility)
# ═══════════════════════════════════════════════════════════

ANATOMY_KEYWORDS = {
    "lung": ["lung", "pulmonary", "lobe", "bronch", "alveol", "parenchym", "hilum", "hilar"],
    "heart": ["heart", "cardiac", "cardiomegaly", "pericardi", "atri", "ventricl", "valv",
              "myocard", "endocard", "mitral", "aortic valve", "tricuspid", "septum"],
    "bone": ["bone", "rib", "spine", "vertebr", "fracture", "sclerotic", "cortical",
             "joint", "cartilage", "ligament", "tendon", "meniscus", "disc"],
    "pleura": ["pleural", "effusion", "costophrenic", "meniscus"],
    "mediastinum": ["mediastin", "aortic", "trachea", "lymph node"],
    "brain": ["brain", "cerebr", "ventricl", "cortex", "white matter", "gray matter",
              "thalamus", "basal ganglia", "cerebellum", "brainstem", "meninges"],
    "abdomen": ["liver", "kidney", "spleen", "pancrea", "bowel", "intestin",
                "gallbladder", "bile duct", "adrenal", "mesentery", "peritoneum"],
    "eye": ["retina", "macula", "fovea", "optic disc", "optic nerve", "choroid",
            "vitreous", "cornea", "lens", "iris", "sclera", "RNFL"],
    "skin": ["skin", "lesion", "epiderm", "dermis", "melanocyt", "keratinoc",
             "nail", "hair follicle", "subcutaneous"],
    "dental": ["tooth", "teeth", "mandib", "maxill", "periodon", "periapic",
               "pulp", "enamel", "dentin", "gingiv", "alveolar bone"],
    "gi_tract": ["esophag", "stomach", "gastric", "duoden", "colon", "rectum",
                 "mucos", "polyp", "ulcer", "stricture", "Barrett"],
    "vascular": ["vessel", "artery", "vein", "stenosis", "aneurysm", "thrombus",
                 "embolus", "collateral", "dissection"],
    "breast": ["breast", "mammary", "nipple", "axillary", "calcification",
               "mass", "density", "BI-RADS"],
}

FINDING_VERBS = ["show", "reveal", "demonstrate", "suggest", "indicate", "consistent",
                 "evidence", "appear", "present", "note", "identif", "detect", "observe"]

NEGATION_PHRASES = ["no ", "no evidence", "without ", "absent", "unremarkable", "negative for",
                    "not seen", "not identified", "not detected", "not observed", "normal",
                    "clear", "intact", "within normal limits", "no significant"]


def extract_findings(text: str) -> list[dict]:
    """Extract structured findings from medical text."""
    findings = []
    for sent in re.split(r'(?<=[.!?])\s+', text):
        sent_lower = sent.lower()
        has_verb = any(v in sent_lower for v in FINDING_VERBS)
        if not has_verb and len(sent.split()) < 4:
            continue
        location = "unspecified"
        for loc_name, keywords in ANATOMY_KEYWORDS.items():
            if any(k in sent_lower for k in keywords):
                location = loc_name
                break
        is_normal = any(n in sent_lower for n in NEGATION_PHRASES)
        severity = "normal" if is_normal else "moderate"
        if any(w in sent_lower for w in ["severe", "significant", "large", "extensive", "critical"]):
            severity = "severe"
        findings.append({"sentence": sent.strip(), "location": location,
                        "is_normal": is_normal, "severity": severity})
    return findings


# ═══════════════════════════════════════════════════════════
#  1. MEDICAL KNOWLEDGE ENGINE
# ═══════════════════════════════════════════════════════════

class MedicalKnowledgeEngine:
    """Causal reasoning via medical knowledge graph."""

    def __init__(self):
        self.kg = MEDICAL_KNOWLEDGE_GRAPH

    def expand_findings(self, findings_list):
        expansions = []
        for finding in findings_list:
            sentence_lower = finding.get("sentence", "").lower()
            for keyword, diagnoses in self.kg["findings_to_dx"].items():
                if keyword in sentence_lower:
                    expansions.append({
                        "trigger": keyword, "location": finding.get("location", "unspecified"),
                        "possible_diagnoses": diagnoses[:5],
                        "source_finding": finding.get("sentence", "")[:100],
                    })
        return expansions

    def validate_diagnosis(self, diagnosis, findings_text):
        dx_lower = diagnosis.lower(); text_lower = findings_text.lower()
        for dx_name, criteria in self.kg["dx_to_findings"].items():
            if dx_name in dx_lower:
                required_found = sum(1 for r in criteria["required"] if r in text_lower)
                supporting_found = sum(1 for s in criteria["supporting"] if s in text_lower)
                total_required = max(len(criteria["required"]), 1)
                score = (required_found / total_required) * 0.6 + min(supporting_found / 3, 1.0) * 0.4
                return {"diagnosis": dx_name, "validation_score": round(score, 3),
                        "required_found": required_found, "required_total": total_required,
                        "supporting_found": supporting_found, "expected_location": criteria.get("location", ""),
                        "severity_markers_present": [m for m in criteria.get("severity_markers", []) if m in text_lower]}
        return {"diagnosis": diagnosis, "validation_score": 0.5, "note": "Not in knowledge graph"}

    def get_differential(self, findings_text, top_k=5):
        text_lower = findings_text.lower()
        dx_scores = {}
        for keyword, diagnoses in self.kg["findings_to_dx"].items():
            if keyword in text_lower:
                for dx in diagnoses:
                    dx_scores[dx] = dx_scores.get(dx, 0) + 1
        ranked = sorted(dx_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"diagnosis": dx, "supporting_findings_count": count,
                 "urgency": self._get_urgency(dx)} for dx, count in ranked]

    def _get_urgency(self, diagnosis):
        for level, conditions in self.kg["urgency"].items():
            if any(c in diagnosis.lower() for c in conditions):
                return level
        return "routine"


# ═══════════════════════════════════════════════════════════
#  2. ENHANCED MEDICAL RAG
# ═══════════════════════════════════════════════════════════

class EnhancedMedicalRAG:
    """Pre-seeded knowledge + PubMed search + guideline retrieval."""

    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.collection = None
        self._initialized = False
        self._embedding_model = embedding_model
        self._pubmed_cache = {}

    def initialize(self):
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            efn = SentenceTransformerEmbeddingFunction(model_name=self._embedding_model)
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection("medical_knowledge_v6", embedding_function=efn)
            if self.collection.count() < len(MEDICAL_KNOWLEDGE_BASE):
                self.collection.add(
                    documents=MEDICAL_KNOWLEDGE_BASE,
                    ids=[f"med_kb_{i}" for i in range(len(MEDICAL_KNOWLEDGE_BASE))],
                    metadatas=[{"source": "medical_guidelines"} for _ in MEDICAL_KNOWLEDGE_BASE],
                )
                logger.info(f"Seeded RAG with {len(MEDICAL_KNOWLEDGE_BASE)} references")
            self._initialized = True
        except Exception as e:
            logger.warning(f"RAG init failed: {e}")
            self._initialized = True

    def retrieve(self, query, top_k=5):
        if not self._initialized:
            self.initialize()
        if not self.collection or self.collection.count() == 0:
            return []
        try:
            results = self.collection.query(query_texts=[query], n_results=min(top_k, self.collection.count()))
            if results and results["documents"] and results["documents"][0]:
                return [{"text": doc, "distance": dist, "source": meta.get("source", "unknown")}
                        for doc, dist, meta in zip(
                            results["documents"][0],
                            results["distances"][0] if results.get("distances") else [0] * len(results["documents"][0]),
                            results["metadatas"][0] if results.get("metadatas") else [{}] * len(results["documents"][0]))]
        except Exception as e:
            logger.warning(f"RAG retrieve failed: {e}")
        return []

    def search_pubmed(self, query, max_results=3):
        if query in self._pubmed_cache:
            return self._pubmed_cache[query]
        try:
            import httpx
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query.replace(' ', '+')}&retmax={max_results}&retmode=json"
            resp = httpx.get(search_url, timeout=5.0)
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(ids)}&retmode=json"
            resp = httpx.get(fetch_url, timeout=5.0)
            results = []
            for uid, data in resp.json().get("result", {}).items():
                if uid == "uids":
                    continue
                results.append({"title": data.get("title", ""), "pubmed_id": uid,
                               "journal": data.get("source", ""), "year": data.get("pubdate", "")[:4]})
            self._pubmed_cache[query] = results
            return results
        except Exception as e:  # noqa: broad-except logged
            return []

    def enrich_prompt(self, question, top_k=3):
        refs = self.retrieve(question, top_k)
        if not refs:
            return question
        ctx = "\n".join([f"[Ref {i+1}]: {r['text']}" for i, r in enumerate(refs)])
        return f"Medical references:\n{ctx}\n\nBased on above guidelines and your medical knowledge, {question}"


# ═══════════════════════════════════════════════════════════
#  3. MEDICAL REASONING ENGINE
# ═══════════════════════════════════════════════════════════

class MedicalReasoningEngine:
    """Structured reasoning — Extract → Reason → Verify → Rank → Explain."""

    def __init__(self):
        self.kg = MedicalKnowledgeEngine()

    def reason(self, model_outputs, modality="xray", rag=None):
        if not model_outputs or not any(o.get("answer") for o in model_outputs):
            return {
                "findings": [], "contradictions": [],
                "differential_diagnosis": [], "confidence": 0.1,
                "evidence": [], "reasoning_chain": ["STEP 1 — No model outputs available."],
                "risk_assessment": {"risk_level": "routine", "urgent_findings": [],
                                   "requires_immediate_attention": False,
                                   "recommend_specialist_review": True,
                                   "safety_message": "⚠️ No models produced output."},
                "finding_count": 0,
            }
        all_findings = self._extract_all_findings(model_outputs)
        contradictions = self._detect_contradictions(all_findings)
        combined_text = " ".join([o.get("answer", "") for o in model_outputs if o.get("answer")])
        differential = self.kg.get_differential(combined_text, top_k=5)
        validated_dx = [{**dx, **self.kg.validate_diagnosis(dx["diagnosis"], combined_text)} for dx in differential]
        confidence = self._compute_true_confidence(model_outputs, all_findings, contradictions, validated_dx)
        evidence = self._retrieve_evidence(differential, rag)
        reasoning_chain = self._build_reasoning_chain(all_findings, contradictions, validated_dx, evidence)
        risk = self._assess_risk(validated_dx, confidence)

        return {
            "findings": all_findings, "contradictions": contradictions,
            "differential_diagnosis": validated_dx, "confidence": confidence,
            "evidence": evidence, "reasoning_chain": reasoning_chain,
            "risk_assessment": risk, "finding_count": len(all_findings),
        }

    def _extract_all_findings(self, outputs):
        all_f = []
        for out in outputs:
            findings = extract_findings(out.get("answer", ""))
            for f in findings:
                f["source_model"] = out.get("model", "unknown")
                all_f.append(f)
        return all_f

    def _detect_contradictions(self, findings):
        by_location = {}
        for f in findings:
            loc = f.get("location", "unspecified")
            by_location.setdefault(loc, []).append(f)
        contradictions = []
        for loc, fs in by_location.items():
            normals = [f for f in fs if f.get("is_normal")]
            abnormals = [f for f in fs if not f.get("is_normal")]
            if normals and abnormals:
                contradictions.append({
                    "location": loc,
                    "normal_by": list(set(f.get("source_model", "?") for f in normals)),
                    "abnormal_by": list(set(f.get("source_model", "?") for f in abnormals)),
                    "resolution": "Favor abnormal finding (higher clinical safety)" if len(abnormals) >= len(normals)
                                  else "Conflicting — recommend clinical correlation",
                })
        return contradictions

    def _compute_true_confidence(self, outputs, findings, contradictions, validated_dx):
        answers = [o.get("answer", "") for o in outputs if o.get("answer")]
        if len(answers) >= 2:
            agreements = []
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    wa = set(re.findall(r'\w+', answers[i].lower()))
                    wb = set(re.findall(r'\w+', answers[j].lower()))
                    if wa and wb:
                        agreements.append(len(wa & wb) / len(wa | wb))
            model_agreement = float(np.mean(agreements)) if agreements else 0.5
        else:
            model_agreement = 0.5
        evidence_strength = float(np.mean([dx.get("validation_score", 0.5) for dx in validated_dx[:3]])) if validated_dx else 0.5
        consistency = max(0.0, 1.0 - min(len(contradictions) * 0.1, 0.3))
        richness = min(len(findings) / 10.0, 1.0)
        confidence = model_agreement * 0.35 + evidence_strength * 0.30 + consistency * 0.20 + richness * 0.15
        return round(min(max(confidence, 0.1), 0.95), 3)

    def _retrieve_evidence(self, differential, rag):
        evidence = []
        if not rag:
            return evidence
        for dx in differential[:3]:
            refs = rag.retrieve(f"{dx['diagnosis']} imaging findings diagnosis", top_k=2)
            if refs:
                evidence.append({"diagnosis": dx["diagnosis"],
                                "references": [r["text"][:200] for r in refs]})
        return evidence

    def _build_reasoning_chain(self, findings, contradictions, validated_dx, evidence):
        chain = []
        abnormal = [f for f in findings if not f.get("is_normal")]
        normal = [f for f in findings if f.get("is_normal")]
        chain.append(f"STEP 1 — Findings: {len(findings)} total ({len(abnormal)} abnormal, {len(normal)} normal)")
        if abnormal:
            locs = list(set(f.get("location", "?") for f in abnormal))
            chain.append(f"  Abnormal locations: {', '.join(locs[:5])}")
        if contradictions:
            chain.append(f"STEP 2 — Contradictions: {len(contradictions)} conflicts")
            for c in contradictions[:2]:
                chain.append(f"  ⚠️ {c['location']}: {c['resolution']}")
        else:
            chain.append("STEP 2 — Contradictions: None ✓")
        chain.append(f"STEP 3 — Differential: {len(validated_dx)} candidates")
        for dx in validated_dx[:3]:
            chain.append(f"  {'→' if dx.get('validation_score', 0) > 0.5 else '?'} {dx['diagnosis']} (score={dx.get('validation_score', 0):.2f})")
        if evidence:
            chain.append(f"STEP 4 — Evidence: {len(evidence)} diagnoses have references")
        if validated_dx:
            chain.append(f"STEP 5 — Primary: {validated_dx[0]['diagnosis']} ({validated_dx[0].get('validation_score', 0):.2f})")
        return chain

    def _assess_risk(self, validated_dx, confidence):
        risk_level = "routine"
        urgent_findings = []
        for dx in validated_dx:
            u = dx.get("urgency", "routine")
            if u == "emergent":
                risk_level = "emergent"
                urgent_findings.append({"diagnosis": dx["diagnosis"], "urgency": "emergent"})
            elif u == "urgent" and risk_level != "emergent":
                risk_level = "urgent"
                urgent_findings.append({"diagnosis": dx["diagnosis"], "urgency": "urgent"})
        needs_review = confidence < 0.5 or risk_level != "routine"
        safety_messages = {
            "emergent": "🔴 CRITICAL: Potentially life-threatening. Immediate physician review required.",
            "urgent": "🟠 URGENT: Requires timely clinical attention within 24-48 hours.",
        }
        if confidence < 0.4:
            msg = "⚠️ LOW CONFIDENCE: AI uncertain. Specialist consultation recommended."
        elif confidence < 0.6:
            msg = "🟡 MODERATE: Correlate with clinical findings."
        else:
            msg = safety_messages.get(risk_level, "🟢 ROUTINE: No findings requiring immediate attention.")
        return {"risk_level": risk_level, "urgent_findings": urgent_findings,
                "requires_immediate_attention": risk_level == "emergent",
                "recommend_specialist_review": needs_review, "safety_message": msg}


# ═══════════════════════════════════════════════════════════
#  4. DYNAMIC FUSION ENGINE
# ═══════════════════════════════════════════════════════════

class DynamicFusionEngine:
    """Uncertainty-aware dynamic fusion with model debate and domain-specialist boost."""

    # v7.0: Domain specialist boost factors (same as fusion module)
    DOMAIN_BOOSTS = {
        "xray": {"chexagent_8b": 0.20, "chexagent_3b": 0.10},
        "ct": {"merlin": 0.15, "med3dvlm": 0.10},
        "mri": {"med3dvlm": 0.12},
        "pathology": {"pathgen": 0.20},
        "cytology": {"pathgen": 0.15},
        "histopathology": {"pathgen": 0.20},
        "fundoscopy": {"retfound": 0.12},
        "oct": {"retfound": 0.12},
    }

    def fuse(self, results, reasoning_output=None, modality="general_medical"):
        successful = [r for r in results if r.get("status") == "success"]
        if not successful:
            return {
                "consensus_answer": (
                    "All analysis models encountered errors. Please verify the input "
                    "file format and retry. If the issue persists, try --complexity simple."
                ),
                "confidence": 0, "best_model": "none", "model_count": 0,
            }

        boosts = self.DOMAIN_BOOSTS.get(modality, {})
        model_analyses = []
        for r in successful:
            res = r.get("result", {}); mk = r["model_key"]
            answer = res.get("answer", "")
            conf = res.get("confidence", 0.5)
            findings = extract_findings(answer)
            reliability = self._compute_reliability(answer, reasoning_output)
            richness = min(1.0, len(findings) / 8.0)

            # v7.0: Domain specialist boost
            specialist_boost = boosts.get(mk, 0.0)
            dynamic_score = (
                conf * 0.30
                + reliability * 0.30
                + richness * 0.25
                + specialist_boost * 0.15 / max(0.01, max(boosts.values()) if boosts else 0.01)
            )

            model_analyses.append({
                "model": mk, "answer": answer, "thinking": res.get("thinking", ""),
                "is_generative": True, "confidence": conf, "findings": findings,
                "reliability": round(reliability, 3), "dynamic_score": round(dynamic_score, 3),
                "specialist_boost": round(specialist_boost, 3),
                "finding_count": len(findings), "excerpt": answer[:300],
            })

        best = max(model_analyses, key=lambda x: x["dynamic_score"])
        debate = self._debate(best, model_analyses) if len(model_analyses) > 1 else {"agreements": [], "challenges": []}

        weights = [m["dynamic_score"] for m in model_analyses]
        confs = [m["confidence"] for m in model_analyses]
        ensemble_conf = float(np.average(confs, weights=weights)) if sum(weights) > 0 else 0.5
        agreement = self._agreement([m["answer"] for m in model_analyses])

        return {
            "consensus_answer": best["answer"], "best_model": best["model"],
            "confidence": round(ensemble_conf, 3), "agreement_score": round(agreement, 3),
            "uncertainty": round(1.0 - ensemble_conf, 3), "model_count": len(model_analyses),
            "modality": modality,
            "all_answers": model_analyses, "individual_results": model_analyses,
            "debate": debate, "dynamic_scores": {m["model"]: m["dynamic_score"] for m in model_analyses},
        }

    def _compute_reliability(self, answer, reasoning):
        if not reasoning:
            return 0.7
        dx_list = reasoning.get("differential_diagnosis", [])
        if not dx_list:
            return 0.7
        answer_lower = answer.lower()
        matches = sum(1 for dx in [d["diagnosis"] for d in dx_list[:3]] if dx in answer_lower)
        return min(0.5 + matches * 0.2, 1.0)

    def _debate(self, best_model, all_models):
        best_findings = set(f.get("location", "") for f in best_model.get("findings", []))
        agreements, challenges = [], []
        for m in all_models:
            if m["model"] == best_model["model"]:
                continue
            m_findings = set(f.get("location", "") for f in m.get("findings", []))
            common = best_findings & m_findings
            if common:
                agreements.append({"model": m["model"], "agrees_on": list(common)})
            unique = m_findings - best_findings
            if unique:
                challenges.append({"model": m["model"], "additional_findings": list(unique)})
        return {"agreements": agreements, "challenges": challenges}

    def _agreement(self, texts):
        if len(texts) < 2:
            return 1.0
        sims = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                wa = set(re.findall(r'\w+', texts[i].lower()))
                wb = set(re.findall(r'\w+', texts[j].lower()))
                if wa and wb:
                    sims.append(len(wa & wb) / len(wa | wb))
        return float(np.mean(sims)) if sims else 0.0


# ═══════════════════════════════════════════════════════════
#  5. CLINICAL SAFETY LAYER
# ═══════════════════════════════════════════════════════════

class ClinicalSafetyLayer:
    """Hallucination detection, confidence gating, mandatory disclaimers."""

    HALLUCINATION_PATTERNS = [
        r"(?:100|99)\s*%\s*(?:certain|confident|sure)",
        r"definitely\s+(?:is|has|shows)\s+(?:cancer|malignant|tumor)",
        r"no\s+(?:need|reason)\s+(?:to|for)\s+(?:further|additional)\s+(?:test|evaluation|workup)",
        r"pathognomonic",
        r"(?:I|the AI)\s+(?:am|is)\s+(?:a|your)\s+doctor",
    ]

    def validate(self, report_text, reasoning_output=None, fused_result=None):
        issues = []
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, report_text, re.IGNORECASE):
                issues.append({"type": "hallucination", "detail": f"Pattern: {pattern[:60]}"})

        confidence = (reasoning_output or {}).get("confidence", (fused_result or {}).get("confidence", 0.5))
        if confidence < 0.3:
            issues.append({"type": "very_low_confidence", "value": confidence})
        elif confidence < 0.5:
            issues.append({"type": "low_confidence", "value": confidence})

        contradictions = (reasoning_output or {}).get("contradictions", [])
        if contradictions:
            issues.append({"type": "model_contradiction", "count": len(contradictions)})

        risk_level = (reasoning_output or {}).get("risk_assessment", {}).get("risk_level", "routine")
        safe_text = re.sub(r'\b(definitely|certainly|100%|guaranteed)\b', 'likely', report_text, flags=re.IGNORECASE)

        return {
            "is_safe": not any(i["type"] in ("hallucination", "very_low_confidence") for i in issues),
            "issues": issues, "risk_level": risk_level,
            "confidence_adequate": confidence >= 0.5,
            "safe_report": safe_text, "issue_count": len(issues),
        }


# ═══════════════════════════════════════════════════════════
#  6. SELF-REFLECTION LOOP
# ═══════════════════════════════════════════════════════════

class SelfReflectionLoop:
    """Post-analysis critique and refinement."""

    def reflect(self, initial_report, model=None):
        critiques = self._nlp_critique(initial_report)

        if model and hasattr(model, 'is_loaded') and model.is_loaded:
            try:
                result = model.analyze(
                    text=f"Review critically. What could be wrong or missed?\n\nReport:\n{initial_report[:500]}",
                    modality="text")
                if result.get("answer"):
                    critiques.append({"type": "model_critique", "content": result["answer"][:300]})
            except Exception as e:  # noqa: broad-except logged
                pass

        improvements = []
        for c in critiques:
            ctype = c.get("type", "")
            if ctype == "missing_section":
                improvements.append(f"Add missing section to report")
            elif ctype == "tunnel_vision":
                improvements.append("Consider differential diagnosis")
            elif ctype == "excessive_hedging":
                improvements.append("Be more specific where evidence supports it")
            elif ctype == "model_critique":
                improvements.append(f"Model suggests: {c['content'][:150]}")

        return {"critiques": critiques, "improvements": improvements, "critique_count": len(critiques)}

    def _nlp_critique(self, text):
        critiques = []
        text_lower = text.lower()
        for section in ["findings", "impression", "recommendation"]:
            if section not in text_lower:
                critiques.append({"type": "missing_section", "content": f"Missing '{section}'"})
        vague_count = sum(1 for v in ["appears to", "may suggest", "cannot exclude", "questionable"] if v in text_lower)
        if vague_count > 3:
            critiques.append({"type": "excessive_hedging", "content": f"{vague_count} hedging phrases"})
        dx_kw = ["pneumonia", "fracture", "tumor", "cancer", "edema", "effusion", "hemorrhage"]
        mentioned = [d for d in dx_kw if d in text_lower]
        if len(mentioned) == 1 and len(text) > 200:
            critiques.append({"type": "tunnel_vision", "content": f"Only {mentioned[0]} mentioned"})
        return critiques


# ═══════════════════════════════════════════════════════════
#  7. MULTI-AGENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

class MultiAgentOrchestrator:
    """Assigns specialist roles to model outputs. v7.0: Domain specialists added."""

    AGENT_ROLES = {
        "radiologist": {
            "models": [
                "hulu_med_7b", "hulu_med_14b", "hulu_med_32b",
                "medgemma_4b", "medgemma_27b",
                "chexagent_8b", "chexagent_3b",
                "radfm",
            ],
        },
        "pathologist": {
            "models": ["pathgen", "medgemma_4b", "hulu_med_7b"],
        },
        "ophthalmologist": {
            "models": ["retfound", "medgemma_4b", "hulu_med_7b"],
        },
        "volumetric_specialist": {
            "models": ["med3dvlm", "merlin", "radfm"],
        },
        "critic": {
            "models": ["medix_r1_2b", "medix_r1_8b", "medix_r1_30b"],
        },
        "researcher": {
            "models": ["biomedclip"],
        },
    }

    def orchestrate(self, model_results, reasoning_output):
        agent_outputs = {}
        for role, config in self.AGENT_ROLES.items():
            role_results = [r for r in model_results
                           if r.get("model_key") in config["models"] and r.get("status") == "success"]
            if role_results:
                agent_outputs[role] = {
                    "role": role,
                    "outputs": [{"model": r["model_key"], "answer": r.get("result", {}).get("answer", "")[:300]}
                               for r in role_results],
                }
        # Decision maker synthesis
        dx = (reasoning_output or {}).get("differential_diagnosis", [])
        risk = (reasoning_output or {}).get("risk_assessment", {})
        rec = []
        if risk.get("risk_level") == "emergent":
            rec.append("IMMEDIATE ACTION: Critical finding requires emergent physician review.")
        if dx:
            rec.append(f"Primary: {dx[0]['diagnosis']} ({dx[0].get('validation_score', 0):.0%})")
            if len(dx) > 1:
                rec.append(f"Differential: {', '.join(d['diagnosis'] for d in dx[1:3])}")
        agent_outputs["decision_maker"] = {"role": "decision_maker", "recommendation": " | ".join(rec)}
        return agent_outputs


# ═══════════════════════════════════════════════════════════
#  8. EXPLAINABILITY ENGINE
# ═══════════════════════════════════════════════════════════

class ExplainabilityEngine:
    """'Why this diagnosis?' — structured feature attribution."""

    def explain(self, reasoning_output, fused_result=None):
        explanations = []
        for dx in (reasoning_output or {}).get("differential_diagnosis", [])[:3]:
            exp = {
                "diagnosis": dx.get("diagnosis", ""),
                "confidence": f"{dx.get('validation_score', 0):.0%}",
                "urgency": dx.get("urgency", "routine"),
                "reasoning": [],
            }
            if dx.get("required_found", 0) > 0:
                exp["reasoning"].append(f"Found {dx['required_found']}/{dx['required_total']} required features")
            if dx.get("supporting_found", 0) > 0:
                exp["reasoning"].append(f"Found {dx['supporting_found']} supporting findings")
            if dx.get("severity_markers_present"):
                exp["reasoning"].append(f"Severity markers: {', '.join(dx['severity_markers_present'])}")
            explanations.append(exp)

        parts = ["── 🧠 AI Reasoning Explanation ──\n"]
        for i, exp in enumerate(explanations):
            icon = '🔴' if exp['urgency'] == 'emergent' else '🟡' if exp['urgency'] == 'urgent' else '🟢'
            parts.append(f"{icon} #{i+1}: {exp['diagnosis']} ({exp['confidence']})")
            for r in exp.get("reasoning", []):
                parts.append(f"   → {r}")
            parts.append("")

        return {"explanations": explanations, "readable": "\n".join(parts)}
