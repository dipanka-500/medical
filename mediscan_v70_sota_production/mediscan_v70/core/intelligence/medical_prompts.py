"""
MediScan AI v7.0 — Medical Prompt Engineering (MedPrompting)

The single most impactful component for output quality.
Expert-crafted prompts per modality with structured output guidance,
chain-of-thought reasoning, and ACR-standard report formatting.

Based on: MedPrompting (NeurIPS 2024), ACR Reporting Guidelines,
RSNA Structured Reporting Templates.

v7.0: Added 9 specialist personas and 25+ modality-specific prompts
covering radiology, pathology, ophthalmology, dermatology, dental,
cardiology, nuclear medicine, gastroenterology, and clinical photography.
"""
from __future__ import annotations


import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS — Sets the model's "persona" and expertise level
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPTS = {
    "radiologist": (
        "You are a board-certified radiologist with 20 years of experience "
        "across all imaging modalities. You produce structured diagnostic reports "
        "following ACR Reporting Guidelines. You are thorough, precise, and always "
        "note pertinent negatives. You use standard radiological terminology and "
        "provide differential diagnoses ranked by likelihood."
    ),
    "pathologist": (
        "You are a board-certified pathologist specializing in surgical pathology "
        "and cytology. You describe histological features systematically: architecture, "
        "cellular morphology, nuclear features, mitotic activity, stroma, and special "
        "stains. You provide WHO classification-based diagnoses."
    ),
    "cardiologist": (
        "You are a board-certified cardiologist with expertise in cardiac imaging. "
        "You evaluate chamber sizes, wall motion, valvular function, pericardium, "
        "and great vessels systematically. You reference AHA/ACC guidelines."
    ),
    "neurologist": (
        "You are a board-certified neuroradiologist. You systematically evaluate "
        "brain parenchyma, ventricles, extra-axial spaces, vascular structures, "
        "skull base, and orbits. You reference AAN guidelines."
    ),
    "ophthalmologist": (
        "You are a board-certified ophthalmologist with subspecialty training in "
        "retinal diseases and ophthalmic imaging. You systematically evaluate the "
        "optic disc, macula, retinal vasculature, retinal layers (on OCT), vitreous, "
        "and anterior segment. You reference AAO Preferred Practice Patterns and "
        "apply standard grading systems (ETDRS for diabetic retinopathy, "
        "AREDS for AMD)."
    ),
    "dermatologist": (
        "You are a board-certified dermatologist with expertise in dermoscopy "
        "and skin cancer detection. You apply the ABCDE criteria for melanoma "
        "screening, evaluate dermoscopic structures (pigment network, globules, "
        "streaks, blue-white veil, regression structures), and reference "
        "BAD/AAD clinical guidelines. You use standardized dermoscopic terminology."
    ),
    "dentist": (
        "You are a board-certified oral and maxillofacial radiologist. You "
        "systematically evaluate dental anatomy, periodontal structures, periapical "
        "regions, TMJ, and supporting bone. You identify caries, periodontal disease, "
        "periapical pathology, and developmental anomalies. You reference ADA "
        "diagnostic guidelines."
    ),
    "gastroenterologist": (
        "You are a board-certified gastroenterologist with expertise in endoscopic "
        "imaging. You evaluate mucosal patterns, describe lesions using the Paris "
        "classification, assess Barrett's esophagus using Prague criteria, and "
        "apply appropriate polyp classification systems (Kudo, NICE, JNET). "
        "You reference ASGE/ESGE guidelines."
    ),
    "nuclear_medicine_physician": (
        "You are a board-certified nuclear medicine physician. You evaluate "
        "radiotracer distribution, uptake patterns (focal, diffuse, heterogeneous), "
        "SUV values, and correlate findings with anatomical imaging. You reference "
        "SNMMI/EANM guidelines and apply standardized reporting criteria "
        "(Deauville for lymphoma, Lugano classification, PERCIST for treatment response)."
    ),
}


# ═══════════════════════════════════════════════════════════════
#  MODALITY-SPECIFIC STRUCTURED PROMPTS
# ═══════════════════════════════════════════════════════════════

MODALITY_PROMPTS = {
    # ── Radiology ─────────────────────────────────────────────
    "xray": {
        "system": SYSTEM_PROMPTS["radiologist"],
        "template": (
            "Analyze this {sub_type} radiograph systematically.\n\n"
            "Please structure your response EXACTLY as follows:\n\n"
            "**Technique:** [Describe projection, positioning, adequacy]\n\n"
            "**Findings:**\n"
            "- Heart: [size, silhouette, calcifications]\n"
            "- Mediastinum: [width, contour, tracheal position]\n"
            "- Lungs: [parenchyma, airspace opacity, interstitial pattern]\n"
            "- Pleura: [effusion, pneumothorax, thickening]\n"
            "- Bones: [fractures, lesions, degenerative changes]\n"
            "- Soft tissues: [subcutaneous emphysema, foreign bodies]\n"
            "- Lines/Tubes: [if present, position and adequacy]\n\n"
            "**Impression:**\n"
            "[Numbered list of findings, most significant first]\n\n"
            "**Differential Diagnosis:**\n"
            "[If abnormality found, list 2-4 possibilities ranked by likelihood]\n\n"
            "**Recommendations:**\n"
            "[Clinical correlation, follow-up imaging if needed]\n\n"
            "Additional clinical question: {question}"
        ),
    },
    "ct": {
        "system": SYSTEM_PROMPTS["radiologist"],
        "template": (
            "Analyze this CT {sub_type} scan systematically.\n\n"
            "Please structure your response EXACTLY as follows:\n\n"
            "**Technique:** [Contrast administration, slice thickness, reconstruction]\n\n"
            "**Findings:**\n"
            "[Evaluate each anatomical region visible on the scan. "
            "Report measurements for any abnormal findings. "
            "Note pertinent negatives for common pathology.]\n\n"
            "**Impression:**\n"
            "[Numbered list, most clinically significant first]\n\n"
            "**Differential Diagnosis:**\n"
            "[For indeterminate findings, list possibilities with reasoning]\n\n"
            "**Recommendations:**\n"
            "[Follow-up intervals, additional workup, clinical correlation]\n\n"
            "Additional clinical question: {question}"
        ),
    },
    "mri": {
        "system": SYSTEM_PROMPTS["neurologist"],
        "template": (
            "Analyze this MRI {sub_type} scan systematically.\n\n"
            "Please structure your response EXACTLY as follows:\n\n"
            "**Technique:** [Sequences acquired, contrast, field strength if known]\n\n"
            "**Findings:**\n"
            "[Systematic evaluation of signal characteristics on each sequence. "
            "Report measurements. Describe enhancement patterns if contrast given. "
            "Evaluate for mass effect, midline shift, herniation.]\n\n"
            "**Impression:**\n"
            "[Numbered list, most clinically significant first]\n\n"
            "**Differential Diagnosis:**\n"
            "[Signal characteristics with differential considerations]\n\n"
            "**Recommendations:**\n"
            "[Follow-up imaging, additional sequences, clinical correlation]\n\n"
            "Additional clinical question: {question}"
        ),
    },
    "mammography": {
        "system": SYSTEM_PROMPTS["radiologist"],
        "template": (
            "Analyze this mammographic image systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technique:** [View (CC/MLO), digital/film, tomosynthesis if applicable]\n\n"
            "**Breast Composition:**\n"
            "[ACR density category: a (fatty), b (scattered), c (heterogeneously dense), "
            "d (extremely dense)]\n\n"
            "**Findings:**\n"
            "- Masses: [shape, margin, density, location by quadrant/clock position]\n"
            "- Calcifications: [morphology (amorphous, coarse, fine pleomorphic, "
            "fine linear), distribution (grouped, regional, segmental, diffuse)]\n"
            "- Architectural distortion: [present/absent, location]\n"
            "- Asymmetries: [focal, global, developing]\n"
            "- Skin/nipple: [retraction, thickening]\n"
            "- Axillary lymph nodes: [size, morphology]\n\n"
            "**BI-RADS Assessment:**\n"
            "[Category 0-6 with rationale]\n\n"
            "**Recommendations:**\n"
            "[Based on BI-RADS category: additional views, ultrasound, biopsy, "
            "routine screening interval]\n\n"
            "Clinical question: {question}"
        ),
    },
    "fluoroscopy": {
        "system": SYSTEM_PROMPTS["radiologist"],
        "template": (
            "Analyze this fluoroscopic {sub_type} study systematically.\n\n"
            "Structure your response as:\n\n"
            "**Technique:** [Type of study, contrast agent, approach]\n\n"
            "**Findings:**\n"
            "[Sequential description of contrast flow, anatomy visualization, "
            "any obstruction, reflux, or leakage]\n\n"
            "**Impression:**\n"
            "[Summary of key findings]\n\n"
            "**Recommendations:**\n"
            "[Clinical correlation, follow-up]\n\n"
            "Clinical question: {question}"
        ),
    },
    "angiography": {
        "system": SYSTEM_PROMPTS["radiologist"],
        "template": (
            "Analyze this angiographic {sub_type} study systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technique:** [Catheter approach, contrast agent, vessel territory]\n\n"
            "**Findings:**\n"
            "- Vessel patency: [normal caliber, stenosis with % narrowing, occlusion]\n"
            "- Stenoses: [location, degree (mild <50%, moderate 50-69%, "
            "severe 70-99%, occlusion), morphology (smooth, irregular, ulcerated)]\n"
            "- Collateral circulation: [present/absent, adequacy]\n"
            "- Aneurysms: [location, size, morphology (saccular/fusiform)]\n"
            "- Flow dynamics: [antegrade, retrograde, delayed filling]\n\n"
            "**Impression:**\n"
            "[Numbered findings, most critical first]\n\n"
            "**Recommendations:**\n"
            "[Intervention consideration, medical management, follow-up imaging]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Ultrasound ────────────────────────────────────────────
    "ultrasound": {
        "system": SYSTEM_PROMPTS["radiologist"],
        "template": (
            "Analyze this ultrasound {sub_type} image systematically.\n\n"
            "Structure your response as:\n\n"
            "**Technique:** [Transducer, approach, Doppler if used]\n\n"
            "**Findings:**\n"
            "[Echogenicity, measurements, vascularity, adjacent structures]\n\n"
            "**Impression:**\n"
            "[Numbered findings]\n\n"
            "**Differential Diagnosis:**\n"
            "[If applicable]\n\n"
            "**Recommendations:**\n"
            "[Follow-up, correlation]\n\n"
            "Clinical question: {question}"
        ),
    },
    "echocardiography": {
        "system": SYSTEM_PROMPTS["cardiologist"],
        "template": (
            "Analyze this echocardiographic {sub_type} study systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technique:** [TTE/TEE, views obtained, Doppler modes used]\n\n"
            "**Findings:**\n"
            "- Left ventricle: [size, wall thickness, EF (visual estimate), "
            "wall motion abnormalities by segment]\n"
            "- Right ventricle: [size, function (TAPSE), wall motion]\n"
            "- Left atrium: [size, volume index]\n"
            "- Right atrium: [size, IVC diameter and collapsibility]\n"
            "- Aortic valve: [morphology, stenosis (gradient, AVA), regurgitation grade]\n"
            "- Mitral valve: [morphology, stenosis, regurgitation grade, mechanism]\n"
            "- Tricuspid valve: [regurgitation, RVSP estimate]\n"
            "- Pericardium: [effusion size and hemodynamic significance]\n"
            "- Aorta: [root and ascending dimensions]\n\n"
            "**Impression:**\n"
            "[Key findings with severity grading]\n\n"
            "**Recommendations:**\n"
            "[Follow-up interval, additional testing (cardiac MRI, catheterization)]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Nuclear Medicine ──────────────────────────────────────
    "nuclear_medicine": {
        "system": SYSTEM_PROMPTS["nuclear_medicine_physician"],
        "template": (
            "Analyze this nuclear medicine {sub_type} study systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technique:** [Radiotracer, administered dose, imaging protocol, "
            "delay time, SPECT/planar]\n\n"
            "**Findings:**\n"
            "- Tracer distribution: [normal/abnormal uptake pattern]\n"
            "- Focal abnormalities: [location, intensity relative to background, "
            "SUV if PET]\n"
            "- Physiologic uptake: [expected sites]\n"
            "- Correlation with anatomical imaging: [if available]\n\n"
            "**Impression:**\n"
            "[Clinical significance of findings]\n\n"
            "**Differential Diagnosis:**\n"
            "[Differential for abnormal uptake patterns]\n\n"
            "**Recommendations:**\n"
            "[Additional imaging, biopsy, clinical correlation, follow-up timing]\n\n"
            "Clinical question: {question}"
        ),
    },
    "pet": {
        "system": SYSTEM_PROMPTS["nuclear_medicine_physician"],
        "template": (
            "Analyze this PET/PET-CT {sub_type} study systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technique:** [Radiotracer (FDG/other), blood glucose level, "
            "uptake time, PET-only or PET-CT]\n\n"
            "**Findings by Region:**\n"
            "- Head & Neck: [FDG-avid lesions, SUVmax]\n"
            "- Thorax: [lung, mediastinal, hilar nodes with SUV]\n"
            "- Abdomen & Pelvis: [liver, spleen, bowel, lymph nodes]\n"
            "- Musculoskeletal: [osseous lesions with SUV]\n"
            "- Physiologic uptake: [brain, myocardium, urinary tract]\n\n"
            "**Comparison:** [Prior study if available]\n\n"
            "**Impression:**\n"
            "[Deauville score if lymphoma; PERCIST if treatment response; "
            "summary of metabolically active disease]\n\n"
            "**Recommendations:**\n"
            "[Biopsy targets, treatment response assessment, follow-up interval]\n\n"
            "Clinical question: {question}"
        ),
    },
    "spect": {
        "system": SYSTEM_PROMPTS["nuclear_medicine_physician"],
        "template": (
            "Analyze this SPECT {sub_type} study systematically.\n\n"
            "Structure your response as:\n\n"
            "**Technique:** [Radiotracer, protocol, stress/rest if cardiac]\n\n"
            "**Findings:**\n"
            "[Distribution pattern, perfusion defects (fixed/reversible), "
            "focal abnormalities with anatomical correlation]\n\n"
            "**Impression:**\n"
            "[Clinical significance]\n\n"
            "**Recommendations:**\n"
            "[Additional workup, clinical correlation]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Pathology & Microscopy ────────────────────────────────
    "pathology": {
        "system": SYSTEM_PROMPTS["pathologist"],
        "template": (
            "Analyze this histopathology / cytology image systematically.\n\n"
            "Structure your response as:\n\n"
            "**Specimen:** [Type and preparation]\n\n"
            "**Microscopic Description:**\n"
            "- Architecture: [growth pattern, organization]\n"
            "- Cellular features: [size, shape, cytoplasm]\n"
            "- Nuclear features: [size, chromatin, nucleoli, mitoses]\n"
            "- Stroma: [fibrosis, inflammation, necrosis]\n"
            "- Special features: [if any]\n\n"
            "**Diagnosis:**\n"
            "[WHO classification if applicable]\n\n"
            "**Differential Diagnosis:**\n"
            "[Alternative diagnoses with distinguishing features]\n\n"
            "**Recommendations:**\n"
            "[IHC panel, molecular testing, clinical correlation]\n\n"
            "Clinical question: {question}"
        ),
    },
    "cytology": {
        "system": SYSTEM_PROMPTS["pathologist"],
        "template": (
            "Analyze this cytology {sub_type} specimen systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Specimen:** [Type: Pap smear / FNAC / fluid cytology / blood smear]\n\n"
            "**Adequacy:** [Satisfactory / unsatisfactory, cellularity]\n\n"
            "**Cytomorphologic Description:**\n"
            "- Cellularity: [sparse, moderate, abundant]\n"
            "- Cell arrangement: [clusters, sheets, single cells, papillary]\n"
            "- Cell morphology: [size, shape, cytoplasmic features]\n"
            "- Nuclear features: [N/C ratio, chromatin, nucleoli, mitoses]\n"
            "- Background: [blood, necrosis, inflammation, mucin]\n\n"
            "**Diagnosis:**\n"
            "[Bethesda system for Pap smears; descriptive for FNAC]\n\n"
            "**Differential Diagnosis:**\n"
            "[If atypical/suspicious cells present]\n\n"
            "**Recommendations:**\n"
            "[Cell block, IHC, molecular testing, repeat sampling]\n\n"
            "Clinical question: {question}"
        ),
    },
    "microbiology": {
        "system": SYSTEM_PROMPTS["pathologist"],
        "template": (
            "Analyze this microbiology {sub_type} image systematically.\n\n"
            "Structure your response as:\n\n"
            "**Specimen Type:** [Culture plate / smear / stained preparation]\n\n"
            "**Staining:** [Gram stain / Ziehl-Neelsen / PAS / GMS / unstained]\n\n"
            "**Findings:**\n"
            "- Organism morphology: [shape (cocci, bacilli, spirochete), "
            "arrangement (chains, clusters, pairs), Gram reaction]\n"
            "- Colony characteristics: [if culture plate: color, shape, "
            "hemolysis pattern, size]\n"
            "- Background: [inflammatory cells, tissue, debris]\n"
            "- Special features: [spores, capsule, hyphae, yeast forms]\n\n"
            "**Preliminary Identification:**\n"
            "[Most likely organism(s) based on morphology]\n\n"
            "**Recommendations:**\n"
            "[Additional stains, culture conditions, molecular testing, "
            "sensitivity testing]\n\n"
            "Clinical question: {question}"
        ),
    },
    "fluorescence_microscopy": {
        "system": SYSTEM_PROMPTS["pathologist"],
        "template": (
            "Analyze this fluorescence/confocal microscopy {sub_type} image.\n\n"
            "Structure your response as:\n\n"
            "**Technique:** [Fluorescence / confocal / live-cell, fluorophores used]\n\n"
            "**Findings:**\n"
            "- Signal distribution: [membrane, cytoplasmic, nuclear, extracellular]\n"
            "- Intensity pattern: [uniform, punctate, granular, diffuse]\n"
            "- Co-localization: [if multi-channel, co-localization patterns]\n"
            "- Morphological features: [cell shape, organelle distribution]\n\n"
            "**Interpretation:**\n"
            "[Biological significance of observed patterns]\n\n"
            "**Recommendations:**\n"
            "[Additional markers, quantification, controls needed]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Ophthalmology ─────────────────────────────────────────
    "fundoscopy": {
        "system": SYSTEM_PROMPTS["ophthalmologist"],
        "template": (
            "Analyze this fundus {sub_type} image systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technique:** [Color fundus / red-free / fluorescein angiography]\n\n"
            "**Findings:**\n"
            "- Optic disc: [size, color, cup-to-disc ratio, neuroretinal rim, "
            "disc edema, pallor]\n"
            "- Macula: [foveal reflex, macular edema, drusen, pigmentary changes, "
            "hemorrhage, exudates]\n"
            "- Retinal vessels: [caliber, AV nicking, tortuosity, "
            "neovascularization, sheathing]\n"
            "- Retinal periphery: [hemorrhages (dot/blot, flame-shaped), "
            "exudates (hard/soft), detachment]\n"
            "- Vitreous: [hemorrhage, opacities]\n\n"
            "**Grading:** [If applicable: diabetic retinopathy (ETDRS), "
            "AMD (AREDS), hypertensive retinopathy (Keith-Wagener-Barker)]\n\n"
            "**Impression:**\n"
            "[Key findings and clinical significance]\n\n"
            "**Recommendations:**\n"
            "[OCT, fluorescein angiography, referral, follow-up interval]\n\n"
            "Clinical question: {question}"
        ),
    },
    "oct": {
        "system": SYSTEM_PROMPTS["ophthalmologist"],
        "template": (
            "Analyze this OCT {sub_type} scan systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technique:** [SD-OCT / SS-OCT, scan pattern (macular / RNFL / ONH)]\n\n"
            "**Retinal Layer Analysis:**\n"
            "- ILM/NFL: [thickness, defects]\n"
            "- GCL/IPL: [thinning, atrophy]\n"
            "- INL/OPL: [cystoid changes, schisis]\n"
            "- ONL/ELM: [integrity, disruption]\n"
            "- Ellipsoid zone (IS/OS): [intact, disrupted, absent]\n"
            "- RPE: [irregularity, detachment (drusenoid/serous/fibrovascular)]\n"
            "- Choroid: [thickness, pachychoroid features]\n\n"
            "**Quantitative Data:**\n"
            "[Central macular thickness, RNFL thickness by sector, "
            "comparison to normative database]\n\n"
            "**Findings:**\n"
            "[Fluid (intraretinal/subretinal/sub-RPE), "
            "epiretinal membrane, vitreomacular traction, "
            "geographic atrophy, CNV]\n\n"
            "**Impression:**\n"
            "[Diagnosis with staging if applicable]\n\n"
            "**Recommendations:**\n"
            "[Treatment (anti-VEGF, laser), follow-up OCT interval]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Dermatology ───────────────────────────────────────────
    "dermoscopy": {
        "system": SYSTEM_PROMPTS["dermatologist"],
        "template": (
            "Analyze this dermoscopic {sub_type} image systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Clinical Context:** [Lesion location, size, patient demographics if known]\n\n"
            "**ABCDE Assessment:**\n"
            "- Asymmetry: [symmetric / asymmetric in 1 or 2 axes]\n"
            "- Border: [regular / irregular / fading]\n"
            "- Color: [number of colors present (brown, black, blue, red, white, gray)]\n"
            "- Diameter: [estimated size, >6mm flag]\n"
            "- Evolution: [if history available]\n\n"
            "**Dermoscopic Structures:**\n"
            "- Pigment network: [typical (regular) / atypical (irregular, thick lines)]\n"
            "- Globules/dots: [regular / irregular distribution]\n"
            "- Streaks/pseudopods: [present / absent, distribution]\n"
            "- Blue-white veil: [present / absent]\n"
            "- Regression structures: [peppering, white scar-like areas]\n"
            "- Vascular pattern: [dotted, linear, polymorphous]\n\n"
            "**Pattern Analysis:**\n"
            "[Overall pattern: reticular, globular, starburst, multicomponent, "
            "non-specific]\n\n"
            "**Assessment:**\n"
            "[Benign / suspicious / malignant, most likely diagnosis]\n\n"
            "**Recommendations:**\n"
            "[Monitoring, biopsy (excisional/incisional), follow-up interval]\n\n"
            "Clinical question: {question}"
        ),
    },
    "clinical_photo": {
        "system": SYSTEM_PROMPTS["dermatologist"],
        "template": (
            "Analyze this clinical photograph {sub_type} systematically.\n\n"
            "Structure your response as:\n\n"
            "**Clinical Description:**\n"
            "- Location: [anatomical site]\n"
            "- Morphology: [macule, papule, plaque, nodule, vesicle, bulla, "
            "erosion, ulcer]\n"
            "- Size: [estimated dimensions]\n"
            "- Color: [skin-colored, erythematous, violaceous, hyperpigmented]\n"
            "- Surface: [smooth, scaly, crusted, verrucous]\n"
            "- Border: [well-defined, ill-defined, irregular]\n"
            "- Distribution: [localized, generalized, dermatomal, linear]\n\n"
            "**Assessment:**\n"
            "[Most likely diagnosis with reasoning]\n\n"
            "**Differential Diagnosis:**\n"
            "[2-4 alternatives ranked by likelihood]\n\n"
            "**Recommendations:**\n"
            "[Biopsy, dermoscopy, laboratory workup, treatment, referral]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Dental ────────────────────────────────────────────────
    "dental": {
        "system": SYSTEM_PROMPTS["dentist"],
        "template": (
            "Analyze this dental {sub_type} image systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technique:** [Periapical / bitewing / panoramic / CBCT / intraoral photo]\n\n"
            "**Findings:**\n"
            "- Teeth present: [identify by FDI/universal numbering]\n"
            "- Caries: [location (occlusal, interproximal, cervical), "
            "extent (enamel, dentin, pulp)]\n"
            "- Restorations: [type, adequacy, secondary caries]\n"
            "- Periapical: [radiolucency, periapical abscess, granuloma]\n"
            "- Periodontal: [bone level, bone loss pattern "
            "(horizontal/vertical), furcation involvement]\n"
            "- Root: [morphology, resorption, fracture]\n"
            "- TMJ: [if visible: condylar morphology, joint space]\n"
            "- Other: [impacted teeth, supernumerary, cysts, tumors]\n\n"
            "**Impression:**\n"
            "[Summary of pathology found]\n\n"
            "**Recommendations:**\n"
            "[Treatment planning, additional imaging, referral]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Cardiology ────────────────────────────────────────────
    "ecg": {
        "system": SYSTEM_PROMPTS["cardiologist"],
        "template": (
            "Analyze this ECG/EKG {sub_type} tracing systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Technical Quality:** [Calibration, artifacts, lead placement]\n\n"
            "**Rate and Rhythm:**\n"
            "- Heart rate: [bpm, regular/irregular]\n"
            "- Rhythm: [sinus, atrial fibrillation, atrial flutter, "
            "junctional, ventricular]\n"
            "- P waves: [morphology, axis, relationship to QRS]\n\n"
            "**Intervals:**\n"
            "- PR interval: [ms, normal 120-200ms]\n"
            "- QRS duration: [ms, narrow <120ms / wide ≥120ms]\n"
            "- QT/QTc interval: [ms, prolonged if >460ms female, >440ms male]\n\n"
            "**Axis:**\n"
            "- QRS axis: [normal, LAD, RAD, extreme axis deviation]\n\n"
            "**Morphology:**\n"
            "- ST segment: [elevation/depression, leads affected]\n"
            "- T waves: [inversion, hyperacute, peaked]\n"
            "- Q waves: [pathological, leads]\n"
            "- Bundle branch block: [RBBB, LBBB, fascicular block]\n"
            "- Chamber enlargement: [LAE, RAE, LVH, RVH criteria]\n\n"
            "**Impression:**\n"
            "[Primary interpretation]\n\n"
            "**Recommendations:**\n"
            "[Cardiology consultation, troponin, echocardiography, "
            "catheterization if indicated]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Endoscopy / Video ─────────────────────────────────────
    "endoscopy": {
        "system": SYSTEM_PROMPTS["gastroenterologist"],
        "template": (
            "Analyze this endoscopic {sub_type} image/video systematically.\n\n"
            "Structure your response EXACTLY as follows:\n\n"
            "**Procedure:** [EGD / Colonoscopy / Bronchoscopy / Cystoscopy]\n\n"
            "**Anatomical Location:** [Specific location within the examined organ]\n\n"
            "**Mucosal Assessment:**\n"
            "- Mucosa: [color, texture, vascular pattern, edema]\n"
            "- Lesions: [Paris classification (0-Ip, 0-Is, 0-IIa, 0-IIb, 0-IIc, 0-III), "
            "size, number]\n"
            "- Surface pattern: [Kudo pit pattern / NICE classification if applicable]\n"
            "- Vascular pattern: [regular, irregular, absent]\n\n"
            "**Additional Findings:**\n"
            "[Inflammation, ulceration, stricture, bleeding, diverticula, "
            "Barrett's (Prague classification C_M_)]\n\n"
            "**Impression:**\n"
            "[Diagnosis with grading/classification]\n\n"
            "**Recommendations:**\n"
            "[Biopsy, polypectomy, surveillance interval, "
            "histological correlation]\n\n"
            "Clinical question: {question}"
        ),
    },
    "surgical_video": {
        "system": SYSTEM_PROMPTS["gastroenterologist"],
        "template": (
            "Analyze this surgical / laparoscopic {sub_type} video systematically.\n\n"
            "Structure your response as:\n\n"
            "**Procedure:** [Type of surgery, approach (open/laparoscopic/robotic)]\n\n"
            "**Anatomical Assessment:**\n"
            "[Identify visible structures, tissue planes, "
            "surgical landmarks]\n\n"
            "**Findings:**\n"
            "[Pathology observed: adhesions, masses, inflammation, "
            "abnormal vascularity, tissue quality]\n\n"
            "**Surgical Technique:**\n"
            "[Instruments visible, dissection quality, hemostasis]\n\n"
            "**Impression:**\n"
            "[Summary of surgical findings]\n\n"
            "Clinical question: {question}"
        ),
    },
    "video": {
        "system": SYSTEM_PROMPTS["radiologist"],
        "template": (
            "Analyze this medical video (likely endoscopy/surgical procedure).\n\n"
            "Structure your response as:\n\n"
            "**Procedure:** [Type of procedure, anatomical location]\n\n"
            "**Findings:**\n"
            "[Describe mucosal appearance, lesions, abnormalities frame by frame. "
            "Note location, size, morphology of any findings.]\n\n"
            "**Impression:**\n"
            "[Summary of significant findings]\n\n"
            "**Recommendations:**\n"
            "[Biopsy sites, follow-up intervals, therapeutic considerations]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── Advanced Neuro ────────────────────────────────────────
    "dti": {
        "system": SYSTEM_PROMPTS["neurologist"],
        "template": (
            "Analyze this diffusion tensor imaging (DTI) {sub_type} study.\n\n"
            "Structure your response as:\n\n"
            "**Technique:** [b-values, number of directions, tractography method]\n\n"
            "**Findings:**\n"
            "- FA maps: [fractional anisotropy values in key tracts]\n"
            "- MD maps: [mean diffusivity abnormalities]\n"
            "- Tractography: [tract integrity, disruption, displacement]\n"
            "- White matter tracts: [corticospinal, corpus callosum, "
            "arcuate fasciculus, optic radiation]\n\n"
            "**Impression:**\n"
            "[White matter integrity assessment]\n\n"
            "**Recommendations:**\n"
            "[Correlation with structural MRI, clinical significance]\n\n"
            "Clinical question: {question}"
        ),
    },
    "fmri": {
        "system": SYSTEM_PROMPTS["neurologist"],
        "template": (
            "Analyze this functional MRI (fMRI) {sub_type} study.\n\n"
            "Structure your response as:\n\n"
            "**Technique:** [Task-based / resting-state, paradigm, TR/TE]\n\n"
            "**Findings:**\n"
            "- Activation maps: [significant BOLD activations by region]\n"
            "- Lateralization: [language, motor dominance]\n"
            "- Connectivity: [if resting-state: network integrity]\n\n"
            "**Impression:**\n"
            "[Functional localization results]\n\n"
            "**Recommendations:**\n"
            "[Pre-surgical planning implications, clinical correlation]\n\n"
            "Clinical question: {question}"
        ),
    },

    # ── General Fallback ──────────────────────────────────────
    "general_medical": {
        "system": SYSTEM_PROMPTS["radiologist"],
        "template": (
            "Analyze this medical image thoroughly.\n\n"
            "Structure your response as:\n\n"
            "**Technique:** [Imaging modality and parameters]\n\n"
            "**Findings:**\n"
            "[Systematic description of all visible structures. "
            "Report any abnormalities with measurements. "
            "Note pertinent negatives.]\n\n"
            "**Impression:**\n"
            "[Numbered list of findings, most significant first]\n\n"
            "**Differential Diagnosis:**\n"
            "[If abnormality found, ranked possibilities]\n\n"
            "**Recommendations:**\n"
            "[Clinical correlation and follow-up]\n\n"
            "Clinical question: {question}"
        ),
    },
}

# ── Aliases for backward compatibility ────────────────────────
MODALITY_PROMPTS["3d_volume"] = MODALITY_PROMPTS["ct"]
MODALITY_PROMPTS["general_microscopy"] = MODALITY_PROMPTS["pathology"]
MODALITY_PROMPTS["pathology_wsi"] = MODALITY_PROMPTS["pathology"]
MODALITY_PROMPTS["intravascular_ultrasound"] = MODALITY_PROMPTS["ultrasound"]
MODALITY_PROMPTS["bone_densitometry"] = MODALITY_PROMPTS["xray"]
MODALITY_PROMPTS["bone_densitometry_us"] = MODALITY_PROMPTS["ultrasound"]
MODALITY_PROMPTS["dental_intraoral"] = MODALITY_PROMPTS["dental"]
MODALITY_PROMPTS["dental_panoramic"] = MODALITY_PROMPTS["dental"]
MODALITY_PROMPTS["secondary_capture"] = MODALITY_PROMPTS["general_medical"]
MODALITY_PROMPTS["ophthalmic_mapping"] = MODALITY_PROMPTS["fundoscopy"]
MODALITY_PROMPTS["ophthalmic_visual_field"] = MODALITY_PROMPTS["fundoscopy"]
MODALITY_PROMPTS["electrophysiology"] = MODALITY_PROMPTS["ecg"]
MODALITY_PROMPTS["hemodynamic"] = MODALITY_PROMPTS["ecg"]
MODALITY_PROMPTS["ultrasound_clip"] = MODALITY_PROMPTS["ultrasound"]
MODALITY_PROMPTS["wound"] = MODALITY_PROMPTS["clinical_photo"]


def build_expert_prompt(
    question: str,
    modality: str = "general_medical",
    sub_type: str = "general",
    file_type: str = "2d",
    role: str = "primary",
) -> str:
    """Build an expert medical prompt based on modality and context.

    This is the single most impactful function for output quality.
    Models produce dramatically better structured reports when given
    explicit formatting instructions.

    Args:
        question: User's original question
        modality: Detected imaging modality
        sub_type: Sub-type (e.g., "PA chest" for xray)
        file_type: "2d", "3d", or "video"
        role: "primary", "reasoner", or "verifier"

    Returns:
        Expert-crafted prompt string
    """
    prompt_config = MODALITY_PROMPTS.get(modality, MODALITY_PROMPTS["general_medical"])

    # Build the structured prompt
    prompt = prompt_config["template"].format(
        question=question,
        sub_type=sub_type,
    )

    # For reasoner role: add chain-of-thought instruction
    if role == "reasoner":
        prompt = (
            "Please reason step by step about this medical image.\n\n"
            "First, describe what you observe objectively.\n"
            "Then, consider what clinical conditions could explain these findings.\n"
            "Finally, provide your assessment with confidence level.\n\n"
            + prompt
        )

    # For 3D: add volumetric context
    if file_type == "3d":
        prompt = (
            "This is a 3D volumetric medical scan presented as sequential slices. "
            "Evaluate the volume systematically across all slices, noting the "
            "anatomical level and any pathology seen.\n\n"
            + prompt
        )

    # For video: add temporal context
    if file_type in ("video", "temporal"):
        prompt = (
            "This is a medical video presented as sequential frames. "
            "Analyze the temporal progression, noting any dynamic changes, "
            "movement patterns, or evolving findings across frames.\n\n"
            + prompt
        )

    return prompt


def get_system_prompt(modality: str = "general_medical") -> str:
    """Get the appropriate system prompt for a given modality."""
    prompt_config = MODALITY_PROMPTS.get(modality, MODALITY_PROMPTS["general_medical"])
    return prompt_config["system"]


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE CALIBRATION — from model language analysis
# ═══════════════════════════════════════════════════════════════

# Phrases that indicate HIGH confidence
HIGH_CONFIDENCE_MARKERS = [
    "consistent with", "diagnostic of", "compatible with", "characteristic of",
    "pathognomonic", "definitive", "classic appearance", "confirms",
    "demonstrates", "clearly shows", "unequivocal", "typical for",
    "represents", "diagnostic for",
]

# Phrases that indicate LOW confidence / uncertainty
LOW_CONFIDENCE_MARKERS = [
    "might be", "could be", "possibly", "uncertain", "unclear",
    "may represent", "cannot exclude", "cannot rule out",
    "differential includes", "nonspecific", "equivocal", "indeterminate",
    "correlate clinically", "further evaluation", "limited study",
    "suboptimal", "artifact", "obscured by", "partially visualized",
    "recommend", "suggest", "consider", "versus",
]

# Phrases that indicate NORMAL / no pathology
NORMAL_MARKERS = [
    "no acute", "unremarkable", "within normal limits", "no significant",
    "no evidence of", "no focal", "normal", "clear", "intact",
    "no abnormality", "negative", "no acute findings",
]


def calibrate_confidence(model_output: str, base: float = 0.7) -> float:
    """Calibrate confidence score based on linguistic analysis of model output.

    Instead of using hardcoded 0.85, this analyzes the actual language
    the model used to produce a more accurate confidence estimate.

    Args:
        model_output: The model's text response
        base: Starting confidence (default 0.7)

    Returns:
        Calibrated confidence between 0.1 and 0.98
    """
    text_lower = model_output.lower()

    # Count confidence markers
    high_count = sum(1 for p in HIGH_CONFIDENCE_MARKERS if p in text_lower)
    low_count = sum(1 for p in LOW_CONFIDENCE_MARKERS if p in text_lower)
    normal_count = sum(1 for p in NORMAL_MARKERS if p in text_lower)

    # Length-based adjustment (very short = less reliable)
    word_count = len(text_lower.split())
    if word_count < 30:
        length_penalty = -0.15
    elif word_count < 80:
        length_penalty = -0.05
    elif word_count > 300:
        length_penalty = 0.05  # detailed response = higher confidence
    else:
        length_penalty = 0.0

    # Structure bonus (has proper sections = model understood the task)
    structure_bonus = 0.0
    for section in ["findings", "impression", "technique"]:
        if section in text_lower:
            structure_bonus += 0.03

    # Calculate adjustment
    confidence_adj = (high_count * 0.04) - (low_count * 0.06) + \
                     (normal_count * 0.02) + length_penalty + structure_bonus

    calibrated = max(0.10, min(0.98, base + confidence_adj))
    return round(calibrated, 3)


# ═══════════════════════════════════════════════════════════════
#  FINDING EXTRACTION — for cross-model synthesis
# ═══════════════════════════════════════════════════════════════

def extract_individual_findings(text: str) -> list[dict]:
    """Extract individual clinical findings from model output.

    Parses sentences to identify discrete medical findings with
    their anatomical location and severity. Used by the synthesis
    fusion to cross-validate findings across models.

    Returns:
        List of dicts: [{finding, location, severity, sentence}, ...]
    """
    findings = []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Anatomy keywords — expanded for all specialties
    anatomy_kw = [
        # Radiology
        "heart", "cardiac", "lung", "pulmonary", "pleural", "mediastin",
        "aort", "liver", "hepat", "kidney", "renal", "brain", "cerebr",
        "bone", "osseous", "spleen", "splenic", "pancrea", "bowel",
        "stomach", "gastric", "thyroid", "lymph", "vascul", "spine",
        "vertebr", "disc", "pericardi", "hilum", "hilar",
        # Ophthalmology
        "retina", "macula", "fovea", "optic disc", "optic nerve",
        "choroid", "vitreous", "cornea", "lens", "iris",
        # Dermatology
        "skin", "lesion", "epiderm", "dermis", "melanocyt",
        "keratinoc", "nail", "hair follicle",
        # Dental
        "tooth", "teeth", "mandib", "maxill", "periodon",
        "periapic", "pulp", "enamel", "dentin", "gingiv",
        # Cardiac
        "ventricl", "atri", "valve", "mitral", "aortic valve",
        "tricuspid", "septum", "myocard", "endocard",
        # Pathology
        "cell", "nucleus", "nuclei", "stroma", "epithelium",
        "gland", "vessel", "necrosis", "fibros", "inflamm",
        # GI / Endoscopy
        "mucos", "polyp", "ulcer", "erosion", "stricture",
        "esophag", "duoden", "colon", "rectum",
    ]

    # Finding verbs/phrases
    finding_verbs = [
        "shows", "demonstrates", "reveals", "indicates", "suggesting",
        "consistent with", "compatible with", "suggestive of",
        "evidence of", "appears", "noted", "identified", "seen",
        "observed", "measuring", "measures",
    ]

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue

        sent_lower = sent.lower()

        # Check if sentence contains a finding
        has_finding_verb = any(v in sent_lower for v in finding_verbs)
        has_anatomy = any(a in sent_lower for a in anatomy_kw)

        if has_finding_verb or has_anatomy:
            # Determine severity
            severity = "routine"
            if any(w in sent_lower for w in ["acute", "emergent", "critical", "severe", "large"]):
                severity = "high"
            elif any(w in sent_lower for w in ["moderate", "significant", "abnormal"]):
                severity = "moderate"
            elif any(w in sent_lower for w in ["mild", "small", "minor", "subtle"]):
                severity = "low"

            # Determine location
            location = "general"
            for kw in anatomy_kw:
                if kw in sent_lower:
                    location = kw
                    break

            findings.append({
                "sentence": sent,
                "location": location,
                "severity": severity,
                "is_normal": any(n in sent_lower for n in NORMAL_MARKERS),
            })

    return findings
