"""
MediScan AI v7.0 — Modality Detector
Auto-detects medical imaging modality from file properties, DICOM tags,
content analysis, and filename heuristics.

Supports 30+ DICOM modality codes and 25+ clinical imaging types including:
  Radiology (X-ray, CT, MRI, Mammography, Fluoroscopy, Angiography)
  Ultrasound (Obstetric, Abdominal, Cardiac, Doppler, Intravascular)
  Nuclear Medicine (PET, SPECT, Bone Scan, Thyroid)
  Pathology (Histopathology, Cytology, WSI, Microbiology)
  Ophthalmology (Fundoscopy, OCT, Visual Field)
  Dermatology (Dermoscopy, Clinical Photography)
  Dental (Intraoral, Panoramic, CBCT)
  Cardiology (Echocardiography, ECG, Angiography)
  Endoscopy / Laparoscopy / Surgical Video
  Advanced (Fluorescence, Confocal, DTI, fMRI)
"""
from __future__ import annotations


import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModalityDetector:
    """Detects imaging modality from image properties, DICOM tags, and heuristics."""

    # ── File extension → likely modality mapping ───────────────────
    EXTENSION_HINTS = {
        # Standard medical
        ".dcm": "dicom",
        ".nii": "3d_volume",
        ".nii.gz": "3d_volume",
        ".nrrd": "3d_volume",
        ".mha": "3d_volume",
        ".mhd": "3d_volume",
        # Video
        ".mp4": "video",
        ".avi": "video",
        ".mov": "video",
        ".mkv": "video",
        ".webm": "video",
        # Whole Slide Imaging (pathology)
        ".svs": "pathology_wsi",
        ".ndpi": "pathology_wsi",
        ".mrxs": "pathology_wsi",
        ".vsi": "pathology_wsi",
        ".scn": "pathology_wsi",
        ".bif": "pathology_wsi",
        ".qptiff": "pathology_wsi",
        # Dental
        ".fdi": "dental",
    }

    # ── DICOM modality codes → internal modality keys ─────────────
    # Reference: DICOM PS3.3 Table C.7.3.1.1.1 (Modality Defined Terms)
    DICOM_MODALITY_MAP = {
        # Radiology
        "CR": "xray",
        "DX": "xray",
        "DR": "xray",
        "CT": "ct",
        "MR": "mri",
        "MG": "mammography",
        "RF": "fluoroscopy",
        "XA": "angiography",
        "XC": "clinical_photo",
        # Ultrasound
        "US": "ultrasound",
        "IVUS": "intravascular_ultrasound",
        "BDUS": "bone_densitometry_us",
        # Nuclear Medicine
        "NM": "nuclear_medicine",
        "PT": "pet",
        "ST": "spect",
        # Endoscopy
        "ES": "endoscopy",
        # Pathology / Microscopy
        "SM": "pathology",
        "GM": "general_microscopy",
        # Ophthalmology
        "OP": "fundoscopy",
        "OPT": "oct",
        "OPM": "ophthalmic_mapping",
        "OPV": "ophthalmic_visual_field",
        "AR": "autorefraction",
        "VA": "visual_acuity",
        "SRF": "subjective_refraction",
        "KER": "keratometry",
        "LEN": "lensometry",
        "IOL": "intraocular_lens",
        # Dental
        "IO": "dental_intraoral",
        "PX": "dental_panoramic",
        # Cardiology
        "ECG": "ecg",
        "EPS": "electrophysiology",
        "HD": "hemodynamic",
        # Bone density
        "BMD": "bone_densitometry",
        # Other
        "SC": "secondary_capture",
        "DOC": "document",
        "SR": "structured_report",
        "AU": "audio",
        "BI": "biomagnetic_imaging",
        "HC": "hard_copy",
        "LS": "laser_scan",
        "RG": "radiographic_imaging",
        "RTIMAGE": "radiotherapy_image",
    }

    # ── Filename patterns → modality hints ────────────────────────
    FILENAME_PATTERNS = [
        (r"(?i)fundus|retina|optic.?disc", "fundoscopy"),
        (r"(?i)oct|retinal.?layer", "oct"),
        (r"(?i)derm|skin.?lesion|mole|melanoma|nevus", "dermoscopy"),
        (r"(?i)dental|tooth|teeth|panoramic|periapical|bitewing", "dental"),
        (r"(?i)mammo|breast|birads", "mammography"),
        (r"(?i)pet.?ct|pet.?scan|suv|fdg", "pet"),
        (r"(?i)spect|scintigraph|thyroid.?scan|bone.?scan", "spect"),
        (r"(?i)echo|cardiac.?us|ejection.?fraction", "echocardiography"),
        (r"(?i)ecg|ekg|electrocardiog", "ecg"),
        (r"(?i)angio|vessel|stenosis|catheter", "angiography"),
        (r"(?i)endoscop|colonoscop|gastroscop|bronchoscop", "endoscopy"),
        (r"(?i)laparoscop|surgical", "surgical_video"),
        (r"(?i)histopath|biopsy|h.?e.?stain|wsi", "pathology"),
        (r"(?i)cytolog|pap.?smear|fnac|blood.?smear", "cytology"),
        (r"(?i)gram.?stain|bacteria|culture|petri|fungal|parasite|malaria", "microbiology"),
        (r"(?i)fluorescen|confocal|live.?cell", "fluorescence_microscopy"),
        (r"(?i)wound|burn|ulcer|surgical.?site", "clinical_photo"),
        (r"(?i)chest|cxr|pa.?view|ap.?view|lateral", "xray"),
        (r"(?i)xray|x.?ray|radiograph|fracture", "xray"),
        (r"(?i)dti|diffusion.?tensor|tractograph|connectome", "dti"),
        (r"(?i)fmri|functional.?mri|bold", "fmri"),
    ]

    def detect(self, data: dict[str, Any]) -> dict[str, Any]:
        """Detect modality from loaded data.

        Returns:
            Dict with keys: modality, confidence, sub_type, dimensions
        """
        file_type = data.get("type", "unknown")
        source_path = data.get("source_path", "")
        metadata = data.get("metadata", {})

        # ── 1. Video files ────────────────────────────────────────
        if file_type == "video":
            sub = self._guess_video_type(metadata, source_path)
            return {
                "modality": sub if sub != "unknown" else "video",
                "sub_type": sub,
                "confidence": 0.9,
                "dimensions": "temporal",
            }

        # ── 2. DICOM files with modality tag ──────────────────────
        dicom_modality = metadata.get("modality")
        if dicom_modality and dicom_modality in self.DICOM_MODALITY_MAP:
            mapped = self.DICOM_MODALITY_MAP[dicom_modality]
            return {
                "modality": mapped,
                "sub_type": dicom_modality,
                "confidence": 0.95,
                "dimensions": "3d" if file_type == "3d" else "2d",
            }

        # ── 3. Filename-based detection ───────────────────────────
        if source_path:
            fn_result = self._classify_by_filename(source_path)
            if fn_result:
                fn_result["dimensions"] = "3d" if file_type == "3d" else "2d"
                return fn_result

        # ── 4. WSI / pathology by extension ───────────────────────
        if source_path:
            suffix = "".join(Path(source_path).suffixes).lower()
            if suffix in self.EXTENSION_HINTS:
                hint = self.EXTENSION_HINTS[suffix]
                if hint == "pathology_wsi":
                    return {
                        "modality": "pathology",
                        "sub_type": "whole_slide_image",
                        "confidence": 0.90,
                        "dimensions": "2d",
                    }

        # ── 5. 3D volumes ─────────────────────────────────────────
        if file_type == "3d":
            return self._classify_3d(data)

        # ── 6. 2D images — pixel-level analysis ───────────────────
        if file_type == "2d":
            return self._classify_2d(data)

        # ── 7. Fallback from file extension ───────────────────────
        if source_path:
            suffix = Path(source_path).suffix.lower()
            if suffix in self.EXTENSION_HINTS:
                hint = self.EXTENSION_HINTS[suffix]
                return {
                    "modality": hint,
                    "sub_type": "unknown",
                    "confidence": 0.5,
                    "dimensions": "unknown",
                }

        return {
            "modality": "unknown",
            "sub_type": "unknown",
            "confidence": 0.0,
            "dimensions": "unknown",
        }

    # ── Filename heuristic classifier ─────────────────────────────

    def _classify_by_filename(self, source_path: str) -> dict[str, Any] | None:
        """Classify modality from filename patterns."""
        filename = Path(source_path).stem.lower()
        for pattern, modality in self.FILENAME_PATTERNS:
            if re.search(pattern, filename):
                return {
                    "modality": modality,
                    "sub_type": "filename_hint",
                    "confidence": 0.65,
                }
        return None

    # ── 2D pixel-level classifier ─────────────────────────────────

    def _classify_2d(self, data: dict[str, Any]) -> dict[str, Any]:
        """Classify 2D medical image by pixel statistics, dimensions, and color."""
        metadata = data.get("metadata", {})
        pixel_array = data.get("pixel_array")

        width = metadata.get("width", 0)
        height = metadata.get("height", 0)
        mode = metadata.get("mode", "")

        # ── Grayscale images ──────────────────────────────────────
        if mode in ("L", "I", "F") or (pixel_array is not None and pixel_array.ndim == 2):
            # Mammography: very large single-view images (typically 3000+)
            if width > 3000 or height > 3000:
                return {"modality": "mammography", "sub_type": "general",
                        "confidence": 0.55, "dimensions": "2d"}
            # ECG: typically wide and short (strip format)
            if width > 1500 and height < 600:
                return {"modality": "ecg", "sub_type": "12_lead",
                        "confidence": 0.50, "dimensions": "2d"}
            # Large grayscale → likely X-ray
            if width > 2000 and height > 2000:
                return {"modality": "xray", "sub_type": "general",
                        "confidence": 0.6, "dimensions": "2d"}
            # Moderate grayscale
            if width > 512 and height > 512:
                return {"modality": "xray", "sub_type": "general",
                        "confidence": 0.5, "dimensions": "2d"}
            # Small grayscale (could be dental or small ROI)
            return {"modality": "xray", "sub_type": "general",
                    "confidence": 0.4, "dimensions": "2d"}

        # ── Color images ──────────────────────────────────────────
        if pixel_array is not None and pixel_array.ndim == 3:
            mean_color = np.mean(pixel_array, axis=(0, 1))

            # ── Pathology: Pink/purple tint (H&E stain) ──────────
            if len(mean_color) >= 3:
                r, g, b = mean_color[0], mean_color[1], mean_color[2]

                # H&E stained tissue: pink-purple dominant, large images
                if r > 120 and b > 80 and g < r and (width > 512 and height > 512):
                    # Very large → likely WSI tile or histopathology
                    if width > 1000 and height > 1000:
                        return {"modality": "pathology", "sub_type": "histopathology",
                                "confidence": 0.65, "dimensions": "2d"}
                    return {"modality": "pathology", "sub_type": "cytology",
                            "confidence": 0.50, "dimensions": "2d"}

                # ── Fundoscopy: dark background, bright disc/vessels ──
                if r > 80 and g < 80 and b < 80 and np.std(pixel_array) > 40:
                    return {"modality": "fundoscopy", "sub_type": "retinal",
                            "confidence": 0.55, "dimensions": "2d"}

                # ── Dermoscopy: relatively uniform skin-tone background ──
                # Dermoscopic images often have circular dark border
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                if aspect_ratio < 1.3 and 300 < width < 2000:
                    # Check for dermoscopic features: skin-colored with dark lesion
                    if r > 100 and g > 60 and b > 40:
                        std_vals = np.std(pixel_array, axis=(0, 1))
                        if np.mean(std_vals) > 30:
                            return {"modality": "dermoscopy", "sub_type": "skin_lesion",
                                    "confidence": 0.45, "dimensions": "2d"}

                # ── Endoscopy: reddish mucosal tissue ─────────────
                if r > 130 and r > g * 1.3 and r > b * 1.5:
                    if 200 < width < 1500 and 200 < height < 1500:
                        return {"modality": "endoscopy", "sub_type": "mucosal",
                                "confidence": 0.45, "dimensions": "2d"}

                # ── Ultrasound: dark background, fan-shaped bright region
                if np.mean(pixel_array) < 80 and np.std(pixel_array) > 50:
                    return {"modality": "ultrasound", "sub_type": "general",
                            "confidence": 0.40, "dimensions": "2d"}

                # ── Clinical photo: natural skin tones, well-lit ──
                if r > 120 and g > 80 and b > 60 and width > 500:
                    return {"modality": "clinical_photo", "sub_type": "general",
                            "confidence": 0.35, "dimensions": "2d"}

        return {"modality": "general_medical", "sub_type": "unknown",
                "confidence": 0.3, "dimensions": "2d"}

    # ── 3D volume classifier ──────────────────────────────────────

    def _classify_3d(self, data: dict[str, Any]) -> dict[str, Any]:
        """Classify 3D volume by intensity range, voxel properties, and shape."""
        volume = data.get("volume")
        metadata = data.get("metadata", {})

        if volume is None:
            return {"modality": "3d_volume", "sub_type": "unknown",
                    "confidence": 0.4, "dimensions": "3d"}

        min_val = float(volume.min())
        max_val = float(volume.max())
        shape = volume.shape

        # CT scans have Hounsfield Units: typically -1024 to +3071
        if min_val < -500 and max_val > 500:
            # Determine CT sub-type by shape/anatomy
            sub = "general"
            if len(shape) >= 3:
                if shape[0] < 50:
                    sub = "head"
                elif shape[0] > 200:
                    sub = "chest_abdomen"
            return {"modality": "ct", "sub_type": sub,
                    "confidence": 0.85, "dimensions": "3d"}

        # MRI: typically 0 to ~4000, no negative values
        if min_val >= 0 and max_val < 5000:
            sub = "general"
            # Check for DTI-like properties (4D)
            if len(shape) == 4:
                if shape[3] > 6:
                    return {"modality": "dti", "sub_type": "diffusion_tensor",
                            "confidence": 0.75, "dimensions": "4d"}
                else:
                    return {"modality": "fmri", "sub_type": "functional",
                            "confidence": 0.60, "dimensions": "4d"}
            return {"modality": "mri", "sub_type": sub,
                    "confidence": 0.7, "dimensions": "3d"}

        # PET/SPECT: often float values, narrow positive range
        if min_val >= 0 and max_val < 50:
            return {"modality": "pet", "sub_type": "suv_map",
                    "confidence": 0.55, "dimensions": "3d"}

        return {"modality": "3d_volume", "sub_type": "unknown",
                "confidence": 0.5, "dimensions": "3d"}

    # ── Video sub-type classifier ─────────────────────────────────

    def _guess_video_type(self, metadata: dict[str, Any],
                          source_path: str = "") -> str:
        """Guess video sub-type from metadata and filename."""
        # Filename hints first
        if source_path:
            fn = Path(source_path).stem.lower()
            if any(kw in fn for kw in ["echo", "cardiac", "heart"]):
                return "echocardiography"
            if any(kw in fn for kw in ["angio", "cath", "vessel"]):
                return "angiography"
            if any(kw in fn for kw in ["colon", "gastro", "broncho", "endoscop"]):
                return "endoscopy"
            if any(kw in fn for kw in ["laparo", "surgical", "surgery"]):
                return "surgical_video"
            if any(kw in fn for kw in ["ultrasound", "us_", "doppler"]):
                return "ultrasound_clip"

        duration = metadata.get("duration_seconds", 0)
        if duration > 600:  # > 10 min → likely surgical
            return "surgical_video"
        if duration < 30:
            return "ultrasound_clip"
        return "endoscopy"

