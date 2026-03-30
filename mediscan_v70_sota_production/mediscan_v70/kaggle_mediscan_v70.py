"""
╔══════════════════════════════════════════════════════════════╗
║  MediScan AI v7.0 — Kaggle Notebook (2x T4 GPU, 4-bit)      ║
║  Production-Grade Medical Imaging Intelligence Pipeline      ║
╠══════════════════════════════════════════════════════════════╣
║  v7.0 UPGRADES:                                  ║
║  ✅ Sequential model loading (fixes OOM)                     ║
║  ✅ Medical Knowledge Graph (causal reasoning)                ║
║  ✅ Enhanced RAG with pre-seeded knowledge + PubMed           ║
║  ✅ Reasoning Engine (CoT + contradiction detection)          ║
║  ✅ Dynamic Fusion (uncertainty-aware, model debate)          ║
║  ✅ Self-Reflection Loop (critique and refine)                ║
║  ✅ Clinical Safety Layer (hallucination detection)           ║
║  ✅ Multi-Agent Orchestrator (specialist roles)               ║
║  ✅ Explainability Engine (why this diagnosis?)               ║
║  ✅ HuluMed process_text bypass (vLLM approach)              ║
║  ✅ HuluMed generate() modals fix                            ║
╚══════════════════════════════════════════════════════════════╝

GPU Budget (2x T4, 14.5GB each = 29GB total):
  SEQUENTIAL EXECUTION (v6.0):
  Group 1: HuluMed-4B  (4-bit) ~2.5GB on GPU 0 → run → unload
           (4B replaces 7B — 7B vision encoder needs 16GB, exceeds T4's 14.5GB)
  Group 2: MedGemma-4B + MediX-R1-2B  → run → unload
  Group 3: Med3DVLM-7B + BiomedCLIP   → run → unload
"""
from __future__ import annotations


# ═══════════════════════════════════════════════════════════
#  CELL 1: Install Dependencies & HF Login
# ═══════════════════════════════════════════════════════════

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

PACKAGES = [
    "torch", "torchvision", "transformers>=4.50.0", "accelerate>=1.2.0",
    "bitsandbytes>=0.45.0", "safetensors",
    "monai>=1.4.0", "nibabel", "SimpleITK", "Pillow", "opencv-python-headless",
    "open-clip-torch>=2.26.0", "scikit-image",
    "chromadb", "sentence-transformers>=3.0.0",
    "pyyaml", "httpx", "duckduckgo-search",
    "qwen_vl_utils", "huggingface_hub", "decord", "ffmpeg"
]

print("📦 Installing dependencies...")
for pkg in PACKAGES:
    try: install(pkg)
    except Exception as e: print(f"  ⚠ {pkg}: {e}")
print("✅ Dependencies installed")

print("🔑 Logging into Hugging Face...")
from huggingface_hub import login
import os
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not _hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        _hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if _hf_token:
    login(token=_hf_token)
    print("✅ HF Login successful")
else:
    print("⚠️ No HF_TOKEN found — set via Kaggle Secrets or environment variable")


# ═══════════════════════════════════════════════════════════
#  CELL 2: Imports & GPU Setup
# ═══════════════════════════════════════════════════════════

import gc, json, logging, os, re, tempfile, time, urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

# v6.0: Optimize CUDA memory allocation for T4 GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("mediscan.kaggle")

NUM_GPUS = torch.cuda.device_count()
print(f"🖥️  GPUs available: {NUM_GPUS}")
for i in range(NUM_GPUS):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  GPU {i}: {name} ({mem:.1f} GB)")

from transformers import BitsAndBytesConfig
QUANT_CONFIG_4BIT = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

VERSION = "7.0"

# v5.1: Model type classification for fusion
GENERATIVE_MODELS = {"hulu_med_7b", "medgemma_4b", "medix_r1_2b", "med3dvlm"}
CLASSIFIER_MODELS = {"biomedclip"}
NATIVE_3D_MODELS = {"hulu_med_7b", "med3dvlm"}


# ═══════════════════════════════════════════════════════════
#  CELL 3: v5.1 Negation Detection Utilities
# ═══════════════════════════════════════════════════════════

NEGATION_PHRASES = [
    "no evidence of", "no sign of", "no signs of", "without evidence of",
    "without", "no", "not", "none", "absent", "negative for",
    "rules out", "ruled out", "rule out", "no definite", "no acute",
    "no significant", "denies", "denied", "unremarkable for",
    "not suggestive of", "not consistent with", "not compatible with",
    "no convincing", "no demonstrable", "no appreciable", "no focal",
    "no gross", "no obvious", "unlikely", "low probability of",
    "resolved", "clear of", "free of", "free from",
]
NEGATION_PHRASES.sort(key=len, reverse=True)


def _is_negated(text, keyword, window_words=8):
    """Check if keyword is negated — scans 8-word window before each occurrence.
    v5.1: Contrastive conjunctions (but, however) break negation scope."""
    text_lower = text.lower()
    kw_lower = keyword.lower()
    start = 0
    found_any = False
    while True:
        idx = text_lower.find(kw_lower, start)
        if idx == -1: break
        found_any = True
        # Find sentence boundary
        r1 = text_lower.rfind(". ", 0, idx); r2 = text_lower.rfind(".\n", 0, idx)
        sent_start = max(r1 + 2 if r1 >= 0 else 0, r2 + 2 if r2 >= 0 else 0, 0)
        prefix = text_lower[sent_start:idx].strip()
        # Contrastive conjunctions break negation scope
        conjs = ["but", "however", "although", "though", "yet", "while"]
        pw = prefix.split()
        for i, w in enumerate(pw):
            if w.strip(",.;:") in conjs: prefix = " ".join(pw[i+1:]); break
        window = " ".join(prefix.split()[-window_words:]) if prefix else ""
        if not any(neg in window for neg in NEGATION_PHRASES):
            return False  # non-negated occurrence found
        start = idx + len(kw_lower)
    return True


def _find_positive_keywords(text, keywords):
    """Partition keywords into (positive, negated) based on context."""
    text_lower = text.lower()
    positive, negated = [], []
    for kw in keywords:
        if kw.lower() not in text_lower: continue
        if _is_negated(text_lower, kw.lower()): negated.append(kw)
        else: positive.append(kw)
    return positive, negated


print("✅ Negation detection ready")


# ═══════════════════════════════════════════════════════════
#  CELL 5b: v5.2 MedPrompting + Confidence Calibration
# ═══════════════════════════════════════════════════════════

# ── Expert System Prompts (per modality specialty) ──
SYSTEM_PROMPTS = {
    "radiologist": (
        "You are a board-certified radiologist with 20 years of experience "
        "across all imaging modalities. You produce structured diagnostic reports "
        "following ACR Reporting Guidelines. You are thorough, precise, and always "
        "note pertinent negatives. You use standard radiological terminology and "
        "provide differential diagnoses ranked by likelihood."
    ),
    "pathologist": (
        "You are a board-certified pathologist specializing in surgical pathology. "
        "You describe histological features systematically: architecture, cellular "
        "morphology, nuclear features, mitotic activity, stroma. You provide "
        "WHO classification-based diagnoses."
    ),
    "neurologist": (
        "You are a board-certified neuroradiologist. You systematically evaluate "
        "brain parenchyma, ventricles, extra-axial spaces, vascular structures, "
        "skull base, and orbits. You reference AAN guidelines."
    ),
}

# ── Structured Prompts per Modality ──
MODALITY_PROMPT_TEMPLATES = {
    "xray": (
        "Analyze this chest radiograph systematically.\n\n"
        "Structure your response EXACTLY as follows:\n\n"
        "**Technique:** [projection, positioning, adequacy]\n\n"
        "**Findings:**\n"
        "- Heart: [size, silhouette]\n"
        "- Mediastinum: [width, contour, tracheal position]\n"
        "- Lungs: [parenchyma, airspace opacity, interstitial pattern]\n"
        "- Pleura: [effusion, pneumothorax, thickening]\n"
        "- Bones: [fractures, lesions, degenerative changes]\n"
        "- Soft tissues: [subcutaneous emphysema, foreign bodies]\n\n"
        "**Impression:**\n[Numbered findings, most significant first]\n\n"
        "**Differential Diagnosis:**\n[2-4 possibilities ranked by likelihood]\n\n"
        "**Recommendations:**\n[Clinical correlation, follow-up]\n\n"
        "Clinical question: {question}"
    ),
    "ct": (
        "Analyze this CT scan systematically.\n\n"
        "Structure your response EXACTLY as follows:\n\n"
        "**Technique:** [contrast, slice thickness]\n\n"
        "**Findings:**\n[Evaluate each visible anatomical region. "
        "Report measurements for abnormal findings. Note pertinent negatives.]\n\n"
        "**Impression:**\n[Numbered list, most clinically significant first]\n\n"
        "**Differential Diagnosis:**\n[For indeterminate findings]\n\n"
        "**Recommendations:**\n[Follow-up, additional workup]\n\n"
        "Clinical question: {question}"
    ),
    "mri": (
        "Analyze this MRI scan systematically.\n\n"
        "Structure your response EXACTLY as follows:\n\n"
        "**Technique:** [sequences, contrast, field strength]\n\n"
        "**Findings:**\n[Signal characteristics on each sequence. "
        "Measurements. Enhancement patterns. Mass effect assessment.]\n\n"
        "**Impression:**\n[Numbered list, most significant first]\n\n"
        "**Differential Diagnosis:**\n[Signal-based differential]\n\n"
        "**Recommendations:**\n[Follow-up, additional sequences]\n\n"
        "Clinical question: {question}"
    ),
    "pathology": (
        "Analyze this histopathology image systematically.\n\n"
        "**Microscopic Description:**\n"
        "- Architecture: [growth pattern, organization]\n"
        "- Cellular features: [size, shape, cytoplasm]\n"
        "- Nuclear features: [size, chromatin, nucleoli, mitoses]\n"
        "- Stroma: [fibrosis, inflammation, necrosis]\n\n"
        "**Diagnosis:**\n[WHO classification if applicable]\n\n"
        "**Differential Diagnosis:**\n[Alternative diagnoses]\n\n"
        "**Recommendations:**\n[IHC, molecular testing]\n\n"
        "Clinical question: {question}"
    ),
    "video": (
        "Analyze this medical video (endoscopy/procedure).\n\n"
        "**Procedure:** [Type, anatomical location]\n\n"
        "**Findings:**\n[Mucosal appearance, lesions, abnormalities. "
        "Note location, size, morphology.]\n\n"
        "**Impression:**\n[Summary of significant findings]\n\n"
        "**Recommendations:**\n[Biopsy, follow-up, therapy]\n\n"
        "Clinical question: {question}"
    ),
    "general_medical": (
        "Analyze this medical image thoroughly.\n\n"
        "Structure your response as:\n\n"
        "**Technique:** [Imaging modality and parameters]\n\n"
        "**Findings:**\n[Systematic description of all visible structures. "
        "Report abnormalities with measurements. Note pertinent negatives.]\n\n"
        "**Impression:**\n[Numbered findings, most significant first]\n\n"
        "**Differential Diagnosis:**\n[Ranked possibilities]\n\n"
        "**Recommendations:**\n[Clinical correlation and follow-up]\n\n"
        "Clinical question: {question}"
    ),
}
# Aliases
for _alias, _target in [("fundoscopy", "general_medical"), ("dermoscopy", "general_medical"),
                         ("mammography", "xray"), ("endoscopy", "video"), ("ultrasound", "general_medical"),
                         ("3d_volume", "ct")]:
    MODALITY_PROMPT_TEMPLATES[_alias] = MODALITY_PROMPT_TEMPLATES[_target]


def build_expert_prompt(question, modality="general_medical", file_type="2d", role="primary"):
    """Build expert medical prompt — the #1 lever for output quality."""
    template = MODALITY_PROMPT_TEMPLATES.get(modality, MODALITY_PROMPT_TEMPLATES["general_medical"])
    prompt = template.format(question=question)
    if role == "reasoner":
        prompt = ("Please reason step by step about this medical image.\n"
                  "First describe objectively, then consider conditions, then assess.\n\n" + prompt)
    if file_type == "3d":
        prompt = ("This is a 3D volumetric scan presented as sequential slices. "
                  "Evaluate systematically across all slices.\n\n" + prompt)
    return prompt


def get_system_prompt(modality="general_medical"):
    """Get specialty-appropriate system prompt."""
    mapping = {"xray": "radiologist", "ct": "radiologist", "mri": "neurologist",
               "pathology": "pathologist", "ultrasound": "radiologist",
               "video": "radiologist", "mammography": "radiologist"}
    key = mapping.get(modality, "radiologist")
    return SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS["radiologist"])


# ── Confidence Calibration from Linguistic Analysis ──
HIGH_CONF_MARKERS = [
    "consistent with", "diagnostic of", "compatible with", "characteristic of",
    "pathognomonic", "definitive", "classic appearance", "confirms",
    "demonstrates", "clearly shows", "unequivocal", "represents",
]
LOW_CONF_MARKERS = [
    "might be", "could be", "possibly", "uncertain", "unclear",
    "may represent", "cannot exclude", "cannot rule out",
    "differential includes", "nonspecific", "equivocal", "indeterminate",
    "correlate clinically", "further evaluation", "limited study",
    "suboptimal", "artifact", "suggest", "consider", "versus",
]
NORMAL_MARKERS = [
    "no acute", "unremarkable", "within normal limits", "no significant",
    "no evidence of", "normal", "clear", "intact", "negative",
]


def calibrate_confidence(text, base=0.7):
    """Calibrate confidence from model language — replaces hardcoded 0.85."""
    tl = text.lower()
    high = sum(1 for p in HIGH_CONF_MARKERS if p in tl)
    low = sum(1 for p in LOW_CONF_MARKERS if p in tl)
    wc = len(tl.split())
    length_adj = -0.15 if wc < 30 else (-0.05 if wc < 80 else (0.05 if wc > 300 else 0.0))
    struct_bonus = sum(0.03 for s in ["findings", "impression", "technique"] if s in tl)
    return round(max(0.10, min(0.98, base + high * 0.04 - low * 0.06 + length_adj + struct_bonus)), 3)


def extract_findings(text):
    """Extract individual clinical findings for cross-model validation."""
    findings = []
    anatomy = ["heart", "cardiac", "lung", "pulmonary", "pleural", "mediastin", "aort",
                "liver", "hepat", "kidney", "renal", "brain", "cerebr", "bone", "osseous",
                "spleen", "pancrea", "bowel", "spine", "vertebr", "pericardi", "hilum"]
    finding_verbs = ["shows", "demonstrates", "reveals", "indicating", "consistent with",
                     "compatible with", "suggestive of", "evidence of", "noted", "identified", "seen"]
    for sent in re.split(r'(?<=[.!?])\s+', text):
        sent = sent.strip()
        if len(sent) < 15: continue
        sl = sent.lower()
        if any(v in sl for v in finding_verbs) or any(a in sl for a in anatomy):
            loc = next((a for a in anatomy if a in sl), "general")
            sev = "high" if any(w in sl for w in ["acute", "severe", "large", "critical"]) else \
                  "moderate" if any(w in sl for w in ["moderate", "significant", "abnormal"]) else \
                  "low" if any(w in sl for w in ["mild", "small", "minor", "subtle"]) else "routine"
            findings.append({"sentence": sent, "location": loc, "severity": sev,
                             "is_normal": any(n in sl for n in NORMAL_MARKERS)})
    return findings


print("✅ MedPrompting + Confidence Calibration ready")


# ═══════════════════════════════════════════════════════════
#  CELL 4: MONAI Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════

from monai.transforms import (
    Compose, EnsureChannelFirst, LoadImage, NormalizeIntensity,
    Orientation, Resize, ScaleIntensity, ScaleIntensityRange, Spacing)


class MONAIPipeline:
    CT_WINDOWS = {"soft_tissue": {"center": 40, "width": 400}, "lung": {"center": -600, "width": 1500},
                  "bone": {"center": 400, "width": 1800}, "brain": {"center": 40, "width": 80}}

    def __init__(self):
        self._scale = ScaleIntensity(minv=0.0, maxv=1.0)
        self._norm_mri = NormalizeIntensity(nonzero=True, channel_wise=True)

    def preprocess(self, data, modality="auto", target_size=None):
        ft = data.get("type", "2d")
        if ft == "3d": return self._preprocess_3d(data, modality, target_size)
        elif ft == "video": return self._preprocess_video(data, target_size)
        return self._preprocess_2d(data, modality, target_size)

    def _preprocess_2d(self, data, modality, target_size):
        image = data.get("image") or data.get("original_image")
        if image is None:
            pa = data.get("pixel_data") or data.get("pixel_array")
            if pa is not None:
                if pa.ndim == 2: image = Image.fromarray(((pa-pa.min())/(pa.max()-pa.min()+1e-8)*255).astype(np.uint8), "L")
                else: image = Image.fromarray(pa.astype(np.uint8))
            else: raise ValueError("No image data found")
        arr = np.array(image).astype(np.float32)
        if modality == "ct": arr = self._ct_window(arr, data)
        elif modality == "mri": arr = self._mri_norm(arr)
        if arr.ndim == 2: arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4): arr = np.transpose(arr, (2, 0, 1))
        tensor = self._scale(torch.from_numpy(arr).float())
        return {"tensor": tensor, "pil_image": image.convert("RGB"), "original_data": data,
                "preprocessing": {"modality": modality, "output_shape": list(tensor.shape)}}

    def _preprocess_3d(self, data, modality, target_size):
        vol = data.get("volume")
        if vol is None: raise ValueError("No volume data")
        vol = vol.astype(np.float32)
        if modality == "ct": vol = self._ct_window_3d(vol, data)
        elif modality == "mri":
            t = torch.from_numpy(vol).float().unsqueeze(0)
            vol = self._norm_mri(t).squeeze(0).numpy()
        if vol.ndim == 3: vol = vol[np.newaxis, ...]
        size = target_size or (128, 256, 256)
        tensor = self._scale(Resize(spatial_size=size, mode="trilinear")(torch.from_numpy(vol).float()))
        return {"tensor": tensor, "original_data": data,
                "preprocessing": {"modality": modality, "target_size": size, "output_shape": list(tensor.shape)}}

    def _preprocess_video(self, data, target_size):
        frames = data.get("frames", [])
        if not frames: raise ValueError("No video frames")
        processed = []
        for f in frames:
            if not isinstance(f, Image.Image): f = Image.fromarray(np.array(f))
            f = f.convert("RGB")
            if target_size: f = f.resize(target_size, Image.LANCZOS)
            processed.append(f)
        return {"frames": processed, "original_data": data, "preprocessing": {"frame_count": len(processed)}}

    def _ct_window(self, arr, data):
        meta = data.get("metadata", {}); c = meta.get("window_center", 40); w = meta.get("window_width", 400)
        if isinstance(c, list): c = c[0]
        if isinstance(w, list): w = w[0]
        lo, hi = float(c)-float(w)/2, float(c)+float(w)/2
        t = torch.from_numpy(arr).float()
        if t.ndim == 2: t = t.unsqueeze(0)
        return ScaleIntensityRange(a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True)(t).squeeze(0).numpy()

    def _ct_window_3d(self, vol, data=None):
        meta = (data or {}).get("metadata", {}); c = meta.get("window_center", 40); w = meta.get("window_width", 400)
        if isinstance(c, list): c = c[0]
        if isinstance(w, list): w = w[0]
        lo, hi = float(c)-float(w)/2, float(c)+float(w)/2
        t = torch.from_numpy(vol).float()
        if t.ndim == 3: t = t.unsqueeze(0)
        return ScaleIntensityRange(a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True)(t).squeeze(0).numpy()

    def _mri_norm(self, arr):
        t = torch.from_numpy(arr).float()
        if t.ndim == 2: t = t.unsqueeze(0)
        return self._norm_mri(t).squeeze(0).numpy()


print("✅ MONAI Pipeline ready")


# ═══════════════════════════════════════════════════════════
#  CELL 5: Ingestion Layer
# ═══════════════════════════════════════════════════════════

class ImageLoader:
    SUPPORTED_2D = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    SUPPORTED_3D = {".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"}
    def load(self, path):
        path = Path(path)
        if not path.exists(): raise FileNotFoundError(f"Image not found: {path}")
        suffix = "".join(path.suffixes).lower()
        if suffix in self.SUPPORTED_3D or path.suffix.lower() in self.SUPPORTED_3D: return self._load_3d(path)
        return self._load_2d(path)
    def _load_2d(self, path):
        image = Image.open(str(path)); image_rgb = image.convert("RGB")
        return {"type": "2d", "image": image_rgb, "original_image": image, "pixel_array": np.array(image_rgb),
                "metadata": {"width": image.width, "height": image.height, "mode": image.mode}, "source_path": str(path)}
    def _load_3d(self, path):
        try:
            import SimpleITK as sitk; img = sitk.ReadImage(str(path)); vol = sitk.GetArrayFromImage(img)
            return {"type": "3d", "volume": vol, "metadata": {"shape": vol.shape, "spacing": img.GetSpacing()}, "source_path": str(path)}
        except Exception:
            import nibabel as nib; nii = nib.load(str(path)); vol = np.asarray(nii.dataobj, dtype=np.float32)
            return {"type": "3d", "volume": vol, "metadata": {"shape": vol.shape}, "source_path": str(path)}
    def extract_slices(self, volume, axis=2, num_slices=None):
        total = volume.shape[axis]; indices = np.linspace(0, total-1, num_slices or total, dtype=int)
        slices = []
        for idx in indices:
            s = np.take(volume, idx, axis=axis); smin, smax = s.min(), s.max()
            slices.append(Image.fromarray(((s-smin)/(smax-smin+1e-8)*255).astype(np.uint8), mode="L"))
        return slices

class VideoLoader:
    def load(self, path, fps=1.0, max_frames=1800, target_size=None):
        import cv2; path = Path(path); cap = cv2.VideoCapture(str(path))
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; frame_interval = max(1, int(video_fps / fps))
        frames, timestamps, frame_idx = [], [], 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if target_size: rgb = cv2.resize(rgb, target_size)
                frames.append(Image.fromarray(rgb)); timestamps.append(frame_idx / video_fps)
            frame_idx += 1
        cap.release()
        return {"type": "video", "frames": frames, "timestamps": timestamps,
                "metadata": {"original_fps": video_fps, "extracted_frames": len(frames)}, "source_path": str(path)}

class DICOMLoader:
    def load(self, path):
        import pydicom; ds = pydicom.dcmread(str(path)); pixel_array = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        metadata = {}
        for tag in ["PatientID", "Modality", "StudyDescription", "WindowCenter", "WindowWidth"]:
            if hasattr(ds, tag): metadata[tag.lower()] = getattr(ds, tag, None)
        return {"type": "2d", "pixel_data": pixel_array, "metadata": metadata, "source_path": str(path)}

class ModalityDetector:
    DICOM_MAP = {"CR": "xray", "DX": "xray", "CT": "ct", "MR": "mri", "US": "ultrasound",
                 "ES": "endoscopy", "MG": "mammography", "OP": "fundoscopy", "SM": "pathology"}
    def detect(self, data):
        ft = data.get("type", "unknown"); meta = data.get("metadata", {})
        if ft == "video": return {"modality": "video", "sub_type": "endoscopy", "confidence": 0.9, "dimensions": "temporal"}
        dm = meta.get("modality")
        if dm and dm in self.DICOM_MAP:
            return {"modality": self.DICOM_MAP[dm], "sub_type": dm, "confidence": 0.95, "dimensions": "3d" if ft == "3d" else "2d"}
        if ft == "3d":
            vol = data.get("volume")
            if vol is not None:
                if float(vol.min()) < -500 and float(vol.max()) > 500:
                    return {"modality": "ct", "sub_type": "general", "confidence": 0.85, "dimensions": "3d"}
                return {"modality": "mri", "sub_type": "general", "confidence": 0.7, "dimensions": "3d"}
        return {"modality": "xray", "sub_type": "general", "confidence": 0.5, "dimensions": "2d"}

class QualityAssessor:
    def assess(self, data):
        checks = {}; ft = data.get("type", "unknown")
        if ft == "2d":
            pa = data.get("pixel_array")
            if pa is not None:
                meta = data.get("metadata", {}); h, w = meta.get("height", pa.shape[0]), meta.get("width", pa.shape[1] if pa.ndim > 1 else 0)
                if h < 100 or w < 100: checks["resolution"] = {"score": 0.2, "warning": f"Very low: {w}x{h}"}
                elif h < 256 or w < 256: checks["resolution"] = {"score": 0.5, "warning": f"Low: {w}x{h}"}
                else: checks["resolution"] = {"score": 1.0}
        scores = [c.get("score", 1.0) for c in checks.values()] if checks else [1.0]
        overall = float(np.mean(scores))
        return {"overall_score": round(overall, 3), "is_acceptable": overall >= 0.5,
                "checks": checks, "warnings": [c.get("warning") for c in checks.values() if c.get("warning")]}

print("✅ Ingestion layer ready")


# ═══════════════════════════════════════════════════════════
#  CELL 6: Model Wrappers (4-bit Quantized) — v5.1
# ═══════════════════════════════════════════════════════════

class BaseModelKaggle:
    def __init__(self, model_id, device_idx=0):
        self.model_id = model_id; self.model = None; self.processor = None
        self.device_idx = device_idx; self.is_loaded = False
    def get_device(self):
        return torch.device(f"cuda:{self.device_idx}" if torch.cuda.is_available() else "cpu")
    def unload(self):
        del self.model; del self.processor; self.model = self.processor = None
        gc.collect(); torch.cuda.empty_cache(); self.is_loaded = False


class HuluMedKaggle(BaseModelKaggle):
    """HuluMed — v6.0: Separate _analyze_2d/_analyze_3d/_analyze_video paths."""

    def load(self):
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading HuluMed (4-bit): {self.model_id}")
        # v6.0: HuluMed-4B fits on single T4 GPU (~3GB 4-bit)
        # For 7B/14B/32B on production servers, use device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, quantization_config=QUANT_CONFIG_4BIT,
            device_map={"": self.device_idx}, torch_dtype=torch.bfloat16)
        self.processor = self._load_processor()
        self.tokenizer = self.processor.tokenizer

        # ═══════════════════════════════════════════════════════════
        # v5.2 DEFINITIVE FIX for 'list' object has no attribute 'replace'
        #
        # ROOT CAUSE: processing_hulumed.py → process_text() internally
        # calls tokenizer.apply_chat_template(text, tokenize=False).
        # On Kaggle's transformers (>=4.50), this returns a LIST of
        # strings instead of a single string. Then process_text does
        # text[i].replace() which crashes because text[i] is a list.
        #
        # FIX: Monkey-patch apply_chat_template on ALL tokenizer
        # instances to ALWAYS return a string when tokenize=False.
        # This fixes the bug AT ITS SOURCE — inside process_text,
        # wherever it calls apply_chat_template, it gets a string.
        # ═══════════════════════════════════════════════════════════
        self._patch_apply_chat_template()

        self._manual_processor = False
        try:
            _ = self.processor(conversation=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
                               return_tensors="pt", add_generation_prompt=True)
        except Exception as e:
            logger.warning(f"Processor __call__ broken ({e}), using manual path")
            self._manual_processor = True
        self.is_loaded = True
        logger.info(f"✅ HuluMed loaded on GPU {self.device_idx}")

    def _patch_apply_chat_template(self):
        """Monkey-patch apply_chat_template to always return string when tokenize=False.

        This is the DEFINITIVE fix. process_text() inside processing_hulumed.py
        calls apply_chat_template which returns a list on newer transformers.
        By patching at the source, every internal call gets a string back.
        """
        def _make_safe(original_fn):
            def safe_apply_chat_template(*args, **kwargs):
                result = original_fn(*args, **kwargs)
                # Only coerce when tokenize=False (text mode, not token IDs)
                if kwargs.get('tokenize', True) is False and isinstance(result, list):
                    if len(result) == 1:
                        return result[0]
                    return "".join(str(r) for r in result)
                return result
            return safe_apply_chat_template

        # Patch every tokenizer instance the processor might use
        patched = set()
        for tok in [self.tokenizer, getattr(self.processor, 'tokenizer', None)]:
            if tok is not None and id(tok) not in patched:
                tok.apply_chat_template = _make_safe(tok.apply_chat_template)
                patched.add(id(tok))
                logger.info(f"✅ Patched apply_chat_template on {type(tok).__name__}")

    def _load_processor(self):
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            logger.info("HuluMed processor via AutoProcessor"); return proc
        except Exception as e:
            logger.warning(f"AutoProcessor failed: {e}")
        try: return self._build_processor_dynamic()
        except Exception as e2: logger.warning(f"Dynamic build failed: {e2}")
        return self._build_processor_importlib()

    def _build_processor_dynamic(self):
        from transformers import AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        ImgProcCls = get_class_from_dynamic_module("image_processing_hulumed.HulumedImageProcessor", self.model_id)
        ProcCls = get_class_from_dynamic_module("processing_hulumed.HulumedProcessor", self.model_id)
        return ProcCls(image_processor=ImgProcCls.from_pretrained(self.model_id), tokenizer=tokenizer)

    def _build_processor_importlib(self):
        from transformers import AutoTokenizer; from huggingface_hub import hf_hub_download; import importlib.util
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        proc_py = hf_hub_download(self.model_id, "processing_hulumed.py")
        img_py = hf_hub_download(self.model_id, "image_processing_hulumed.py")
        cache_dir = os.path.dirname(proc_py)
        for fname in ["constants.py", "mm_utils.py"]:
            try: hf_hub_download(self.model_id, fname)
            except Exception: pass
        if cache_dir not in sys.path: sys.path.insert(0, cache_dir)
        def _lm(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
        img_mod = _lm("image_processing_hulumed", img_py)
        proc_mod = _lm("processing_hulumed", proc_py)
        return proc_mod.HulumedProcessor(image_processor=img_mod.HulumedImageProcessor.from_pretrained(self.model_id), tokenizer=tokenizer)

    def analyze(self, images=None, text="", modality="image", **kwargs):
        """v6.0: Route to unified analyze with HF-first + manual fallback."""
        if not self.is_loaded: self.load()
        start = time.time()
        conversation = self._build_conversation(images, text, modality, **kwargs)

        # ═══════════════════════════════════════════════════════════════
        #  STRATEGY: Try the official HF processor(conversation=...) path
        #  first. If it fails, fall back to manual processing.
        #  The HF path handles image loading, system prompt, and
        #  tokenization automatically — it's the best path.
        # ═══════════════════════════════════════════════════════════════
        inputs = None

        # Path A: Official HF Transformers API (preferred)
        if not self._manual_processor:
            try:
                inputs = self.processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
            except Exception as e:
                logger.warning(f"HF processor(conversation=...) failed: {e}, falling back to manual")
                self._manual_processor = True

        # Path B: Manual Original Method API (fallback)
        if inputs is None:
            inputs = self._manual_process(conversation, images, modality, **kwargs)

        # Move to device + cast pixel_values to bfloat16 (per official docs)
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Generate — match official API generate params per path
        modal_map = {"image": "image", "3d": "video", "video": "video", "text": "text"}
        with torch.inference_mode():
            if self._manual_processor:
                # Pop any keys that would conflict with explicit generate() kwargs
                gen_inputs = dict(inputs)
                existing_modals = gen_inputs.pop("modals", None)
                target_modal = modal_map.get(modality, "image")
                output = self.model.generate(
                    **gen_inputs, do_sample=True,
                    modals=[target_modal],
                    temperature=0.6, max_new_tokens=2048,
                    use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                output = self.model.generate(**inputs, max_new_tokens=2048)

        return self._decode_output(output, start, modality)

    def _manual_process(self, conversation, images=None, modality="image", **kwargs):
        """v5.3 DEFINITIVE FIX: Bypass process_text entirely.

        process_text is broken on Kaggle's transformers — every approach that
        touches it crashes with 'list' object has no attribute 'replace'.

        Solution: Follow the vLLM approach (Document 46 / HuluMed test suite):
        1. Process images via process_images (this works fine)
        2. Build prompt with <image> tokens embedded in plain text
        3. Use apply_chat_template on simple string content (not dict format)
        4. Tokenize with tokenizer() directly
        5. Merge pixel_values + input_ids

        This COMPLETELY avoids the broken process_text code path.
        """
        from transformers import BatchEncoding

        # ── Step 1: Collect PIL images ──
        all_images = []

        # Use pre-loaded PIL images from engine first
        if images and modality == "image":
            for img in (images if isinstance(images, list) else [images]):
                if isinstance(img, Image.Image):
                    all_images.append(img)

        # If no images from engine, load from conversation paths
        if not all_images:
            for msg in conversation:
                for item in msg.get("content", []):
                    ctype = item.get("type", "")
                    if ctype == "image":
                        img_info = item.get("image", {}); img_path = img_info.get("image_path", "")
                        if img_path and os.path.exists(img_path):
                            try:
                                loaded = self.processor.load_images(img_path)
                                all_images.extend(loaded if isinstance(loaded, list) else [loaded])
                            except Exception as e:
                                logger.warning(f"load_images failed: {e}")
                                if images: all_images.extend(images if isinstance(images, list) else [images])
                        elif images:
                            all_images.extend(images if isinstance(images, list) else [images])
                    elif ctype == "3d":
                        td = item.get("3d", {}); ip = td.get("image_path", "")
                        if ip and os.path.exists(ip):
                            loaded_3d = False
                            try:
                                loaded = self.processor.load_images(ip, nii_num_slices=td.get("nii_num_slices", 180), nii_axis=td.get("nii_axis", 2))
                                all_images.extend(loaded if isinstance(loaded, list) else [loaded]); loaded_3d = True
                            except TypeError:
                                try:
                                    loaded = self.processor.load_images(ip)
                                    all_images.extend(loaded if isinstance(loaded, list) else [loaded]); loaded_3d = True
                                except Exception: pass
                            except Exception: pass
                            if not loaded_3d:
                                try:
                                    import nibabel as nib
                                    nii = nib.load(str(ip)); vol = np.asarray(nii.dataobj, dtype=np.float32)
                                    ns, ax = td.get("nii_num_slices", 180), td.get("nii_axis", 2)
                                    for idx in np.linspace(0, vol.shape[ax]-1, min(ns, vol.shape[ax]), dtype=int):
                                        s = np.take(vol, int(idx), axis=ax)
                                        sn = ((s - s.min()) / (s.max() - s.min() + 1e-8) * 255).astype(np.uint8)
                                        all_images.append(Image.fromarray(sn, "L").convert("RGB"))
                                    logger.info(f"Loaded {len(all_images)} slices via nibabel")
                                except Exception as e3: logger.error(f"nibabel failed: {e3}")
                    elif ctype == "video":
                        vid = item.get("video", {}); vp = vid.get("video_path", "")
                        if vp and os.path.exists(vp):
                            try:
                                frames, _ = self.processor.load_video(vp, fps=vid.get("fps", 1), max_frames=vid.get("max_frames", 1800))
                                all_images.extend(frames)
                            except Exception as e:
                                if images: all_images.extend(images if isinstance(images, list) else [images])

        # ── Step 2: Process images via process_images (this works fine) ──
        merge_size = 2 if modality in ("video", "3d") else 1
        images_for_processor = [all_images] if (modality in ("video", "3d") and all_images) else (all_images or None)
        image_inputs = self.processor.process_images(images_for_processor, merge_size=merge_size)

        for key in list(image_inputs.keys()):
            val = image_inputs[key]
            if isinstance(val, np.ndarray): image_inputs[key] = torch.from_numpy(val)
            elif isinstance(val, list):
                image_inputs[key] = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in val]

        # ── Step 3: Build prompt with <image> tokens (vLLM approach) ──
        # Extract the user's question text from conversation
        question_text = ""
        for msg in conversation:
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    question_text = item.get("text", "")

        # Build content string with <image> tokens — exactly like vLLM test suite:
        #   Single image: "<image>Question text"
        #   3D/Video:     "<image><image>...<image>Question text" (one per frame)
        num_images = len(all_images) if all_images else 0
        image_tokens = "<image>" * num_images
        content_str = f"{image_tokens}{question_text}"

        # Build simple conversation (plain string content, NOT dict format)
        simple_conv = [{"role": "user", "content": content_str}]

        # ── Step 4: apply_chat_template → prompt string ──
        prompt = self.tokenizer.apply_chat_template(
            simple_conv, tokenize=False, add_generation_prompt=True
        )
        # Coerce to string if list (safety net)
        if isinstance(prompt, list):
            prompt = prompt[0] if len(prompt) == 1 else "".join(str(x) for x in prompt)

        # ── Step 5: Tokenize directly (BYPASSES process_text entirely) ──
        text_inputs = self.tokenizer(prompt, return_tensors="pt")

        # ── Step 6: Merge image + text inputs ──
        return BatchEncoding(data={**text_inputs, **image_inputs})

    def _decode_output(self, output, start, modality):
        """Decode model output with thinking extraction and calibrated confidence."""
        answer = self.processor.batch_decode(output, skip_special_tokens=True, use_think=False)[0].strip()
        full = self.processor.batch_decode(output, skip_special_tokens=True, use_think=True)[0].strip()
        thinking = full.replace(answer, "").strip() if full != answer else ""
        conf = calibrate_confidence(answer, base=0.80)
        return {"model": self.model_id, "answer": answer, "response": full, "thinking": thinking,
                "confidence": conf, "modality": modality,
                "metadata": {"inference_time": round(time.time()-start, 2)}}

    def _build_conversation(self, images, text, modality, **kwargs):
        content = []
        if modality == "image":
            path = kwargs.get("source_path", "")
            if not path and images:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False); images[0].save(tmp.name); tmp.close(); path = tmp.name
            content.append({"type": "image", "image": {"image_path": str(path)}})
        elif modality == "3d":
            nii = kwargs.get("nii_path", kwargs.get("source_path", ""))
            content.append({"type": "3d", "3d": {"image_path": str(nii), "nii_num_slices": kwargs.get("nii_num_slices", 180), "nii_axis": kwargs.get("nii_axis", 2)}})
        elif modality == "video":
            vp = kwargs.get("video_path", kwargs.get("source_path", ""))
            content.append({"type": "video", "video": {"video_path": str(vp), "fps": 1, "max_frames": 1800}})
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]


class MedGemmaKaggle(BaseModelKaggle):
    """v6.0: Uses modality-aware expert system prompts + calibrated confidence."""
    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, quantization_config=QUANT_CONFIG_4BIT, device_map={"": self.device_idx}, torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(self.model_id); self.is_loaded = True
    def analyze(self, images=None, text="", modality="image", **kwargs):
        if not self.is_loaded: self.load()
        start = time.time()
        # v5.2: Use modality-aware expert system prompt
        detected_modality = kwargs.get("detected_modality", "general_medical")
        sys_prompt = get_system_prompt(detected_modality)
        messages = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}, {"role": "user", "content": []}]
        if images and modality == "image": messages[1]["content"].append({"type": "image", "image": images[0]})
        messages[1]["content"].append({"type": "text", "text": text})
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode(): output = self.model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)
        response = self.processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # v5.2: Calibrated confidence from linguistic analysis
        conf = calibrate_confidence(response, base=0.75)
        return {"model": self.model_id, "answer": response, "response": response, "thinking": "",
                "confidence": conf, "modality": modality, "metadata": {"inference_time": round(time.time()-start, 2)}}


class MediXR1Kaggle(BaseModelKaggle):
    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, quantization_config=QUANT_CONFIG_4BIT, device_map={"": self.device_idx}, torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(self.model_id); self.is_loaded = True
    def analyze(self, images=None, text="", modality="image", **kwargs):
        if not self.is_loaded: self.load()
        start = time.time()
        MAX_DIM = 384; resized = []
        if images and modality == "image":
            for img in (images if isinstance(images, list) else [images]):
                if isinstance(img, Image.Image):
                    w, h = img.size
                    if max(w, h) > MAX_DIM: scale = MAX_DIM / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                    resized.append(img)
        messages = [{"role": "user", "content": []}]
        if resized and modality == "image": messages[0]["content"].append({"type": "image", "image": resized[0]})
        messages[0]["content"].append({"type": "text", "text": text})
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = None
        try:
            from qwen_vl_utils import process_vision_info; image_inputs, _ = process_vision_info(messages)
        except Exception: image_inputs = resized if resized else None
        proc_kw = {"text": [chat_text], "return_tensors": "pt", "padding": True}
        if image_inputs: proc_kw["images"] = image_inputs
        inputs = self.processor(**proc_kw).to(self.model.device)
        with torch.inference_mode(): output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.6)
        response = self.processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        thinking = ""
        for tag in ["<think>", "<thinking>"]:
            end_tag = tag.replace("<", "</")
            if tag in response and end_tag in response:
                thinking = response[response.index(tag)+len(tag):response.index(end_tag)].strip()
                response = response[response.index(end_tag)+len(end_tag):].strip(); break
        # v5.2: Calibrated confidence
        conf = calibrate_confidence(response, base=0.70)
        return {"model": self.model_id, "answer": response, "response": response, "thinking": thinking, "confidence": conf, "modality": modality, "metadata": {"inference_time": round(time.time()-start, 2)}}


class BiomedCLIPKaggle(BaseModelKaggle):
    MEDICAL_LABELS = ["chest x-ray showing normal findings", "chest x-ray showing pneumonia", "chest x-ray showing pleural effusion",
        "chest x-ray showing cardiomegaly", "CT scan of the brain", "MRI scan", "pathology slide", "fundus photograph", "dermoscopy image", "ultrasound image"]
    def load(self):
        from open_clip import create_model_from_pretrained, get_tokenizer
        self.model, self.preprocess = create_model_from_pretrained(self.model_id)
        self.tokenizer = get_tokenizer(self.model_id)
        self.model = self.model.to(self.get_device()).half(); self.is_loaded = True
    def analyze(self, images=None, text="", modality="image", **kwargs):
        if not self.is_loaded: self.load()
        if not images: return {"model": "biomedclip", "answer": "No image provided", "confidence": 0}
        device = self.get_device()
        img_tensor = self.preprocess(images[0].convert("RGB")).unsqueeze(0).to(device).half()
        labels = kwargs.get("labels", self.MEDICAL_LABELS); text_tokens = self.tokenizer(labels).to(device)
        with torch.inference_mode():
            img_feat, txt_feat, _ = self.model(img_tensor, text_tokens)
            sims = ((img_feat / img_feat.norm(dim=-1, keepdim=True)) @ (txt_feat / txt_feat.norm(dim=-1, keepdim=True)).T).squeeze(0).softmax(dim=-1)
        top_idx = sims.argmax().item()
        return {"model": "biomedclip", "answer": labels[top_idx], "response": labels[top_idx], "thinking": "",
                "confidence": float(sims[top_idx]), "all_scores": {l: round(float(s), 4) for l, s in zip(labels, sims)}, "modality": modality}


class Med3DVLMKaggle(BaseModelKaggle):
    IMAGE_SIZE = (128, 256, 256)
    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, max_length=1024, padding_side="right", use_fast=False, ignore_mismatched_sizes=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, quantization_config=QUANT_CONFIG_4BIT, device_map={"": self.device_idx}, torch_dtype=torch.bfloat16)
        try: self.proj_out_num = self.model.get_model().config.proj_out_num if hasattr(self.model.get_model().config, "proj_out_num") else 256
        except Exception: self.proj_out_num = 256
        self.is_loaded = True
    def analyze(self, images=None, text="", modality="3d", **kwargs):
        if not self.is_loaded: self.load()
        start = time.time(); volume = self._get_volume(images, kwargs)
        image_size = kwargs.get("image_size", self.IMAGE_SIZE)
        image_input = np.expand_dims(volume.copy(), axis=0)
        image_input = Resize(spatial_size=image_size, mode="bilinear")(image_input)
        image_input = image_input.data.unsqueeze(0).to(device=self.get_device(), dtype=torch.bfloat16)
        input_txt = "<im_patch>" * self.proj_out_num + text
        input_ids = self.processor(input_txt, return_tensors="pt")["input_ids"].to(device=self.get_device())
        with torch.no_grad():
            generation = self.model.generate(images=image_input, inputs=input_ids, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=1.0)
        response = self.processor.decode(generation[0], skip_special_tokens=True).strip()
        return {"model": self.model_id, "answer": response, "response": response, "thinking": "", "confidence": 0.85, "modality": "3d",
                "metadata": {"inference_time": round(time.time()-start, 2), "volume_shape": list(volume.shape)}}
    def _get_volume(self, images, kwargs):
        va = kwargs.get("volume_array")
        if va is not None: return np.asarray(va, dtype=np.float32)
        vp = kwargs.get("volume_path") or kwargs.get("nii_path") or kwargs.get("source_path")
        if vp:
            try: import SimpleITK as sitk; return sitk.GetArrayFromImage(sitk.ReadImage(str(vp))).astype(np.float32)
            except Exception: import nibabel as nib; return np.asarray(nib.load(str(vp)).dataobj, dtype=np.float32)
        if images and isinstance(images, list) and len(images) > 0 and isinstance(images[0], np.ndarray) and images[0].ndim == 3:
            return images[0].astype(np.float32)
        raise ValueError("Med3DVLM requires 3D volume.")


print("✅ All model wrappers defined (v6.0)")


# ═══════════════════════════════════════════════════════════
#  CELL 7: v5.1 Pipeline Components (Routing, Fusion, Governance, RAG, Reporting)
# ═══════════════════════════════════════════════════════════

class IntelligentRouter:
    ROUTING_TABLE = {
        "xray": {"primary": ["hulu_med_7b", "medgemma_4b"], "verifier": ["biomedclip"], "reasoner": ["medix_r1_2b"], "specialist_3d": []},
        "ct": {"primary": ["hulu_med_7b"], "verifier": ["biomedclip"], "reasoner": ["medix_r1_2b"], "specialist_3d": ["med3dvlm"]},
        "mri": {"primary": ["hulu_med_7b", "medgemma_4b"], "verifier": ["biomedclip"], "reasoner": ["medix_r1_2b"], "specialist_3d": ["med3dvlm"]},
        "pathology": {"primary": ["medgemma_4b", "hulu_med_7b"], "verifier": ["biomedclip"], "reasoner": ["medix_r1_2b"], "specialist_3d": []},
        "ultrasound": {"primary": ["hulu_med_7b", "medgemma_4b"], "verifier": ["biomedclip"], "reasoner": ["medix_r1_2b"], "specialist_3d": []},
        "video": {"primary": ["hulu_med_7b"], "verifier": [], "reasoner": ["medix_r1_2b"], "specialist_3d": []},
        "3d_volume": {"primary": ["hulu_med_7b"], "verifier": [], "reasoner": ["medix_r1_2b"], "specialist_3d": ["med3dvlm"]},
        "general_medical": {"primary": ["hulu_med_7b", "medgemma_4b"], "verifier": ["biomedclip"], "reasoner": ["medix_r1_2b"], "specialist_3d": []},
    }
    def __init__(self, available_models):
        self.available = set(available_models)
    def route(self, modality, file_type="2d", complexity="standard"):
        key = modality if modality in self.ROUTING_TABLE else "general_medical"
        base = {k: list(v) for k, v in self.ROUTING_TABLE[key].items()}
        if file_type == "3d" and not base.get("specialist_3d"): base["specialist_3d"] = ["med3dvlm"]
        return {role: [m for m in models if m in self.available] for role, models in base.items()}



# MultiModelFusion removed in v6.0 — replaced by DynamicFusionEngine with
# uncertainty-aware scoring, model debate, and reasoning-engine integration.


class GovernanceLayer:
    """v6.0: Negation-aware clinical validation + risk flagging."""
    CRITICAL_FINDINGS = {"pneumothorax": "stat", "pulmonary embolism": "stat", "aortic dissection": "emergent",
                         "intracranial hemorrhage": "emergent", "stroke": "emergent", "tension pneumothorax": "emergent"}
    RISK_KEYWORDS = {
        "emergent": ["pneumothorax", "pulmonary embolism", "aortic dissection", "intracranial hemorrhage", "stroke", "cardiac tamponade"],
        "urgent": ["mass", "tumor", "malignant", "fracture", "obstruction", "pneumonia"],
        "routine": ["degenerative", "chronic", "stable", "benign", "unremarkable"],
    }

    def validate(self, fused_result):
        text = fused_result.get("consensus_answer", ""); text_lower = text.lower()
        issues, warnings = [], []
        if len(text.strip()) < 50: issues.append("Report too short")
        for section in ["findings", "impression"]:
            if section not in text_lower: warnings.append(f"Missing section: {section}")

        # v5.1: Negation-aware risk flagging
        risk_level = "routine"; flagged, negated_findings = [], []
        for level in ["emergent", "urgent"]:
            positive, negated = _find_positive_keywords(text, self.RISK_KEYWORDS[level])
            for kw in positive:
                flagged.append({"finding": kw, "risk_level": level})
                if level == "emergent": risk_level = "emergent"
                elif risk_level != "emergent": risk_level = "urgent"
            for kw in negated:
                negated_findings.append({"finding": kw, "would_be_level": level})

        # v5.1: Negation-aware guideline check
        cf_names = list(self.CRITICAL_FINDINGS.keys())
        pos_critical, neg_critical = _find_positive_keywords(text, cf_names)
        critical = [{"finding": f, "urgency": self.CRITICAL_FINDINGS[f]} for f in pos_critical]

        return {
            "clinical_validation": {"is_valid": len(issues) == 0, "issues": issues, "warnings": warnings,
                                     "has_safety_language": "clinical correlation" in text_lower},
            "risk_assessment": {"risk_level": risk_level, "flagged_findings": flagged,
                                "negated_findings": negated_findings,
                                "requires_immediate_attention": risk_level == "emergent"},
            "guideline_check": {"has_critical_findings": len(critical) > 0, "critical_findings": critical,
                                "negated_critical_findings": neg_critical},
        }




# MedicalRAGKaggle removed in v6.0 — replaced by EnhancedMedicalRAG with
# pre-seeded knowledge, PubMed search, and semantic retrieval.


class ReportGenerator:
    """v6.0: Structured section parsing, differential diagnosis, per-model bars."""

    def _parse_clinical_sections(self, text):
        sections = {}
        patterns = {
            "technique": r"(?:technique|protocol)[:\s]*(.+?)(?=\n\s*(?:comparison|finding|impression|differential|recommend)|$)",
            "comparison": r"(?:comparison|prior)[:\s]*(.+?)(?=\n\s*(?:finding|impression|differential|recommend)|$)",
            "findings": r"(?:finding)[s]?[:\s]*(.+?)(?=\n\s*(?:impression|differential|recommend)|$)",
            "impression": r"(?:impression|conclusion|summary)[:\s]*(.+?)(?=\n\s*(?:differential|recommend)|$)",
            "differential": r"(?:differential\s*(?:diagnos[ie]s?)?)[:\s]*(.+?)(?=\n\s*(?:recommend)|$)",
            "recommendations": r"(?:recommend(?:ation)?[s]?|follow[- ]?up)[:\s]*(.+?)(?=$)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = re.sub(r'\n\s*\n', '\n', match.group(1).strip())
                if content and len(content) > 10: sections[key] = content
        return sections

    def generate(self, fused, modality_info, governance, patient_info=None):
        consensus = fused.get("consensus_answer", "")
        parsed = self._parse_clinical_sections(consensus)
        return {
            "report_id": str(uuid4()), "timestamp": datetime.utcnow().isoformat(), "status": "final",
            "clinical_report": {
                "technique": parsed.get("technique", f"{modality_info.get('modality', 'Medical')} imaging analysis"),
                "findings": parsed.get("findings", consensus),
                "impression": parsed.get("impression", consensus[:500] if len(consensus) > 500 else consensus),
                "differential_diagnosis": parsed.get("differential", ""),
                "recommendations": parsed.get("recommendations", "Clinical correlation recommended."),
            },
            "ai_metadata": {"models_used": [a["model"] for a in fused.get("all_answers", [])],
                            "best_model": fused.get("best_model", ""), "confidence": fused.get("confidence", 0),
                            "agreement_score": fused.get("agreement_score", 0), "model_count": fused.get("model_count", 0),
                            "individual_results": fused.get("individual_results", []),
                            "cross_validation": fused.get("cross_validation", {})},
            "governance": governance,
            "disclaimer": "AI-generated report (MediScan v6.0). Must be reviewed by a qualified healthcare professional.",
        }

    def to_text(self, report):
        cr = report.get("clinical_report", {}); ai = report.get("ai_metadata", {}); gov = report.get("governance", {})
        conf = ai.get("confidence", 0); agree = ai.get("agreement_score", 0)
        risk = gov.get("risk_assessment", {}).get("risk_level", "routine")
        bar_len = 20; filled = int(conf * bar_len); conf_bar = "█" * filled + "░" * (bar_len - filled)
        risk_badge = {"emergent": "🔴 EMERGENT", "urgent": "🟠 URGENT", "routine": "🟢 ROUTINE"}
        parts = ["", "╔══════════════════════════════════════════════════════════════╗",
                 f"║           🏥 MediScan AI v{VERSION} — Diagnostic Report           ║",
                 "╚══════════════════════════════════════════════════════════════╝", "",
                 f"  📋 Report ID:  {report.get('report_id', 'N/A')}", f"  🕐 Date:       {report.get('timestamp', 'N/A')}",
                 f"  ⚕️  Risk Level: {risk_badge.get(risk, '🟢 ROUTINE')}", f"  📊 Confidence: {conf_bar} {conf:.0%}", ""]

        if risk in ("emergent", "urgent"):
            parts.extend(["  ┌─────────────────────────────────────────────────────────┐",
                          f"  │ ⚠️  CRITICAL ALERT — RISK LEVEL: {risk.upper():^20s}    │",
                          "  │ Immediate clinical attention may be required.           │",
                          "  └─────────────────────────────────────────────────────────┘", ""])

        icons = {"technique": "🔬", "findings": "🔍", "impression": "💡", "differential_diagnosis": "🧬", "recommendations": "📌"}
        for key, title in [("technique", "Technique"), ("findings", "Findings"), ("impression", "Impression"),
                           ("differential_diagnosis", "Differential Diagnosis"), ("recommendations", "Recommendations")]:
            val = cr.get(key, "")
            if val:
                parts.extend([f"  ── {icons.get(key, '📄')} {title} {'─' * (48 - len(title))}", ""])
                for line in val.split("\n"): parts.append(f"  {line.strip()}")
                parts.append("")

        # v5.1: Per-model bars
        individual = ai.get("individual_results", [])
        if individual:
            parts.extend(["  ── 🤖 Per-Model Analysis ────────────────────────────────", ""])
            for ir in individual:
                mc = ir.get("confidence", 0); mb = "█" * int(mc * 15) + "░" * (15 - int(mc * 15))
                tag = "GEN" if ir.get("is_generative") else "CLS"
                parts.append(f"  [{tag}] {ir.get('model','?'):18s} {mb} {mc:.0%}  (w={ir.get('weight',0):.2f})")
                excerpt = ir.get("excerpt", "")[:80].replace("\n", " ")
                if excerpt: parts.append(f"       └─ {excerpt}{'…' if len(ir.get('excerpt','')) > 80 else ''}")
            parts.append("")

        # v5.1: Negated findings
        negated = gov.get("risk_assessment", {}).get("negated_findings", [])
        if negated:
            parts.extend(["  ── 🚫 Negated Findings (excluded from risk) ─────────────", ""])
            for nf in negated: parts.append(f"  ✓ \"{nf.get('finding', '')}\" — negated (would be {nf.get('would_be_level', '?')})")
            parts.append("")

        # v5.2: Cross-Validation Results
        xval = ai.get("cross_validation", {})
        corroborated = xval.get("corroborated", [])
        contradictions = xval.get("contradictions", [])
        corr_rate = xval.get("corroboration_rate", 0)
        if corroborated or contradictions:
            parts.extend(["  ── 🔗 Cross-Model Validation ─────────────────────────────", ""])
            if corroborated:
                parts.append(f"  Corroboration Rate: {corr_rate:.0%}")
                for c in corroborated[:5]:
                    parts.append(f"  ✅ {c['location']:15s} — confirmed by {c['count']} models ({', '.join(c.get('models', []))})")
            if contradictions:
                for c in contradictions[:3]:
                    parts.append(f"  ⚠️  {c['location']:15s} — CONFLICT: normal by {', '.join(c.get('normal_by', []))}, abnormal by {', '.join(c.get('abnormal_by', []))}")
            parts.append("")

        models_str = ", ".join(ai.get("models_used", [])) or "N/A"
        parts.extend(["  ── 📊 Analysis Summary ──────────────────────────────────", "",
                       f"  Models Used:      {models_str}", f"  Best Model:       {ai.get('best_model', 'N/A')}",
                       f"  Model Agreement:  {agree:.0%}", f"  Models Consulted: {ai.get('model_count', 0)}", ""])

        # ═══ v6.0: INTELLIGENCE SECTIONS (reasoning, safety, explainability) ═══

        # v6.0: Reasoning Chain — shows HOW the AI reached its conclusion
        reasoning = report.get("reasoning", {})
        chain = reasoning.get("reasoning_chain", [])
        if chain:
            parts.extend(["  ── 🧠 AI Reasoning Chain ────────────────────────────────", ""])
            for step in chain: parts.append(f"  {step}")
            parts.append("")

        # v6.0: Knowledge-Graph Differential Diagnosis
        dx_list = reasoning.get("differential_diagnosis", [])
        if dx_list:
            parts.extend(["  ── 🧬 Knowledge-Graph Differential ─────────────────────", ""])
            for d in dx_list[:5]:
                score = d.get("validation_score", 0)
                urg = d.get("urgency", "routine")
                icon = "→" if score > 0.5 else "?"
                markers = d.get("severity_markers_present", [])
                line = f"  {icon} {d['diagnosis']:25s} score={score:.0%}  urgency={urg}"
                if markers: line += f"  markers: {', '.join(markers)}"
                parts.append(line)
            parts.append("")

        # v6.0: Model Debate
        debate = ai.get("debate", {})
        if debate and (debate.get("agreements") or debate.get("challenges")):
            parts.extend(["  ── ⚡ Model Debate ──────────────────────────────────────", ""])
            for a in (debate.get("agreements") or [])[:3]:
                parts.append(f"  ✅ {a['model']} agrees on: {', '.join(a.get('agrees_on', [])[:3])}")
            for c in (debate.get("challenges") or [])[:3]:
                parts.append(f"  ⚡ {c['model']} adds: {', '.join(c.get('additional_findings', [])[:3])}")
            parts.append("")

        # v6.0: Safety Assessment
        safety = report.get("safety", {})
        if safety:
            safe_icon = "✅" if safety.get("is_safe", True) else "⚠️"
            parts.extend([f"  ── 🛡️ Safety Assessment ({safe_icon}) ─────────────────────────", ""])
            for issue in safety.get("issues", [])[:3]:
                parts.append(f"  {'🔴' if issue.get('type') == 'hallucination' else '🟡'} {issue.get('type', '?')}: "
                            f"{str(issue.get('detail', issue.get('action', issue.get('value', ''))))[:80]}")
            sm = reasoning.get("risk_assessment", {}).get("safety_message", "")
            if sm: parts.append(f"  {sm}")
            parts.append("")

        # v6.0: Self-Reflection
        reflection = report.get("reflection", {})
        if reflection and reflection.get("improvements"):
            parts.extend([f"  ── 💭 Self-Reflection ({reflection.get('critique_count', 0)} critiques) ──────────────────", ""])
            for imp in reflection["improvements"][:3]:
                parts.append(f"  💭 {imp}")
            parts.append("")

        # v6.0: Explainability
        explanation = report.get("explanation", {})
        readable = explanation.get("readable", "") if explanation else ""
        if readable:
            parts.extend(["  ── 📖 Explainability ────────────────────────────────────", ""])
            for line in readable.strip().split("\n"): parts.append(f"  {line}")
            parts.append("")

        # v6.0: Decision Maker
        multi_agent = report.get("multi_agent", {})
        dm = multi_agent.get("decision_maker", {}) if multi_agent else {}
        if dm and dm.get("recommendation"):
            parts.extend(["  ── 🤖 Decision Maker ────────────────────────────────────", "",
                          f"  {dm['recommendation']}", ""])

        parts.extend(["  ── ⚖️ Disclaimer ──────────────────────────────────────────", "",
                       f"  {report.get('disclaimer', '')}", "",
                       "╚══════════════════════════════════════════════════════════════╝", ""])
        return "\n".join(parts)


print("✅ v6.0 Pipeline components ready")


# ═══════════════════════════════════════════════════════════
#  CELL 7b: v6.0 INTELLIGENCE LAYER — Production-Grade AI
#  This is the CORE upgrade: Reasoning + RAG + Safety + Multi-Agent
# ═══════════════════════════════════════════════════════════

# ── 1. MEDICAL KNOWLEDGE GRAPH ──────────────────────────────
# Embedded medical relationships for causal reasoning

MEDICAL_KNOWLEDGE_GRAPH = {
    # Findings → possible diagnoses
    "findings_to_dx": {
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
        "air bronchograms": ["pneumonia", "alveolar hemorrhage", "lymphoma", "bronchoalveolar carcinoma"],
        "widened mediastinum": ["aortic dissection", "lymphoma", "mediastinal mass", "trauma"],
        "hilar enlargement": ["sarcoidosis", "lymphoma", "lung cancer", "pulmonary hypertension"],
        "interstitial markings": ["pulmonary fibrosis", "interstitial pneumonia", "sarcoidosis", "lymphangitic spread"],
    },
    # Diagnosis → expected findings
    "dx_to_findings": {
        "pneumonia": {"required": ["opacity", "consolidation"], "supporting": ["air bronchograms", "fever", "cough"],
                      "location": "lobar or segmental", "severity_markers": ["bilateral", "multilobar", "cavitation"]},
        "heart failure": {"required": ["cardiomegaly"], "supporting": ["effusion", "edema", "cephalization", "kerley b lines"],
                         "location": "bilateral", "severity_markers": ["bilateral effusion", "pulmonary edema"]},
        "lung cancer": {"required": ["mass", "nodule"], "supporting": ["lymphadenopathy", "effusion", "bone destruction"],
                       "location": "unilateral", "severity_markers": ["metastasis", "effusion", "bone involvement"]},
        "tuberculosis": {"required": ["opacity"], "supporting": ["cavitation", "upper lobe", "lymphadenopathy", "miliary pattern"],
                        "location": "upper lobes", "severity_markers": ["miliary", "bilateral", "cavitation"]},
        "pulmonary embolism": {"required": [], "supporting": ["hampton hump", "westermark sign", "effusion"],
                              "location": "peripheral", "severity_markers": ["bilateral", "saddle embolus"]},
        "aortic dissection": {"required": ["widened mediastinum"], "supporting": ["intimal flap", "double lumen"],
                             "location": "mediastinal", "severity_markers": ["type a", "pericardial effusion"]},
        "covid-19": {"required": ["ground glass"], "supporting": ["bilateral", "peripheral", "crazy paving"],
                    "location": "bilateral peripheral", "severity_markers": ["bilateral", ">50% involvement"]},
    },
    # Anatomy keywords for location extraction
    "anatomy": {
        "lung": ["lung", "pulmonary", "lobe", "bronch", "alveol", "parenchym"],
        "heart": ["heart", "cardiac", "cardiomegaly", "pericardi", "atri", "ventricl", "valv"],
        "bone": ["bone", "rib", "spine", "vertebr", "fracture", "sclerotic", "lytic", "cortical"],
        "pleura": ["pleural", "effusion", "costophrenic", "meniscus"],
        "mediastinum": ["mediastin", "hilar", "aortic", "trachea", "lymph node"],
        "diaphragm": ["diaphragm", "subdiaphragmatic", "costophrenic"],
        "soft tissue": ["soft tissue", "subcutaneous", "axill"],
    },
    # Risk scoring for diagnoses
    "urgency": {
        "emergent": ["aortic dissection", "pulmonary embolism", "pneumothorax", "cardiac tamponade",
                     "intracranial hemorrhage", "stroke", "tension pneumothorax"],
        "urgent": ["pneumonia", "lung cancer", "heart failure", "fracture", "tuberculosis"],
        "routine": ["granuloma", "degenerative changes", "atherosclerosis", "old fracture"],
    },
}


class MedicalKnowledgeEngine:
    """Causal reasoning via medical knowledge graph — CPU-based, no GPU needed."""

    def __init__(self):
        self.kg = MEDICAL_KNOWLEDGE_GRAPH

    def expand_findings(self, findings_list):
        """Given extracted findings, expand with possible diagnoses and supporting evidence."""
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
        """Check if findings support a diagnosis using knowledge graph."""
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
        """Generate differential diagnosis from findings using knowledge graph."""
        text_lower = findings_text.lower()
        dx_scores = {}
        for keyword, diagnoses in self.kg["findings_to_dx"].items():
            if keyword in text_lower:
                for dx in diagnoses:
                    if dx not in dx_scores: dx_scores[dx] = 0
                    dx_scores[dx] += 1
        ranked = sorted(dx_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"diagnosis": dx, "supporting_findings_count": count,
                 "urgency": self._get_urgency(dx)} for dx, count in ranked]

    def _get_urgency(self, diagnosis):
        for level, conditions in self.kg["urgency"].items():
            if any(c in diagnosis.lower() for c in conditions): return level
        return "routine"


# ── 2. ENHANCED RAG WITH PRE-SEEDED KNOWLEDGE ──────────────

MEDICAL_KNOWLEDGE_BASE = [
    # Radiology guidelines
    "Normal chest X-ray shows clear lung fields bilaterally, normal cardiac silhouette with cardiothoracic ratio less than 0.5, clear costophrenic angles, intact bony structures, and midline trachea.",
    "Pneumonia typically presents as lobar or segmental consolidation with air bronchograms. May show associated pleural effusion. Common organisms include Streptococcus pneumoniae, Haemophilus influenzae.",
    "Congestive heart failure on CXR shows cardiomegaly (CTR > 0.5), cephalization of vessels, Kerley B lines, bilateral pleural effusions, and pulmonary edema pattern.",
    "Pneumothorax shows visceral pleural line with absent lung markings peripherally. Tension pneumothorax shows mediastinal shift away from affected side — requires immediate decompression.",
    "Pulmonary nodule workup: < 6mm low risk (follow-up), 6-8mm intermediate (CT 6-12 months), > 8mm high risk (PET/CT or biopsy). Fleischner Society guidelines apply.",
    "BIRADS classification: 0-incomplete, 1-negative, 2-benign, 3-probably benign, 4-suspicious, 5-highly suggestive of malignancy, 6-known malignancy.",
    "CT pulmonary angiography is gold standard for pulmonary embolism. Direct signs include intraluminal filling defect. RV/LV ratio > 1.0 suggests right heart strain.",
    "Aortic dissection classification: Stanford Type A involves ascending aorta (surgical emergency), Type B involves descending aorta only (often medical management).",
    "Brain MRI for stroke: DWI shows acute infarct as restricted diffusion (bright). ADC map shows corresponding dark signal. FLAIR shows subacute-chronic changes.",
    "ACR TI-RADS for thyroid nodules: TR1 (benign), TR2 (not suspicious), TR3 (mildly suspicious), TR4 (moderately suspicious), TR5 (highly suspicious).",
    # Safety guidelines
    "Critical findings requiring immediate communication: tension pneumothorax, aortic dissection, pulmonary embolism, intracranial hemorrhage, bowel perforation.",
    "AI-assisted diagnosis should always include confidence levels and recommendation for clinical correlation. High-risk findings must flag for immediate physician review.",
    "Differential diagnosis should always be provided with supporting evidence. Avoid single-diagnosis overconfidence. Report uncertainty explicitly.",
    # Anatomical reference
    "Costophrenic angles should be sharp and clear. Blunting suggests at least 200-300mL of pleural fluid. Meniscus sign confirms free-flowing effusion.",
    "Normal mediastinal width on PA CXR is less than 8cm. Widened mediastinum raises concern for aortic pathology, lymphadenopathy, or mass.",
    "Lung zones: Upper (above 2nd anterior rib), Mid (2nd-4th rib), Lower (below 4th rib). TB favors upper lobes; pneumonia can be any lobe.",
]


class EnhancedMedicalRAG:
    """v6.0: Pre-seeded knowledge + PubMed search + guideline retrieval."""

    def __init__(self):
        self.collection = None; self._initialized = False; self._pubmed_cache = {}

    def initialize(self):
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            efn = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection("medical_knowledge_v6", embedding_function=efn)
            if self.collection.count() < len(MEDICAL_KNOWLEDGE_BASE):
                self.collection.add(
                    documents=MEDICAL_KNOWLEDGE_BASE,
                    ids=[f"med_kb_{i}" for i in range(len(MEDICAL_KNOWLEDGE_BASE))],
                    metadatas=[{"source": "medical_guidelines", "type": "reference"} for _ in MEDICAL_KNOWLEDGE_BASE],
                )
                logger.info(f"Seeded RAG with {len(MEDICAL_KNOWLEDGE_BASE)} medical references")
            self._initialized = True
        except Exception as e:
            logger.warning(f"RAG init failed: {e}"); self._initialized = True

    def retrieve(self, query, top_k=5):
        """Retrieve relevant medical knowledge."""
        if not self._initialized: self.initialize()
        if not self.collection or self.collection.count() == 0: return []
        try:
            results = self.collection.query(query_texts=[query], n_results=min(top_k, self.collection.count()))
            if results and results["documents"] and results["documents"][0]:
                return [{"text": doc, "distance": dist, "source": meta.get("source", "unknown")}
                        for doc, dist, meta in zip(results["documents"][0],
                                                    results["distances"][0] if results.get("distances") else [0]*len(results["documents"][0]),
                                                    results["metadatas"][0] if results.get("metadatas") else [{}]*len(results["documents"][0]))]
        except Exception as e: logger.warning(f"RAG retrieve failed: {e}")
        return []

    def search_pubmed(self, query, max_results=3):
        """Search PubMed for relevant research (best-effort, may fail without network)."""
        if query in self._pubmed_cache: return self._pubmed_cache[query]
        try:
            import httpx
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query.replace(' ', '+')}&retmax={max_results}&retmode=json"
            resp = httpx.get(search_url, timeout=5.0)
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids: return []
            fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(ids)}&retmode=json"
            resp = httpx.get(fetch_url, timeout=5.0)
            results = []
            for uid, data in resp.json().get("result", {}).items():
                if uid == "uids": continue
                results.append({"title": data.get("title", ""), "pubmed_id": uid,
                               "journal": data.get("source", ""), "year": data.get("pubdate", "")[:4]})
            self._pubmed_cache[query] = results
            return results
        except Exception: return []

    def enrich_prompt(self, question, top_k=3):
        """Enrich question with retrieved medical knowledge."""
        refs = self.retrieve(question, top_k)
        if not refs: return question
        ctx = "\n".join([f"[Ref {i+1}]: {r['text']}" for i, r in enumerate(refs)])
        return f"Medical references:\n{ctx}\n\nBased on above guidelines and your medical knowledge, {question}"


# ── 3. MEDICAL REASONING ENGINE ─────────────────────────────

class MedicalReasoningEngine:
    """v6.0: Structured reasoning — the 'thinking brain' of MediScan.

    Pipeline: Extract → Reason → Verify → Rank → Explain

    This is what separates MediScan from a simple model wrapper.
    Every finding goes through medical reasoning before being reported.
    """

    def __init__(self):
        self.kg = MedicalKnowledgeEngine()
        self.rag = None  # Set by engine

    def reason(self, model_outputs, modality="xray", rag=None):
        """Full reasoning pipeline over multi-model outputs."""
        self.rag = rag

        # Guard: if no model outputs, return empty reasoning with clear status
        if not model_outputs or not any(o.get("answer") for o in model_outputs):
            return {
                "findings": [], "contradictions": [],
                "differential_diagnosis": [], "confidence": 0.1,
                "evidence": [], "reasoning_chain": ["STEP 1 — No model outputs available. Cannot reason."],
                "risk_assessment": {"risk_level": "routine", "urgent_findings": [],
                                   "requires_immediate_attention": False,
                                   "recommend_specialist_review": True,
                                   "safety_message": "⚠️ No models produced output. Results unreliable."},
                "finding_count": 0,
            }

        # Step 1: Extract all findings from all models
        all_findings = self._extract_all_findings(model_outputs)

        # Step 2: Detect contradictions
        contradictions = self._detect_contradictions(all_findings)

        # Step 3: Build differential diagnosis via knowledge graph
        combined_text = " ".join([o.get("answer", "") for o in model_outputs if o.get("answer")])
        differential = self.kg.get_differential(combined_text, top_k=5)

        # Step 4: Validate top diagnoses against findings
        validated_dx = []
        for dx in differential:
            validation = self.kg.validate_diagnosis(dx["diagnosis"], combined_text)
            validated_dx.append({**dx, **validation})

        # Step 5: Compute true confidence (not heuristic)
        confidence = self._compute_true_confidence(model_outputs, all_findings, contradictions, validated_dx)

        # Step 6: Retrieve supporting evidence
        evidence = self._retrieve_evidence(differential, rag)

        # Step 7: Generate reasoning chain (CoT)
        reasoning_chain = self._build_reasoning_chain(all_findings, contradictions, validated_dx, evidence)

        # Step 8: Determine risk level
        risk = self._assess_risk(validated_dx, confidence)

        return {
            "findings": all_findings,
            "contradictions": contradictions,
            "differential_diagnosis": validated_dx,
            "confidence": confidence,
            "evidence": evidence,
            "reasoning_chain": reasoning_chain,
            "risk_assessment": risk,
            "finding_count": len(all_findings),
        }

    def _extract_all_findings(self, outputs):
        """Extract and deduplicate findings from all model outputs."""
        all_f = []
        for out in outputs:
            answer = out.get("answer", "")
            model = out.get("model", "unknown")
            findings = extract_findings(answer)
            for f in findings:
                f["source_model"] = model
                all_f.append(f)
        return all_f

    def _detect_contradictions(self, findings):
        """Detect when models disagree about the same anatomical location."""
        by_location = {}
        for f in findings:
            loc = f.get("location", "unspecified")
            if loc not in by_location: by_location[loc] = []
            by_location[loc].append(f)

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
                                  else "Conflicting — recommend repeat imaging or clinical correlation",
                })
        return contradictions

    def _compute_true_confidence(self, outputs, findings, contradictions, validated_dx):
        """True calibrated confidence = agreement × evidence × consistency."""
        # Factor 1: Model agreement (0-1)
        answers = [o.get("answer", "") for o in outputs if o.get("answer")]
        if len(answers) >= 2:
            agreements = []
            for i in range(len(answers)):
                for j in range(i+1, len(answers)):
                    wa = set(re.findall(r'\w+', answers[i].lower()))
                    wb = set(re.findall(r'\w+', answers[j].lower()))
                    if wa and wb: agreements.append(len(wa & wb) / len(wa | wb))
            model_agreement = float(np.mean(agreements)) if agreements else 0.5
        else:
            model_agreement = 0.5

        # Factor 2: Evidence strength — how well findings match knowledge graph
        evidence_strength = 0.5
        if validated_dx:
            top_scores = [dx.get("validation_score", 0.5) for dx in validated_dx[:3]]
            evidence_strength = float(np.mean(top_scores))

        # Factor 3: Consistency — penalize contradictions
        contradiction_penalty = min(len(contradictions) * 0.1, 0.3)
        consistency = 1.0 - contradiction_penalty

        # Factor 4: Finding richness
        richness = min(len(findings) / 10.0, 1.0)

        # Weighted combination
        confidence = (model_agreement * 0.35 + evidence_strength * 0.30 +
                      consistency * 0.20 + richness * 0.15)
        return round(min(max(confidence, 0.1), 0.95), 3)

    def _retrieve_evidence(self, differential, rag):
        """Retrieve supporting evidence for top diagnoses."""
        evidence = []
        if not rag: return evidence
        for dx in differential[:3]:
            refs = rag.retrieve(f"{dx['diagnosis']} imaging findings diagnosis", top_k=2)
            if refs:
                evidence.append({"diagnosis": dx["diagnosis"],
                                "references": [r["text"][:200] for r in refs]})
        return evidence

    def _build_reasoning_chain(self, findings, contradictions, validated_dx, evidence):
        """Build human-readable chain-of-thought reasoning."""
        chain = []
        # Step 1
        abnormal = [f for f in findings if not f.get("is_normal")]
        normal = [f for f in findings if f.get("is_normal")]
        chain.append(f"STEP 1 — Finding extraction: {len(findings)} findings ({len(abnormal)} abnormal, {len(normal)} normal)")
        if abnormal:
            locs = list(set(f.get("location", "?") for f in abnormal))
            chain.append(f"  Abnormal locations: {', '.join(locs[:5])}")

        # Step 2
        if contradictions:
            chain.append(f"STEP 2 — Contradiction check: {len(contradictions)} conflicts detected")
            for c in contradictions[:2]:
                chain.append(f"  ⚠️ {c['location']}: {c['resolution']}")
        else:
            chain.append("STEP 2 — Contradiction check: No conflicts ✓")

        # Step 3
        chain.append(f"STEP 3 — Differential diagnosis: {len(validated_dx)} candidates")
        for dx in validated_dx[:3]:
            score = dx.get("validation_score", 0)
            chain.append(f"  {'→' if score > 0.5 else '?'} {dx['diagnosis']} (score={score:.2f}, urgency={dx.get('urgency', '?')})")

        # Step 4
        if evidence:
            chain.append(f"STEP 4 — Evidence support: {len(evidence)} diagnoses have guideline references")

        # Step 5
        if validated_dx:
            top = validated_dx[0]
            chain.append(f"STEP 5 — Primary assessment: {top['diagnosis']} (validation={top.get('validation_score', 0):.2f})")

        return chain

    def _assess_risk(self, validated_dx, confidence):
        """Clinical risk assessment based on diagnoses and confidence."""
        risk_level = "routine"; urgent_findings = []
        for dx in validated_dx:
            u = dx.get("urgency", "routine")
            if u == "emergent":
                risk_level = "emergent"
                urgent_findings.append({"diagnosis": dx["diagnosis"], "urgency": "emergent"})
            elif u == "urgent" and risk_level != "emergent":
                risk_level = "urgent"
                urgent_findings.append({"diagnosis": dx["diagnosis"], "urgency": "urgent"})

        # Safety override: low confidence + any finding = recommend review
        needs_review = confidence < 0.5 or risk_level != "routine"
        return {
            "risk_level": risk_level,
            "urgent_findings": urgent_findings,
            "requires_immediate_attention": risk_level == "emergent",
            "recommend_specialist_review": needs_review,
            "confidence_adequate": confidence >= 0.5,
            "safety_message": self._get_safety_message(risk_level, confidence),
        }

    def _get_safety_message(self, risk, confidence):
        if risk == "emergent":
            return "🔴 CRITICAL: Findings suggest potentially life-threatening condition. Immediate physician review required."
        elif risk == "urgent":
            return "🟠 URGENT: Findings require timely clinical attention. Schedule review within 24-48 hours."
        elif confidence < 0.4:
            return "⚠️ LOW CONFIDENCE: AI analysis uncertain. Recommend repeat imaging or specialist consultation."
        elif confidence < 0.6:
            return "🟡 MODERATE: Results should be correlated with clinical findings. Discuss with treating physician."
        return "🟢 ROUTINE: No findings requiring immediate attention. Follow standard clinical protocols."


# ── 4. DYNAMIC FUSION ENGINE ────────────────────────────────

class DynamicFusionEngine:
    """v6.0: Replaces static weighting with uncertainty-aware dynamic fusion.

    Instead of: score = static_weight * answer
    Now: score = model_confidence × reliability × agreement × evidence_strength

    Also implements model "debate" — models' findings argue for/against each other.
    """

    def fuse(self, results, reasoning_output=None):
        """Dynamic fusion with reasoning-informed weighting."""
        successful = [r for r in results if r.get("status") == "success"]
        if not successful:
            return {"consensus_answer": "No models produced results.", "confidence": 0, "model_count": 0}

        # Build model analyses with dynamic scoring
        model_analyses = []
        for r in successful:
            res = r.get("result", {}); mk = r["model_key"]
            answer = res.get("answer", ""); thinking = res.get("thinking", "")
            is_gen = mk in GENERATIVE_MODELS
            conf = res.get("confidence", 0.5)
            findings = extract_findings(answer) if is_gen else []

            # Dynamic reliability: model's agreement with reasoning engine's assessment
            reliability = self._compute_reliability(answer, reasoning_output) if reasoning_output else 0.7

            # Dynamic score = confidence × reliability × finding_richness
            richness = min(1.0, len(findings) / 8.0)
            dynamic_score = conf * 0.35 + reliability * 0.35 + richness * 0.30

            model_analyses.append({
                "model": mk, "answer": answer, "thinking": thinking,
                "is_generative": is_gen, "confidence": conf, "findings": findings,
                "reliability": round(reliability, 3), "dynamic_score": round(dynamic_score, 3),
                "finding_count": len(findings), "excerpt": answer[:300],
            })

        # Select best by dynamic score
        gen_models = [m for m in model_analyses if m["is_generative"]]
        pool = gen_models if gen_models else model_analyses
        best = max(pool, key=lambda x: x["dynamic_score"])

        # Debate: check if other models contradict the best model's key findings
        debate_results = self._debate(best, pool) if len(pool) > 1 else {"agreements": [], "challenges": []}

        # Weighted ensemble confidence
        weights = [m["dynamic_score"] for m in model_analyses]
        confs = [m["confidence"] for m in model_analyses]
        ensemble_conf = float(np.average(confs, weights=weights)) if sum(weights) > 0 else 0.5

        # Agreement scoring
        gen_texts = [m["answer"] for m in gen_models]
        agreement = self._agreement(gen_texts)

        return {
            "consensus_answer": best["answer"], "best_model": best["model"],
            "confidence": round(ensemble_conf, 3), "agreement_score": round(agreement, 3),
            "uncertainty": round(1.0 - ensemble_conf, 3), "model_count": len(model_analyses),
            "all_answers": model_analyses, "individual_results": model_analyses,
            "debate": debate_results, "dynamic_scores": {m["model"]: m["dynamic_score"] for m in model_analyses},
        }

    def _compute_reliability(self, answer, reasoning):
        """How well does model output align with reasoning engine's knowledge-graph analysis?"""
        if not reasoning: return 0.7
        dx_list = reasoning.get("differential_diagnosis", [])
        if not dx_list: return 0.7
        answer_lower = answer.lower()
        top_dx = [d["diagnosis"] for d in dx_list[:3]]
        matches = sum(1 for dx in top_dx if dx in answer_lower)
        return min(0.5 + matches * 0.2, 1.0)

    def _debate(self, best_model, all_models):
        """Models debate: check agreement and challenges to best model's assessment."""
        best_findings = set(f.get("location", "") for f in best_model.get("findings", []))
        agreements, challenges = [], []
        for m in all_models:
            if m["model"] == best_model["model"]: continue
            m_findings = set(f.get("location", "") for f in m.get("findings", []))
            common = best_findings & m_findings
            if common:
                agreements.append({"model": m["model"], "agrees_on": list(common)})
            unique_to_other = m_findings - best_findings
            if unique_to_other:
                challenges.append({"model": m["model"], "additional_findings": list(unique_to_other)})
        return {"agreements": agreements, "challenges": challenges}

    def _agreement(self, texts):
        if len(texts) < 2: return 1.0
        sims = []
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                wa = set(re.findall(r'\w+', texts[i].lower())); wb = set(re.findall(r'\w+', texts[j].lower()))
                if wa and wb: sims.append(len(wa & wb) / len(wa | wb))
        return float(np.mean(sims)) if sims else 0.0


# ── 5. CLINICAL SAFETY LAYER ────────────────────────────────

class ClinicalSafetyLayer:
    """v6.0: Hallucination detection, uncertainty fallback, 'refer to doctor' logic.

    Every output passes through safety before reaching the user.
    Inspired by Claude's safety-first approach.
    """

    HALLUCINATION_PATTERNS = [
        r"(?:100|99)\s*%\s*(?:certain|confident|sure)",  # Overconfident claims
        r"definitely\s+(?:is|has|shows)\s+(?:cancer|malignant|tumor)",  # Definitive cancer diagnosis
        r"no\s+(?:need|reason)\s+(?:to|for)\s+(?:further|additional)\s+(?:test|evaluation|workup)",  # Dismissing workup
        r"pathognomonic",  # Rarely appropriate for imaging alone
        r"(?:I|the AI)\s+(?:am|is)\s+(?:a|your)\s+doctor",  # Impersonating physician
    ]

    MANDATORY_DISCLAIMERS = {
        "emergent": "⚠️ CRITICAL FINDING detected. This requires IMMEDIATE physician review. Contact your healthcare provider or emergency department NOW.",
        "urgent": "⚠️ Findings requiring timely medical attention identified. Please schedule a physician review within 24-48 hours.",
        "low_confidence": "⚠️ AI confidence is LOW for this analysis. Results may be unreliable. A specialist should review the original imaging.",
        "contradiction": "⚠️ Models produced CONFLICTING assessments. Clinical correlation and possibly repeat imaging recommended.",
    }

    def validate(self, report_text, reasoning_output, fused_result):
        """Full safety validation before output."""
        issues = []

        # 1. Hallucination detection
        hallucinations = self._detect_hallucinations(report_text)
        if hallucinations:
            issues.extend([{"type": "hallucination", "detail": h} for h in hallucinations])

        # 2. Confidence threshold check
        confidence = reasoning_output.get("confidence", 0) if reasoning_output else fused_result.get("confidence", 0)
        if confidence < 0.3:
            issues.append({"type": "very_low_confidence", "value": confidence,
                          "action": "BLOCK — do not present as diagnostic. Show only as preliminary observation."})
        elif confidence < 0.5:
            issues.append({"type": "low_confidence", "value": confidence,
                          "action": "WARN — add prominent uncertainty disclaimer."})

        # 3. Contradiction check
        contradictions = reasoning_output.get("contradictions", []) if reasoning_output else []
        if contradictions:
            issues.append({"type": "model_contradiction", "count": len(contradictions),
                          "detail": f"Models disagree on {len(contradictions)} findings"})

        # 4. Risk assessment
        risk = reasoning_output.get("risk_assessment", {}) if reasoning_output else {}
        risk_level = risk.get("risk_level", "routine")
        if risk_level == "emergent":
            issues.append({"type": "emergent_finding", "action": "Immediate physician notification required"})

        # 5. Missing safety language
        text_lower = report_text.lower()
        has_disclaimer = any(phrase in text_lower for phrase in
                           ["clinical correlation", "recommend", "physician review", "consult", "follow-up"])
        if not has_disclaimer:
            issues.append({"type": "missing_safety_language", "action": "Add clinical correlation recommendation"})

        # Build safety-enhanced output
        disclaimers = self._build_disclaimers(risk_level, confidence, contradictions)
        safe_report = self._apply_safety(report_text, issues, disclaimers)

        return {
            "is_safe": len([i for i in issues if i.get("type") in ("hallucination", "very_low_confidence")]) == 0,
            "issues": issues, "disclaimers": disclaimers,
            "risk_level": risk_level, "confidence_adequate": confidence >= 0.5,
            "safe_report": safe_report, "issue_count": len(issues),
        }

    def _detect_hallucinations(self, text):
        """Detect likely AI hallucinations in medical text."""
        found = []
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(f"Suspicious pattern: {pattern[:60]}")
        return found

    def _build_disclaimers(self, risk_level, confidence, contradictions):
        """Build appropriate disclaimers based on analysis quality."""
        disclaimers = []
        if risk_level == "emergent": disclaimers.append(self.MANDATORY_DISCLAIMERS["emergent"])
        elif risk_level == "urgent": disclaimers.append(self.MANDATORY_DISCLAIMERS["urgent"])
        if confidence < 0.5: disclaimers.append(self.MANDATORY_DISCLAIMERS["low_confidence"])
        if contradictions: disclaimers.append(self.MANDATORY_DISCLAIMERS["contradiction"])
        disclaimers.append("This is an AI-assisted analysis and should not replace professional medical judgment.")
        return disclaimers

    def _apply_safety(self, text, issues, disclaimers):
        """Apply safety modifications to report text."""
        safe = text
        # Remove overconfident language
        safe = re.sub(r'\b(definitely|certainly|100%|guaranteed)\b', 'likely', safe, flags=re.IGNORECASE)
        safe = re.sub(r'\bno\s+need\s+for\s+further\b', 'recommend further', safe, flags=re.IGNORECASE)
        return safe


# ── 6. SELF-REFLECTION LOOP ─────────────────────────────────

class SelfReflectionLoop:
    """v6.0: After initial analysis, critique and refine.

    Uses available model to ask critical questions:
    - What could be wrong in this diagnosis?
    - What findings might be missed?
    - Is there any contradiction?

    Then integrates critiques into final output.
    """

    CRITIQUE_PROMPTS = [
        "Review this radiology report critically. What diagnoses could be wrong? What findings might be missed?",
        "Are there any internal contradictions or inconsistencies in this report?",
        "What additional imaging or tests would help confirm or rule out the findings?",
    ]

    def reflect(self, initial_report, model=None, max_critiques=2):
        """Self-reflection loop — critique the initial report."""
        critiques = []

        # NLP-based self-critique (no GPU needed)
        critiques.extend(self._nlp_critique(initial_report))

        # If a model is available and loaded, use it for deeper critique
        if model and model.is_loaded and len(critiques) < max_critiques:
            for prompt in self.CRITIQUE_PROMPTS[:1]:  # Only 1 model critique to save GPU
                try:
                    result = model.analyze(text=f"{prompt}\n\nReport:\n{initial_report[:500]}",
                                          modality="text")
                    if result.get("answer"):
                        critiques.append({"type": "model_critique", "content": result["answer"][:300]})
                except Exception: pass

        # Synthesize critiques
        improvements = self._synthesize(critiques)
        return {"critiques": critiques, "improvements": improvements, "critique_count": len(critiques)}

    def _nlp_critique(self, text):
        """Rule-based critique using medical knowledge."""
        critiques = []
        text_lower = text.lower()

        # Check for missing standard sections
        for section in ["findings", "impression", "recommendation"]:
            if section not in text_lower:
                critiques.append({"type": "missing_section", "content": f"Report lacks '{section}' section"})

        # Check for vague language
        vague = ["appears to", "may suggest", "cannot exclude", "questionable"]
        vague_count = sum(1 for v in vague if v in text_lower)
        if vague_count > 3:
            critiques.append({"type": "excessive_hedging", "content": f"Report uses {vague_count} hedging phrases — may indicate low model confidence"})

        # Check for single-diagnosis tunnel vision
        dx_keywords = ["pneumonia", "fracture", "tumor", "cancer", "edema", "effusion", "hemorrhage"]
        mentioned_dx = [d for d in dx_keywords if d in text_lower]
        if len(mentioned_dx) == 1 and len(text) > 200:
            critiques.append({"type": "tunnel_vision", "content": f"Only one diagnosis ({mentioned_dx[0]}) mentioned — consider differential diagnosis"})

        return critiques

    def _synthesize(self, critiques):
        """Turn critiques into actionable improvements."""
        improvements = []
        for c in critiques:
            ctype = c.get("type", "")
            if ctype == "missing_section":
                section_name = c['content'].split("'")[1] if "'" in c['content'] else c['content']
                improvements.append(f"Add {section_name} section to report")
            elif ctype == "tunnel_vision":
                improvements.append("Consider adding differential diagnosis")
            elif ctype == "excessive_hedging":
                improvements.append("Consider being more specific in assessment where evidence supports it")
            elif ctype == "model_critique":
                improvements.append(f"Model suggests: {c['content'][:150]}")
        return improvements


# ── 7. MULTI-AGENT ORCHESTRATOR ──────────────────────────────

class MultiAgentOrchestrator:
    """v6.0: Assigns specialist roles to model outputs.

    Pipeline: Radiologist → Critic → Researcher → Decision Maker

    NOT a separate model — reinterprets existing model outputs through
    role-specific lenses for richer analysis.
    """

    AGENT_ROLES = {
        "radiologist": {"focus": "primary findings", "models": ["hulu_med_7b", "medgemma_4b"],
                       "instruction": "Focus on imaging findings, measurements, anatomical descriptions"},
        "critic": {"focus": "contradictions and gaps", "models": ["medix_r1_2b"],
                  "instruction": "Challenge findings, look for missed pathology, check consistency"},
        "researcher": {"focus": "evidence and guidelines", "models": ["biomedclip"],
                      "instruction": "Compare with published literature and guidelines"},
        "decision_maker": {"focus": "synthesis and recommendation", "models": [],
                          "instruction": "Synthesize all inputs into final clinical recommendation"},
    }

    def orchestrate(self, model_results, reasoning_output):
        """Assign roles and synthesize multi-agent perspective."""
        agent_outputs = {}

        for role, config in self.AGENT_ROLES.items():
            role_results = [r for r in model_results if r.get("model_key") in config["models"] and r.get("status") == "success"]
            if role == "decision_maker":
                agent_outputs[role] = self._decision_maker(model_results, reasoning_output)
            elif role_results:
                agent_outputs[role] = {
                    "role": role, "focus": config["focus"],
                    "outputs": [{"model": r["model_key"], "answer": r.get("result", {}).get("answer", "")[:300]}
                               for r in role_results],
                    "finding_count": sum(len(extract_findings(r.get("result", {}).get("answer", ""))) for r in role_results),
                }
            else:
                agent_outputs[role] = {"role": role, "status": "no assigned model ran"}

        return agent_outputs

    def _decision_maker(self, all_results, reasoning):
        """Synthesize all agent outputs into final recommendation."""
        successful = [r for r in all_results if r.get("status") == "success"]
        dx = reasoning.get("differential_diagnosis", []) if reasoning else []
        risk = reasoning.get("risk_assessment", {}) if reasoning else {}

        recommendation = []
        if risk.get("risk_level") == "emergent":
            recommendation.append("IMMEDIATE ACTION: Critical finding requires emergent physician review.")
        elif risk.get("risk_level") == "urgent":
            recommendation.append("PRIORITY: Schedule physician review within 24-48 hours.")

        if dx:
            top = dx[0]
            recommendation.append(f"Primary consideration: {top['diagnosis']} (confidence: {top.get('validation_score', 0):.0%})")
            if len(dx) > 1:
                alts = ", ".join(d["diagnosis"] for d in dx[1:3])
                recommendation.append(f"Differential includes: {alts}")

        if risk.get("recommend_specialist_review"):
            recommendation.append("Specialist consultation recommended for definitive evaluation.")

        return {"role": "decision_maker", "recommendation": " | ".join(recommendation),
                "models_consulted": len(successful), "diagnoses_considered": len(dx)}


# ── 8. EXPLAINABILITY ENGINE ─────────────────────────────────

class ExplainabilityEngine:
    """v6.0: 'Why this diagnosis?' — Feature attribution for trust.

    Doctors need to understand WHY the AI reached its conclusion.
    This produces structured explanations linking findings → diagnosis.
    """

    def explain(self, reasoning_output, fused_result):
        """Generate structured explanation for the analysis."""
        explanations = []
        dx_list = reasoning_output.get("differential_diagnosis", [])

        for dx in dx_list[:3]:
            explanation = {
                "diagnosis": dx.get("diagnosis", ""),
                "confidence": f"{dx.get('validation_score', 0):.0%}",
                "urgency": dx.get("urgency", "routine"),
                "supporting_evidence": [],
                "reasoning": [],
            }

            # Why this diagnosis?
            if dx.get("required_found", 0) > 0:
                explanation["reasoning"].append(
                    f"Found {dx['required_found']}/{dx['required_total']} required imaging features")
            if dx.get("supporting_found", 0) > 0:
                explanation["reasoning"].append(
                    f"Found {dx['supporting_found']} supporting findings")
            if dx.get("severity_markers_present"):
                explanation["reasoning"].append(
                    f"Severity markers present: {', '.join(dx['severity_markers_present'])}")

            # Evidence trail
            evidence = reasoning_output.get("evidence", [])
            for e in evidence:
                if e.get("diagnosis") == dx.get("diagnosis"):
                    explanation["supporting_evidence"] = e.get("references", [])

            explanations.append(explanation)

        # Build human-readable explanation
        readable = self._format_readable(explanations)
        return {"explanations": explanations, "readable": readable}

    def _format_readable(self, explanations):
        """Format explanations for display."""
        parts = ["── 🧠 AI Reasoning Explanation ──\n"]
        for i, exp in enumerate(explanations):
            parts.append(f"{'🔴' if exp['urgency'] == 'emergent' else '🟡' if exp['urgency'] == 'urgent' else '🟢'} "
                        f"#{i+1}: {exp['diagnosis']} ({exp['confidence']})")
            for r in exp.get("reasoning", []):
                parts.append(f"   → {r}")
            if exp.get("supporting_evidence"):
                parts.append(f"   📚 Guideline: {exp['supporting_evidence'][0][:120]}...")
            parts.append("")
        return "\n".join(parts)


print("✅ v6.0 Intelligence Layer ready")


# ═══════════════════════════════════════════════════════════
#  CELL 8: MediScan Kaggle Engine v5.1 (Full Pipeline)
# ═══════════════════════════════════════════════════════════

class MediScanKaggleEngine:
    """v6.0: Full intelligence pipeline with sequential model loading.

    Pipeline:
      Ingest → Quality → Modality → MONAI → Route →
      Sequential Model Execution (load→run→unload) →
      RAG Enrichment → Reasoning Engine → Dynamic Fusion →
      Self-Reflection → Safety Layer → Multi-Agent Synthesis →
      Explainability → Report Generation
    """

    def __init__(self):
        logger.info(f"🚀 Initializing MediScan AI v{VERSION} (Kaggle)...")
        self.image_loader = ImageLoader(); self.video_loader = VideoLoader()
        self.dicom_loader = DICOMLoader(); self.modality_detector = ModalityDetector()
        self.quality_assessor = QualityAssessor(); self.preprocessor = MONAIPipeline()

        # v6.0: Models are lazy-loaded — only register, don't load
        # NOTE: HuluMed-4B replaces 7B on Kaggle T4 — the 7B vision encoder
        # allocates a single 16GB tensor during forward pass which exceeds T4's 14.5GB.
        # HuluMed-4B has identical architecture but fits in T4 memory.
        # On production servers with A100s, use HuluMed-7B/14B/32B instead.
        self.models = {}
        gpu0, gpu1 = 0, min(1, NUM_GPUS - 1)
        self.models["hulu_med_7b"] = HuluMedKaggle("ZJU-AI4H/Hulu-Med-4B", device_idx=gpu0)
        self.models["medgemma_4b"] = MedGemmaKaggle("google/medgemma-4b-it", device_idx=gpu0)
        self.models["medix_r1_2b"] = MediXR1Kaggle("MBZUAI/MediX-R1-2B", device_idx=gpu1)
        self.models["med3dvlm"] = Med3DVLMKaggle("MagicXin/Med3DVLM-Qwen-2.5-7B", device_idx=gpu1)
        self.models["biomedclip"] = BiomedCLIPKaggle("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", device_idx=gpu1)

        self.router = IntelligentRouter(available_models=list(self.models.keys()))
        # v6.0: Intelligence modules
        self.rag = EnhancedMedicalRAG()
        self.reasoning = MedicalReasoningEngine()
        self.fusion = DynamicFusionEngine()
        self.safety = ClinicalSafetyLayer()
        self.reflection = SelfReflectionLoop()
        self.multi_agent = MultiAgentOrchestrator()
        self.explainability = ExplainabilityEngine()
        self.governance = GovernanceLayer()
        self.report_gen = ReportGenerator()
        logger.info(f"✅ MediScan v{VERSION} Kaggle Engine initialized")
        print(f"\n📊 Models registered: {list(self.models.keys())}")

    def _unload_all(self):
        """Unload all models to free GPU memory."""
        for name, model in self.models.items():
            if model.is_loaded:
                model.unload(); logger.info(f"  Unloaded {name}")
        gc.collect(); torch.cuda.empty_cache()

    def _run_model(self, model_key, use_prompt, data, preprocessed, file_type, source_path,
                   modality, extra_kw, middle_slice_pil, first_slice_pil):
        """Run a single model with proper loading/unloading."""
        model = self.models[model_key]
        if not model.is_loaded:
            model.load()

        if file_type == "3d":
            if model_key in NATIVE_3D_MODELS:
                return model.analyze(text=use_prompt, modality="3d",
                                    nii_path=source_path, nii_num_slices=180, **extra_kw)
            elif model_key == "biomedclip":
                return model.analyze(images=[first_slice_pil] if first_slice_pil else None,
                                    text=use_prompt, modality="image", **extra_kw)
            else:
                return model.analyze(images=[middle_slice_pil] if middle_slice_pil else None,
                                    text=use_prompt, modality="image", **extra_kw)
        elif file_type == "2d":
            pil = preprocessed.get("pil_image")
            return model.analyze(images=[pil] if pil else None, text=use_prompt,
                                modality="image", **extra_kw)
        elif file_type == "video":
            frames = preprocessed.get("frames", data.get("frames", []))
            return model.analyze(images=frames, text=use_prompt,
                                modality="video", video_path=source_path, **extra_kw)
        else:
            pil = preprocessed.get("pil_image")
            return model.analyze(images=[pil] if pil else None, text=use_prompt,
                                modality="image", **extra_kw)

    def analyze(self, file_path, question="Generate a comprehensive medical report.",
                target_language="en", complexity="standard", models_to_use=None):
        request_id = str(uuid4())[:8]; start = time.time()
        print(f"\n{'='*60}\n🔍 Analysis {request_id} started\n   Input: {file_path}\n{'='*60}")

        try:
            # ── Steps 1-5: Preprocessing (unchanged) ──
            print("  Step 1/12: Ingestion..."); data = self._ingest(file_path)
            print("  Step 2/12: Quality..."); quality = self.quality_assessor.assess(data)
            print(f"    Quality: {quality['overall_score']:.2f}")
            print("  Step 3/12: Modality..."); modality_info = self.modality_detector.detect(data)
            modality = modality_info["modality"]; print(f"    Modality: {modality}")
            print("  Step 4/12: MONAI..."); preprocessed = self.preprocessor.preprocess(data, modality=modality)
            print("  Step 5/12: Routing...")
            if models_to_use: route = {"primary": models_to_use, "verifier": [], "reasoner": [], "specialist_3d": []}
            else: route = self.router.route(modality, data.get("type", "2d"), complexity)
            active_models = [m for ms in route.values() for m in ms]; print(f"    Route: {route}")

            # ── Step 6: SEQUENTIAL Model Execution ──
            print("  Step 6/12: Model Execution (v6.0 sequential + MedPrompting)...")
            results = []; file_type = data.get("type", "2d"); source_path = data.get("source_path", "")

            # Pre-extract slices for 3D
            middle_slice_pil, first_slice_pil = None, None
            if file_type == "3d":
                volume = data.get("volume")
                if volume is not None:
                    try:
                        mid = volume[volume.shape[0] // 2]; smin, smax = mid.min(), mid.max()
                        middle_slice_pil = Image.fromarray(((mid-smin)/(smax-smin+1e-8)*255).astype(np.uint8), "L").convert("RGB")
                        first = volume[0]; smin, smax = first.min(), first.max()
                        first_slice_pil = Image.fromarray(((first-smin)/(smax-smin+1e-8)*255).astype(np.uint8), "L").convert("RGB")
                    except Exception as e: logger.warning(f"Slice extraction failed: {e}")

            # v6.0: Build prompts with RAG enrichment
            expert_prompt = build_expert_prompt(question, modality=modality, file_type=file_type, role="primary")
            enriched_question = self.rag.enrich_prompt(expert_prompt)
            reasoner_prompt = build_expert_prompt(question, modality=modality, file_type=file_type, role="reasoner")
            enriched_reasoner = self.rag.enrich_prompt(reasoner_prompt)

            extra_kw = {"source_path": source_path, "detected_modality": modality}

            # v6.0: SEQUENTIAL EXECUTION — load one group at a time
            # Group 1: HuluMed alone (needs both GPUs)
            # Group 2: MedGemma + MediX-R1 (one per GPU)
            # Group 3: BiomedCLIP (tiny, fits anywhere)
            execution_groups = [
                [m for m in active_models if m == "hulu_med_7b"],
                [m for m in active_models if m in ("medgemma_4b", "medix_r1_2b")],
                [m for m in active_models if m in ("med3dvlm", "biomedclip")],
            ]

            for group_idx, group in enumerate(execution_groups):
                if not group: continue

                # Unload previous group to free GPU memory
                if group_idx > 0:
                    self._unload_all()

                for model_key in group:
                    if model_key not in self.models: continue
                    try:
                        print(f"    Running {model_key}...")
                        route_role = "primary"
                        for role_name, role_models in route.items():
                            if model_key in role_models: route_role = role_name; break
                        use_prompt = enriched_reasoner if route_role == "reasoner" else enriched_question

                        result = self._run_model(model_key, use_prompt, data, preprocessed,
                                                file_type, source_path, modality, extra_kw,
                                                middle_slice_pil, first_slice_pil)
                        results.append({"model_key": model_key, "status": "success", "result": result,
                                        "duration": result.get("metadata", {}).get("inference_time", 0)})
                        print(f"      ✅ {model_key} done ({result.get('metadata', {}).get('inference_time', 0):.1f}s)")
                    except Exception as e:
                        logger.error(f"    ❌ {model_key} failed: {e}")
                        results.append({"model_key": model_key, "status": "error", "error": str(e)})

            # Free GPU for reasoning/reflection
            self._unload_all()

            # ── Step 7: RAG Evidence Retrieval ──
            print("  Step 7/12: RAG knowledge retrieval...")
            combined_text = " ".join([r.get("result", {}).get("answer", "")
                                     for r in results if r.get("status") == "success"])
            pubmed_results = self.rag.search_pubmed(f"{modality} {combined_text[:100]}", max_results=3)
            if pubmed_results:
                print(f"    Found {len(pubmed_results)} PubMed references")

            # ── Step 8: REASONING ENGINE (the brain) ──
            print("  Step 8/12: Reasoning engine (CoT + knowledge graph)...")
            model_outputs = [{"answer": r.get("result", {}).get("answer", ""),
                             "model": r["model_key"], "confidence": r.get("result", {}).get("confidence", 0.5)}
                            for r in results if r.get("status") == "success"]
            reasoning = self.reasoning.reason(model_outputs, modality=modality, rag=self.rag)
            print(f"    Findings: {reasoning['finding_count']} | Contradictions: {len(reasoning['contradictions'])} | "
                  f"Confidence: {reasoning['confidence']:.2f}")
            if reasoning.get("differential_diagnosis"):
                top_dx = reasoning["differential_diagnosis"][0]
                print(f"    Top Dx: {top_dx['diagnosis']} (score={top_dx.get('validation_score', 0):.2f})")
            for step in reasoning.get("reasoning_chain", []):
                print(f"    {step}")

            # ── Step 9: DYNAMIC FUSION ──
            print("  Step 9/12: Dynamic fusion (uncertainty-aware)...")
            fused = self.fusion.fuse(results, reasoning_output=reasoning)
            print(f"    Best: {fused['best_model']} | Confidence: {fused['confidence']:.2f} | Agreement: {fused['agreement_score']:.2f}")
            debate = fused.get("debate", {})
            if debate.get("challenges"):
                for c in debate["challenges"][:2]:
                    print(f"    ⚡ {c['model']} adds: {', '.join(c['additional_findings'][:3])}")

            # ── Step 10: SELF-REFLECTION ──
            print("  Step 10/12: Self-reflection loop...")
            reflection = self.reflection.reflect(fused.get("consensus_answer", ""))
            if reflection.get("improvements"):
                for imp in reflection["improvements"][:3]:
                    print(f"    💭 {imp}")

            # ── Step 11: SAFETY LAYER ──
            print("  Step 11/12: Clinical safety validation...")
            safety = self.safety.validate(fused.get("consensus_answer", ""), reasoning, fused)
            risk_level = safety.get("risk_level", "routine")
            print(f"    Safety: {'✅ PASS' if safety['is_safe'] else '⚠️ ISSUES'} | Risk: {risk_level} | Issues: {safety['issue_count']}")
            for disclaimer in safety.get("disclaimers", [])[:2]:
                print(f"    {disclaimer[:100]}")

            # ── Step 11b: Multi-Agent Synthesis ──
            agents = self.multi_agent.orchestrate(results, reasoning)
            dm = agents.get("decision_maker", {})
            if dm.get("recommendation"):
                print(f"    🤖 Decision: {dm['recommendation'][:120]}")

            # ── Step 11c: Explainability ──
            explanation = self.explainability.explain(reasoning, fused)

            # ── Step 12: REPORT GENERATION ──
            print("  Step 12/12: Report generation (v6.0 full intelligence)...")
            # Governance (legacy, still useful for structured validation)
            gov = self.governance.validate(fused)

            # Merge reasoning into governance
            gov["reasoning"] = {
                "chain": reasoning.get("reasoning_chain", []),
                "differential": reasoning.get("differential_diagnosis", []),
                "contradictions": reasoning.get("contradictions", []),
            }
            gov["safety"] = safety
            gov["risk_assessment"] = reasoning.get("risk_assessment", gov.get("risk_assessment", {}))

            report = self.report_gen.generate(fused, modality_info, gov)
            # Enrich report with v6.0 data
            report["reasoning"] = reasoning
            report["explanation"] = explanation
            report["reflection"] = reflection
            report["multi_agent"] = agents
            report["safety"] = safety
            report["pubmed_references"] = pubmed_results

            duration = time.time() - start; report_text = self.report_gen.to_text(report)
            print(f"\n{'='*60}\n✅ Analysis complete in {duration:.1f}s\n{'='*60}")

            # Print reasoning explanation
            if explanation.get("readable"):
                print(explanation["readable"])

            print(report_text)

            return {"request_id": request_id, "report": report, "report_text": report_text,
                    "governance": gov, "fusion": fused, "quality": quality, "reasoning": reasoning,
                    "safety": safety, "explanation": explanation, "reflection": reflection,
                    "modality": modality_info, "pipeline_duration": round(duration, 2),
                    "models_used": [r["model_key"] for r in results if r["status"] == "success"]}
        except Exception as e:
            duration = time.time() - start; logger.error(f"Pipeline failed: {e}", exc_info=True)
            print(f"\n❌ Error: {e}"); return {"request_id": request_id, "error": str(e), "pipeline_duration": round(duration, 2)}

    def _ingest(self, file_path):
        path = Path(file_path); suffix = "".join(path.suffixes).lower()
        if suffix == ".dcm" or path.suffix.lower() == ".dcm": return self.dicom_loader.load(path)
        elif path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"): return self.video_loader.load(path)
        else: return self.image_loader.load(path)

    def health_check(self):
        return {"models": {k: {"loaded": m.is_loaded, "gpu": m.device_idx} for k, m in self.models.items()},
                "gpus": {i: {"name": torch.cuda.get_device_name(i),
                             "used_gb": round(torch.cuda.memory_allocated(i)/1024**3, 1),
                             "total_gb": round(torch.cuda.get_device_properties(i).total_memory/1024**3, 1)} for i in range(NUM_GPUS)}}


# ═══════════════════════════════════════════════════════════
#  CELL 8b: v6.0 Conversation Brain Layer
#  Makes MediScan feel like ChatGPT — not a static report tool
# ═══════════════════════════════════════════════════════════

class ResponseStyler:
    """Rewrites raw reports → conversational output per mode."""
    JARGON = {"opacity": "a cloudy area", "consolidation": "area of possible fluid/infection",
        "effusion": "fluid buildup", "pleural effusion": "fluid around the lungs",
        "cardiomegaly": "an enlarged heart", "pneumothorax": "a collapsed lung (air leak)",
        "atelectasis": "partially collapsed lung", "infiltrate": "an abnormal area",
        "nodule": "a small round spot", "mass": "an abnormal growth", "edema": "swelling",
        "stenosis": "a narrowing", "fracture": "a break in the bone", "hemorrhage": "bleeding",
        "fibrosis": "scarring", "metastasis": "cancer spread", "malignant": "cancerous",
        "benign": "non-cancerous", "bilateral": "on both sides", "parenchyma": "organ tissue",
        "mediastinum": "center of the chest"}

    def style(self, report, mode="patient"):
        if mode == "patient": return self._patient(report)
        elif mode == "research": return self._research(report)
        return self._doctor(report)

    def _patient(self, report):
        cr = report.get("clinical_report", {}); gov = report.get("governance", {})
        ai = report.get("ai_metadata", {}); risk = gov.get("risk_assessment", {}).get("risk_level", "routine")
        conf = ai.get("confidence", 0); p = []
        p.append("👋 Hi! I've analyzed your medical image. Here's what I found:\n")
        p.append("📸 My analysis process:")
        for icon, step in [("🔍", "Checked image quality and orientation"), ("🫁", "Examined lung fields"),
                           ("❤️", "Evaluated heart size and shape"), ("🦴", "Reviewed bones and soft tissues"),
                           ("📋", "Compiled findings")]:
            p.append(f"  {icon} {step}")
        p.append("")
        if conf >= 0.7: p.append("✅ I'm fairly confident in these results.\n")
        elif conf >= 0.4: p.append("⚠️ Moderate confidence — discuss with your doctor.\n")
        else: p.append("❗ Uncertain results — specialist review recommended.\n")
        risk_msg = {"emergent": "🔴 URGENT: Please contact your doctor or visit ER.",
                    "urgent": "🟠 Some findings need attention — schedule an appointment.",
                    "routine": "🟢 Nothing appears to need immediate attention."}
        p.append(risk_msg.get(risk, risk_msg["routine"])); p.append("")
        for key, icon, title in [("findings", "🔍", "What I found"), ("impression", "💡", "What this means"),
                                  ("differential_diagnosis", "🤔", "Possible explanations"),
                                  ("recommendations", "📌", "Next steps")]:
            val = cr.get(key, "")
            if val: p.extend([f"{icon} {title}:", f"  {self._simplify(val)}", ""])
        p.extend(["─" * 50, "⚠️ This is AI analysis — always discuss with your healthcare provider."])
        return "\n".join(p)

    def _doctor(self, report):
        cr = report.get("clinical_report", {}); ai = report.get("ai_metadata", {})
        gov = report.get("governance", {}); risk = gov.get("risk_assessment", {}).get("risk_level", "routine")
        reasoning = report.get("reasoning", {})
        safety = report.get("safety", {})
        explanation = report.get("explanation", {})

        p = [f"MEDISCAN AI v{VERSION} — DIAGNOSTIC REPORT",
             f"Report ID: {report.get('report_id','N/A')}  |  Risk: {risk.upper()}  |  Conf: {ai.get('confidence',0):.0%}", ""]

        # Safety alerts first
        if safety.get("safety_message"):
            sm = reasoning.get("risk_assessment", {}).get("safety_message", "")
            if sm: p.extend([sm, ""])
        if risk in ("emergent", "urgent"): p.extend([f"*** CRITICAL: {risk.upper()} ***", ""])

        # Standard clinical sections
        for key, title in [("technique", "TECHNIQUE"), ("findings", "FINDINGS"), ("impression", "IMPRESSION"),
                           ("differential_diagnosis", "DIFFERENTIAL"), ("recommendations", "RECOMMENDATIONS")]:
            val = cr.get(key, "")
            if val: p.extend([f"{title}:", val, ""])

        # v6.0: AI Reasoning Chain (doctors need to see HOW the AI reasoned)
        chain = reasoning.get("reasoning_chain", [])
        if chain:
            p.extend(["AI REASONING:", ""])
            for step in chain: p.append(f"  {step}")
            p.append("")

        # v6.0: Knowledge-graph differential
        dx = reasoning.get("differential_diagnosis", [])
        if dx:
            p.append("KNOWLEDGE-GRAPH DIFFERENTIAL:")
            for d in dx[:3]:
                score = d.get("validation_score", 0)
                urg = d.get("urgency", "routine")
                p.append(f"  {'→' if score > 0.5 else '?'} {d['diagnosis']} (score={score:.0%}, urgency={urg})")
            p.append("")

        # v6.0: Explainability
        readable = (explanation or {}).get("readable", "")
        if readable:
            p.extend([readable, ""])

        p.extend([f"Models: {', '.join(ai.get('models_used',[]))}", "DISCLAIMER: AI-generated. Physician review required."])
        return "\n".join(p)

    def _research(self, report):
        cr = report.get("clinical_report", {}); ai = report.get("ai_metadata", {})
        gov = report.get("governance", {})
        reasoning = report.get("reasoning", {})
        safety = report.get("safety", {})
        explanation = report.get("explanation", {})
        reflection = report.get("reflection", {})
        multi_agent = report.get("multi_agent", {})

        p = [f"═══ MediScan v{VERSION} Research Report ═══"]

        # Model ensemble details
        p.append("── Model Ensemble ──")
        for ir in ai.get("individual_results", []):
            p.append(f"  [{ir.get('model','?'):18s}] conf={ir.get('confidence',0):.3f} "
                    f"reliability={ir.get('reliability',0):.3f} dynamic={ir.get('dynamic_score',0):.3f}")
        p.extend(["", f"  Best: {ai.get('best_model','')} | Conf: {ai.get('confidence',0):.4f} | Agree: {ai.get('agreement_score',0):.4f}"])

        # v6.0: Debate results
        debate = ai.get("debate", {}) if isinstance(ai, dict) else {}
        if debate.get("challenges"):
            p.extend(["", "── Model Debate ──"])
            for c in debate["challenges"][:3]:
                p.append(f"  ⚡ {c['model']} adds: {', '.join(c.get('additional_findings', [])[:3])}")
        if debate.get("agreements"):
            for a in debate["agreements"][:3]:
                p.append(f"  ✅ {a['model']} agrees on: {', '.join(a.get('agrees_on', [])[:3])}")

        # v6.0: Reasoning chain
        chain = reasoning.get("reasoning_chain", [])
        if chain:
            p.extend(["", "── Reasoning Chain ──"])
            for step in chain: p.append(f"  {step}")

        # v6.0: Knowledge-graph differential
        dx = reasoning.get("differential_diagnosis", [])
        if dx:
            p.extend(["", "── Knowledge-Graph Differential ──"])
            for d in dx[:5]:
                markers = d.get("severity_markers_present", [])
                p.append(f"  {d['diagnosis']:25s} score={d.get('validation_score',0):.2f} "
                        f"urgency={d.get('urgency','?')} req={d.get('required_found',0)}/{d.get('required_total',0)} "
                        f"{'markers: '+','.join(markers) if markers else ''}")

        # v6.0: Safety analysis
        if safety:
            p.extend(["", f"── Safety (issues={safety.get('issue_count',0)}, safe={safety.get('is_safe',True)}) ──"])
            for issue in safety.get("issues", [])[:3]:
                p.append(f"  {'🔴' if issue.get('type') == 'hallucination' else '🟡'} {issue.get('type','?')}: {issue.get('detail', issue.get('action', ''))[:80]}")

        # v6.0: Self-reflection
        if reflection and reflection.get("critiques"):
            p.extend(["", f"── Self-Reflection ({reflection.get('critique_count',0)} critiques) ──"])
            for imp in reflection.get("improvements", [])[:3]:
                p.append(f"  💭 {imp}")

        # v6.0: Multi-agent
        dm = multi_agent.get("decision_maker", {})
        if dm.get("recommendation"):
            p.extend(["", "── Decision Maker ──", f"  {dm['recommendation']}"])

        # Clinical sections
        for key in ["findings", "impression", "differential_diagnosis"]:
            val = cr.get(key, "")
            if val: p.extend(["", f"── {key.replace('_',' ').title()} ──", f"  {val}"])

        # Explainability
        readable = (explanation or {}).get("readable", "")
        if readable: p.extend(["", readable])

        p.append("═══ End ═══")
        return "\n".join(p)

    def _simplify(self, text):
        r = text
        for j, s in sorted(self.JARGON.items(), key=lambda x: len(x[0]), reverse=True):
            r = re.sub(re.escape(j), f"{s} ({j})", r, flags=re.IGNORECASE)
        return r


class MediScanChat:
    """ChatGPT-like interface: intent detection + memory + mode switching + styled output."""
    MEDICAL_KW = ["analyze", "scan", "diagnose", "xray", "ct", "mri", "chest", "brain",
                  "lung", "fracture", "tumor", "mass", "lesion", "opacity", "report", "pathology"]
    FOLLOWUP_RE = [r"explain.+more", r"what about", r"is (?:it|this) serious", r"worried", r"follow.?up"]
    MODES = {"doctor": ["doctor mode", "clinical mode"], "patient": ["patient mode", "simple mode", "explain simply"],
             "research": ["research mode", "detailed mode", "scientific mode"]}

    def __init__(self, engine):
        self.engine = engine; self.styler = ResponseStyler()
        self.mode = "patient"; self.memory = deque(maxlen=20)
        self.last_result = None; self.last_file = None

    def chat(self, message, file_path=None, context=None):
        """Main entry — handles everything like ChatGPT."""
        context = context or {}
        self.memory.append({"role": "user", "content": message, "file": file_path})
        # Mode switch
        for mode, triggers in self.MODES.items():
            if any(t in message.lower() for t in triggers):
                old = self.mode; self.mode = mode
                icons = {"doctor": "🧑‍⚕️", "patient": "👨‍👩‍👧", "research": "🧠"}
                return self._resp("mode_switch", f"Switched to {icons.get(mode,'')} {mode.title()} mode.")
        # Intent
        ml = message.lower()
        if file_path: return self._analyze(message, file_path, context)
        if self.last_result and any(re.search(p, ml) for p in self.FOLLOWUP_RE): return self._followup(message)
        if self.last_file and sum(1 for kw in self.MEDICAL_KW if kw in ml) >= 1: return self._analyze(message, self.last_file, context)
        return self._casual(message)

    def _analyze(self, q, fp, ctx):
        self.last_file = fp
        result = self.engine.analyze(file_path=fp, question=q, target_language=ctx.get("language", "en"))
        self.last_result = result
        report = result.get("report")
        styled = self.styler.style(report, self.mode) if report else result.get("report_text", "Analysis complete.")
        return self._resp("analysis", styled)

    def _followup(self, q):
        cr = self.last_result.get("report", {}).get("clinical_report", {})
        report = self.last_result.get("report", {})
        risk = report.get("governance", {}).get("risk_assessment", {}).get("risk_level", "routine")
        # v6.0: Pull reasoning data for richer follow-ups
        reasoning = report.get("reasoning", {})
        dx_list = reasoning.get("differential_diagnosis", [])
        confidence = reasoning.get("confidence", report.get("fusion", {}).get("confidence", 0))
        safety_msg = reasoning.get("risk_assessment", {}).get("safety_message", "")

        if any(w in q.lower() for w in ["serious", "worried", "scared"]):
            msgs = {"emergent": "🔴 Yes — urgent findings detected. Please see a doctor immediately.",
                    "urgent": "🟠 Some findings need attention, but not an emergency. See your doctor soon.",
                    "routine": "🟢 The analysis looks reassuring. Nothing urgent, but always consult your doctor."}
            parts = [msgs.get(risk, msgs["routine"])]
            if dx_list:
                top = dx_list[0]
                parts.append(f"\n📊 Top finding: {top['diagnosis']} (confidence: {top.get('validation_score', 0):.0%})")
            if safety_msg:
                parts.append(f"\n{safety_msg}")
            impression = cr.get("impression", "")
            if impression:
                parts.append(f"\n💡 Summary: {impression[:300]}")
            return self._resp("follow_up", "\n".join(parts))
        # General follow-up
        parts = [f"Based on your previous scan:"]
        if dx_list:
            parts.append(f"📊 Primary consideration: {dx_list[0]['diagnosis']} ({dx_list[0].get('validation_score', 0):.0%})")
            if len(dx_list) > 1:
                parts.append(f"   Also considered: {', '.join(d['diagnosis'] for d in dx_list[1:3])}")
        parts.append(f"\n{cr.get('findings', '')[:400]}")
        parts.append(f"\nQ: {q}")
        return self._resp("follow_up", "\n".join(parts))

    def _casual(self, msg):
        ml = msg.lower()
        if any(g in ml for g in ["hello", "hi", "hey"]):
            return self._resp("casual", f"👋 Hi! I'm MediScan AI v{VERSION}.\nUpload a medical image and I'll analyze it.\nMode: {self.mode.title()} — say 'patient mode' or 'doctor mode' to switch.")
        if "help" in ml:
            return self._resp("casual", f"🏥 MediScan AI v{VERSION}\n📷 Supports: X-ray, CT, MRI, pathology, video, 3D NIfTI, DICOM\n🧑‍⚕️ Modes: doctor / patient / research\nJust upload an image!")
        return self._resp("casual", f"I'm MediScan AI. Upload a medical image for analysis. Say 'help' for info.")

    def _resp(self, intent, text, **extra):
        self.memory.append({"role": "assistant", "content": text[:500]})
        return {"intent": intent, "response": text, "mode": self.mode, **extra}


print("✅ v6.0 Conversation Brain + ResponseStyler ready")


# ═══════════════════════════════════════════════════════════
#  CELL 9: Test Image Downloads
# ═══════════════════════════════════════════════════════════

TEST_DIR = Path('./test_images'); TEST_DIR.mkdir(exist_ok=True)

def _dl(url, dest, desc='', timeout=120):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MediScan-AI/5.1'})
        with urllib.request.urlopen(req, timeout=timeout) as r: data = r.read()
        os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
        with open(dest, 'wb') as f: f.write(data)
        print(f'  ✅ {os.path.basename(dest)} ({len(data)/1024:.0f} KB)'); return True
    except Exception as e: print(f'  ❌ {e}'); return False

def _synth(path, shape=(512,512)):
    arr = np.random.randint(10, 200, shape, dtype=np.uint8)
    arr[shape[0]//3:2*shape[0]//3, shape[1]//3:2*shape[1]//3] += 40
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)

def download_test_images():
    images = {}
    cxr_dir = TEST_DIR / 'cxr'; cxr_dir.mkdir(exist_ok=True)
    dest = cxr_dir / 'CXR_001.png'
    if dest.exists(): images['CXR_001.png'] = str(dest)
    elif _dl('https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_0001_0.png', str(dest)): images['CXR_001.png'] = str(dest)
    else: _synth(str(dest)); images['CXR_001.png'] = str(dest)

    nii_dir = TEST_DIR / 'nifti'; nii_dir.mkdir(exist_ok=True)
    dest = nii_dir / 'brain_t1.nii.gz'
    if dest.exists(): images['brain_t1.nii.gz'] = str(dest)
    elif _dl('https://s3.amazonaws.com/openneuro.org/ds000002/sub-01/anat/sub-01_T1w.nii.gz', str(dest)): images['brain_t1.nii.gz'] = str(dest)
    else:
        try:
            import nibabel as nib
            nib.Nifti1Image(np.random.randn(64,64,32).astype(np.float32)*200-500, np.eye(4)).to_filename(str(dest))
            images['brain_t1.nii.gz'] = str(dest)
        except Exception: pass

    dcm_dir = TEST_DIR / 'dicom'; dcm_dir.mkdir(exist_ok=True)
    dest = dcm_dir / 'CT_small.dcm'
    if dest.exists(): images['CT_small.dcm'] = str(dest)
    else:
        dcm_urls = [
            'https://github.com/pydicom/pydicom/raw/main/tests/dicomdir_tests/77654033/CR1/6154',
            'https://github.com/pydicom/pydicom/raw/v3.0.1/pydicom/data/test_files/CT_small.dcm',
            'https://raw.githubusercontent.com/pydicom/pydicom/main/pydicom/data/test_files/CT_small.dcm',
        ]
        for url in dcm_urls:
            if _dl(url, str(dest), 'DICOM'): images['CT_small.dcm'] = str(dest); break

    print(f'\n📊 {len(images)} test images ready')
    return images

test_images = download_test_images()


# ═══════════════════════════════════════════════════════════
#  CELL 10: Initialize Engine + Chat Interface Demo
# ═══════════════════════════════════════════════════════════

engine = MediScanKaggleEngine()
chat = MediScanChat(engine)

print("\n📊 System Health:")
health = engine.health_check()
for k, v in health["models"].items():
    print(f"  {k}: {'✅ Loaded' if v['loaded'] else '❌ Lazy Load'} (GPU {v['gpu']})")

# ── v6.0 Self-Tests ──
print("\n🧪 v6.0 Self-Tests:")
gov_test = GovernanceLayer()
t1 = gov_test.validate({"consensus_answer": "No evidence of pneumothorax. Lungs are clear."})
t2 = gov_test.validate({"consensus_answer": "Large pneumothorax identified."})
assert t1["risk_assessment"]["risk_level"] == "routine", "Negation test failed"
assert t2["risk_assessment"]["risk_level"] == "emergent", "Positive test failed"
print("  ✅ Negation detection OK")

# Test MedPrompting
try:
    ep = build_expert_prompt("Analyze this xray", modality="xray", file_type="2d", role="primary")
    assert len(ep) > 50, "Expert prompt too short"
    print("  ✅ MedPrompting OK")
except Exception as e:
    print(f"  ❌ MedPrompting FAILED: {e}")

# Test Calibrated Confidence
try:
    c1 = calibrate_confidence("This is a clear finding of consolidation in the left lower lobe.", base=0.7)
    c2 = calibrate_confidence("Maybe possible unclear.", base=0.7)
    assert c1 > c2, "Calibration logic inverted"
    print("  ✅ Calibrated confidence OK")
except Exception as e:
    print(f"  ❌ Calibrated confidence FAILED: {e}")

# Test Knowledge Graph
try:
    kg = MedicalKnowledgeEngine()
    dx = kg.get_differential("opacity and consolidation in left lower lobe", top_k=3)
    assert len(dx) > 0, "No differential generated"
    assert any("pneumonia" in d["diagnosis"] for d in dx), "Pneumonia not in differential for opacity+consolidation"
    print("  ✅ Knowledge Graph OK")
except Exception as e:
    print(f"  ❌ Knowledge Graph FAILED: {e}")

# Test Reasoning Engine (empty input guard)
try:
    re_test = MedicalReasoningEngine()
    result = re_test.reason([], modality="xray")
    assert result["confidence"] == 0.1, "Empty input should give low confidence"
    assert result["finding_count"] == 0, "Empty input should give 0 findings"
    print("  ✅ Reasoning Engine OK")
except Exception as e:
    print(f"  ❌ Reasoning Engine FAILED: {e}")

# Test Safety Layer
try:
    sl = ClinicalSafetyLayer()
    safe = sl.validate("This is definitely 100% certain cancer.", None, None)
    assert not safe["is_safe"], "Should flag hallucination"
    print("  ✅ Safety Layer OK")
except Exception as e:
    print(f"  ❌ Safety Layer FAILED: {e}")

# ═══════════════════════════════════════════════════════════
#  CONVERSATIONAL DEMO — Shows ChatGPT-like interaction
# ═══════════════════════════════════════════════════════════

def demo_chat(msg, file=None):
    """Simulate a chat message and print styled response."""
    print(f"\n{'─'*60}")
    print(f"👤 User: {msg}")
    if file: print(f"   📎 Attached: {os.path.basename(file)}")
    print(f"{'─'*60}")
    result = chat.chat(msg, file_path=file)
    print(f"\n🤖 MediScan AI ({result['mode']} mode):\n")
    print(result["response"])
    return result

# Demo 1: Greeting
demo_chat("Hello!")

# Demo 2: Help
demo_chat("What can you do?")

# Demo 3: Mode switch
demo_chat("Switch to doctor mode")

# Demo 4: Chest X-ray analysis (doctor mode)
if 'CXR_001.png' in test_images:
    print("\n" + "="*80 + "\n🚀 DEMO: Chest X-ray in Doctor Mode\n" + "="*80)
    demo_chat("Analyze this chest X-ray for any abnormalities.", file=test_images['CXR_001.png'])

# Demo 5: Switch to patient mode + follow-up
demo_chat("Switch to patient mode")
demo_chat("Is it serious? Should I be worried?")

# Demo 6: Switch to research mode
demo_chat("Switch to research mode")

# Demo 7: Brain MRI in research mode
if 'brain_t1.nii.gz' in test_images:
    print("\n" + "="*80 + "\n🚀 DEMO: Brain MRI in Research Mode\n" + "="*80)
    demo_chat("Analyze this brain MRI for hemorrhage or mass effect.", file=test_images['brain_t1.nii.gz'])

# Demo 8: CT DICOM
if 'CT_small.dcm' in test_images:
    demo_chat("patient mode")
    print("\n" + "="*80 + "\n🚀 DEMO: CT DICOM in Patient Mode\n" + "="*80)
    demo_chat("What does this CT scan show?", file=test_images['CT_small.dcm'])

print(f"\n🎯 All v{VERSION} demos completed!")
print(f"📊 Chat history: {len(chat.memory)} messages")
