"""
MediScan AI v7.0 — DICOM Loader
Handles loading and parsing of DICOM files with full metadata extraction.
"""
from __future__ import annotations


import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pydicom
from PIL import Image

logger = logging.getLogger(__name__)


class DICOMLoader:
    """Production DICOM loader with windowing, multi-frame, and series support."""

    # Standard DICOM window presets (center, width)
    WINDOW_PRESETS = {
        "lung": (-600, 1500),
        "mediastinum": (40, 400),
        "bone": (400, 1800),
        "brain": (40, 80),
        "subdural": (75, 215),
        "stroke": (32, 8),
        "liver": (60, 150),
        "abdomen": (40, 350),
        "soft_tissue": (50, 400),
        "spine": (50, 250),
    }

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a DICOM file and return structured data."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DICOM file not found: {path}")

        try:
            ds = pydicom.dcmread(str(path), force=True)
            result = {
                "type": "2d",  # Single DICOM file = 2D image
                "pixel_data": self._extract_pixel_data(ds),
                "metadata": self._extract_metadata(ds),
                "modality": self._detect_modality(ds),
                "patient_info": self._extract_patient_info(ds),
                "study_info": self._extract_study_info(ds),
                "source_path": str(path),
            }
            logger.info(f"DICOM loaded: {path.name} | Modality: {result['modality']}")
            return result
        except Exception as e:
            logger.error(f"Failed to load DICOM {path}: {e}")
            raise

    def load_series(self, directory: str | Path) -> list[dict[str, Any]]:
        """Load an entire DICOM series from a directory."""
        directory = Path(directory)
        dcm_files = sorted(directory.glob("*.dcm")) + sorted(directory.glob("*.DCM"))
        if not dcm_files:
            # Try files without extension (common in DICOM)
            dcm_files = []
            for f in sorted(directory.iterdir()):
                if f.is_file():
                    try:
                        pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                        dcm_files.append(f)
                    except Exception as e:  # noqa: broad-except logged
                        continue

        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files found in: {directory}")

        series = []
        for dcm_path in dcm_files:
            try:
                series.append(self.load(dcm_path))
            except Exception as e:
                logger.warning(f"Skipping {dcm_path.name}: {e}")

        logger.info(f"Loaded DICOM series: {len(series)} slices from {directory}")
        return series

    def series_to_volume(self, series: list[dict[str, Any]]) -> np.ndarray:
        """Stack a DICOM series into a 3D volume, sorted by slice location."""
        slices_with_loc = []
        for s in series:
            loc = s["metadata"].get("slice_location", 0.0)
            pixel = s["pixel_data"]
            if pixel is not None:
                slices_with_loc.append((loc, pixel))

        slices_with_loc.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices_with_loc], axis=0)
        return volume

    def apply_windowing(
        self, pixel_data: np.ndarray, window: str | tuple[float, float] = "soft_tissue"
    ) -> np.ndarray:
        """Apply HU windowing to CT data."""
        if isinstance(window, str):
            if window not in self.WINDOW_PRESETS:
                logger.warning(f"Unknown window preset '{window}', using soft_tissue")
                window = "soft_tissue"
            center, width = self.WINDOW_PRESETS[window]
        else:
            center, width = window

        lower = center - width / 2
        upper = center + width / 2
        windowed = np.clip(pixel_data, lower, upper)
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        return windowed

    def _extract_pixel_data(self, ds: pydicom.Dataset) -> Optional[np.ndarray]:
        """Extract pixel data with rescale slope/intercept applied."""
        if not hasattr(ds, "PixelData"):
            return None

        try:
            pixel_array = ds.pixel_array.astype(np.float32)

            # Apply rescale slope and intercept (for CT Hounsfield Units)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            pixel_array = pixel_array * slope + intercept

            return pixel_array
        except Exception as e:
            logger.warning(f"Could not extract pixel data: {e}")
            return None

    def _extract_metadata(self, ds: pydicom.Dataset) -> dict[str, Any]:
        """Extract comprehensive DICOM metadata."""
        metadata = {}
        fields = {
            "sop_class_uid": "SOPClassUID",
            "study_instance_uid": "StudyInstanceUID",
            "series_instance_uid": "SeriesInstanceUID",
            "sop_instance_uid": "SOPInstanceUID",
            "modality": "Modality",
            "manufacturer": "Manufacturer",
            "institution": "InstitutionName",
            "station_name": "StationName",
            "study_description": "StudyDescription",
            "series_description": "SeriesDescription",
            "body_part": "BodyPartExamined",
            "slice_thickness": "SliceThickness",
            "slice_location": "SliceLocation",
            "pixel_spacing": "PixelSpacing",
            "rows": "Rows",
            "columns": "Columns",
            "bits_allocated": "BitsAllocated",
            "bits_stored": "BitsStored",
            "photometric_interpretation": "PhotometricInterpretation",
            "window_center": "WindowCenter",
            "window_width": "WindowWidth",
            "rescale_slope": "RescaleSlope",
            "rescale_intercept": "RescaleIntercept",
            "kvp": "KVP",
            "exposure": "Exposure",
            "image_position": "ImagePositionPatient",
            "image_orientation": "ImageOrientationPatient",
            "acquisition_date": "AcquisitionDate",
            "acquisition_time": "AcquisitionTime",
            "content_date": "ContentDate",
        }

        for key, tag in fields.items():
            val = getattr(ds, tag, None)
            if val is not None:
                if hasattr(val, "original_string"):
                    metadata[key] = str(val)
                elif isinstance(val, pydicom.multival.MultiValue):
                    metadata[key] = [float(v) for v in val]
                elif isinstance(val, pydicom.uid.UID):
                    metadata[key] = str(val)
                else:
                    try:
                        metadata[key] = float(val) if isinstance(val, (int, float)) else str(val)
                    except (ValueError, TypeError):
                        metadata[key] = str(val)

        return metadata

    def _detect_modality(self, ds: pydicom.Dataset) -> str:
        """Detect imaging modality from DICOM tags."""
        modality = getattr(ds, "Modality", "UNKNOWN")
        modality_map = {
            "CR": "xray", "DX": "xray", "DR": "xray",
            "CT": "ct", "MR": "mri", "MG": "mammography",
            "US": "ultrasound", "NM": "nuclear_medicine",
            "PT": "pet", "XA": "angiography",
            "RF": "fluoroscopy", "ES": "endoscopy",
            "OP": "fundoscopy", "OPT": "oct",
            "SM": "slide_microscopy",
        }
        return modality_map.get(str(modality), str(modality).lower())

    def _extract_patient_info(self, ds: pydicom.Dataset) -> dict[str, Any]:
        """Extract patient demographics and anonymize PHI before returning.

        HIPAA: Raw PHI must NEVER be passed downstream without anonymization.
        Only sex and age (non-identifying) are returned in cleartext.
        """
        from mediscan_v70.core.security.hipaa import HIPAACompliance

        raw_info = {
            "patient_id": str(getattr(ds, "PatientID", "")),
            "patient_name": str(getattr(ds, "PatientName", "")),
            "birth_date": str(getattr(ds, "PatientBirthDate", "")),
            "sex": str(getattr(ds, "PatientSex", "")),
            "age": str(getattr(ds, "PatientAge", "")),
            "weight": str(getattr(ds, "PatientWeight", "")),
        }
        # Anonymize PHI fields before they leave the ingestion layer
        hipaa = HIPAACompliance()
        return hipaa.anonymize(raw_info)

    def _extract_study_info(self, ds: pydicom.Dataset) -> dict[str, Any]:
        """Extract study-level information."""
        return {
            "study_date": str(getattr(ds, "StudyDate", "")),
            "study_time": str(getattr(ds, "StudyTime", "")),
            "accession_number": str(getattr(ds, "AccessionNumber", "")),
            "referring_physician": str(getattr(ds, "ReferringPhysicianName", "")),
            "study_description": str(getattr(ds, "StudyDescription", "")),
        }

    def to_pil(self, pixel_data: np.ndarray, window: str = "soft_tissue") -> Image.Image:
        """Convert DICOM pixel data to PIL Image with windowing."""
        windowed = self.apply_windowing(pixel_data, window)
        if windowed.ndim == 2:
            return Image.fromarray(windowed, mode="L")
        return Image.fromarray(windowed)
