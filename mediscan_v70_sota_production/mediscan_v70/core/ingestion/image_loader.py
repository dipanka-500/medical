"""
MediScan AI v7.0 — Image Loader
Handles loading of standard medical images (JPEG, PNG, TIFF, etc.) and NIfTI 3D volumes.
"""
from __future__ import annotations


import logging
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image

logger = logging.getLogger(__name__)


class ImageLoader:
    """Unified loader for 2D images and 3D NIfTI volumes."""

    SUPPORTED_2D = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    SUPPORTED_3D = {".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"}

    def load(self, path: str | Path) -> dict[str, Any]:
        """Auto-detect and load any supported image format."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        suffix = "".join(path.suffixes).lower()  # handles .nii.gz
        if suffix in self.SUPPORTED_3D or path.suffix.lower() in self.SUPPORTED_3D:
            return self._load_3d(path)
        elif path.suffix.lower() in self.SUPPORTED_2D:
            return self._load_2d(path)
        else:
            # Try 2D first, then 3D
            try:
                return self._load_2d(path)
            except Exception as e:  # noqa: broad-except logged
                return self._load_3d(path)

    def _load_2d(self, path: Path) -> dict[str, Any]:
        """Load a 2D medical image."""
        image = Image.open(str(path))
        image_rgb = image.convert("RGB") if image.mode != "RGB" else image

        result = {
            "type": "2d",
            "image": image_rgb,
            "original_image": image,
            "pixel_array": np.array(image_rgb),
            "metadata": {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "size_bytes": path.stat().st_size,
            },
            "source_path": str(path),
        }
        logger.info(f"2D image loaded: {path.name} ({image.width}x{image.height})")
        return result

    def _load_3d(self, path: Path) -> dict[str, Any]:
        """Load a 3D NIfTI or other volumetric image."""
        try:
            # Primary: SimpleITK (more robust for medical formats)
            sitk_image = sitk.ReadImage(str(path))
            volume = sitk.GetArrayFromImage(sitk_image)
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = sitk_image.GetDirection()

            result = {
                "type": "3d",
                "volume": volume,
                "sitk_image": sitk_image,
                "metadata": {
                    "shape": volume.shape,
                    "spacing": spacing,
                    "origin": origin,
                    "direction": direction,
                    "dtype": str(volume.dtype),
                    "min_value": float(volume.min()),
                    "max_value": float(volume.max()),
                    "size_bytes": path.stat().st_size,
                },
                "source_path": str(path),
            }
            logger.info(f"3D volume loaded: {path.name} | Shape: {volume.shape}")
            return result

        except Exception as e:
            logger.warning(f"SimpleITK failed for {path}, trying nibabel: {e}")
            # Fallback: nibabel (NIfTI-specific)
            nii = nib.load(str(path))
            volume = np.asarray(nii.dataobj, dtype=np.float32)
            affine = nii.affine
            header = nii.header

            # Transpose from nibabel (X, Y, Z) to SimpleITK convention
            # (Z, Y, X) so that depth is always axis 0 — downstream code
            # (main.py slice extraction) assumes depth-first ordering.
            if volume.ndim == 3:
                volume = np.transpose(volume, (2, 1, 0))

            result = {
                "type": "3d",
                "volume": volume,
                "nifti_obj": nii,
                "metadata": {
                    "shape": volume.shape,
                    "voxel_size": header.get_zooms()[:3] if hasattr(header, "get_zooms") else None,
                    "affine": affine.tolist(),
                    "dtype": str(volume.dtype),
                    "min_value": float(volume.min()),
                    "max_value": float(volume.max()),
                    "size_bytes": path.stat().st_size,
                },
                "source_path": str(path),
            }
            logger.info(f"3D volume loaded (nibabel): {path.name} | Shape: {volume.shape}")
            return result

    def extract_slices(
        self,
        volume: np.ndarray,
        axis: int = 2,
        num_slices: Optional[int] = None,
    ) -> list[Image.Image]:
        """Extract 2D slices from a 3D volume along a given axis.

        Args:
            volume: 3D numpy array
            axis: 0=sagittal, 1=coronal, 2=axial
            num_slices: Number of evenly-spaced slices to extract. None = all.
        """
        total = volume.shape[axis]
        if num_slices and num_slices < total:
            indices = np.linspace(0, total - 1, num_slices, dtype=int)
        else:
            indices = range(total)

        slices = []
        for idx in indices:
            if axis == 0:
                s = volume[idx, :, :]
            elif axis == 1:
                s = volume[:, idx, :]
            else:
                s = volume[:, :, idx]

            # Normalize to 0-255
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                s_norm = ((s - s_min) / (s_max - s_min) * 255).astype(np.uint8)
            else:
                s_norm = np.zeros_like(s, dtype=np.uint8)

            slices.append(Image.fromarray(s_norm, mode="L"))

        return slices

    def load_multi_images(self, paths: list[str | Path]) -> list[dict[str, Any]]:
        """Load multiple images at once."""
        return [self.load(p) for p in paths]
