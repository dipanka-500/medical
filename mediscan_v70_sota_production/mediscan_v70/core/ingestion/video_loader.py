"""
MediScan AI v7.0 — Video Loader
Handles loading medical videos (endoscopy, ultrasound, surgical) with frame extraction.
"""
from __future__ import annotations


import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VideoLoader:
    """Medical video loader with intelligent frame sampling."""

    SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    def load(
        self,
        path: str | Path,
        fps: float = 1.0,
        max_frames: int = 1800,
        target_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, Any]:
        """Load a video and extract frames at specified FPS.

        Args:
            path: Path to video file
            fps: Frames per second to extract (1.0 = 1 frame/second)
            max_frames: Maximum number of frames to extract
            target_size: Optional (width, height) to resize frames
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame interval
        frame_interval = max(1, int(video_fps / fps))

        frames = []
        timestamps = []
        frame_idx = 0

        try:
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if target_size:
                        frame_rgb = cv2.resize(frame_rgb, target_size)

                    pil_frame = Image.fromarray(frame_rgb)
                    frames.append(pil_frame)
                    timestamps.append(frame_idx / video_fps)

                frame_idx += 1
        finally:
            cap.release()

        result = {
            "type": "video",
            "frames": frames,
            "timestamps": timestamps,
            "metadata": {
                "original_fps": video_fps,
                "extracted_fps": fps,
                "total_original_frames": total_frames,
                "extracted_frames": len(frames),
                "duration_seconds": duration,
                "resolution": (width, height),
                "size_bytes": path.stat().st_size,
            },
            "source_path": str(path),
        }

        logger.info(
            f"Video loaded: {path.name} | {len(frames)} frames "
            f"({duration:.1f}s @ {fps} fps)"
        )
        return result

    def extract_keyframes(
        self, path: str | Path, threshold: float = 30.0, max_frames: int = 100
    ) -> dict[str, Any]:
        """Extract keyframes based on scene change detection.

        Uses frame difference to detect significant changes — useful for
        surgical videos where most frames are similar.
        """
        path = Path(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        timestamps = []
        prev_gray = None
        frame_idx = 0

        try:
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is None:
                    # Always include first frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                    timestamps.append(frame_idx / video_fps)
                else:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = np.mean(diff)
                    if mean_diff > threshold:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                        timestamps.append(frame_idx / video_fps)

                prev_gray = gray
                frame_idx += 1
        finally:
            cap.release()

        logger.info(f"Keyframes extracted: {len(frames)} from {path.name}")
        return {
            "type": "video",
            "frames": frames,
            "timestamps": timestamps,
            "metadata": {"extraction_method": "keyframe", "threshold": threshold},
            "source_path": str(path),
        }
