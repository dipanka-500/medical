"""
MediScan AI v7.0 — Hulu-Med Model Wrapper
Based STRICTLY on official HuggingFace Transformers API from:
  https://github.com/ZJUI-AI4H/Hulu-Med
  https://huggingface.co/ZJU-AI4H/Hulu-Med-7B

CRITICAL NOTES from official docs:

  TWO DISTINCT APIs exist:
  ─────────────────────────
  1. HF Transformers API (new, recommended):
     - processor(conversation=..., add_system_prompt=True, ...)
     - model.generate(**inputs, max_new_tokens=N) for IMAGE/3D/VIDEO — NO modals=
     - model.generate(**inputs, modals=["text"], do_sample=True, ...) for TEXT-ONLY

  2. Original Method API (old, fallback):
     - HulumedProcessor(images=[frames], text=conversation, merge_size=2)
     - model.generate(**inputs, modals=[modal], do_sample=True, ...) for ALL modalities

  Both APIs require:
  - pixel_values must be cast to bfloat16
  - use_think in batch_decode controls thinking process output

Supports: text, 2D image, multi-image, interleaved, 3D NIfTI, video
Architecture: SigLIP vision encoder + Qwen LLM decoder
7B/32B → hulumed_qwen2 (Qwen2.5), 14B → hulumed_qwen3 (Qwen3)

Official pinned versions: transformers==4.51.2, accelerate==1.7.0,
                          torch==2.4.0, flash-attn==2.7.3
"""
from __future__ import annotations


import atexit
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from ...wrappers.base_model import BaseModel

logger = logging.getLogger(__name__)


class HuluMedModel(BaseModel):
    """Hulu-Med: A Transparent Generalist Model for Holistic Medical
    Vision-Language Understanding.

    Official API: AutoModelForCausalLM + AutoProcessor (trust_remote_code=True)
    Models: Hulu-Med-7B, Hulu-Med-14B, Hulu-Med-32B
    """

    def load(self) -> None:
        """Load Hulu-Med model using official HuggingFace API.

        From official docs:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype="bfloat16", device_map="auto",
                attn_implementation="flash_attention_2",
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer
        """
        from transformers import AutoModelForCausalLM

        start = time.time()
        logger.info(f"Loading Hulu-Med: {self.model_id}")

        # Check transformers version compatibility
        self._check_transformers_version()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.get_dtype(),
            device_map=self.config.get("device_map", "auto"),
            attn_implementation=self.config.get("attn_implementation", "flash_attention_2"),
        )

        # Load processor — prefer AutoProcessor, fail fast if incompatible
        self.processor = self._load_processor()

        self.tokenizer = self.processor.tokenizer
        self.device = next(self.model.parameters()).device
        self.is_loaded = True
        self._load_time = time.time() - start

        # Track temp files for cleanup
        self._temp_files: list[str] = []
        atexit.register(self._cleanup_temp_files)

        # Monkey-patch apply_chat_template for newer transformers compatibility
        # (Returns list instead of string when tokenize=False, which breaks
        #  processing_hulumed.py's process_text() that does text[i].replace())
        self._patch_apply_chat_template()

        # Test if processor.__call__ works (may break on newer transformers)
        self._manual_processor = False
        try:
            _ = self.processor(
                conversation=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
                return_tensors="pt", add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"Processor __call__ broken ({e}), will use manual invoke path")
            self._manual_processor = True

        logger.info(f"Hulu-Med loaded in {self._load_time:.1f}s on {self.device}")

    def _check_transformers_version(self):
        """Warn if transformers version doesn't match official requirements."""
        try:
            import transformers
            version = transformers.__version__
            # Official: transformers==4.51.2
            if not version.startswith("4.51"):
                logger.warning(
                    f"HuluMed officially requires transformers==4.51.2, "
                    f"found {version}. This may cause compatibility issues. "
                    f"Monkey-patches are applied as workarounds."
                )
        except Exception:
            pass

    def _patch_apply_chat_template(self):
        """Monkey-patch apply_chat_template to return string when tokenize=False.

        This is the definitive fix for 'list' object has no attribute 'replace'
        in processing_hulumed.py's process_text() method.

        Known to be needed on transformers >= 4.52.x where apply_chat_template
        changed return type behavior.
        """
        def _make_safe(original_fn):
            def safe_apply(*args, **kwargs):
                result = original_fn(*args, **kwargs)
                if kwargs.get('tokenize', True) is False and isinstance(result, list):
                    # Flatten nested lists and return single string
                    parts = []
                    for item in result:
                        if isinstance(item, list):
                            parts.extend(str(x) for x in item)
                        else:
                            parts.append(str(item))
                    return parts[0] if len(parts) == 1 else "".join(parts)
                return result
            return safe_apply

        patched = set()
        for tok in [self.tokenizer, getattr(self.processor, 'tokenizer', None)]:
            if tok is not None and id(tok) not in patched:
                tok.apply_chat_template = _make_safe(tok.apply_chat_template)
                patched.add(id(tok))
                logger.info(f"Patched apply_chat_template on {type(tok).__name__}")

    def _load_processor(self):
        """Load HuluMed processor — prefer official API, single fallback.

        Official API (recommended):
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        If AutoProcessor fails (version incompatibility), try manual construction.
        """
        # Attempt 1: Standard AutoProcessor (official recommended path)
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            logger.info("HuluMed processor loaded via AutoProcessor")
            return proc
        except Exception as e:
            logger.warning(
                f"AutoProcessor failed ({type(e).__name__}: {e}). "
                f"Ensure transformers>=4.51.2 is installed. "
                f"Trying manual construction..."
            )

        # Attempt 2: Manual construction via dynamic module loading
        try:
            from transformers import AutoTokenizer
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            ImgProcCls = get_class_from_dynamic_module(
                "image_processing_hulumed.HulumedImageProcessor", self.model_id
            )
            ProcCls = get_class_from_dynamic_module(
                "processing_hulumed.HulumedProcessor", self.model_id
            )
            img_proc = ImgProcCls.from_pretrained(self.model_id)
            proc = ProcCls(image_processor=img_proc, tokenizer=tokenizer)
            logger.info("HuluMed processor built manually (dynamic_module)")
            return proc
        except Exception as e2:
            logger.error(
                f"All processor loading attempts failed. Last error: {e2}. "
                f"Please ensure: pip install transformers==4.51.2 accelerate==1.7.0"
            )
            raise RuntimeError(
                f"Cannot load HuluMed processor for {self.model_id}. "
                f"Check transformers version compatibility."
            ) from e2

    def analyze(
        self,
        images: Optional[list] = None,
        text: str = "",
        modality: str = "image",
        use_think: bool = True,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        **kwargs,
    ) -> dict[str, Any]:
        """Run Hulu-Med inference following the official API.

        CRITICAL: Two different generation modes exist:
          1. Text-only: modals=["text"], do_sample=True, temperature
          2. HF Transformers visual: model.generate(**inputs, max_new_tokens=N)
             + do_sample, temperature for quality generation
          3. Manual/Original visual: modals=[modal_str], do_sample=True, temperature

        Args:
            images: List of PIL Images (used as fallback if no source_path)
            text: The medical question/prompt
            modality: "text", "image", "multi_image", "interleaved", "3d", "video"
            use_think: If True, include reasoning/thinking process in output
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: source_path, nii_path, nii_num_slices, nii_axis,
                      video_path, fps, max_frames, interleaved_content
        """
        if not self.is_loaded:
            self.load()

        start_time = time.time()

        # Build conversation in official format
        conversation = self._build_conversation(images, text, modality, **kwargs)

        # ── Process inputs ──
        if self._manual_processor:
            inputs = self._manual_process(conversation, images, modality, **kwargs)
        else:
            # Official HF API: processor(conversation=..., add_system_prompt=True, ...)
            processor_kwargs = {
                "conversation": conversation,
                "return_tensors": "pt",
                "add_generation_prompt": True,
            }
            if modality != "text":
                processor_kwargs["add_system_prompt"] = True
            inputs = self.processor(**processor_kwargs)

        # ── Move tensors to device ──
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # ── Cast pixel values to bfloat16 (CRITICAL per official docs) ──
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.get_dtype())

        # ── Generate ──
        with torch.inference_mode():
            # Pop any conflicting keys that would collide with explicit kwargs
            gen_inputs = dict(inputs)
            gen_inputs.pop("modals", None)

            if modality == "text":
                # Text-only: use modals=["text"] with sampling (both APIs agree)
                output_ids = self.model.generate(
                    **gen_inputs,
                    do_sample=True,
                    modals=["text"],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            elif self._manual_processor:
                # Original Method API: pass modals=[modal_str] for ALL modalities
                # This matches the official original-method inference scripts
                modal_str = "video" if modality in ("3d", "video") else "image"
                output_ids = self.model.generate(
                    **gen_inputs,
                    do_sample=True,
                    modals=[modal_str],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                # HF Transformers API for visual modalities:
                # Official simple examples show just max_new_tokens, but the older
                # inference scripts show do_sample + temperature are intended.
                # Include generation params for quality output.
                output_ids = self.model.generate(
                    **gen_inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        # ── Decode ──
        # Decode separately with use_think=True and use_think=False
        # This is the safest way per official docs to isolate chain-of-thought
        response_full = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, use_think=True
        )[0].strip()

        response_answer = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, use_think=False
        )[0].strip()

        # Extract thinking using safer approach than naive string replacement
        thinking = self._extract_thinking(response_full, response_answer)

        inference_time = time.time() - start_time

        return {
            "model": self.model_id,
            "response": response_answer if not use_think else response_full,
            "answer": response_answer,
            "thinking": thinking,
            "modality": modality,
            "confidence": self.estimate_confidence(response_answer, base_confidence=0.82),
            "metadata": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "use_think": use_think,
                "inference_time": round(inference_time, 2),
                "processor_mode": "manual" if self._manual_processor else "auto",
            },
        }

    def _extract_thinking(self, full_response: str, answer_only: str) -> str:
        """Extract thinking/reasoning from full response vs answer-only.

        Uses rfind to safely isolate the chain-of-thought, avoiding issues
        when the answer text appears multiple times in the full response.
        """
        if not full_response or not answer_only:
            return ""
        if full_response == answer_only:
            return ""

        # Find the LAST occurrence of the answer in the full response
        idx = full_response.rfind(answer_only)
        if idx > 0:
            thinking = full_response[:idx].strip()
            # Clean up common thinking markers
            for marker in ["</think>", "<think>", "---"]:
                thinking = thinking.replace(marker, "").strip()
            return thinking
        elif idx == 0:
            # Answer is at the start, thinking might be after (unlikely)
            remainder = full_response[len(answer_only):].strip()
            return remainder if remainder else ""
        else:
            # answer_only not found in full — just return the difference
            return full_response.replace(answer_only, "").strip()

    def _manual_process(
        self,
        conversation: list[dict],
        images: Optional[list] = None,
        modality: str = "image",
        **kwargs,
    ) -> dict:
        """Bypass processor.__call__ — invoke internal methods directly.
        This avoids the broken _merge_kwargs in newer transformers.

        Uses the ORIGINAL METHOD API from official HuluMed docs:
            processor(images=[slices], text=conversation, merge_size=2, return_tensors="pt")
        """
        import numpy as np
        from transformers import BatchEncoding

        # 1. Load images from paths in conversation, or use PIL images directly
        all_images = []
        for msg in conversation:
            for item in msg.get("content", []):
                ctype = item.get("type", "")
                if ctype == "image":
                    img_info = item.get("image", {})
                    img_path = img_info.get("image_path", "")
                    if img_path and Path(img_path).exists():
                        try:
                            loaded = self.processor.load_images(img_path)
                            if isinstance(loaded, list):
                                all_images.extend(loaded)
                            else:
                                all_images.append(loaded)
                        except Exception as e:
                            logger.warning(f"load_images failed for {img_path}: {e}")
                            if images:
                                all_images.extend(images if isinstance(images, list) else [images])
                    elif images:
                        all_images.extend(images if isinstance(images, list) else [images])
                elif ctype == "3d":
                    td_info = item.get("3d", {})
                    img_path = td_info.get("image_path", "")
                    nii_num_slices = td_info.get("nii_num_slices", 180)
                    nii_axis = td_info.get("nii_axis", 2)
                    if img_path and Path(img_path).exists():
                        loaded_3d = False
                        # Try processor.load_images with NIfTI kwargs
                        try:
                            loaded = self.processor.load_images(
                                img_path, nii_num_slices=nii_num_slices, nii_axis=nii_axis
                            )
                            if isinstance(loaded, list):
                                all_images.extend(loaded)
                            else:
                                all_images.append(loaded)
                            loaded_3d = True
                        except TypeError:
                            logger.info("load_images doesn't accept NIfTI kwargs, trying plain")
                            try:
                                loaded = self.processor.load_images(img_path)
                                if isinstance(loaded, list):
                                    all_images.extend(loaded)
                                else:
                                    all_images.append(loaded)
                                loaded_3d = True
                            except Exception as e2:
                                logger.warning(f"load_images(plain) also failed: {e2}")
                        except Exception as e:
                            logger.warning(f"load_images(3d) failed: {e}")

                        if not loaded_3d:
                            # Manual nibabel fallback (explicitly documented: pip install nibabel)
                            logger.info("Falling back to manual nibabel NIfTI loading")
                            try:
                                import nibabel as nib
                                nii = nib.load(str(img_path))
                                vol = np.asarray(nii.dataobj, dtype=np.float32)
                                total = vol.shape[nii_axis]
                                indices = np.linspace(0, total - 1, min(nii_num_slices, total), dtype=int)
                                for idx in indices:
                                    s = np.take(vol, int(idx), axis=nii_axis)
                                    smin, smax = float(s.min()), float(s.max())
                                    s_norm = ((s - smin) / (smax - smin + 1e-8) * 255).astype(np.uint8)
                                    from PIL import Image as PILImage
                                    all_images.append(PILImage.fromarray(s_norm, mode="L").convert("RGB"))
                                logger.info(f"Loaded {len(indices)} slices via nibabel fallback")
                            except ImportError:
                                logger.error(
                                    "nibabel not installed. Required for 3D NIfTI processing. "
                                    "Install with: pip install nibabel"
                                )
                            except Exception as e3:
                                logger.error(f"nibabel fallback failed: {e3}")
                elif ctype == "video":
                    vid_info = item.get("video", {})
                    vid_path = vid_info.get("video_path", "")
                    fps = vid_info.get("fps", 1)
                    max_frames = vid_info.get("max_frames", 1800)
                    if vid_path and Path(vid_path).exists():
                        try:
                            frames, _ = self.processor.load_video(
                                vid_path, fps=fps, max_frames=max_frames
                            )
                            all_images.extend(frames)
                        except Exception as e:
                            logger.warning(f"load_video failed: {e}")

        # 2. Process images per Official Original Method API:
        # - Image: process_images(all_images, merge_size=1) → each PIL is separate
        # - Video/3D: process_images([all_images], merge_size=2) → WRAPPED in outer list
        # This matches: processor(images=[slices], merge_size=2, ...) from official docs
        merge_size = 2 if modality in ("video", "3d") else 1
        if all_images:
            images_for_processor = [all_images] if modality in ("video", "3d") else all_images
        else:
            images_for_processor = None
        image_inputs = self.processor.process_images(
            images_for_processor, merge_size=merge_size
        )

        # 3. Convert numpy arrays to torch tensors
        for key in list(image_inputs.keys()):
            val = image_inputs[key]
            if isinstance(val, np.ndarray):
                image_inputs[key] = torch.from_numpy(val)
            elif isinstance(val, list):
                image_inputs[key] = [
                    torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for v in val
                ]

        # 4. Build prompt with <image> tokens (Original Method approach)
        question_text = ""
        for msg in conversation:
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    question_text = item.get("text", "")

        num_images = len(all_images) if all_images else 0
        content_str = "<image>" * num_images + question_text
        simple_conv = [{"role": "user", "content": content_str}]

        prompt = self.tokenizer.apply_chat_template(
            simple_conv, tokenize=False, add_generation_prompt=True
        )
        if isinstance(prompt, list):
            prompt = prompt[0] if len(prompt) == 1 else "".join(str(x) for x in prompt)

        # 5. Tokenize directly (BYPASSES broken process_text entirely)
        text_inputs = self.tokenizer(str(prompt), return_tensors="pt")

        return BatchEncoding(data={**text_inputs, **image_inputs})

    def _build_conversation(
        self,
        images: Optional[list],
        text: str,
        modality: str,
        **kwargs,
    ) -> list[dict]:
        """Build conversation in Hulu-Med's EXACT official format.

        Official conversation formats from docs:
        - Text:        [{"type": "text", "text": "..."}]
        - Image:       [{"type": "image", "image": {"image_path": "..."}}, {"type": "text", ...}]
        - Multi-Image: [{"type": "image", ...}, {"type": "image", ...}, {"type": "text", ...}]
        - Interleaved: [{"type": "text", ...}, {"type": "image", ...}, {"type": "text", ...}, ...]
        - 3D:          [{"type": "3d", "3d": {"image_path": "...", ...}}, {"type": "text", ...}]
        - Video:       [{"type": "video", "video": {"video_path": "...", ...}}, {"type": "text", ...}]

        CRITICAL: Do NOT silently downgrade multimodal to text-only.
        If multimodal is requested but media is missing, raise ValueError.
        """
        content = []

        if modality == "text":
            # ── Text-only ──
            content.append({"type": "text", "text": text})

        elif modality in ("image", "multi_image", "3d", "video", "interleaved") and not (
            images or kwargs.get("source_path") or kwargs.get("nii_path")
            or kwargs.get("video_path") or kwargs.get("interleaved_content")
        ):
            # ── EXPLICIT ERROR instead of silent fallback ──
            raise ValueError(
                f"Missing input for multimodal modality '{modality}'. "
                f"Provide images, source_path, nii_path, video_path, "
                f"or interleaved_content as appropriate."
            )

        elif modality == "image":
            # ── Single 2D image ──
            # Official: {"type": "image", "image": {"image_path": "./demo/demo.jpg"}}
            image_path = self._resolve_image_path(images, **kwargs)
            content.append({
                "type": "image",
                "image": {"image_path": str(image_path)},
            })
            content.append({"type": "text", "text": text})

        elif modality == "multi_image":
            # ── Multiple images ──
            # Official: multiple {"type": "image"} entries
            if images:
                for i, img in enumerate(images):
                    img_path = self._resolve_single_image_path(img, index=i, **kwargs)
                    content.append({
                        "type": "image",
                        "image": {"image_path": str(img_path)},
                    })
            content.append({"type": "text", "text": text})

        elif modality == "interleaved":
            # ── Interleaved text + images ──
            # Official "Interleaved Example":
            #   [{"type": "text", "text": "Image A:"},
            #    {"type": "image", "image": {"image_path": "./XRay.jpg"}},
            #    {"type": "text", "text": "Image B:"},
            #    {"type": "image", "image": {"image_path": "./pathology.png"}},
            #    {"type": "text", "text": "Which is the pathology slide?"}]
            interleaved_content = kwargs.get("interleaved_content", [])
            if interleaved_content:
                content.extend(interleaved_content)
            else:
                # Fallback: build from images + text (non-interleaved)
                if images:
                    for i, img in enumerate(images):
                        img_path = self._resolve_single_image_path(img, index=i, **kwargs)
                        content.append({
                            "type": "text",
                            "text": f"Image {chr(65 + i)}:",
                        })
                        content.append({
                            "type": "image",
                            "image": {"image_path": str(img_path)},
                        })
                content.append({"type": "text", "text": text})

        elif modality == "3d":
            # ── 3D NIfTI volume ──
            # Official: {"type": "3d", "3d": {"image_path": "...",
            #            "nii_num_slices": 180, "nii_axis": 2}}
            nii_path = kwargs.get("nii_path", kwargs.get("source_path", ""))
            num_slices = kwargs.get("nii_num_slices", 180)
            axis = kwargs.get("nii_axis", 2)  # 0=sagittal, 1=coronal, 2=axial
            content.append({
                "type": "3d",
                "3d": {
                    "image_path": str(nii_path),
                    "nii_num_slices": num_slices,
                    "nii_axis": axis,
                },
            })
            content.append({"type": "text", "text": text})

        elif modality == "video":
            # ── Video ──
            # Official: {"type": "video", "video": {"video_path": "...",
            #            "fps": 1, "max_frames": 1800}}
            video_path = kwargs.get("video_path", kwargs.get("source_path", ""))
            fps = kwargs.get("fps", 1)
            max_frames = kwargs.get("max_frames", 1800)
            content.append({
                "type": "video",
                "video": {
                    "video_path": str(video_path),
                    "fps": fps,
                    "max_frames": max_frames,
                },
            })
            content.append({"type": "text", "text": text})

        return [{"role": "user", "content": content}]

    def _resolve_image_path(self, images: Optional[list], **kwargs) -> str:
        """Resolve image path for the processor.

        The Hulu-Med processor loads images internally from file paths.
        If we only have a PIL image, save it to a temp file first.
        """
        # Prefer source_path (original file on disk)
        source_path = kwargs.get("source_path", kwargs.get("file_path", ""))
        if source_path and Path(source_path).exists():
            return str(source_path)

        # Fallback: save PIL image to temp file
        if images and len(images) > 0:
            return self._save_pil_to_temp(images[0])

        return ""

    def _resolve_single_image_path(self, img, index: int = 0, **kwargs) -> str:
        """Resolve a single image to a path."""
        if isinstance(img, (str, Path)):
            return str(img)
        if isinstance(img, Image.Image):
            return self._save_pil_to_temp(img, suffix=f"_{index}")
        return ""

    def _save_pil_to_temp(self, pil_image: Image.Image, suffix: str = "") -> str:
        """Save a PIL image to a temporary file and return the path.

        Temp files are tracked and cleaned up on exit or via cleanup().
        """
        tmp = tempfile.NamedTemporaryFile(
            suffix=f"{suffix}.jpg", prefix="hulumed_", delete=False
        )
        pil_image.save(tmp.name)
        tmp.close()
        self._temp_files.append(tmp.name)
        logger.debug(f"Saved PIL image to temp: {tmp.name}")
        return tmp.name

    def _cleanup_temp_files(self) -> None:
        """Clean up all temporary files created during inference."""
        cleaned = 0
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
                    cleaned += 1
            except Exception as e:
                logger.debug(f"Could not remove temp file {f}: {e}")
        if cleaned:
            logger.debug(f"Cleaned up {cleaned} temp files")
        self._temp_files.clear()

    def unload(self) -> None:
        """Free model from GPU memory and clean up temp files."""
        self._cleanup_temp_files()
        super().unload()
