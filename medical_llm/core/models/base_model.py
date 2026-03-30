"""
Base Model — Abstract base class for all LLM model wrappers.
Provides lazy loading, GPU management, generation config, health checks,
and a unified generate() interface.
"""

from __future__ import annotations

import gc
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for all medical LLM wrappers.

    Provides:
        - Lazy model loading with configurable device/dtype
        - Unified generate() interface
        - GPU memory management (unload/cleanup)
        - Health check and status reporting
        - Generation config management (temperature, top_p, etc.)

    Subclasses must implement:
        - _load_model(): Load the model and tokenizer
        - _generate_impl(): Core generation logic
    """

    def __init__(self, model_id: str, config: dict[str, Any] | None = None):
        self.model_id = model_id
        self.config = config or {}

        # Model state
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._load_time: float = 0.0
        self._generation_count: int = 0
        self._total_tokens_generated: int = 0

        # Generation defaults from config
        self.max_new_tokens = self.config.get("max_new_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.3)
        self.top_p = self.config.get("top_p", 0.9)
        self.top_k = self.config.get("top_k", 50)
        self.repetition_penalty = self.config.get("repetition_penalty", 1.15)
        self.do_sample = self.config.get("do_sample", True)
        self.device_map = self.config.get("device_map", "auto")
        self.torch_dtype = self.config.get("torch_dtype", "bfloat16")
        self.trust_remote_code = self.config.get("trust_remote_code", False)
        self.weight = self.config.get("weight", 0.5)
        self.role = self.config.get("role", "primary")
        self.use_vllm = self.config.get("use_vllm", False)

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._is_loaded

    def load(self) -> None:
        """Load the model into memory. Idempotent — safe to call multiple times."""
        if self._is_loaded:
            logger.debug(f"{self.model_id} already loaded")
            return

        start = time.time()
        logger.info(f"Loading model: {self.model_id}")

        try:
            self._load_model()
            self._is_loaded = True
            self._load_time = time.time() - start
            logger.info(
                f"Model loaded: {self.model_id} in {self._load_time:.1f}s"
            )
        except Exception as e:
            logger.error(f"Failed to load {self.model_id}: {e}")
            raise

    @abstractmethod
    def _load_model(self) -> None:
        """Subclass implementation: load model and tokenizer.

        Must set self._model and self._tokenizer.
        """
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        do_sample: bool | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Override max tokens for this call
            temperature: Override temperature
            top_p: Override top_p
            top_k: Override top_k
            repetition_penalty: Override repetition penalty
            do_sample: Override sampling flag
            system_prompt: Optional system prompt to prepend
            **kwargs: Additional model-specific parameters

        Returns:
            Dict with keys: text, tokens_generated, latency, model
        """
        if not self._is_loaded:
            self.load()

        gen_config = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "top_k": top_k if top_k is not None else self.top_k,
            "repetition_penalty": repetition_penalty or self.repetition_penalty,
            "do_sample": do_sample if do_sample is not None else self.do_sample,
        }

        # Build full prompt with system message
        full_prompt = prompt
        if system_prompt:
            full_prompt = self._build_chat_prompt(system_prompt, prompt)

        start = time.time()
        try:
            result = self._generate_impl(full_prompt, gen_config, **kwargs)
            latency = time.time() - start

            self._generation_count += 1
            tokens = result.get("tokens_generated", 0)
            self._total_tokens_generated += tokens

            result.update({
                "model": self.model_id,
                "model_key": self.config.get("key", self.model_id),
                "latency": round(latency, 3),
                "weight": self.weight,
                "role": self.role,
            })

            logger.info(
                f"Generated {tokens} tokens from {self.model_id} "
                f"in {latency:.2f}s"
            )
            return result

        except Exception as e:
            latency = time.time() - start
            logger.error(f"Generation failed for {self.model_id}: {e}")
            return {
                "text": "",
                "error": str(e),
                "model": self.model_id,
                "latency": round(latency, 3),
                "tokens_generated": 0,
            }

    @abstractmethod
    def _generate_impl(
        self, prompt: str, gen_config: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Subclass implementation: core generation logic.

        Must return dict with at least: text, tokens_generated
        """
        raise NotImplementedError

    def _build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Build a chat-formatted prompt. Override for model-specific templates."""
        return f"{system_prompt}\n\n{user_prompt}"

    def _get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)

    def unload(self) -> None:
        """Unload model from GPU/CPU memory."""
        if not self._is_loaded:
            return

        logger.info(f"Unloading model: {self.model_id}")

        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._is_loaded = False
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        logger.info(f"Model unloaded: {self.model_id}")

    def health_check(self) -> dict[str, Any]:
        """Return model health status."""
        return {
            "model_id": self.model_id,
            "is_loaded": self._is_loaded,
            "load_time": round(self._load_time, 2),
            "generation_count": self._generation_count,
            "total_tokens": self._total_tokens_generated,
            "role": self.role,
            "weight": self.weight,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"<{self.__class__.__name__}({self.model_id}, {status})>"


class HuggingFaceLLM(BaseLLM):
    """Generic HuggingFace Transformers LLM wrapper.

    Handles standard AutoModelForCausalLM + AutoTokenizer loading
    and generation. Most model engines can inherit from this.
    """

    def _load_model(self) -> None:
        """Load model using HuggingFace Transformers."""
        if self.use_vllm:
            from vllm import LLM

            self._model = LLM(
                model=self.model_id,
                trust_remote_code=self.trust_remote_code,
                dtype=self.torch_dtype,
                max_model_len=self.config.get("max_model_len", 8192),
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.90),
                tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
            )
            self._tokenizer = self._model.get_tokenizer()
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": self._get_torch_dtype(),
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }

        # BUG-5 FIX: Properly check Flash Attention 2 availability
        try:
            import importlib
            if importlib.util.find_spec("flash_attn") is not None:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled")
            else:
                logger.info("Flash Attention 2 not available, using default attention")
        except Exception:
            logger.info("Flash Attention 2 check failed, using default attention")

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        self._model.eval()

    def _generate_impl(
        self, prompt: str, gen_config: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Generate using HuggingFace Transformers."""
        if self.use_vllm:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=gen_config["max_new_tokens"],
                temperature=max(gen_config["temperature"], 0.01),
                top_p=gen_config["top_p"],
                top_k=gen_config["top_k"],
                repetition_penalty=gen_config["repetition_penalty"],
            )

            outputs = self._model.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            token_count = len(outputs[0].outputs[0].token_ids)

            return {
                "text": generated_text,
                "tokens_generated": token_count,
            }

        import torch

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._get_max_context_length() - gen_config["max_new_tokens"],
            padding=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=gen_config["max_new_tokens"],
                temperature=max(gen_config["temperature"], 0.01),
                top_p=gen_config["top_p"],
                top_k=gen_config["top_k"],
                repetition_penalty=gen_config["repetition_penalty"],
                do_sample=gen_config["do_sample"],
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "text": text,
            "tokens_generated": len(generated_ids),
        }

    def _get_max_context_length(self) -> int:
        """Get maximum context length for this model."""
        if hasattr(self._model, "config"):
            for attr in ["max_position_embeddings", "n_positions", "max_seq_len"]:
                val = getattr(self._model.config, attr, None)
                if val:
                    return val
        return 4096


class VLLMEngine(BaseLLM):
    """vLLM-based wrapper for high-throughput inference.

    Uses vLLM's LLM class for optimized batched inference
    with PagedAttention and continuous batching.
    """

    def _load_model(self) -> None:
        """Load model using vLLM."""
        from vllm import LLM

        self._model = LLM(
            model=self.model_id,
            trust_remote_code=self.trust_remote_code,
            dtype=self.torch_dtype,
            max_model_len=self.config.get("max_model_len", 8192),
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.90),
            tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
        )
        self._tokenizer = self._model.get_tokenizer()

    def _generate_impl(
        self, prompt: str, gen_config: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Generate using vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=gen_config["max_new_tokens"],
            temperature=max(gen_config["temperature"], 0.01),
            top_p=gen_config["top_p"],
            top_k=gen_config["top_k"],
            repetition_penalty=gen_config["repetition_penalty"],
        )

        outputs = self._model.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        token_count = len(outputs[0].outputs[0].token_ids)

        return {
            "text": generated_text,
            "tokens_generated": token_count,
        }
