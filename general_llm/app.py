"""
General LLM Service — OpenAI-compatible API for the conversational backbone.

Serves a local open-source model (Qwen2.5-7B-Instruct by default) with an
OpenAI-compatible /v1/chat/completions endpoint. The platform gateway calls
this exactly like it would call OpenAI — zero code changes needed.

Supports:
    - CPU and GPU inference (auto-detects)
    - bfloat16/float16/float32 dtype selection
    - Concurrent request handling with thread pool
    - Health check endpoint
    - Configurable via environment variables
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ─────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.1"))
PORT = int(os.environ.get("PORT", "8004"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))

# ── Logging ──────────────────────────────────────────────────────────────

log_format = os.environ.get("LOG_FORMAT", "console")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s" if log_format == "console"
    else '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("general-llm")

# ── System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are MedAI, an advanced AI-powered medical assistant platform. You provide \
helpful, accurate, and empathetic responses to users.

Your capabilities include:
- Answering medical questions with evidence-based information
- Explaining medical conditions, symptoms, treatments, and medications
- Helping users understand their medical reports and test results
- Providing general health and wellness information
- Natural conversation (greetings, follow-ups, clarifications)

Guidelines:
- Be conversational and natural. Respond to any type of message appropriately.
- For medical questions, provide thorough, evidence-based answers.
- Always include appropriate disclaimers: you are an AI assistant and users should \
consult healthcare professionals for medical decisions.
- If you're unsure about something, say so honestly.
- Format responses with markdown for readability.
- Be concise but thorough.
- Never fabricate medical data, statistics, or citations.
- For serious symptoms or emergencies, advise seeking immediate medical attention.
"""

# ── Global State ─────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_device = None
_lock = Lock()
_executor: ThreadPoolExecutor | None = None


def _load_model():
    """Load model and tokenizer (called once at startup)."""
    global _model, _tokenizer, _device

    logger.info("Loading model: %s", MODEL_ID)
    start = time.monotonic()

    # Detect device
    if torch.cuda.is_available():
        _device = "cuda"
        dtype = torch.bfloat16
        device_map = "auto"
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        _device = "cpu"
        dtype = torch.float32
        device_map = "cpu"
        logger.info("No GPU detected — using CPU (inference will be slower)")

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Load model
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    _model.eval()

    elapsed = time.monotonic() - start
    logger.info("Model loaded in %.1fs [device=%s, dtype=%s]", elapsed, _device, dtype)


# ── Inference ────────────────────────────────────────────────────────────

def _generate_sync(messages: list[dict], max_tokens: int, temperature: float) -> dict:
    """Synchronous generation (runs in thread pool)."""
    with _lock:  # Serialize access to model (GPU memory safety)
        # Build prompt using the tokenizer's chat template
        if hasattr(_tokenizer, "apply_chat_template"):
            # Prepend system message
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            prompt_ids = _tokenizer.apply_chat_template(
                full_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            # Fallback for tokenizers without chat template
            prompt_parts = [f"System: {SYSTEM_PROMPT}\n"]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}\n")
                elif role == "user":
                    prompt_parts.append(f"User: {content}\n")
                else:
                    prompt_parts.append(f"Assistant: {content}\n")
            prompt_parts.append("Assistant: ")
            prompt_text = "".join(prompt_parts)
            prompt_ids = _tokenizer.encode(prompt_text, return_tensors="pt")

        prompt_ids = prompt_ids.to(_model.device)
        input_len = prompt_ids.shape[1]

        # Generate
        with torch.inference_mode():
            output_ids = _model.generate(
                prompt_ids,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),  # Avoid division by zero
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=temperature > 0,
                pad_token_id=_tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (skip prompt)
        generated_ids = output_ids[0][input_len:]
        answer = _tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "answer": answer.strip(),
            "prompt_tokens": input_len,
            "completion_tokens": len(generated_ids),
        }


# ── Request/Response Models (OpenAI-compatible) ─────────────────────────

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=MAX_NEW_TOKENS, le=8192)
    temperature: float = Field(default=TEMPERATURE, ge=0, le=2)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


# ── FastAPI App ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor
    _load_model()
    _executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    logger.info("General LLM service ready [port=%d, max_concurrent=%d]", PORT, MAX_CONCURRENT)
    yield
    if _executor:
        _executor.shutdown(wait=False)
    logger.info("General LLM service shut down")


app = FastAPI(
    title="MedAI General LLM",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(body: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if _model is None or _tokenizer is None:
        raise HTTPException(503, detail="Model not loaded yet")

    if body.stream:
        raise HTTPException(400, detail="Streaming not yet supported")

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    # Run inference in thread pool to not block the event loop
    import asyncio
    loop = asyncio.get_event_loop()
    start = time.monotonic()

    try:
        result = await loop.run_in_executor(
            _executor,
            _generate_sync,
            messages,
            body.max_tokens,
            body.temperature,
        )
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(500, detail=f"Generation failed: {str(e)[:200]}")

    latency = (time.monotonic() - start) * 1000
    logger.info(
        "Generated %d tokens in %.0fms (prompt=%d tokens)",
        result["completion_tokens"], latency, result["prompt_tokens"],
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=MODEL_ID,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=result["answer"]),
            ),
        ],
        usage=UsageInfo(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        ),
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy" if _model is not None else "loading",
        "model": MODEL_ID,
        "device": _device or "unknown",
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }
