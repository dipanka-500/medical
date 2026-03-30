# JarvisLabs Launch Plan

This deployment is tuned for a JarvisLabs template instance with `8 x RTX A6000 48GB` and `2TB` persistent storage.

The defaults intentionally keep persistent data under `/home` and use non-default local DB ports to avoid collisions with package-managed services.

## Port layout

- `6006`: public MedAI platform entrypoint on JarvisLabs
- `8001`: `mediscan_v70` internal API
- `8002`: `medical_llm` internal API
- `8003`: `medicscan_ocr` internal API
- `8004`: `vLLM` OpenAI-compatible general LLM

Only `6006` needs to be exposed publicly. The platform calls the other services on `127.0.0.1`.

## GPU layout

- GPU `0`: always-on `vLLM` server for the general conversational model
- GPUs `1,2,3,4`: `medical_llm` with vLLM-backed heavy text models
- GPUs `5,6,7`: `mediscan_v70`
- OCR stays CPU-first unless its selected backend needs GPU

The Jarvis profile defaults the vLLM text stack to `float16`, which is the safer default for RTX A6000 instances than carrying over an A100-style `bfloat16` assumption.

## What changed in code

- `medical_llm` now supports LRU-style model residency, optional auto-unload after each call, and sequential execution for heavy models.
- `mediscan_v70` now supports the same pattern for its heavier vision models.
- `medical_llm/config/model_config.jarvislabs_a6000x8.yaml` enables vLLM for the text-model stack with 4-way tensor parallelism for 70B models.

## First-time setup

1. Install host packages if the template image does not already have them:

```bash
bash deployment/jarvislabs/bootstrap_host.sh
```

2. Create the env file:

```bash
cp deployment/jarvislabs/jarvislabs.env.example deployment/jarvislabs/jarvislabs.env
```

The example uses `ENVIRONMENT=staging` on purpose so the platform runs without uvicorn auto-reload.

3. Build the Python environments:

```bash
bash deployment/jarvislabs/bootstrap_envs.sh
```

If `python3.12` is not present in the template, the bootstrap script now falls back automatically to `python3.11`, `python3.10`, or `python3`.

4. Download the models you actually plan to serve into `HF_HOME` / `MODELS_DIR`.

At minimum you should cache:

- `Qwen/Qwen2.5-7B-Instruct`
- `BAAI/bge-large-en-v1.5`
- the `medical_llm` models you keep enabled
- the `mediscan_v70` models you want routed
- OCR backends you want local instead of remote

5. Start the full stack:

```bash
bash deployment/jarvislabs/launch_stack.sh
```

6. Stop it later with:

```bash
bash deployment/jarvislabs/stop_stack.sh
```

## Notes

- The launcher uses local PostgreSQL and Redis, because JarvisLabs templates do not allow nested Docker.
- The platform is configured to use `faiss` locally, so Qdrant is not required for this JarvisLabs path.
- `medical_llm` still needs its spaCy / SciSpaCy model assets if you want the NER layer fully active.
- The OCR bootstrap now installs `transformers`, `accelerate`, and `safetensors` so the local FireRed backend is actually runnable after environment setup.
- OCR now skips backends that are registered but not actually runnable in the current environment, and its health endpoint reports real backend availability instead of just registration.
- Remote OCR preference defaults to off unless `SARVAM_API_KEY` is present; set `MEDISCAN_PREFER_REMOTE_API=true` if you want Sarvam prioritized.
- The launcher now waits for true readiness, not just open ports: `medical_llm` must report `/ready`, `mediscan` must report healthy status, OCR must expose at least one ready backend, and the platform must reach overall `healthy`.
- If you expose extra JarvisLabs ports, you can move the platform back to `8000`, but `6006` is the default public endpoint in the provided docs.
