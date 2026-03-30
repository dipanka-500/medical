# Medical LLM Engine

Research-oriented multi-model medical LLM pipeline with:

- layered query analysis and smart routing
- retrieval-augmented generation over local, PubMed, and web evidence
- multi-model fusion, safety checks, and audit logging
- production API plus CLI and container deployment support

## Quick Start

```bash
python -m pip install -r requirements.txt
python main.py --interactive
```

## API

Run the production API locally:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Optional environment variables:

- `MEDICAL_LLM_API_KEY` to require `X-API-Key` on protected endpoints
- `MEDICAL_LLM_INIT_ON_STARTUP=true|false` to control eager initialization
- `MEDICAL_LLM_INGEST_BUILTIN=true|false` to preload built-in knowledge
- `MEDICAL_LLM_MAX_CONCURRENT_REQUESTS=1` to cap concurrent requests per process
- `MEDICAL_LLM_QUEUE_TIMEOUT_SECONDS=30` to limit how long requests wait for a worker slot
- `MEDICAL_LLM_MAX_QUEUE_DEPTH=4` to shed load before the queue grows unbounded
- `MEDICAL_LLM_REDIS_URL=redis://...` to enable shared cache and conversation state across replicas
- `MEDICAL_LLM_SHARED_STATE_ENABLED=true|false` to turn shared cache/session state on or off
- `MEDICAL_LLM_DISTRIBUTED_MAX_CONCURRENT_REQUESTS=2` to enforce a cluster-wide in-flight cap across replicas
- `MEDICAL_LLM_CACHE_TTL_SECONDS=300` and `MEDICAL_LLM_SESSION_TTL_SECONDS=3600` to tune shared-state retention

Example request:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What are the side effects of metformin?\",\"mode\":\"patient\"}"
```

## Testing

```bash
python -m pytest
```

## Notes

- This project is intended for research and decision support workflows.
- It must not be used as the sole basis for clinical decisions.
