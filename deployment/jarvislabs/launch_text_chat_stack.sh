#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/deployment/jarvislabs/jarvislabs.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

PLATFORM_PORT="${TEXT_STACK_PLATFORM_PORT:-8000}"
FRONTEND_PORT="${TEXT_STACK_FRONTEND_PORT:-851}"
GENERAL_LLM_PORT="${GENERAL_LLM_PORT:-8004}"
MEDICAL_LLM_PORT="${MEDICAL_LLM_PORT:-8002}"
POSTGRES_PORT="${POSTGRES_PORT:-15432}"
POSTGRES_USER="${POSTGRES_USER:-medai}"
POSTGRES_DB="${POSTGRES_DB:-medai}"
REDIS_PORT="${REDIS_PORT:-16379}"
DATA_ROOT="${DATA_ROOT:-/home/medai-stack}"
MODELS_DIR="${MODELS_DIR:-/home/medai-models}"
HF_HOME="${HF_HOME:-$MODELS_DIR}"
PG_RUNTIME_USER="${PG_RUNTIME_USER:-postgres}"

RUN_DIR="$DATA_ROOT/run"
LOG_DIR="$DATA_ROOT/logs"
PGDATA="$DATA_ROOT/postgres"
REDIS_DATA="$DATA_ROOT/redis"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$PGDATA" "$REDIS_DATA" "$MODELS_DIR" "$DATA_ROOT/vector_store"

gen_secret() {
  python3 - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
}

JWT_SECRET_KEY="${JWT_SECRET_KEY:-$(gen_secret)}"
ENCRYPTION_KEY="${ENCRYPTION_KEY:-$(gen_secret)}"
DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://${POSTGRES_USER}@127.0.0.1:${POSTGRES_PORT}/${POSTGRES_DB}}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:${REDIS_PORT}/0}"

resolve_pg_bin() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return
  fi
  local candidate
  for candidate in /usr/lib/postgresql/*/bin/"$name"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo ""
}

resolve_cache_bin() {
  local candidate
  for candidate in "$@"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done
  echo ""
}

run_pg_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    if ! id "$PG_RUNTIME_USER" >/dev/null 2>&1; then
      echo "Missing PostgreSQL runtime user: $PG_RUNTIME_USER" >&2
      exit 1
    fi
    runuser -u "$PG_RUNTIME_USER" -- "$@"
    return
  fi
  "$@"
}

INITDB_BIN="$(resolve_pg_bin initdb)"
PG_CTL_BIN="$(resolve_pg_bin pg_ctl)"
PG_ISREADY_BIN="$(resolve_pg_bin pg_isready)"
CREATEDB_BIN="$(resolve_pg_bin createdb)"
CACHE_SERVER_BIN="$(resolve_cache_bin valkey-server redis-server)"
CACHE_CLI_BIN="$(resolve_cache_bin valkey-cli redis-cli)"

start_postgres() {
  if [[ -z "$INITDB_BIN" || -z "$PG_CTL_BIN" || -z "$PG_ISREADY_BIN" || -z "$CREATEDB_BIN" ]]; then
    echo "PostgreSQL binaries not found. Run deployment/jarvislabs/bootstrap_host.sh first." >&2
    exit 1
  fi

  if [[ "$(id -u)" -eq 0 ]]; then
    install -d -m 0755 -o "$PG_RUNTIME_USER" -g "$PG_RUNTIME_USER" "$PGDATA"
    install -d -m 0755 -o "$PG_RUNTIME_USER" -g "$PG_RUNTIME_USER" "$LOG_DIR"
    touch "$LOG_DIR/postgres.log"
    chown "$PG_RUNTIME_USER:$PG_RUNTIME_USER" "$LOG_DIR/postgres.log"
  fi

  if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
    run_pg_cmd "$INITDB_BIN" -D "$PGDATA" -A trust -U "$POSTGRES_USER" >/dev/null
  fi

  if [[ -f "$RUN_DIR/postgres.pid" ]] && kill -0 "$(cat "$RUN_DIR/postgres.pid")" 2>/dev/null; then
    return
  fi

  run_pg_cmd "$PG_CTL_BIN" -D "$PGDATA" -l "$LOG_DIR/postgres.log" -o "-p ${POSTGRES_PORT}" start >/dev/null
  echo "$(head -n 1 "$PGDATA/postmaster.pid")" > "$RUN_DIR/postgres.pid"

  for _ in $(seq 1 60); do
    if "$PG_ISREADY_BIN" -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$POSTGRES_USER" >/dev/null 2>&1; then
      "$CREATEDB_BIN" -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$POSTGRES_USER" "$POSTGRES_DB" >/dev/null 2>&1 || true
      return
    fi
    sleep 1
  done

  echo "PostgreSQL did not become ready." >&2
  exit 1
}

start_cache() {
  if [[ -z "$CACHE_SERVER_BIN" ]]; then
    echo "valkey-server/redis-server not found. Run deployment/jarvislabs/bootstrap_host.sh first." >&2
    exit 1
  fi

  if [[ -f "$RUN_DIR/redis.pid" ]] && kill -0 "$(cat "$RUN_DIR/redis.pid")" 2>/dev/null; then
    return
  fi

  "$CACHE_SERVER_BIN" \
    --bind 127.0.0.1 \
    --port "$REDIS_PORT" \
    --dir "$REDIS_DATA" \
    --save "" \
    --appendonly no \
    --daemonize yes \
    --pidfile "$RUN_DIR/redis.pid" \
    --logfile "$LOG_DIR/redis.log"

  for _ in $(seq 1 30); do
    if [[ -n "$CACHE_CLI_BIN" ]] && "$CACHE_CLI_BIN" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done

  echo "Cache server did not become ready." >&2
  exit 1
}

require_venv() {
  local dir="$1"
  if [[ ! -x "$dir/.venv/bin/python" ]]; then
    echo "Missing virtualenv in $dir. Run deployment/jarvislabs/bootstrap_envs.sh first." >&2
    exit 1
  fi
}

start_process() {
  local name="$1"
  local workdir="$2"
  local logfile="$3"
  shift 3
  (
    cd "$workdir"
    nohup "$@" >"$logfile" 2>&1 &
    echo $! > "$RUN_DIR/${name}.pid"
  )
}

wait_for_http() {
  local name="$1"
  local url="$2"
  for _ in $(seq 1 600); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "Timed out waiting for ${name} via ${url}" >&2
  exit 1
}

ensure_platform_schema() {
  (
    cd "$ROOT/platform"
    env \
      ENVIRONMENT=development \
      DATABASE_URL="$DATABASE_URL" \
      "$ROOT/platform/.venv/bin/python" - <<'PY'
import asyncio
from db.session import ensure_schema
asyncio.run(ensure_schema())
PY
  )
}

require_venv "$ROOT/platform"
require_venv "$ROOT/medical_llm"
require_venv "$ROOT/general_llm"

start_postgres
start_cache
ensure_platform_schema

if [[ -f "$RUN_DIR/general-llm.pid" ]] && kill -0 "$(cat "$RUN_DIR/general-llm.pid")" 2>/dev/null; then
  :
else
  start_process "general-llm" "$ROOT/general_llm" "$LOG_DIR/general-llm.log" \
    env \
      HF_HOME="$HF_HOME" \
      TRANSFORMERS_CACHE="$MODELS_DIR" \
      CUDA_VISIBLE_DEVICES="${GENERAL_LLM_CUDA_DEVICES:-0}" \
      "$ROOT/general_llm/.venv/bin/vllm" serve "${GENERAL_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
      --host 127.0.0.1 \
      --port "$GENERAL_LLM_PORT" \
      --dtype "${GENERAL_LLM_DTYPE:-float16}" \
      --gpu-memory-utilization "${GENERAL_LLM_GPU_MEMORY_UTILIZATION:-0.90}" \
      --max-model-len "${GENERAL_LLM_MAX_MODEL_LEN:-8192}"
fi
wait_for_http "general-llm" "http://127.0.0.1:${GENERAL_LLM_PORT}/v1/models"

if [[ -f "$RUN_DIR/medical-llm.pid" ]] && kill -0 "$(cat "$RUN_DIR/medical-llm.pid")" 2>/dev/null; then
  :
else
  start_process "medical-llm" "$ROOT/medical_llm" "$LOG_DIR/medical-llm.log" \
    env \
      ENVIRONMENT="${ENVIRONMENT:-staging}" \
      HF_HOME="$HF_HOME" \
      TRANSFORMERS_CACHE="$MODELS_DIR" \
      CUDA_VISIBLE_DEVICES="${MEDICAL_LLM_CUDA_DEVICES:-1}" \
      MEDICAL_LLM_MODEL_CONFIG="$ROOT/medical_llm/config/model_config.jarvislabs_text.yaml" \
      MEDICAL_LLM_PIPELINE_CONFIG="$ROOT/medical_llm/config/pipeline_config.jarvislabs_text.yaml" \
      MEDICAL_LLM_REDIS_URL="$REDIS_URL" \
      MEDICAL_LLM_VECTOR_STORE="faiss" \
      MEDICAL_LLM_INIT_ON_STARTUP="false" \
      MEDICAL_LLM_INGEST_BUILTIN="false" \
      MEDICAL_LLM_MAX_RESIDENT_MODELS="1" \
      MEDICAL_LLM_AUTO_UNLOAD_AFTER_REQUEST="false" \
      MEDICAL_LLM_SEQUENTIAL_HEAVY_MODELS="false" \
      MEDICAL_LLM_MAX_CONCURRENT_REQUESTS="1" \
      MEDICAL_LLM_MAX_QUEUE_DEPTH="2" \
      "$ROOT/medical_llm/.venv/bin/python" -m uvicorn app:app \
      --host 127.0.0.1 \
      --port "$MEDICAL_LLM_PORT"
fi
wait_for_http "medical-llm" "http://127.0.0.1:${MEDICAL_LLM_PORT}/health"

if [[ -f "$RUN_DIR/platform.pid" ]] && kill -0 "$(cat "$RUN_DIR/platform.pid")" 2>/dev/null; then
  :
else
  start_process "platform" "$ROOT/platform" "$LOG_DIR/platform.log" \
    env \
      ENVIRONMENT="${ENVIRONMENT:-staging}" \
      PORT="$PLATFORM_PORT" \
      WORKERS="${WORKERS:-1}" \
      DATABASE_URL="$DATABASE_URL" \
      REDIS_URL="$REDIS_URL" \
      JWT_SECRET_KEY="$JWT_SECRET_KEY" \
      ENCRYPTION_KEY="$ENCRYPTION_KEY" \
      MEDICAL_LLM_URL="http://127.0.0.1:${MEDICAL_LLM_PORT}" \
      MEDISCAN_VLM_URL="http://127.0.0.1:8001" \
      MEDISCAN_OCR_URL="http://127.0.0.1:8003" \
      LLM_PROVIDER="openai_compatible" \
      LLM_API_KEY="local-no-key-needed" \
      LLM_MODEL="${GENERAL_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
      LLM_BASE_URL="http://127.0.0.1:${GENERAL_LLM_PORT}" \
      VECTOR_DB_TYPE="faiss" \
      VECTOR_DB_PATH="$DATA_ROOT/vector_store" \
      ENABLE_SEARCH_RAG="false" \
      ENABLE_GRANITE_VISION="false" \
      ENABLE_OPENRAG="false" \
      OPENRAG_URL="${OPENRAG_URL:-http://127.0.0.1:8006}" \
      ENABLE_CONTEXT_GRAPH="false" \
      CONTEXT_GRAPH_URL="${CONTEXT_GRAPH_URL:-http://127.0.0.1:8007}" \
      ENABLE_CONTEXT1_AGENT="false" \
      CONTEXT1_URL="${CONTEXT1_URL:-http://127.0.0.1:8008}" \
      VOICE_ASR_PROVIDER="disabled" \
      VOICE_TTS_PROVIDER="disabled" \
      "$ROOT/platform/.venv/bin/python" main.py
fi
wait_for_http "platform" "http://127.0.0.1:${PLATFORM_PORT}/api/v1/health/ready"

echo "Minimal text chat backend is running."
echo "Platform API: http://127.0.0.1:${PLATFORM_PORT}"
echo "General LLM:  http://127.0.0.1:${GENERAL_LLM_PORT}"
echo "Medical LLM:  http://127.0.0.1:${MEDICAL_LLM_PORT}"
echo "Frontend URL after starting Vite on port ${FRONTEND_PORT}: https://c3cb303897${FRONTEND_PORT}.notebooks.jarvislabs.net"
echo "Logs: $LOG_DIR"
