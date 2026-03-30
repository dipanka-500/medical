#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/deployment/jarvislabs/jarvislabs.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

DATA_ROOT="${DATA_ROOT:-/home/medai-stack}"
RUN_DIR="$DATA_ROOT/run"
PG_RUNTIME_USER="${PG_RUNTIME_USER:-postgres}"

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

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

run_pg_cmd() {
  if [[ "$(id -u)" -eq 0 ]] && id "$PG_RUNTIME_USER" >/dev/null 2>&1; then
    runuser -u "$PG_RUNTIME_USER" -- "$@"
    return
  fi
  "$@"
}

for name in platform ocr mediscan medical-llm general-llm; do
  stop_pid_file "$RUN_DIR/${name}.pid"
done

for cache_cli in valkey-cli redis-cli; do
  if command -v "$cache_cli" >/dev/null 2>&1; then
    "$cache_cli" -p "${REDIS_PORT:-16379}" shutdown >/dev/null 2>&1 || true
    break
  fi
done

PG_CTL_BIN="$(resolve_pg_bin pg_ctl)"
if [[ -n "$PG_CTL_BIN" ]]; then
  run_pg_cmd "$PG_CTL_BIN" -D "$DATA_ROOT/postgres" stop >/dev/null 2>&1 || true
fi

echo "MedAI stack stopped."
