#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  MediScan AI — Full Stack Launcher for JarvisLabs (NO DOCKER)
#  PyTorch template, 8 × A6000
#
#  API endpoint: https://c3cb303897851.notebooks.jarvislabs.net
#  SSH port:     389785
#
#  USAGE:
#    bash run_all.sh            # start everything
#    bash run_all.sh --stop     # kill all services
#    bash run_all.sh --status   # show what's running
#    bash run_all.sh --logs     # tail all logs
#
#  Services started (in order):
#    [1] redis       :6379  — session cache / rate limiting
#    [2] postgres    :5432  — user/auth/audit data
#    [3] qdrant      :6333  — vector DB for RAG
#    [4] neo4j       :7687  — patient longitudinal graph
#    [5] general-llm :8004  — Qwen2.5-7B backbone (GPU 7)
#    [6] mediscan-vlm:8001  — 16-model VLM engine  (GPU 0-7)
#    [7] medical-llm :8002  — MediX-R1 reasoning   (GPU 5,6)
#    [8] mediscan-ocr:8003  — Document OCR pipeline (CPU)
#    [9] granite     :8005  — Granite Vision 3B     (GPU 7)
#   [10] openrag     :8006  — Hybrid RAG service    (CPU)
#   [11] ctx-graph   :8007  — Neo4j patient graph   (CPU)
#   [12] ctx1-agent  :8008  — Multi-hop RAG agent   (CPU)
#   [13] platform    :8000  — API Gateway            (CPU)
#   [14] frontend    :6006  — React UI (nginx)       (public URL)
#
#  Access:
#    Frontend & API:  https://c3cb303897851.notebooks.jarvislabs.net
#    API Docs:        https://c3cb303897851.notebooks.jarvislabs.net/api/v1/docs
#    Alt port 8080:   https://c3cb303897851-8080.notebooks.jarvislabs.net
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # medical_ai_startup - Copy (2)/
VENV="$ROOT/mediscan_v70_sota_production/deploy/jarvislabs/../../../venv_medai"
LOG_DIR="/home/logs"
DATA_DIR="/home/data"
MODELS_DIR="/home/models"
HF_CACHE="/home/hf_cache"
SESSION="medai"   # tmux session name

mkdir -p "$LOG_DIR" "$DATA_DIR" "$MODELS_DIR"

# ── Colours ────────────────────────────────────────────────────────────
GRN='\033[0;32m'; YLW='\033[1;33m'; RED='\033[0;31m'; CYN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${GRN}[INFO]${NC} $*"; }
warn() { echo -e "${YLW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR] ${NC} $*"; }
hdr()  { echo -e "\n${CYN}══ $* ══${NC}"; }

# ── CLI flags ──────────────────────────────────────────────────────────
STOP=0; STATUS=0; LOGS=0
for arg in "$@"; do
    case "$arg" in
        --stop)   STOP=1 ;;
        --status) STATUS=1 ;;
        --logs)   LOGS=1 ;;
    esac
done

# ── Stop all ───────────────────────────────────────────────────────────
if [[ $STOP -eq 1 ]]; then
    info "Stopping all MediAI services…"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    pkill -f "uvicorn.*800" 2>/dev/null || true
    pkill -f "redis-server"  2>/dev/null || true
    pkill -f "postgres"      2>/dev/null || true
    pkill -f "qdrant"        2>/dev/null || true
    pkill -f "neo4j"         2>/dev/null || true
    info "All services stopped."
    exit 0
fi

# ── Status ─────────────────────────────────────────────────────────────
if [[ $STATUS -eq 1 ]]; then
    echo ""
    echo "══ MediAI Service Status ══"
    for port_name in "6379:Redis" "5432:PostgreSQL" "6333:Qdrant" "7687:Neo4j" \
                     "8000:Platform" "8001:MediScan-VLM" "8002:Medical-LLM" \
                     "8003:OCR" "8004:General-LLM" "8005:Granite-Vision" \
                     "8006:OpenRAG" "8007:Ctx-Graph" "8008:Ctx1-Agent" "6006:Frontend"; do
        port="${port_name%%:*}"; name="${port_name##*:}"
        if nc -z localhost "$port" 2>/dev/null; then
            echo -e "  ${GRN}✓${NC} $name :$port"
        else
            echo -e "  ${RED}✗${NC} $name :$port"
        fi
    done
    echo ""
    exit 0
fi

# ── Logs ───────────────────────────────────────────────────────────────
if [[ $LOGS -eq 1 ]]; then
    tail -f "$LOG_DIR"/*.log 2>/dev/null || echo "No logs yet."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════
#  PREREQUISITES CHECK
# ══════════════════════════════════════════════════════════════════════
hdr "Checking prerequisites"

command -v tmux   &>/dev/null || { apt-get install -y -qq tmux;   }
command -v nginx  &>/dev/null || { apt-get install -y -qq nginx;  }
command -v node   &>/dev/null || { apt-get install -y -qq nodejs npm; }
command -v psql   &>/dev/null || { apt-get install -y -qq postgresql postgresql-client; }
command -v redis-server &>/dev/null || { apt-get install -y -qq redis-server; }
command -v nc     &>/dev/null || { apt-get install -y -qq netcat-openbsd; }
command -v java   &>/dev/null || { apt-get install -y -qq openjdk-17-jre-headless; }

# Qdrant binary
if ! command -v qdrant &>/dev/null && [[ ! -f /usr/local/bin/qdrant ]]; then
    info "Downloading Qdrant…"
    curl -L "https://github.com/qdrant/qdrant/releases/download/v1.13.4/qdrant-x86_64-unknown-linux-musl.tar.gz" \
        | tar -xz -C /usr/local/bin/
fi

# ══════════════════════════════════════════════════════════════════════
#  ENVIRONMENT FILE
# ══════════════════════════════════════════════════════════════════════
hdr "Loading environment"

ENV_FILE="$ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    warn ".env not found — copying from .env.example"
    cp "$ROOT/.env.example" "$ENV_FILE"
fi

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

# Override docker service hostnames → localhost for native run
export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER:-medai}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB:-medai}"
export REDIS_URL="redis://:${REDIS_PASSWORD}@localhost:6379/0"
export MEDISCAN_VLM_URL="http://localhost:8001"
export MEDICAL_LLM_URL="http://localhost:8002"
export MEDISCAN_OCR_URL="http://localhost:8003"
export LLM_BASE_URL="http://localhost:8004"
export GRANITE_VLLM_URL="http://localhost:8005/v1"
export OPENRAG_URL="http://localhost:8006"
export CONTEXT_GRAPH_URL="http://localhost:8007"
export CONTEXT1_URL="http://localhost:8008"
export VECTOR_DB_URL="http://localhost:6333"
export OPENRAG_QDRANT_URL="http://localhost:6333"
export OPENRAG_OPENSEARCH_URL="http://localhost:9200"
export NEO4J_URI="bolt://localhost:7687"
export MEDICAL_LLM_REDIS_URL="redis://:${REDIS_PASSWORD}@localhost:6379/0"
export MEDICAL_LLM_QDRANT_URL="http://localhost:6333"

# HuggingFace — use 2TB /home disk
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export MODELS_DIR="$MODELS_DIR"
export HF_HUB_OFFLINE=0    # allow downloads on first run
export TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

# CORS — allow JarvisLabs public URL
export MEDISCAN_CORS_ORIGINS="https://c3cb303897851.notebooks.jarvislabs.net,http://localhost:3000,http://localhost:6006"
export CORS_ALLOW_ORIGINS="$MEDISCAN_CORS_ORIGINS"
export VITE_API_URL="/api/v1"

info "Environment loaded from $ENV_FILE"

# ══════════════════════════════════════════════════════════════════════
#  PYTHON VENV
# ══════════════════════════════════════════════════════════════════════
hdr "Python environment"
VENV="/home/venv"
if [[ ! -d "$VENV" ]]; then
    warn "venv not found — run setup.sh first!"
    exit 1
fi
source "$VENV/bin/activate"
info "venv active: $(python --version)"

# ══════════════════════════════════════════════════════════════════════
#  TMUX SESSION
# ══════════════════════════════════════════════════════════════════════
hdr "Starting tmux session: $SESSION"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Helper: open a new tmux window and run a command
new_win() {
    local name="$1"; local cmd="$2"
    tmux new-window -t "$SESSION" -n "$name"
    tmux send-keys -t "$SESSION:$name" "cd $ROOT && $cmd" Enter
}

# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 — INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════
hdr "Phase 1: Starting infrastructure"

# ── Redis (Valkey) ────────────────────────────────────────────────────
info "Starting Redis on :6379…"
new_win "redis" "redis-server \
    --requirepass '${REDIS_PASSWORD}' \
    --maxmemory 512mb \
    --maxmemory-policy allkeys-lru \
    --appendonly yes \
    --loglevel notice \
    2>&1 | tee $LOG_DIR/redis.log"
sleep 3

# ── PostgreSQL ─────────────────────────────────────────────────────────
info "Starting PostgreSQL on :5432…"
PG_DATA="/home/postgres_data"
if [[ ! -d "$PG_DATA" ]]; then
    mkdir -p "$PG_DATA"
    chown postgres:postgres "$PG_DATA" 2>/dev/null || true
    sudo -u postgres initdb -D "$PG_DATA" 2>/dev/null \
        || initdb -D "$PG_DATA" 2>/dev/null || true
fi
new_win "postgres" "sudo -u postgres postgres -D $PG_DATA \
    -c 'log_min_duration_statement=500' \
    -c 'max_connections=200' \
    2>&1 | tee $LOG_DIR/postgres.log"
sleep 5

# Create DB and user if not exist
sudo -u postgres psql -c "CREATE USER ${POSTGRES_USER:-medai} WITH PASSWORD '${POSTGRES_PASSWORD}';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE ${POSTGRES_DB:-medai} OWNER ${POSTGRES_USER:-medai};" 2>/dev/null || true
# Run platform migrations
(cd "$ROOT/platform" && python -m alembic upgrade head 2>&1 | tee "$LOG_DIR/migration.log") || warn "DB migration failed — check logs"

# ── Qdrant ─────────────────────────────────────────────────────────────
info "Starting Qdrant on :6333…"
mkdir -p /home/qdrant_storage
new_win "qdrant" "qdrant \
    --storage-path /home/qdrant_storage \
    2>&1 | tee $LOG_DIR/qdrant.log"
sleep 5

# ── Neo4j ──────────────────────────────────────────────────────────────
if command -v neo4j &>/dev/null || [[ -f /home/neo4j/bin/neo4j ]]; then
    info "Starting Neo4j on :7687…"
    NEO4J_HOME="${NEO4J_HOME:-/home/neo4j}"
    new_win "neo4j" "NEO4J_AUTH=${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD:-changeme} \
        $NEO4J_HOME/bin/neo4j console \
        2>&1 | tee $LOG_DIR/neo4j.log"
    sleep 10
else
    warn "Neo4j not found — skipping (context-graph will be unavailable)"
    warn "Install: bash $SCRIPT_DIR/install_neo4j.sh"
fi

info "Infrastructure ready. Waiting 3s…"
sleep 3

# ══════════════════════════════════════════════════════════════════════
#  PHASE 2 — AI SERVICES
# ══════════════════════════════════════════════════════════════════════
hdr "Phase 2: Starting AI services"

# GPU assignments (matches gpu_allocation.py)
# GPU 0,1 → hulu_med_32b
# GPU 2,3 → medgemma_27b / medix_r1_30b
# GPU 4,5 → hulu_med_14b / medix_r1_8b / hulu_med_7b
# GPU 6   → chexagent / radfm / medix_r1_2b
# GPU 7   → general-llm + pathgen + retfound + merlin + biomedclip

# ── General LLM (Qwen2.5-7B) — GPU 7 ─────────────────────────────────
info "Starting General LLM (Qwen2.5-7B) on :8004…"
new_win "general-llm" "CUDA_VISIBLE_DEVICES=7 \
    HF_HOME=$HF_CACHE \
    MODEL_ID=${GENERAL_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
    PORT=8004 \
    python $ROOT/general_llm/app.py \
    2>&1 | tee $LOG_DIR/general_llm.log"

# ── MediScan VLM Engine (16 models) — GPU 0-6 ─────────────────────────
info "Starting MediScan VLM on :8001 (GPU 0-6, pre-loading 16 models)…"
new_win "mediscan-vlm" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    HF_HOME=$HF_CACHE \
    MEDISCAN_MAX_RESIDENT_MODELS=0 \
    MEDISCAN_AUTO_UNLOAD_AFTER_INFERENCE=0 \
    MEDISCAN_SEQUENTIAL_HEAVY_MODELS=1 \
    python -m uvicorn mediscan_v70.api_server:app \
        --host 0.0.0.0 --port 8001 \
        --workers 1 --log-level info \
    2>&1 | tee $LOG_DIR/mediscan_vlm.log"

# ── Medical LLM (MediX-R1) — GPU 5,6 ─────────────────────────────────
info "Starting Medical LLM on :8002…"
new_win "medical-llm" "CUDA_VISIBLE_DEVICES=5,6 \
    HF_HOME=$HF_CACHE \
    MEDICAL_LLM_REDIS_URL=${MEDICAL_LLM_REDIS_URL} \
    MEDICAL_LLM_QDRANT_URL=${MEDICAL_LLM_QDRANT_URL} \
    PORT=8002 \
    python -m uvicorn medical_llm.app:app \
        --host 0.0.0.0 --port 8002 \
        --workers 1 --log-level info \
    2>&1 | tee $LOG_DIR/medical_llm.log"

# ── MediScan OCR — CPU ─────────────────────────────────────────────────
info "Starting MediScan OCR on :8003…"
new_win "mediscan-ocr" "SARVAM_API_KEY=${SARVAM_API_KEY:-} \
    MEDISCAN_CORS_ORIGINS='${MEDISCAN_CORS_ORIGINS}' \
    python -m uvicorn medicscan_ocr.app:app \
        --host 0.0.0.0 --port 8003 \
        --workers 2 --log-level info \
    2>&1 | tee $LOG_DIR/mediscan_ocr.log"

# ── Granite Vision 3B — GPU 7 ─────────────────────────────────────────
info "Starting Granite Vision sidecar on :8005…"
new_win "granite" "CUDA_VISIBLE_DEVICES=7 \
    HF_HOME=$HF_CACHE \
    GRANITE_ENABLED=${GRANITE_ENABLED:-true} \
    python $ROOT/granite_vision_sidecar/start_server.py \
        --port 8005 \
    2>&1 | tee $LOG_DIR/granite.log"

# ── OpenRAG Service ────────────────────────────────────────────────────
info "Starting OpenRAG on :8006…"
new_win "openrag" "OPENRAG_QDRANT_URL=http://localhost:6333 \
    OPENRAG_OPENSEARCH_URL=http://localhost:9200 \
    OPENRAG_LLM_URL=http://localhost:8004 \
    GRANITE_VLLM_URL=http://localhost:8005/v1 \
    HF_HOME=$HF_CACHE \
    python -m uvicorn openrag_service.app:app \
        --host 0.0.0.0 --port 8006 \
        --workers 1 --log-level info \
    2>&1 | tee $LOG_DIR/openrag.log"

# ── Context Graph Service ──────────────────────────────────────────────
info "Starting Context Graph on :8007…"
new_win "ctx-graph" "NEO4J_URI=bolt://localhost:7687 \
    NEO4J_USER=${NEO4J_USER:-neo4j} \
    NEO4J_PASSWORD=${NEO4J_PASSWORD:-changeme} \
    python -m uvicorn context_graph_service.app:app \
        --host 0.0.0.0 --port 8007 \
        --workers 1 --log-level info \
    2>&1 | tee $LOG_DIR/ctx_graph.log"

# ── Context1 Agent ─────────────────────────────────────────────────────
info "Starting Context1 Agent on :8008…"
new_win "ctx1-agent" "CONTEXT1_OPENRAG_URL=http://localhost:8006 \
    CONTEXT1_LLM_URL=http://localhost:8004 \
    CONTEXT1_LLM_MODEL=${GENERAL_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct} \
    python -m uvicorn context1_agent.app:app \
        --host 0.0.0.0 --port 8008 \
        --workers 1 --log-level info \
    2>&1 | tee $LOG_DIR/ctx1_agent.log"

info "AI services launched. Waiting 15s for General LLM to start…"
sleep 15

# ══════════════════════════════════════════════════════════════════════
#  PHASE 3 — PLATFORM GATEWAY
# ══════════════════════════════════════════════════════════════════════
hdr "Phase 3: Starting Platform Gateway"

info "Starting Platform API Gateway on :8000…"
new_win "platform" "DATABASE_URL='${DATABASE_URL}' \
    REDIS_URL='${REDIS_URL}' \
    JWT_SECRET_KEY='${JWT_SECRET_KEY}' \
    ENCRYPTION_KEY='${ENCRYPTION_KEY}' \
    MEDISCAN_VLM_URL=http://localhost:8001 \
    MEDICAL_LLM_URL=http://localhost:8002 \
    MEDISCAN_OCR_URL=http://localhost:8003 \
    LLM_BASE_URL=http://localhost:8004 \
    GRANITE_VLLM_URL=http://localhost:8005/v1 \
    OPENRAG_URL=http://localhost:8006 \
    CONTEXT_GRAPH_URL=http://localhost:8007 \
    CONTEXT1_URL=http://localhost:8008 \
    VECTOR_DB_URL=http://localhost:6333 \
    HF_HOME=$HF_CACHE \
    UVICORN_WORKERS=1 \
    python -m uvicorn platform.main:app \
        --host 0.0.0.0 --port 8000 \
        --workers 1 --log-level info \
    2>&1 | tee $LOG_DIR/platform.log"

sleep 8

# ══════════════════════════════════════════════════════════════════════
#  PHASE 4 — FRONTEND + NGINX
# ══════════════════════════════════════════════════════════════════════
hdr "Phase 4: Building frontend and starting nginx"

# Build React frontend
FRONTEND_DIR="$ROOT/frontend"
if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    info "Installing frontend dependencies…"
    (cd "$FRONTEND_DIR" && npm install --silent)
fi

info "Building React frontend…"
(cd "$FRONTEND_DIR" && VITE_API_URL=/api/v1 npm run build) \
    && info "Frontend built." \
    || warn "Frontend build failed — check $LOG_DIR/frontend_build.log"

# ─────────────────────────────────────────────────────────────────────
# API 1 (port 6006) → https://c3cb303897851.notebooks.jarvislabs.net
#   React Frontend + proxies /api/v1 to platform:8000
#
# API 2 (port 8080) → https://c3cb303897852.notebooks.jarvislabs.net
#   Platform FastAPI directly — raw REST + Swagger at /docs
# ─────────────────────────────────────────────────────────────────────
NGINX_CONF="/etc/nginx/sites-available/medai"
cat > "$NGINX_CONF" <<'NGINX'

# ── API 1: React Frontend (port 6006) ────────────────────────────────
server {
    listen 6006;
    server_name _;

    root /home/medai_frontend;
    index index.html;

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Frontend → Platform gateway (internal)
    location /api/v1/ {
        proxy_pass         http://127.0.0.1:8000/api/v1/;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Upgrade           $http_upgrade;
        proxy_set_header   Connection        "upgrade";
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        client_max_body_size 500M;
    }

    # WebSocket streaming
    location /ws/ {
        proxy_pass         http://127.0.0.1:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_read_timeout 3600s;
    }
}

# ── API 2: Raw Platform API + Swagger Docs (port 8080) ───────────────
server {
    listen 8080;
    server_name _;

    # Full platform API with no frontend — direct REST + Swagger UI
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Upgrade           $http_upgrade;
        proxy_set_header   Connection        "upgrade";
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        client_max_body_size 500M;
    }
}
NGINX

# Copy built frontend to nginx root
mkdir -p /home/medai_frontend
cp -r "$FRONTEND_DIR/dist/." /home/medai_frontend/

# Enable site and restart nginx
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/medai
rm -f /etc/nginx/sites-enabled/default
nginx -t && nginx -s reload 2>/dev/null || systemctl restart nginx
info "nginx serving frontend on :6006"

# ══════════════════════════════════════════════════════════════════════
#  STARTUP COMPLETE
# ══════════════════════════════════════════════════════════════════════

echo ""
echo -e "${GRN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GRN}║  MediScan AI — All Services Running                              ║${NC}"
echo -e "${GRN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GRN}║  Frontend + API  https://c3cb303897851.notebooks.jarvislabs.net  ║${NC}"
echo -e "${GRN}║  API Docs        .../api/v1/docs                                 ║${NC}"
echo -e "${GRN}║  Platform        http://localhost:8000                            ║${NC}"
echo -e "${GRN}║  MediScan VLM    http://localhost:8001  (loading ~10 min)         ║${NC}"
echo -e "${GRN}║  Medical LLM     http://localhost:8002                            ║${NC}"
echo -e "${GRN}║  OCR             http://localhost:8003                            ║${NC}"
echo -e "${GRN}║  General LLM     http://localhost:8004                            ║${NC}"
echo -e "${GRN}║  Granite Vision  http://localhost:8005                            ║${NC}"
echo -e "${GRN}║  OpenRAG         http://localhost:8006                            ║${NC}"
echo -e "${GRN}║  Context Graph   http://localhost:8007                            ║${NC}"
echo -e "${GRN}║  Redis           localhost:6379                                   ║${NC}"
echo -e "${GRN}║  Qdrant          http://localhost:6333                            ║${NC}"
echo -e "${GRN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GRN}║  tmux attach:  tmux attach -t medai                               ║${NC}"
echo -e "${GRN}║  Check status: bash run_all.sh --status                           ║${NC}"
echo -e "${GRN}║  Stop all:     bash run_all.sh --stop                             ║${NC}"
echo -e "${GRN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "MediScan VLM (16 models) loads in background — takes 10-30 min on first run."
info "Use 'tmux attach -t medai' then Ctrl+B then W to switch between service windows."
