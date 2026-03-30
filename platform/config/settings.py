"""
Central configuration for the MedAI Platform.

Production-grade features:
    - Pydantic-settings for environment variable loading
    - Strict validation of secrets at startup
    - HIPAA-compliance settings (session timeouts, PHI audit levels)
    - Observability configuration (Prometheus, OTLP, structured logging)
    - Feature flags for gradual rollout
    - Connection pool tuning for PostgreSQL and Redis
    - Computed properties for derived settings
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_INSECURE_DEFAULTS = {
    "CHANGE-ME-IN-PRODUCTION-use-openssl-rand-hex-64",
    "CHANGE-ME-32-byte-key-for-aes256",
    "change-me-in-production",
}


class Settings(BaseSettings):
    """All platform configuration — loaded from env vars or .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── App ──────────────────────────────────────────────────────────
    app_name: str = "MedAI Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # ── Server ───────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── Database ─────────────────────────────────────────────────────
    # Default uses Docker Compose service name; override for local dev
    # NOTE: Override via DATABASE_URL env var in all environments.
    # This default uses Docker Compose service name and requires POSTGRES_PASSWORD to be set.
    database_url: str = "postgresql+asyncpg://medai:CHANGE_ME@postgres:5432/medai"
    database_echo: bool = False
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_pool_recycle: int = 3600         # Recycle connections hourly
    database_pool_timeout: int = 30           # Wait 30s for a connection
    database_statement_timeout_ms: int = 30000  # 30s query timeout

    # ── Redis / Valkey (rate-limiting, caching, sessions) ────────────
    redis_url: str = "redis://redis:6379/0"
    redis_max_connections: int = 50
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0
    redis_retry_on_timeout: bool = True

    # ── JWT / Auth ───────────────────────────────────────────────────
    jwt_secret_key: str = "CHANGE-ME-IN-PRODUCTION-use-openssl-rand-hex-64"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    require_email_verification: bool = False

    # ── Encryption (AES-256 for patient data) ────────────────────────
    encryption_key: str = "CHANGE-ME-32-byte-key-for-aes256"
    encryption_key_version: int = 1           # Increment on key rotation

    # ── Rate Limiting ────────────────────────────────────────────────
    rate_limit_default: str = "60/minute"
    rate_limit_auth: str = "10/minute"
    rate_limit_ai: str = "30/minute"
    rate_limit_register: str = "5/hour"
    rate_limit_upload: str = "20/hour"
    rate_limit_ip_blocklist_threshold: int = 100  # Auto-block after N violations

    # ── Concurrency / Backpressure ──────────────────────────────────
    max_concurrent_ai_queries: int = 10
    max_active_requests: int = 200
    ai_queue_timeout_seconds: float = 30.0

    # ── Free tier ────────────────────────────────────────────────────
    free_tier_daily_queries: int = 5

    # ── File Uploads ─────────────────────────────────────────────────
    max_upload_size_mb: int = 50
    max_voice_upload_size_mb: int = 25
    allowed_file_types: list[str] = [
        "application/pdf", "image/jpeg", "image/png", "image/tiff",
        "application/dicom",
    ]

    # ── Voice (optional open-source speech stack) ────────────────────
    voice_model_cache_dir: str = ""
    voice_asr_provider: Literal["disabled", "faster_whisper"] = "disabled"
    voice_asr_model: str = "large-v3"
    voice_asr_device: Literal["auto", "cpu", "cuda"] = "auto"
    voice_asr_compute_type: str = "int8"
    voice_asr_beam_size: int = 5
    voice_tts_provider: Literal["disabled", "coqui"] = "disabled"
    voice_tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    voice_tts_device: Literal["auto", "cpu", "cuda"] = "auto"
    voice_tts_language: str = "en"
    voice_tts_speaker: str = ""
    voice_tts_speaker_wav: str = ""

    # ── Engine URLs (internal service addresses) ─────────────────────
    # Defaults use Docker Compose service names; override with env vars
    # for local dev (e.g. MEDISCAN_VLM_URL=http://localhost:8001)
    mediscan_vlm_url: str = "http://mediscan-vlm:8001"
    medical_llm_url: str = "http://medical-llm:8002"
    mediscan_ocr_url: str = "http://mediscan-ocr:8003"
    granite_vllm_url: str = "http://granite-vision:8005/v1"
    openrag_url: str = "http://openrag-service:8006"
    context_graph_url: str = "http://context-graph:8007"
    context1_url: str = "http://context1-agent:8008"
    engine_timeout_seconds: float = 120.0     # Per-engine call timeout
    engine_retry_max_attempts: int = 2        # Retry on transient failure
    engine_retry_backoff_seconds: float = 1.0 # Backoff between retries

    # ── Vector DB ────────────────────────────────────────────────────
    vector_db_type: Literal["faiss", "qdrant", "weaviate", "chroma"] = "qdrant"
    vector_db_url: str = "http://qdrant:6333"
    vector_db_path: str = "./data/vector_store"
    search_cache_ttl_seconds: int = 600
    searxng_url: str = ""
    trusted_medical_domains: list[str] = [
        "pubmed.ncbi.nlm.nih.gov",
        "who.int",
        "nih.gov",
        "cdc.gov",
        "fda.gov",
        "nice.org.uk",
        "cochranelibrary.com",
        "bmj.com",
        "thelancet.com",
        "nejm.org",
    ]

    # ── General LLM (conversational backbone — local open-source) ─────
    # Default: points to local general-llm service (Qwen2.5-7B-Instruct)
    # Also supports external: "anthropic" (Claude), "openai", "openai_compatible"
    llm_provider: Literal["anthropic", "openai", "openai_compatible"] = "openai_compatible"
    llm_api_key: str = "local-no-key-needed"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_base_url: str = "http://general-llm:8004"  # Local service (override for external APIs)
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7

    # ── External APIs ────────────────────────────────────────────────
    pubmed_api_key: str = ""
    sarvam_api_key: str = ""

    # ── Email (verification, 2FA) ────────────────────────────────────
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "noreply@medai.health"
    smtp_use_tls: bool = True

    # ── Temp file storage ────────────────────────────────────────────
    upload_temp_dir: str = ""  # empty = system default

    # ── HIPAA / Compliance ───────────────────────────────────────────
    session_idle_timeout_minutes: int = 30    # Auto-logout after inactivity
    require_consent_for_ai: bool = True       # Require patient consent before AI analysis
    data_retention_days: int = 2555           # ~7 years (HIPAA minimum)
    phi_audit_level: Literal["basic", "detailed", "full"] = "detailed"
    require_2fa_for_doctors: bool = True      # Enforce 2FA for medical staff
    max_login_attempts: int = 5               # Account lockout threshold
    lockout_duration_minutes: int = 15        # Lockout duration
    password_history_count: int = 5           # Prevent password reuse

    # ── Observability ────────────────────────────────────────────────
    log_format: Literal["json", "console"] = "json"
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    otlp_endpoint: str = ""                   # OpenTelemetry collector
    otlp_service_name: str = "medai-platform"

    # ── Neo4j / Context Graph ─────────────────────────────────────────
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"
    neo4j_database: str = "neo4j"

    # ── Feature Flags ────────────────────────────────────────────────
    enable_field_encryption: bool = True      # Encrypt PHI at field level
    enable_search_rag: bool = True            # Enable search/RAG pipeline
    enable_streaming: bool = True             # Enable SSE streaming
    enable_virus_scan: bool = False           # ClamAV integration
    enable_audit_hash_chain: bool = True      # Tamper-proof audit chain
    enable_granite_vision: bool = True        # Granite 4.0 3B document extraction sidecar
    enable_openrag: bool = True               # OpenRAG agentic ingestion + hybrid search
    enable_context_graph: bool = True         # Neo4j longitudinal patient memory
    enable_context1_agent: bool = False       # Context-1 multi-hop (stub — set True when harness released)

    # ── Computed Properties ──────────────────────────────────────────

    @computed_field  # type: ignore[misc]
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @computed_field  # type: ignore[misc]
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @computed_field  # type: ignore[misc]
    @property
    def max_voice_upload_size_bytes(self) -> int:
        return self.max_voice_upload_size_mb * 1024 * 1024

    # ── Validation ───────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_secrets(self) -> "Settings":
        """Block production startup with insecure default secrets."""
        if self.environment == "production":
            if self.jwt_secret_key in _INSECURE_DEFAULTS:
                raise ValueError(
                    "FATAL: jwt_secret_key is still the insecure default. "
                    "Set JWT_SECRET_KEY env var (use `openssl rand -hex 64`)."
                )
            if self.encryption_key in _INSECURE_DEFAULTS:
                raise ValueError(
                    "FATAL: encryption_key is still the insecure default. "
                    "Set ENCRYPTION_KEY env var (use `openssl rand -hex 32`)."
                )
            if self.debug:
                raise ValueError("FATAL: debug=True is not allowed in production.")
            if self.database_echo:
                raise ValueError(
                    "FATAL: database_echo=True leaks SQL (including patient data) to logs."
                )
            if len(self.jwt_secret_key) < 32:
                raise ValueError(
                    "FATAL: jwt_secret_key too short. Use at least 32 characters."
                )
            if len(self.encryption_key) < 16:
                raise ValueError(
                    "FATAL: encryption_key too short. Use at least 16 characters."
                )
        elif self.jwt_secret_key in _INSECURE_DEFAULTS:
            warnings.warn(
                "Using insecure default JWT secret — acceptable only for local development.",
                stacklevel=2,
            )

        # Staging validations
        if self.environment == "staging":
            if self.jwt_secret_key in _INSECURE_DEFAULTS:
                raise ValueError(
                    "FATAL: jwt_secret_key must be set for staging environment."
                )

        # SMTP validation: warn if email features are expected but not configured
        if self.environment in ("production", "staging"):
            if self.require_email_verification and not self.smtp_host:
                warnings.warn(
                    "SMTP not configured but require_email_verification=True. "
                    "Email sending will fail. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD.",
                    stacklevel=2,
                )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
