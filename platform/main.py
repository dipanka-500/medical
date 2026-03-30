"""
MedAI Platform — Production Entry Point.

Security & reliability features:
    - Structured JSON logging (production) / colored text (dev)
    - Signal handling for graceful shutdown
    - SSL/TLS support for direct termination
    - Worker / concurrency tuning from settings
    - Startup banner with environment info

Usage:
    uvicorn main:app                        (development)
    uvicorn main:app --workers 4             (production)
"""

from __future__ import annotations

import logging
import logging.config
import os
import signal
import sys
from typing import Any

import uvicorn

from config import settings

# ── Structured Logging Configuration ─────────────────────────────────────

_LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "logging.Formatter",
            "format": '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
                      '"logger":"%(name)s","message":"%(message)s",'
                      '"module":"%(module)s","func":"%(funcName)s"}',
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
        "console": {
            "()": "logging.Formatter",
            "format": "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "json" if settings.environment == "production" else "console",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "DEBUG" if settings.debug else "INFO",
        "handlers": ["default"],
    },
    "loggers": {
        "uvicorn": {"level": "INFO", "handlers": ["default"], "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {
            "level": "INFO" if settings.environment != "production" else "WARNING",
            "handlers": ["default"],
            "propagate": False,
        },
        "sqlalchemy.engine": {
            "level": "WARNING",
            "handlers": ["default"],
            "propagate": False,
        },
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
    },
}


def _configure_logging() -> None:
    """Apply structured logging configuration."""
    logging.config.dictConfig(_LOG_CONFIG)


# ── Signal Handling ──────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger = logging.getLogger("medai_platform.main")
    logger.info("Received %s — initiating graceful shutdown", sig_name)
    _shutdown_requested = True
    sys.exit(0)


# ── Startup Banner ───────────────────────────────────────────────────────

def _print_banner() -> None:
    """Print startup information."""
    logger = logging.getLogger("medai_platform.main")
    logger.info("=" * 60)
    logger.info("  🚀 MedAI Platform v%s", settings.app_version)
    logger.info("  Environment : %s", settings.environment)
    logger.info("  Host        : %s:%d", settings.host, settings.port)
    logger.info("  Workers     : %d", 1 if settings.environment == "development" else settings.workers)
    logger.info("  Debug       : %s", settings.debug)
    logger.info("  DB Pool     : %d (+%d overflow)", settings.database_pool_size, settings.database_max_overflow)
    logger.info("  Log Format  : %s", "json" if settings.environment == "production" else "console")
    logger.info("=" * 60)


# ── Application Factory ─────────────────────────────────────────────────

_configure_logging()

from gateway.app import create_app  # noqa: E402 — must import after logging

app = create_app()


# ── CLI Entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _print_banner()

    is_dev = settings.environment == "development"

    uvicorn_config: dict[str, Any] = {
        "app": "main:app",
        "host": settings.host,
        "port": settings.port,
        "workers": 1 if is_dev else settings.workers,
        "reload": is_dev,
        "log_level": "debug" if settings.debug else "info",
        "access_log": is_dev,
        "proxy_headers": True,
        "forwarded_allow_ips": "*" if is_dev else os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        "server_header": False,       # Don't leak server info
        "date_header": True,
        "limit_concurrency": 100,     # Max concurrent connections
        "backlog": 2048,              # TCP backlog queue
        "timeout_keep_alive": 5,      # Keep-alive timeout
    }

    # SSL/TLS support (for direct termination without reverse proxy)
    ssl_cert = os.getenv("SSL_CERTFILE")
    ssl_key = os.getenv("SSL_KEYFILE")
    if ssl_cert and ssl_key:
        uvicorn_config["ssl_certfile"] = ssl_cert
        uvicorn_config["ssl_keyfile"] = ssl_key
        logging.getLogger("medai_platform.main").info("TLS enabled: cert=%s", ssl_cert)

    uvicorn.run(**uvicorn_config)
