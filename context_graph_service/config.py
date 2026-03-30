"""
Context Graph Service — Configuration.

Neo4j connection, memory tiers, and POLE+O model settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ContextGraphConfig:
    """Configuration for the Neo4j context graph service."""

    # ── Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8007

    # ── Neo4j ─────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: float = 30.0

    # ── Memory Tiers ──────────────────────────────────────────────────
    short_term_ttl_hours: int = 24
    long_term_enabled: bool = True
    reasoning_memory_enabled: bool = True

    # ── Embeddings (for vector similarity in Neo4j) ───────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"

    # ── POLE+O Entity Types ───────────────────────────────────────────
    entity_types: tuple = ("Person", "Organization", "Location", "Event", "Object")

    # ── Security ──────────────────────────────────────────────────────
    phi_audit_enabled: bool = True
    encrypt_patient_data: bool = True


def load_config() -> ContextGraphConfig:
    return ContextGraphConfig(
        host=os.getenv("CONTEXT_GRAPH_HOST", "0.0.0.0"),
        port=int(os.getenv("CONTEXT_GRAPH_PORT", "8007")),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        short_term_ttl_hours=int(os.getenv("CONTEXT_GRAPH_STM_TTL_HOURS", "24")),
        long_term_enabled=os.getenv("CONTEXT_GRAPH_LTM_ENABLED", "true").lower() in {"1", "true"},
        reasoning_memory_enabled=os.getenv("CONTEXT_GRAPH_REASONING_ENABLED", "true").lower() in {"1", "true"},
        embedding_model=os.getenv("CONTEXT_GRAPH_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        phi_audit_enabled=os.getenv("CONTEXT_GRAPH_PHI_AUDIT", "true").lower() in {"1", "true"},
        encrypt_patient_data=os.getenv("CONTEXT_GRAPH_ENCRYPT_PHI", "true").lower() in {"1", "true"},
    )
