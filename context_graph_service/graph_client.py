"""
Neo4j Graph Client — Three-Memory System.

Implements the POLE+O knowledge graph model:
  1. Short-term memory: Conversation history and document content
  2. Long-term memory: Entity knowledge graph (Person, Organization, Location, Event, Object)
  3. Reasoning memory: Decision traces with full provenance

Designed as a DERIVED READ MODEL — not the source of truth.
Data flows in from the platform, not written back into patient records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from context_graph_service.config import ContextGraphConfig

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A POLE+O entity in the knowledge graph."""
    entity_id: str
    entity_type: str  # Person, Organization, Location, Event, Object
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_doc_id: str = ""
    created_at: str = ""


@dataclass
class Relationship:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_doc_id: str = ""


@dataclass
class ConversationMessage:
    """A message in short-term memory."""
    message_id: str
    session_id: str
    patient_id: str
    role: str  # user, assistant, system
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    """A reasoning trace for decision provenance."""
    trace_id: str
    patient_id: str
    query: str
    thought_chain: List[str]
    tool_calls: List[Dict[str, Any]]
    conclusion: str
    confidence: float
    sources: List[str]
    timestamp: str


@dataclass
class CareTimeline:
    """A patient care timeline entry."""
    event_id: str
    patient_id: str
    event_type: str  # admission, discharge, procedure, medication, lab_result, diagnosis
    event_date: str
    description: str
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Schema Initialization Cypher ──────────────────────────────────────────

SCHEMA_INIT_CYPHER = [
    # Constraints
    "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT message_id IF NOT EXISTS FOR (m:Message) REQUIRE m.message_id IS UNIQUE",
    "CREATE CONSTRAINT trace_id IF NOT EXISTS FOR (t:DecisionTrace) REQUIRE t.trace_id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:CareEvent) REQUIRE e.event_id IS UNIQUE",
    "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE",

    # Indexes for performance
    "CREATE INDEX patient_name IF NOT EXISTS FOR (p:Patient) ON (p.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX message_session IF NOT EXISTS FOR (m:Message) ON (m.session_id)",
    "CREATE INDEX event_date IF NOT EXISTS FOR (e:CareEvent) ON (e.event_date)",
    "CREATE INDEX event_patient IF NOT EXISTS FOR (e:CareEvent) ON (e.patient_id)",

    # POLE+O entity labels
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.entity_id IS UNIQUE",
    "CREATE CONSTRAINT org_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.entity_id IS UNIQUE",
    "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.entity_id IS UNIQUE",
    "CREATE CONSTRAINT event_entity_id IF NOT EXISTS FOR (e:Event) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT object_id IF NOT EXISTS FOR (o:Object) REQUIRE o.entity_id IS UNIQUE",
]


class Neo4jGraphClient:
    """Production Neo4j client with three-memory architecture."""

    def __init__(self, config: ContextGraphConfig) -> None:
        self._config = config
        self._driver = None

    def connect(self) -> None:
        """Establish connection to Neo4j."""
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            self._config.neo4j_uri,
            auth=(self._config.neo4j_user, self._config.neo4j_password),
            max_connection_pool_size=self._config.neo4j_max_connection_pool_size,
            connection_timeout=self._config.neo4j_connection_timeout,
        )
        # Verify connectivity
        self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", self._config.neo4j_uri)

        # Initialize schema
        self._init_schema()

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    def _init_schema(self) -> None:
        """Create constraints and indexes."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            for cypher in SCHEMA_INIT_CYPHER:
                try:
                    session.run(cypher)
                except Exception as e:
                    # Constraints may already exist
                    logger.debug("Schema init (may already exist): %s — %s", cypher[:60], e)
        logger.info("Neo4j schema initialized")

    # ── Short-Term Memory (Conversation History) ──────────────────────

    def store_message(self, message: ConversationMessage) -> None:
        """Store a conversation message in short-term memory."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            session.run(
                """
                MERGE (p:Patient {patient_id: $patient_id})
                MERGE (s:Session {session_id: $session_id})
                MERGE (p)-[:HAS_SESSION]->(s)
                CREATE (m:Message {
                    message_id: $message_id,
                    session_id: $session_id,
                    patient_id: $patient_id,
                    role: $role,
                    content: $content,
                    timestamp: $timestamp
                })
                MERGE (s)-[:CONTAINS_MESSAGE]->(m)
                WITH m, s
                OPTIONAL MATCH (s)-[:CONTAINS_MESSAGE]->(prev:Message)
                WHERE prev.timestamp < m.timestamp AND prev.message_id <> m.message_id
                WITH m, prev ORDER BY prev.timestamp DESC LIMIT 1
                FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (prev)-[:FOLLOWED_BY]->(m)
                )
                """,
                message_id=message.message_id,
                session_id=message.session_id,
                patient_id=message.patient_id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
            )

    def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            result = session.run(
                """
                MATCH (s:Session {session_id: $session_id})-[:CONTAINS_MESSAGE]->(m:Message)
                RETURN m.message_id AS id, m.role AS role, m.content AS content,
                       m.timestamp AS timestamp
                ORDER BY m.timestamp ASC
                LIMIT $limit
                """,
                session_id=session_id,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ── Long-Term Memory (Entity Knowledge Graph — POLE+O) ───────────

    def upsert_entity(self, entity: Entity) -> None:
        """Upsert a POLE+O entity into the knowledge graph."""
        label = entity.entity_type if entity.entity_type in self._config.entity_types else "Entity"
        with self._driver.session(database=self._config.neo4j_database) as session:
            session.run(
                f"""
                MERGE (e:{label} {{entity_id: $entity_id}})
                SET e.name = $name,
                    e.entity_type = $entity_type,
                    e.properties = $properties,
                    e.source_doc_id = $source_doc_id,
                    e.updated_at = $updated_at
                ON CREATE SET e.created_at = $updated_at
                """,
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                properties=json.dumps(entity.properties),
                source_doc_id=entity.source_doc_id,
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

    def create_relationship(self, rel: Relationship) -> None:
        """Create a relationship between two entities."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            session.run(
                """
                MATCH (a {entity_id: $source_id})
                MATCH (b {entity_id: $target_id})
                MERGE (a)-[r:RELATED_TO {relation_type: $relation_type}]->(b)
                SET r.properties = $properties,
                    r.source_doc_id = $source_doc_id,
                    r.updated_at = $updated_at
                """,
                source_id=rel.source_id,
                target_id=rel.target_id,
                relation_type=rel.relation_type,
                properties=json.dumps(rel.properties),
                source_doc_id=rel.source_doc_id,
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

    def get_patient_entities(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all entities related to a patient."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            result = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})-[*1..3]-(e)
                WHERE e:Person OR e:Organization OR e:Location OR e:Event OR e:Object
                RETURN DISTINCT e.entity_id AS id, e.name AS name,
                       e.entity_type AS type, e.properties AS properties
                LIMIT 100
                """,
                patient_id=patient_id,
            )
            return [dict(record) for record in result]

    def link_entity_to_patient(self, patient_id: str, entity_id: str, relation: str) -> None:
        """Link an entity to a patient."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            session.run(
                """
                MERGE (p:Patient {patient_id: $patient_id})
                MATCH (e {entity_id: $entity_id})
                MERGE (p)-[:HAS_ENTITY {relation: $relation}]->(e)
                """,
                patient_id=patient_id,
                entity_id=entity_id,
                relation=relation,
            )

    # ── Reasoning Memory (Decision Traces) ────────────────────────────

    def store_decision_trace(self, trace: DecisionTrace) -> None:
        """Store a decision trace with full provenance."""
        if not self._config.reasoning_memory_enabled:
            return

        with self._driver.session(database=self._config.neo4j_database) as session:
            session.run(
                """
                MERGE (p:Patient {patient_id: $patient_id})
                CREATE (t:DecisionTrace {
                    trace_id: $trace_id,
                    patient_id: $patient_id,
                    query: $query,
                    thought_chain: $thought_chain,
                    tool_calls: $tool_calls,
                    conclusion: $conclusion,
                    confidence: $confidence,
                    sources: $sources,
                    timestamp: $timestamp
                })
                MERGE (p)-[:HAS_TRACE]->(t)
                """,
                trace_id=trace.trace_id,
                patient_id=trace.patient_id,
                query=trace.query,
                thought_chain=json.dumps(trace.thought_chain),
                tool_calls=json.dumps(trace.tool_calls),
                conclusion=trace.conclusion,
                confidence=trace.confidence,
                sources=json.dumps(trace.sources),
                timestamp=trace.timestamp,
            )

    def get_decision_traces(self, patient_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get decision traces for a patient."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            result = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})-[:HAS_TRACE]->(t:DecisionTrace)
                RETURN t.trace_id AS id, t.query AS query, t.conclusion AS conclusion,
                       t.confidence AS confidence, t.timestamp AS timestamp,
                       t.sources AS sources
                ORDER BY t.timestamp DESC
                LIMIT $limit
                """,
                patient_id=patient_id,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ── Care Timeline ─────────────────────────────────────────────────

    def add_care_event(self, event: CareTimeline) -> None:
        """Add an event to a patient's care timeline."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            session.run(
                """
                MERGE (p:Patient {patient_id: $patient_id})
                CREATE (e:CareEvent {
                    event_id: $event_id,
                    patient_id: $patient_id,
                    event_type: $event_type,
                    event_date: $event_date,
                    description: $description,
                    provider: $provider,
                    metadata: $metadata
                })
                MERGE (p)-[:HAS_EVENT]->(e)
                WITH p, e
                OPTIONAL MATCH (p)-[:HAS_EVENT]->(prev:CareEvent)
                WHERE prev.event_date <= e.event_date AND prev.event_id <> e.event_id
                WITH e, prev ORDER BY prev.event_date DESC LIMIT 1
                FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (prev)-[:FOLLOWED_BY]->(e)
                )
                """,
                event_id=event.event_id,
                patient_id=event.patient_id,
                event_type=event.event_type,
                event_date=event.event_date,
                description=event.description,
                provider=event.provider,
                metadata=json.dumps(event.metadata),
            )

    def get_care_timeline(
        self, patient_id: str, event_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get a patient's care timeline."""
        type_filter = "AND e.event_type = $event_type" if event_type else ""
        with self._driver.session(database=self._config.neo4j_database) as session:
            result = session.run(
                f"""
                MATCH (p:Patient {{patient_id: $patient_id}})-[:HAS_EVENT]->(e:CareEvent)
                WHERE TRUE {type_filter}
                RETURN e.event_id AS id, e.event_type AS type, e.event_date AS date,
                       e.description AS description, e.provider AS provider
                ORDER BY e.event_date DESC
                LIMIT $limit
                """,
                patient_id=patient_id,
                event_type=event_type,
                limit=limit,
            )
            return [dict(record) for record in result]

    # ── Graph Queries (Multi-hop reasoning) ───────────────────────────

    def find_related_patients(self, patient_id: str, relation_types: List[str] = None) -> List[Dict[str, Any]]:
        """Find patients related through shared entities (e.g., same provider, condition)."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            result = session.run(
                """
                MATCH (p1:Patient {patient_id: $patient_id})-[:HAS_ENTITY]->(e)<-[:HAS_ENTITY]-(p2:Patient)
                WHERE p1 <> p2
                RETURN DISTINCT p2.patient_id AS patient_id, e.name AS shared_entity,
                       e.entity_type AS entity_type
                LIMIT 20
                """,
                patient_id=patient_id,
            )
            return [dict(record) for record in result]

    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get a comprehensive patient summary from the graph."""
        with self._driver.session(database=self._config.neo4j_database) as session:
            # Count entities, events, traces, sessions
            result = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})
                OPTIONAL MATCH (p)-[:HAS_ENTITY]->(e)
                OPTIONAL MATCH (p)-[:HAS_EVENT]->(ev:CareEvent)
                OPTIONAL MATCH (p)-[:HAS_TRACE]->(t:DecisionTrace)
                OPTIONAL MATCH (p)-[:HAS_SESSION]->(s:Session)
                RETURN p.patient_id AS patient_id,
                       count(DISTINCT e) AS entity_count,
                       count(DISTINCT ev) AS event_count,
                       count(DISTINCT t) AS trace_count,
                       count(DISTINCT s) AS session_count
                """,
                patient_id=patient_id,
            )
            record = result.single()
            if record is None:
                return {"patient_id": patient_id, "exists": False}
            return {**dict(record), "exists": True}

    # ── Health / Stats ────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Check Neo4j connectivity and return stats."""
        try:
            with self._driver.session(database=self._config.neo4j_database) as session:
                result = session.run(
                    """
                    CALL db.labels() YIELD label
                    RETURN collect(label) AS labels
                    """
                )
                labels = result.single()["labels"]

                count_result = session.run(
                    "MATCH (n) RETURN count(n) AS total_nodes"
                )
                total_nodes = count_result.single()["total_nodes"]

            return {
                "status": "ok",
                "neo4j_uri": self._config.neo4j_uri,
                "database": self._config.neo4j_database,
                "total_nodes": total_nodes,
                "labels": labels,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
