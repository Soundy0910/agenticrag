"""
backend/scripts/reset_neo4j.py

Clear all Neo4j nodes/relationships and recreate constraints for the new schema.

Usage:
  python -m backend.scripts.reset_neo4j           # prompts for confirmation
  python -m backend.scripts.reset_neo4j --force   # no prompt
"""

import argparse
import logging
import sys

from neo4j import GraphDatabase

import backend.config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constraints for the new schema
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    ("company_id_unique",  "Company",         "company_id"),
    ("filing_id_unique",   "Filing",          "filing_id"),
    ("segment_id_unique",  "BusinessSegment", "segment_id"),
    ("metric_id_unique",   "Metric",          "metric_id"),
    ("risk_id_unique",     "RiskFactor",      "risk_id"),
    ("topic_id_unique",    "Topic",           "topic_id"),
    ("chunk_id_unique",    "Chunk",           "chunk_id"),
]


def _delete_all(session) -> tuple[int, int]:
    """Delete all nodes and relationships in batches. Returns (nodes, rels) deleted."""
    total_nodes = total_rels = 0
    while True:
        result = session.run(
            "MATCH (n) WITH n LIMIT 10000 "
            "DETACH DELETE n "
            "RETURN count(n) AS deleted"
        )
        deleted = result.single()["deleted"]
        total_nodes += deleted
        if deleted == 0:
            break
    # count remaining relationships (should be 0)
    result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
    total_rels = result.single()["cnt"]
    return total_nodes, total_rels


def _create_constraints(session) -> None:
    for constraint_name, label, prop in CONSTRAINTS:
        session.run(
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )
        logger.info("  constraint: %s.%s", label, prop)


def reset(force: bool = False) -> None:
    if not cfg.NEO4J_URI:
        logger.error("NEO4J_URI not set — check your .env file.")
        sys.exit(1)

    if not force:
        confirm = input(
            "This will DELETE ALL nodes and relationships in Neo4j. "
            f"Database: {cfg.NEO4J_DATABASE!r} at {cfg.NEO4J_URI!r}\n"
            "Type 'yes' to continue: "
        ).strip()
        if confirm.lower() != "yes":
            logger.info("Aborted.")
            return

    driver = GraphDatabase.driver(
        cfg.NEO4J_URI,
        auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
    )

    try:
        with driver.session(database=cfg.NEO4J_DATABASE) as session:
            logger.info("Deleting all nodes and relationships...")
            nodes_deleted, rels_remaining = _delete_all(session)
            logger.info("Deleted %d nodes. Relationships remaining: %d", nodes_deleted, rels_remaining)

            logger.info("Creating constraints for new schema...")
            _create_constraints(session)
            logger.info("Done. Neo4j is clean and ready for re-ingestion.")
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset Neo4j graph and recreate schema constraints.")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()
    reset(force=args.force)
