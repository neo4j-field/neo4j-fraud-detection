"""
Phase 4 — Create Neo4j Schema: Constraints and Indexes
========================================================
Creates all uniqueness constraints and indexes for the fraud detection graph.
Safe to run multiple times (IF NOT EXISTS guards).

Usage:
    python src/graph/create_schema.py
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    # Unique constraints (implicitly create an index)
    (
        "transaction_id_unique",
        "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS "
        "FOR (t:Transaction) REQUIRE t.transactionId IS UNIQUE",
    ),
    (
        "card_id_unique",
        "CREATE CONSTRAINT card_id_unique IF NOT EXISTS "
        "FOR (c:Card) REQUIRE c.cardId IS UNIQUE",
    ),
    (
        "email_domain_unique",
        "CREATE CONSTRAINT email_domain_unique IF NOT EXISTS "
        "FOR (e:EmailDomain) REQUIRE e.domain IS UNIQUE",
    ),
    (
        "billing_address_unique",
        "CREATE CONSTRAINT billing_address_unique IF NOT EXISTS "
        "FOR (b:BillingAddress) REQUIRE b.addrKey IS UNIQUE",
    ),
    (
        "device_key_unique",
        "CREATE CONSTRAINT device_key_unique IF NOT EXISTS "
        "FOR (d:Device) REQUIRE d.deviceKey IS UNIQUE",
    ),
]

INDEXES = [
    (
        "transaction_fraud_idx",
        "CREATE INDEX transaction_fraud_idx IF NOT EXISTS "
        "FOR (t:Transaction) ON (t.isFraud)",
    ),
    (
        "transaction_dt_idx",
        "CREATE INDEX transaction_dt_idx IF NOT EXISTS "
        "FOR (t:Transaction) ON (t.transactionDT)",
    ),
    (
        "transaction_amt_idx",
        "CREATE INDEX transaction_amt_idx IF NOT EXISTS "
        "FOR (t:Transaction) ON (t.transactionAmt)",
    ),
    (
        "transaction_product_idx",
        "CREATE INDEX transaction_product_idx IF NOT EXISTS "
        "FOR (t:Transaction) ON (t.productCD)",
    ),
]


def create_schema(driver, database: str):
    """Apply all constraints and indexes."""
    with driver.session(database=database) as session:
        log.info("Creating constraints ...")
        for name, cypher in CONSTRAINTS:
            try:
                session.run(cypher)
                log.info(f"  OK: {name}")
            except Exception as e:
                log.warning(f"  SKIP {name}: {e}")

        log.info("Creating indexes ...")
        for name, cypher in INDEXES:
            try:
                session.run(cypher)
                log.info(f"  OK: {name}")
            except Exception as e:
                log.warning(f"  SKIP {name}: {e}")

    log.info("Schema creation complete.")


def verify_schema(driver, database: str):
    """List all constraints and indexes to confirm they were created."""
    with driver.session(database=database) as session:
        log.info("Current constraints:")
        result = session.run("SHOW CONSTRAINTS")
        for row in result:
            log.info(f"  {row['name']} | {row['type']} | {row['labelsOrTypes']} | {row['properties']}")

        log.info("Current indexes (non-constraint):")
        result = session.run("SHOW INDEXES WHERE type <> 'LOOKUP' AND NOT owningConstraint IS NOT NULL")
        for row in result:
            log.info(f"  {row['name']} | {row['labelsOrTypes']} | {row['properties']}")


def main():
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    log.info(f"Connecting to: {uri}")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    log.info("Connected.")

    create_schema(driver, database)
    verify_schema(driver, database)

    driver.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
