"""
Phase 4 — Load Fraud Detection Graph into Neo4j
=================================================
Loads the IEEE-CIS dataset into Neo4j Aura following the chosen graph model:

  (Transaction)-[:USED_CARD]->(Card)
  (Transaction)-[:PAYER_EMAIL]->(EmailDomain)
  (Transaction)-[:RECIPIENT_EMAIL]->(EmailDomain)
  (Transaction)-[:BILLED_TO]->(BillingAddress)
  (Transaction)-[:USED_DEVICE]->(Device)

Design principles:
- All writes are idempotent (MERGE not CREATE)
- Data loaded in configurable batch sizes to control memory
- Progress logged every batch
- Failures logged and skipped, not fatal
- Entity nodes created first, then Transaction nodes, then relationships

Usage:
    python src/graph/load_graph.py
    python src/graph/load_graph.py --batch-size 500 --limit 10000
"""

import logging
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Iterator

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase, Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

DATA_DIR = ROOT
DEFAULT_BATCH_SIZE = 500
DEFAULT_CHUNK_SIZE = 50_000  # rows read from CSV at a time


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_device(raw: str) -> str:
    """
    Normalize DeviceInfo strings to a canonical device key.
    Strips build numbers and minor version noise.
    """
    if not raw or pd.isna(raw):
        return None
    s = str(raw).strip()
    # Strip Android build suffixes: "SAMSUNG SM-G892A Build/NRD90M" → "SAMSUNG SM-G892A"
    if " Build/" in s:
        s = s.split(" Build/")[0].strip()
    # Standardize common desktop/browser identifiers
    s = s.replace("Trident/7.0", "Windows IE11")
    return s[:200]  # cap length


def safe_str(val) -> str | None:
    """Return string or None for NaN/None values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val).strip()


def safe_float(val) -> float | None:
    """Return float or None for NaN/None values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def make_addr_key(addr1, addr2) -> str | None:
    """Composite billing address key."""
    a1 = safe_float(addr1)
    a2 = safe_float(addr2)
    if a1 is None or a2 is None:
        return None
    return f"{int(a1)}|{int(a2)}"


# ---------------------------------------------------------------------------
# Batch writers
# ---------------------------------------------------------------------------

def _run_batch(session: Session, cypher: str, batch: list, operation_name: str) -> int:
    """Execute a MERGE batch and return count of rows processed."""
    try:
        session.run(cypher, batch=batch)
        return len(batch)
    except Exception as e:
        log.error(f"  Batch failed [{operation_name}]: {e}")
        return 0


def load_transactions(session: Session, rows: list[dict]) -> int:
    """MERGE Transaction nodes with properties."""
    cypher = """
    UNWIND $batch AS row
    MERGE (t:Transaction {transactionId: row.transactionId})
    SET t.transactionDT     = row.transactionDT,
        t.transactionAmt    = row.transactionAmt,
        t.productCD         = row.productCD,
        t.isFraud           = row.isFraud,
        t.card4             = row.card4,
        t.card6             = row.card6,
        t.addr1             = row.addr1,
        t.addr2             = row.addr2,
        t.dist1             = row.dist1,
        t.hasIdentity       = row.hasIdentity
    """
    return _run_batch(session, cypher, rows, "Transaction")


def load_cards_and_rels(session: Session, rows: list[dict]) -> int:
    """MERGE Card nodes and USED_CARD relationships."""
    cypher = """
    UNWIND $batch AS row
    MERGE (c:Card {cardId: row.cardId})
    WITH c, row
    MATCH (t:Transaction {transactionId: row.transactionId})
    MERGE (t)-[:USED_CARD]->(c)
    """
    return _run_batch(session, cypher, rows, "Card+USED_CARD")


def load_payer_email_rels(session: Session, rows: list[dict]) -> int:
    """MERGE EmailDomain nodes and PAYER_EMAIL relationships."""
    cypher = """
    UNWIND $batch AS row
    MERGE (e:EmailDomain {domain: row.domain})
    WITH e, row
    MATCH (t:Transaction {transactionId: row.transactionId})
    MERGE (t)-[:PAYER_EMAIL]->(e)
    """
    return _run_batch(session, cypher, rows, "EmailDomain+PAYER_EMAIL")


def load_recipient_email_rels(session: Session, rows: list[dict]) -> int:
    """MERGE EmailDomain nodes and RECIPIENT_EMAIL relationships."""
    cypher = """
    UNWIND $batch AS row
    MERGE (e:EmailDomain {domain: row.domain})
    WITH e, row
    MATCH (t:Transaction {transactionId: row.transactionId})
    MERGE (t)-[:RECIPIENT_EMAIL]->(e)
    """
    return _run_batch(session, cypher, rows, "EmailDomain+RECIPIENT_EMAIL")


def load_billing_address_rels(session: Session, rows: list[dict]) -> int:
    """MERGE BillingAddress nodes and BILLED_TO relationships."""
    cypher = """
    UNWIND $batch AS row
    MERGE (b:BillingAddress {addrKey: row.addrKey})
    WITH b, row
    MATCH (t:Transaction {transactionId: row.transactionId})
    MERGE (t)-[:BILLED_TO]->(b)
    """
    return _run_batch(session, cypher, rows, "BillingAddress+BILLED_TO")


def load_device_rels(session: Session, rows: list[dict]) -> int:
    """MERGE Device nodes and USED_DEVICE relationships."""
    cypher = """
    UNWIND $batch AS row
    MERGE (d:Device {deviceKey: row.deviceKey})
    WITH d, row
    MATCH (t:Transaction {transactionId: row.transactionId})
    MERGE (t)-[:USED_DEVICE]->(d)
    """
    return _run_batch(session, cypher, rows, "Device+USED_DEVICE")


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------

def process_transaction_chunk(
    session: Session,
    tt_chunk: pd.DataFrame,
    ti_chunk: pd.DataFrame | None,
    batch_size: int,
) -> dict:
    """Process one chunk of transactions and write to Neo4j."""
    stats = {
        "transactions": 0,
        "cards": 0,
        "payer_emails": 0,
        "recipient_emails": 0,
        "billing_addresses": 0,
        "devices": 0,
    }

    # Merge identity data if available
    if ti_chunk is not None and len(ti_chunk) > 0:
        df = tt_chunk.merge(ti_chunk, on="TransactionID", how="left")
    else:
        df = tt_chunk.copy()
        for col in ["DeviceInfo"]:
            if col not in df.columns:
                df[col] = None

    # --- Pass 1: Transaction nodes ---
    tx_rows = []
    card_rows = []
    payer_rows = []
    recipient_rows = []
    addr_rows = []
    device_rows = []

    for _, row in df.iterrows():
        tid = int(row["TransactionID"])

        tx_rows.append({
            "transactionId": tid,
            "transactionDT":  safe_float(row.get("TransactionDT")),
            "transactionAmt": safe_float(row.get("TransactionAmt")),
            "productCD":      safe_str(row.get("ProductCD")),
            "isFraud":        int(row.get("isFraud", 0)) if not pd.isna(row.get("isFraud", 0)) else None,
            "card4":          safe_str(row.get("card4")),
            "card6":          safe_str(row.get("card6")),
            "addr1":          safe_float(row.get("addr1")),
            "addr2":          safe_float(row.get("addr2")),
            "dist1":          safe_float(row.get("dist1")),
            "hasIdentity":    bool("DeviceType" in row and not pd.isna(row.get("DeviceType", float("nan")))),
        })

        # Card
        card1 = safe_str(row.get("card1"))
        if card1:
            card_rows.append({"transactionId": tid, "cardId": card1})

        # Payer email
        p_email = safe_str(row.get("P_emaildomain"))
        if p_email:
            payer_rows.append({"transactionId": tid, "domain": p_email.lower()})

        # Recipient email
        r_email = safe_str(row.get("R_emaildomain"))
        if r_email:
            recipient_rows.append({"transactionId": tid, "domain": r_email.lower()})

        # Billing address
        addr_key = make_addr_key(row.get("addr1"), row.get("addr2"))
        if addr_key:
            addr_rows.append({"transactionId": tid, "addrKey": addr_key})

        # Device
        device_raw = row.get("DeviceInfo")
        device_key = normalize_device(device_raw)
        if device_key:
            device_rows.append({"transactionId": tid, "deviceKey": device_key})

    # Write in batches
    for i in range(0, len(tx_rows), batch_size):
        stats["transactions"] += load_transactions(session, tx_rows[i:i+batch_size])
    for i in range(0, len(card_rows), batch_size):
        stats["cards"] += load_cards_and_rels(session, card_rows[i:i+batch_size])
    for i in range(0, len(payer_rows), batch_size):
        stats["payer_emails"] += load_payer_email_rels(session, payer_rows[i:i+batch_size])
    for i in range(0, len(recipient_rows), batch_size):
        stats["recipient_emails"] += load_recipient_email_rels(session, recipient_rows[i:i+batch_size])
    for i in range(0, len(addr_rows), batch_size):
        stats["billing_addresses"] += load_billing_address_rels(session, addr_rows[i:i+batch_size])
    for i in range(0, len(device_rows), batch_size):
        stats["devices"] += load_device_rels(session, device_rows[i:i+batch_size])

    return stats


# ---------------------------------------------------------------------------
# Validation queries
# ---------------------------------------------------------------------------

def run_validation(driver, database: str):
    """Run post-load sanity checks."""
    queries = [
        ("Transaction count", "MATCH (t:Transaction) RETURN count(t) AS n"),
        ("Fraud transactions", "MATCH (t:Transaction {isFraud: 1}) RETURN count(t) AS n"),
        ("Card count", "MATCH (c:Card) RETURN count(c) AS n"),
        ("EmailDomain count", "MATCH (e:EmailDomain) RETURN count(e) AS n"),
        ("BillingAddress count", "MATCH (b:BillingAddress) RETURN count(b) AS n"),
        ("Device count", "MATCH (d:Device) RETURN count(d) AS n"),
        ("USED_CARD rels", "MATCH ()-[r:USED_CARD]->() RETURN count(r) AS n"),
        ("PAYER_EMAIL rels", "MATCH ()-[r:PAYER_EMAIL]->() RETURN count(r) AS n"),
        ("RECIPIENT_EMAIL rels", "MATCH ()-[r:RECIPIENT_EMAIL]->() RETURN count(r) AS n"),
        ("BILLED_TO rels", "MATCH ()-[r:BILLED_TO]->() RETURN count(r) AS n"),
        ("USED_DEVICE rels", "MATCH ()-[r:USED_DEVICE]->() RETURN count(r) AS n"),
    ]

    log.info("=== Graph Validation ===")
    results = {}
    with driver.session(database=database) as session:
        for label, cypher in queries:
            r = session.run(cypher).single()
            n = r["n"] if r else 0
            log.info(f"  {label}: {n:,}")
            results[label] = n

    # Example neighborhood query
    log.info("=== Example: Top 5 cards by fraud transaction count ===")
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (c:Card)<-[:USED_CARD]-(t:Transaction {isFraud: 1})
            RETURN c.cardId AS card, count(t) AS fraudTxCount
            ORDER BY fraudTxCount DESC LIMIT 5
        """)
        for row in result:
            log.info(f"  Card {row['card']}: {row['fraudTxCount']} fraud transactions")

    log.info("=== Example: Top 5 devices by fraud transaction count ===")
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (d:Device)<-[:USED_DEVICE]-(t:Transaction {isFraud: 1})
            RETURN d.deviceKey AS device, count(t) AS fraudTxCount
            ORDER BY fraudTxCount DESC LIMIT 5
        """)
        for row in result:
            log.info(f"  Device '{row['device']}': {row['fraudTxCount']} fraud transactions")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load IEEE-CIS fraud graph into Neo4j")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of rows per Neo4j write transaction")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Number of CSV rows to load into memory at once")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total rows (for testing; default=all)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip post-load validation queries")
    args = parser.parse_args()

    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    log.info(f"Connecting to: {uri}")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    log.info("Connected.")

    # Pre-load identity table (it's small enough: 144K rows, 41 cols, 25MB)
    log.info("Pre-loading train_identity.csv (144K rows) ...")
    ti_full = pd.read_csv(DATA_DIR / "train_identity.csv")
    log.info(f"  Identity shape: {ti_full.shape}")

    # Stream transaction file in chunks
    log.info(f"Loading train_transaction.csv in chunks of {args.chunk_size:,} ...")

    total_stats = {
        "transactions": 0, "cards": 0, "payer_emails": 0,
        "recipient_emails": 0, "billing_addresses": 0, "devices": 0,
    }

    rows_processed = 0
    chunk_num = 0
    start_time = time.time()

    with driver.session(database=database) as session:
        for tt_chunk in pd.read_csv(
            DATA_DIR / "train_transaction.csv",
            chunksize=args.chunk_size,
        ):
            chunk_num += 1

            # Limit for testing
            if args.limit and rows_processed >= args.limit:
                break
            if args.limit:
                remaining = args.limit - rows_processed
                tt_chunk = tt_chunk.head(remaining)

            # Get matching identity rows for this chunk
            chunk_ids = tt_chunk["TransactionID"].values
            ti_chunk = ti_full[ti_full["TransactionID"].isin(chunk_ids)]

            chunk_stats = process_transaction_chunk(
                session, tt_chunk, ti_chunk, args.batch_size
            )

            for k in total_stats:
                total_stats[k] += chunk_stats[k]

            rows_processed += len(tt_chunk)
            elapsed = time.time() - start_time
            rate = rows_processed / elapsed if elapsed > 0 else 0
            log.info(
                f"  Chunk {chunk_num}: {rows_processed:,} rows "
                f"({rate:.0f} rows/s) | "
                f"tx={total_stats['transactions']:,} cards={total_stats['cards']:,} "
                f"emails={total_stats['payer_emails']+total_stats['recipient_emails']:,} "
                f"devices={total_stats['devices']:,}"
            )

    elapsed = time.time() - start_time
    log.info(f"\nLoad complete in {elapsed:.1f}s")
    log.info(f"Total stats: {total_stats}")

    if not args.skip_validation:
        run_validation(driver, database)

    driver.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
