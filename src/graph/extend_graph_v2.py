"""
Graph V2 Extension
==================
Extends the existing graph with three new structural elements:

1. OSBrowser nodes + HAS_OS_BROWSER relationships
   Source: id_30 (OS) + id_31 (browser) from train_identity
   Coverage: ~24% of transactions (identity-linked only)
   Value: Soft device fingerprint; certain OS+browser combos have 2-12% fraud rates

2. ProxyType nodes + VIA_PROXY relationships
   Source: id_23 from train_identity
   Coverage: ~3.6% of identity-linked transactions
   Value: IP_PROXY:ANONYMOUS has 13.7% fraud rate (4x baseline)

3. PREV_ON_CARD temporal edges between consecutive transactions on the same card
   Source: card1 + TransactionDT from train_transaction
   Coverage: ~74% of transactions (those with a prior transaction on same card)
   Value: If prev transaction on same card was fraud → 50% chance current is fraud (14x baseline)

All writes are idempotent (MERGE / MERGE+ON CREATE SET).

Usage:
    python src/graph/extend_graph_v2.py
    python src/graph/extend_graph_v2.py --skip-temporal   # skip the expensive T→T edges
    python src/graph/extend_graph_v2.py --limit 10000     # test on subset
"""

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
DATA_DIR = ROOT
BATCH_SIZE = 500
CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_str(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val).strip()


def normalize_os_browser(os_val, browser_val) -> str | None:
    """Canonical OS+Browser key."""
    os_s = safe_str(os_val)
    br_s = safe_str(browser_val)
    if os_s is None and br_s is None:
        return None
    os_s = os_s or "unknown_os"
    br_s = br_s or "unknown_browser"
    return f"{os_s}|{br_s}"[:200]


def run_batch(session, cypher, batch, label):
    try:
        session.run(cypher, batch=batch)
        return len(batch)
    except Exception as e:
        log.error(f"  Batch failed [{label}]: {e}")
        return 0


# ---------------------------------------------------------------------------
# 1. OSBrowser nodes + HAS_OS_BROWSER relationships
# ---------------------------------------------------------------------------

def load_os_browser(session, rows):
    cypher = """
    UNWIND $batch AS row
    MERGE (ob:OSBrowser {osBrowserKey: row.osBrowserKey})
    WITH ob, row
    MATCH (t:Transaction {transactionId: row.transactionId})
    MERGE (t)-[:HAS_OS_BROWSER]->(ob)
    """
    return run_batch(session, cypher, rows, "OSBrowser+HAS_OS_BROWSER")


def add_os_browser_nodes(driver, database, ti: pd.DataFrame):
    """Add OSBrowser nodes and relationships from identity data."""
    log.info("=== Adding OSBrowser nodes ===")

    rows = []
    for _, row in ti.iterrows():
        key = normalize_os_browser(row.get("id_30"), row.get("id_31"))
        if key and key != "unknown_os|unknown_browser":
            rows.append({
                "transactionId": int(row["TransactionID"]),
                "osBrowserKey": key,
            })

    log.info(f"  OSBrowser rows to write: {len(rows):,}")
    total = 0
    with driver.session(database=database) as session:
        for i in range(0, len(rows), BATCH_SIZE):
            total += load_os_browser(session, rows[i:i+BATCH_SIZE])

    log.info(f"  OSBrowser rels written: {total:,}")


# ---------------------------------------------------------------------------
# 2. ProxyType nodes + VIA_PROXY relationships
# ---------------------------------------------------------------------------

def load_proxy_type(session, rows):
    cypher = """
    UNWIND $batch AS row
    MERGE (p:ProxyType {proxyLabel: row.proxyLabel})
    WITH p, row
    MATCH (t:Transaction {transactionId: row.transactionId})
    MERGE (t)-[:VIA_PROXY]->(p)
    """
    return run_batch(session, cypher, rows, "ProxyType+VIA_PROXY")


def add_proxy_nodes(driver, database, ti: pd.DataFrame):
    """Add ProxyType nodes and relationships from identity data."""
    log.info("=== Adding ProxyType nodes ===")

    proxy_rows = ti[ti["id_23"].notna()][["TransactionID", "id_23"]].copy()
    proxy_rows = proxy_rows[~proxy_rows["id_23"].astype(str).str.lower().isin(
        ["nan", "notfound", ""]
    )]
    log.info(f"  Transactions with proxy data: {len(proxy_rows):,}")

    rows = [
        {"transactionId": int(r["TransactionID"]), "proxyLabel": str(r["id_23"]).strip()}
        for _, r in proxy_rows.iterrows()
    ]

    total = 0
    with driver.session(database=database) as session:
        for i in range(0, len(rows), BATCH_SIZE):
            total += load_proxy_type(session, rows[i:i+BATCH_SIZE])

    log.info(f"  ProxyType rels written: {total:,}")


# ---------------------------------------------------------------------------
# 3. PREV_ON_CARD temporal edges
# ---------------------------------------------------------------------------

def load_prev_on_card(session, rows):
    """
    Create directed PREV_ON_CARD edge: (t_prev)-[:PREV_ON_CARD]->(t_curr)
    Properties: dt_gap_seconds (time between the two transactions)
    """
    cypher = """
    UNWIND $batch AS row
    MATCH (t_prev:Transaction {transactionId: row.prevId})
    MATCH (t_curr:Transaction {transactionId: row.currId})
    MERGE (t_prev)-[r:PREV_ON_CARD]->(t_curr)
    ON CREATE SET r.dtGapSeconds = row.dtGap
    """
    return run_batch(session, cypher, rows, "PREV_ON_CARD")


def add_temporal_card_edges(driver, database, tt: pd.DataFrame, window_hours: int = 24, limit=None):
    """
    For each card, create PREV_ON_CARD edges between consecutive transactions
    that fall within `window_hours` of each other.

    Strategy: sort all transactions by (card1, TransactionDT), then for each
    consecutive pair on the same card, emit an edge if gap <= window.
    """
    log.info(f"=== Adding PREV_ON_CARD edges (window={window_hours}h) ===")
    window_sec = window_hours * 3600

    # Sort by card + time
    df = tt[["TransactionID", "card1", "TransactionDT"]].dropna().copy()
    df = df.sort_values(["card1", "TransactionDT"]).reset_index(drop=True)

    # Create consecutive pairs within the same card
    df["next_id"] = df.groupby("card1")["TransactionID"].shift(-1)
    df["next_dt"] = df.groupby("card1")["TransactionDT"].shift(-1)
    df["next_card"] = df.groupby("card1")["card1"].shift(-1)

    # Keep only same-card consecutive pairs within window
    edges = df[
        (df["next_card"] == df["card1"]) &
        (df["next_dt"] - df["TransactionDT"] <= window_sec) &
        (df["next_dt"] - df["TransactionDT"] >= 0)
    ][["TransactionID", "next_id", "TransactionDT", "next_dt"]].copy()

    edges["dtGap"] = (edges["next_dt"] - edges["TransactionDT"]).astype(int)

    if limit:
        edges = edges.head(limit)

    log.info(f"  PREV_ON_CARD pairs to create: {len(edges):,}")

    rows = [
        {
            "prevId": int(r["TransactionID"]),
            "currId": int(r["next_id"]),
            "dtGap": int(r["dtGap"]),
        }
        for _, r in edges.iterrows()
    ]

    total = 0
    t0 = time.time()
    with driver.session(database=database) as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i+BATCH_SIZE]
            total += load_prev_on_card(session, batch)
            if (i // BATCH_SIZE) % 100 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = total / elapsed
                log.info(f"  Progress: {total:,}/{len(rows):,} edges ({rate:.0f}/s)")

    log.info(f"  PREV_ON_CARD edges written: {total:,} in {time.time()-t0:.1f}s")
    return total


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(driver, database):
    queries = [
        ("OSBrowser nodes", "MATCH (n:OSBrowser) RETURN count(n) AS n"),
        ("ProxyType nodes", "MATCH (n:ProxyType) RETURN count(n) AS n"),
        ("HAS_OS_BROWSER rels", "MATCH ()-[r:HAS_OS_BROWSER]->() RETURN count(r) AS n"),
        ("VIA_PROXY rels", "MATCH ()-[r:VIA_PROXY]->() RETURN count(r) AS n"),
        ("PREV_ON_CARD rels", "MATCH ()-[r:PREV_ON_CARD]->() RETURN count(r) AS n"),
    ]
    log.info("=== Validation ===")
    with driver.session(database=database) as session:
        for label, cypher in queries:
            n = session.run(cypher).single()["n"]
            log.info(f"  {label}: {n:,}")

    log.info("=== ProxyType fraud rates ===")
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (p:ProxyType)<-[:VIA_PROXY]-(t:Transaction)
            RETURN p.proxyLabel AS proxy,
                   count(t) AS total,
                   sum(CASE WHEN t.isFraud=1 THEN 1 ELSE 0 END) AS fraudCount,
                   round(100.0 * sum(CASE WHEN t.isFraud=1 THEN 1 ELSE 0 END) / count(t), 2) AS fraudPct
            ORDER BY fraudPct DESC
        """)
        for row in result:
            log.info(f"  {row['proxy']}: {row['fraudPct']}% fraud ({row['fraudCount']}/{row['total']})")

    log.info("=== PREV_ON_CARD: fraud follow-fraud rate ===")
    with driver.session(database=database) as session:
        result = session.run("""
            MATCH (t1:Transaction {isFraud:1})-[:PREV_ON_CARD]->(t2:Transaction)
            RETURN count(t2) AS following_fraud_txns,
                   sum(CASE WHEN t2.isFraud=1 THEN 1 ELSE 0 END) AS also_fraud,
                   round(100.0 * sum(CASE WHEN t2.isFraud=1 THEN 1 ELSE 0 END) / count(t2), 2) AS also_fraud_pct
        """).single()
        if result:
            log.info(
                f"  Txns following a fraud on same card: {result['following_fraud_txns']:,} | "
                f"also fraud: {result['also_fraud']:,} ({result['also_fraud_pct']}%)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extend graph with V2 nodes and relationships")
    parser.add_argument("--skip-temporal", action="store_true",
                        help="Skip PREV_ON_CARD edge creation (faster for testing)")
    parser.add_argument("--window-hours", type=int, default=24,
                        help="Time window for PREV_ON_CARD edges (default: 24h)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit PREV_ON_CARD edges (for testing)")
    args = parser.parse_args()

    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    log.info(f"Connected to: {uri}")

    # Load identity table (small — 144K rows)
    log.info("Loading train_identity.csv ...")
    ti = pd.read_csv(DATA_DIR / "train_identity.csv",
                     usecols=["TransactionID", "id_23", "id_30", "id_31"])
    log.info(f"  Shape: {ti.shape}")

    # 1. OSBrowser
    add_os_browser_nodes(driver, database, ti)

    # 2. ProxyType
    add_proxy_nodes(driver, database, ti)

    # 3. PREV_ON_CARD (most expensive)
    if not args.skip_temporal:
        log.info("Loading train_transaction.csv for temporal edges ...")
        tt = pd.read_csv(DATA_DIR / "train_transaction.csv",
                         usecols=["TransactionID", "card1", "TransactionDT"])
        add_temporal_card_edges(driver, database, tt,
                                window_hours=args.window_hours,
                                limit=args.limit)
    else:
        log.info("Skipping PREV_ON_CARD edges (--skip-temporal)")

    # Add index for PREV_ON_CARD traversal performance
    with driver.session(database=database) as session:
        session.run(
            "CREATE INDEX os_browser_idx IF NOT EXISTS FOR (n:OSBrowser) ON (n.osBrowserKey)"
        )
        session.run(
            "CREATE INDEX proxy_type_idx IF NOT EXISTS FOR (n:ProxyType) ON (n.proxyLabel)"
        )
        log.info("Created OSBrowser and ProxyType indexes.")

    run_validation(driver, database)
    driver.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
