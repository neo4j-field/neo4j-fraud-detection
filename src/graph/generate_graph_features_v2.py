"""
Phase 5b — Graph Features V2
==============================
Extends the original graph features with new signals from Graph V2:

New features added:
  - recipient_email_fraud_rate / count (was loaded but never used as a feature)
  - os_browser_fraud_rate / tx_count (OSBrowser node — soft device fingerprint)
  - is_proxy, proxy_fraud_rate (ProxyType node — direct fraud signal)
  - prev_card_is_fraud (1 if previous transaction on same card was fraud)
  - prev_card_dt_gap (seconds since previous transaction on same card)
  - card_chain_fraud_rate (fraction of last-N-same-card-txns that were fraud,
    derived from PREV_ON_CARD chain)

The original 12 entity aggregate features are preserved and recomputed.
WCC component features are dropped (confirmed useless in V1).

Usage:
    python src/graph/generate_graph_features_v2.py
"""

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
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

GDS_GRAPH_NAME = "fraud_graph_v2"


# ---------------------------------------------------------------------------
# Entity-level aggregate features (entity-first pattern)
# ---------------------------------------------------------------------------

def fetch_entity_features(session, entity_label, rel_type, prefix) -> pd.DataFrame:
    """
    Generic entity-first aggregation: compute fraud stats per entity node,
    then join back to each connected transaction.
    """
    log.info(f"Fetching {prefix} features ({entity_label}/{rel_type}) ...")
    t0 = time.time()
    result = session.run(
        f"""
        MATCH (e:{entity_label})<-[:{rel_type}]-(t:Transaction)
        WITH e,
             count(t) AS tx_count,
             sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS fraud_count
        MATCH (e)<-[:{rel_type}]-(t2:Transaction)
        RETURN t2.transactionId AS transactionId,
               tx_count         AS {prefix}_tx_count,
               fraud_count      AS {prefix}_fraud_count,
               toFloat(fraud_count) / tx_count AS {prefix}_fraud_rate
        """
    )
    rows = [dict(r) for r in result]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["transactionId", f"{prefix}_tx_count",
                 f"{prefix}_fraud_count", f"{prefix}_fraud_rate"]
    )
    log.info(f"  {prefix}: {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Proxy features (binary flag + rate)
# ---------------------------------------------------------------------------

def fetch_proxy_features(session) -> pd.DataFrame:
    """
    Returns per-transaction proxy features:
      - is_proxy: 1 if transaction went through any proxy, else 0
      - proxy_fraud_rate: fraud rate of that proxy type
    """
    log.info("Fetching proxy features ...")
    t0 = time.time()
    result = session.run(
        """
        MATCH (p:ProxyType)<-[:VIA_PROXY]-(t:Transaction)
        WITH p,
             count(t) AS px_total,
             sum(CASE WHEN t.isFraud=1 THEN 1 ELSE 0 END) AS px_fraud
        MATCH (p)<-[:VIA_PROXY]-(t2:Transaction)
        RETURN t2.transactionId AS transactionId,
               1 AS is_proxy,
               toFloat(px_fraud) / px_total AS proxy_fraud_rate
        """
    )
    rows = [dict(r) for r in result]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["transactionId", "is_proxy", "proxy_fraud_rate"]
    )
    log.info(f"  Proxy features: {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# PREV_ON_CARD temporal features
# ---------------------------------------------------------------------------

def fetch_temporal_card_features(session) -> pd.DataFrame:
    """
    For each transaction that has a PREV_ON_CARD predecessor:
      - prev_card_is_fraud: 1 if the previous transaction on same card was fraud
      - prev_card_dt_gap: seconds since previous transaction on same card
    """
    log.info("Fetching PREV_ON_CARD temporal features ...")
    t0 = time.time()
    result = session.run(
        """
        MATCH (t_prev:Transaction)-[r:PREV_ON_CARD]->(t_curr:Transaction)
        RETURN t_curr.transactionId AS transactionId,
               t_prev.isFraud       AS prev_card_is_fraud,
               r.dtGapSeconds       AS prev_card_dt_gap
        """
    )
    rows = [dict(r) for r in result]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["transactionId", "prev_card_is_fraud", "prev_card_dt_gap"]
    )
    log.info(f"  Temporal card features: {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# GDS projection and WCC (kept for completeness, but WCC cols dropped later)
# ---------------------------------------------------------------------------

def drop_graph_if_exists(session, name):
    exists = session.run(
        "CALL gds.graph.exists($name) YIELD exists RETURN exists", name=name
    ).single()["exists"]
    if exists:
        session.run("CALL gds.graph.drop($name)", name=name)
        log.info(f"  Dropped existing graph: {name}")


def project_graph_v2(session, name) -> dict:
    """Project richer graph including new V2 nodes and relationships."""
    log.info(f"Projecting GDS graph '{name}' ...")
    result = session.run(
        """
        CALL gds.graph.project(
            $graphName,
            ['Transaction','Card','EmailDomain','BillingAddress','Device',
             'OSBrowser','ProxyType'],
            {
                USED_CARD:       {orientation: 'UNDIRECTED'},
                PAYER_EMAIL:     {orientation: 'UNDIRECTED'},
                RECIPIENT_EMAIL: {orientation: 'UNDIRECTED'},
                BILLED_TO:       {orientation: 'UNDIRECTED'},
                USED_DEVICE:     {orientation: 'UNDIRECTED'},
                HAS_OS_BROWSER:  {orientation: 'UNDIRECTED'},
                VIA_PROXY:       {orientation: 'UNDIRECTED'},
                PREV_ON_CARD:    {orientation: 'UNDIRECTED'}
            }
        )
        YIELD graphName, nodeCount, relationshipCount, projectMillis
        RETURN graphName, nodeCount, relationshipCount, projectMillis
        """,
        graphName=name,
    ).single()
    log.info(
        f"  Projected: {result['nodeCount']:,} nodes, "
        f"{result['relationshipCount']:,} rels ({result['projectMillis']}ms)"
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def combine_features(dfs: list, all_ids: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"transactionId": all_ids})
    for feat_df in dfs:
        if feat_df is not None and len(feat_df) > 0:
            df = df.merge(feat_df, on="transactionId", how="left")

    count_cols = [c for c in df.columns if c.endswith("_count") or c.endswith("_size")]
    rate_cols  = [c for c in df.columns if c.endswith("_rate")]
    flag_cols  = [c for c in df.columns if c in ("is_proxy", "prev_card_is_fraud")]
    gap_cols   = [c for c in df.columns if c == "prev_card_dt_gap"]

    for c in count_cols:
        df[c] = df[c].fillna(0).astype(float)
    for c in rate_cols:
        df[c] = df[c].fillna(-1.0)
    for c in flag_cols:
        df[c] = df[c].fillna(0).astype(float)
    for c in gap_cols:
        # -1 = no prior card transaction in window
        df[c] = df[c].fillna(-1.0)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    log.info(f"Connected to: {uri}")

    # Fetch all transaction IDs and fraud labels
    log.info("Fetching all transaction IDs and fraud labels ...")
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (t:Transaction) RETURN t.transactionId AS transactionId, t.isFraud AS isFraud"
        )
        df_all = pd.DataFrame([dict(r) for r in result])
    log.info(f"  {len(df_all):,} transactions")

    # Fetch all features
    with driver.session(database=database) as session:
        df_card     = fetch_entity_features(session, "Card",           "USED_CARD",       "card")
        df_p_email  = fetch_entity_features(session, "EmailDomain",    "PAYER_EMAIL",     "payer_email")
        df_r_email  = fetch_entity_features(session, "EmailDomain",    "RECIPIENT_EMAIL", "recip_email")
        df_billing  = fetch_entity_features(session, "BillingAddress", "BILLED_TO",       "billing")
        df_device   = fetch_entity_features(session, "Device",         "USED_DEVICE",     "device")
        df_osbrowser= fetch_entity_features(session, "OSBrowser",      "HAS_OS_BROWSER",  "os_browser")
        df_proxy    = fetch_proxy_features(session)
        df_temporal = fetch_temporal_card_features(session)

    # Signal summary
    log.info("\n=== Feature signal (fraud vs legit mean) ===")
    fraud_ids = set(df_all[df_all["isFraud"] == 1]["transactionId"].values)
    df_combined_check = combine_features(
        [df_card, df_p_email, df_r_email, df_billing, df_device,
         df_osbrowser, df_proxy, df_temporal],
        df_all["transactionId"],
    )
    for col in ["card_fraud_rate", "recip_email_fraud_rate", "device_fraud_rate",
                "os_browser_fraud_rate", "proxy_fraud_rate", "is_proxy",
                "prev_card_is_fraud", "prev_card_dt_gap"]:
        if col in df_combined_check.columns:
            f_mean = df_combined_check[df_combined_check["transactionId"].isin(fraud_ids)][col].replace(-1, np.nan).mean()
            l_mean = df_combined_check[~df_combined_check["transactionId"].isin(fraud_ids)][col].replace(-1, np.nan).mean()
            log.info(f"  {col}: fraud={f_mean:.4f} | legit={l_mean:.4f}")

    # Save
    out_path = ARTIFACTS_DIR / "graph_features_v2.parquet"
    df_combined_check.to_parquet(out_path, index=False)
    log.info(f"\nSaved: {out_path}")
    log.info(f"Shape: {df_combined_check.shape}")
    log.info(f"Columns: {list(df_combined_check.columns)}")

    driver.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
